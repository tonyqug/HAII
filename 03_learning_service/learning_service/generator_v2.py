from __future__ import annotations

import copy
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Settings
from .generation import (
    ArtifactValidationError,
    EvidenceAccessor,
    GroundedGenerator as HeuristicGroundedGenerator,
    NeedsUserInputError,
    OptionalGeminiClient,
    _json_text_candidates,
)
from .utils import (
    dedupe_citations,
    distinct_slide_numbers,
    first_sentence,
    infer_concept_label,
    normalize_whitespace,
    safe_excerpt,
    take_sentences,
    utc_now_iso,
    make_id,
)

ALLOWED_SUPPORT_STATUSES = {
    "slide_grounded",
    "inferred_from_slides",
    "external_supplement",
    "insufficient_evidence",
}
ALLOWED_RESPONSE_STYLES = {"concise", "standard", "step_by_step"}
ALLOWED_QUESTION_TYPES = {"short_answer", "long_answer"}
ALLOWED_DIFFICULTIES = {"easier", "mixed", "harder"}

LOGGER = logging.getLogger(__name__)


class GeminiPrimaryClient(OptionalGeminiClient):
    def generate_json(self, system_instruction: str, user_prompt: str, max_output_tokens: int = 2048) -> Optional[Dict[str, Any]]:
        text = self._generate_content(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
        )
        if not text:
            return None
        for candidate in _json_text_candidates(text):
            try:
                payload = json.loads(candidate)
            except ValueError:
                continue
            if isinstance(payload, dict):
                return payload
        self.last_call_info["failure_reason"] = "invalid_response_json"
        self.last_call_info["failure_detail"] = "Gemini returned text that could not be parsed as JSON."
        self.last_call_info["raw_response_preview"] = self.last_call_info.get("raw_response_preview") or text[:400]
        LOGGER.warning("Gemini returned text that could not be parsed as JSON. Preview: %s", text[:400])
        return None


class GroundedGenerator(HeuristicGroundedGenerator):
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.gemini = GeminiPrimaryClient(settings)

    def _record_gemini_failure(self, reason: str, detail: str, *, payload: Any = None) -> None:
        info = copy.deepcopy(getattr(self.gemini, "last_call_info", {}) or {})
        info["failure_reason"] = reason
        info["failure_detail"] = detail
        if payload is not None and not info.get("raw_response_preview"):
            try:
                info["raw_response_preview"] = safe_excerpt(json.dumps(payload, ensure_ascii=False), 400)
            except Exception:
                info["raw_response_preview"] = safe_excerpt(str(payload), 400)
        self.gemini.last_call_info = info

    def _reject_gemini_output(self, artifact_name: str, detail: str, *, payload: Any = None, reason: str = "invalid_response_json_payload") -> None:
        self._record_gemini_failure(reason, detail, payload=payload)
        preview = (getattr(self.gemini, "last_call_info", {}) or {}).get("raw_response_preview")
        if preview:
            LOGGER.warning("Rejecting Gemini %s output: %s Preview: %s", artifact_name, detail, preview)
            return None
        LOGGER.warning("Rejecting Gemini %s output: %s", artifact_name, detail)
        return None

    def _log_gemini_fallback(self, artifact_name: str, fallback_name: str) -> None:
        gemini_failure = copy.deepcopy(getattr(self.gemini, "last_call_info", {}) or {})
        LOGGER.warning(
            "Falling back to %s for %s because Gemini failed with reason=%s detail=%s after models=%s.",
            fallback_name,
            artifact_name,
            gemini_failure.get("failure_reason") or "llm_generation_failed",
            gemini_failure.get("failure_detail"),
            gemini_failure.get("attempted_models") or [],
        )

    # --------------------------
    # Gemini-backed study plans
    # --------------------------
    def build_study_plan(
        self,
        *,
        bundle: Dict[str, Any],
        topic_text: Optional[str],
        time_budget_minutes: int,
        grounding_mode: str,
        student_context: Optional[Dict[str, Any]] = None,
        parent_study_plan_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        artifact = None
        if self.gemini.configured:
            artifact = self._build_study_plan_via_gemini(
                bundle=bundle,
                topic_text=topic_text,
                time_budget_minutes=time_budget_minutes,
                grounding_mode=grounding_mode,
                student_context=student_context or {},
                parent_study_plan_id=parent_study_plan_id,
            )
        if artifact is not None:
            artifact.setdefault("_meta", {})["generation_path"] = "gemini"
            return artifact
        if self.gemini.configured:
            self._log_gemini_fallback("study plan generation", "heuristic fallback")
        artifact = super().build_study_plan(
            bundle=bundle,
            topic_text=topic_text,
            time_budget_minutes=time_budget_minutes,
            grounding_mode=grounding_mode,
            student_context=student_context,
            parent_study_plan_id=parent_study_plan_id,
        )
        artifact.setdefault("_meta", {})["generation_path"] = "heuristic_fallback"
        return artifact

    def _build_study_plan_via_gemini(
        self,
        *,
        bundle: Dict[str, Any],
        topic_text: Optional[str],
        time_budget_minutes: int,
        grounding_mode: str,
        student_context: Dict[str, Any],
        parent_study_plan_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        accessor = EvidenceAccessor(bundle)
        concept_records = accessor.concept_records(accessor.material_ids)
        if not concept_records:
            return None
        citation_index = self._citation_index(bundle)
        normalized_context = self._normalize_student_context(student_context)
        inferred_topic = not normalize_whitespace(topic_text or "")
        resolved_topic = normalize_whitespace(topic_text or accessor.infer_topic(accessor.material_ids))
        ranked_records = self._rank_concept_records_for_plan(
            concept_records,
            resolved_topic,
            normalized_context,
        )
        evidence_records = self._select_study_sequence_records(
            ranked_records,
            max(time_budget_minutes, 90),
        )
        if not evidence_records:
            evidence_records = concept_records[:6]

        evidence_lines: List[str] = []
        for idx, record in enumerate(evidence_records[:8], start=1):
            citation_ids = [citation["citation_id"] for citation in record.get("citations", []) if citation.get("citation_id")]
            evidence_lines.append(
                f"{idx}. concept={record['concept_name']} | slide={record['slide_number']} | citations={citation_ids} | summary={record['summary_text']}"
            )

        prompt = (
            f"Grounding mode: {grounding_mode}\n"
            f"Topic: {resolved_topic}\n"
            f"Time budget minutes: {time_budget_minutes}\n"
            f"Student context: {json.dumps(normalized_context, ensure_ascii=False)}\n"
            f"Allowed citation ids: {sorted(citation_index)}\n"
            "Evidence digest:\n"
            + "\n".join(evidence_lines)
            + "\n\n"
            "Return a JSON object with keys topic_text, prerequisites, study_sequence, common_mistakes, uncertainty.\n"
            "Rules:\n"
            "- Use only citation_ids from the allowed list.\n"
            "- Tailor the plan to the supplied topic, time budget, prior knowledge, weak areas, and goals when those inputs are present.\n"
            "- Produce a sequential checklist-style study plan with meaningful milestones rather than a generic summary.\n"
            "- Include at least 3 prerequisites, at least 1 study_sequence step, and exactly 3 common_mistakes.\n"
            "- Each prerequisite object must have concept_name, why_needed, support_status, citation_ids.\n"
            "- Each study_sequence object must have title, objective, recommended_time_minutes, tasks, milestone, depends_on_prereq_indexes, support_status, citation_ids.\n"
            "- Each common_mistakes object must have pattern, why_it_happens, prevention_advice, support_status, citation_ids.\n"
            "- strict_lecture_only must not use external_supplement.\n"
            "- Do not invent citation ids or unsupported lecture claims.\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You generate grounded study plans from lecture evidence. Return valid JSON only. "
                "Every grounded claim must be tied to allowed citation ids."
            ),
            user_prompt=prompt,
            max_output_tokens=1800,
        )
        if not isinstance(payload, dict):
            return self._reject_gemini_output("study plan", "Expected a top-level JSON object for the study plan response.", payload=payload)

        prerequisites_raw = payload.get("prerequisites") or []
        sequence_raw = payload.get("study_sequence") or []
        mistakes_raw = payload.get("common_mistakes") or []
        if len(prerequisites_raw) < 3 or len(sequence_raw) < 1 or len(mistakes_raw) < 3:
            return self._reject_gemini_output(
                "study plan",
                (
                    "Expected at least 3 prerequisites, 1 study_sequence step, and 3 common_mistakes, "
                    f"but received {len(prerequisites_raw)}, {len(sequence_raw)}, and {len(mistakes_raw)}."
                ),
                payload=payload,
            )

        prerequisites: List[Dict[str, Any]] = []
        prereq_ids: List[str] = []
        for index, raw in enumerate(prerequisites_raw[:5], start=1):
            if not isinstance(raw, dict):
                return self._reject_gemini_output(
                    "study plan",
                    f"Prerequisite #{index} was not a JSON object.",
                    payload=payload,
                )
            citations = self._citation_objects(raw.get("citation_ids"), citation_index)
            raw_status = self._safe_text(raw.get("support_status"))
            if raw_status in {"slide_grounded", "inferred_from_slides"} and not citations:
                return self._reject_gemini_output(
                    "study plan",
                    f"Prerequisite #{index} claimed grounded support but did not reference any allowed citation_ids.",
                    payload=payload,
                )
            status = self._normalize_support_status(raw.get("support_status"), citations, grounding_mode, allow_external=True)
            prereq_id = make_id("prereq")
            prereq_ids.append(prereq_id)
            prerequisites.append(
                {
                    "item_id": prereq_id,
                    "concept_name": self._safe_text(raw.get("concept_name"), fallback="Key prerequisite"),
                    "why_needed": self._safe_text(raw.get("why_needed"), fallback="This concept supports later lecture material."),
                    "support_status": status,
                    "citations": citations,
                }
            )

        study_sequence: List[Dict[str, Any]] = []
        for index, raw in enumerate(sequence_raw[:6], start=1):
            if not isinstance(raw, dict):
                return self._reject_gemini_output(
                    "study plan",
                    f"Study sequence step #{index} was not a JSON object.",
                    payload=payload,
                )
            citations = self._citation_objects(raw.get("citation_ids"), citation_index)
            raw_status = self._safe_text(raw.get("support_status"))
            if raw_status in {"slide_grounded", "inferred_from_slides"} and not citations:
                return self._reject_gemini_output(
                    "study plan",
                    f"Study sequence step #{index} claimed grounded support but did not reference any allowed citation_ids.",
                    payload=payload,
                )
            status = self._normalize_support_status(raw.get("support_status"), citations, grounding_mode, allow_external=True)
            depends_on_indexes = raw.get("depends_on_prereq_indexes") or []
            depends_on = [
                prereq_ids[int(value) - 1]
                for value in depends_on_indexes
                if isinstance(value, int) and 1 <= value <= len(prereq_ids)
            ]
            tasks = [self._safe_text(task) for task in (raw.get("tasks") or []) if self._safe_text(task)]
            if not tasks:
                tasks = ["Review the cited lecture evidence and restate the idea in your own words."]
            minutes = self._bounded_minutes(raw.get("recommended_time_minutes"), time_budget_minutes, len(sequence_raw[:6]))
            study_sequence.append(
                {
                    "step_id": make_id("step"),
                    "order_index": index,
                    "title": self._safe_text(raw.get("title"), fallback=f"Study step {index}"),
                    "objective": self._safe_text(raw.get("objective"), fallback="Review the grounded lecture idea and connect it to earlier material."),
                    "recommended_time_minutes": minutes,
                    "tasks": tasks,
                    "milestone": self._safe_text(
                        raw.get("milestone"),
                        fallback=self._step_milestone(
                            self._safe_text(raw.get("title"), fallback=f"study step {index}"),
                            resolved_topic,
                            normalized_context.get("goals", ""),
                        ),
                    ),
                    "depends_on": depends_on,
                    "support_status": status,
                    "citations": citations,
                }
            )

        common_mistakes: List[Dict[str, Any]] = []
        for index, raw in enumerate(mistakes_raw[:3], start=1):
            if not isinstance(raw, dict):
                return self._reject_gemini_output(
                    "study plan",
                    f"Common mistake #{index} was not a JSON object.",
                    payload=payload,
                )
            citations = self._citation_objects(raw.get("citation_ids"), citation_index)
            raw_status = self._safe_text(raw.get("support_status"))
            if raw_status in {"slide_grounded", "inferred_from_slides"} and not citations:
                return self._reject_gemini_output(
                    "study plan",
                    f"Common mistake #{index} claimed grounded support but did not reference any allowed citation_ids.",
                    payload=payload,
                )
            status = self._normalize_support_status(raw.get("support_status"), citations, grounding_mode, allow_external=True)
            common_mistakes.append(
                {
                    "item_id": make_id("mistake"),
                    "pattern": self._safe_text(raw.get("pattern"), fallback="Potential misunderstanding from the lecture material"),
                    "why_it_happens": self._safe_text(raw.get("why_it_happens"), fallback="Students can lose the connection between the lecture idea and its conditions or examples."),
                    "prevention_advice": self._safe_text(raw.get("prevention_advice"), fallback="Check the cited lecture slides and restate the distinction in your own words."),
                    "support_status": status,
                    "citations": citations,
                }
            )

        uncertainty = self._sanitize_uncertainty(payload.get("uncertainty"))
        if inferred_topic:
            uncertainty.insert(
                0,
                {
                    "code": "topic_inferred",
                    "message": "No explicit topic_text was provided, so the service inferred the study topic from the grounded lecture evidence.",
                },
            )
        if accessor.summary["low_confidence_item_count"] > 0 and not any(note.get("code") == "low_confidence_evidence" for note in uncertainty):
            uncertainty.append(
                {
                    "code": "low_confidence_evidence",
                    "message": "Some evidence items were marked low confidence, so treat their related plan items with extra care.",
                }
            )

        all_citations: List[Dict[str, Any]] = []
        for section in prerequisites + study_sequence + common_mistakes:
            all_citations.extend(section.get("citations") or [])
        cited_slides = distinct_slide_numbers(all_citations)
        if not cited_slides and concept_records:
            return self._reject_gemini_output(
                "study plan",
                "The study plan response did not preserve any grounded slide citations.",
                payload=payload,
            )
        omitted_or_low = sorted(
            set(accessor.distinct_slide_numbers()) - set(cited_slides)
            | set(accessor.low_confidence_slides())
        )
        tailoring_summary = self._build_tailoring_summary(
            accessor=accessor,
            topic_text=resolved_topic,
            inferred_topic=inferred_topic,
            time_budget_minutes=time_budget_minutes,
            grounding_mode=grounding_mode,
            student_context=normalized_context,
            citations=all_citations,
        )
        return {
            "study_plan_id": make_id("study_plan"),
            "parent_study_plan_id": parent_study_plan_id,
            "workspace_id": bundle.get("workspace_id"),
            "created_at": utc_now_iso(),
            "topic_text": self._safe_text(payload.get("topic_text"), fallback=resolved_topic),
            "time_budget_minutes": time_budget_minutes,
            "grounding_mode": grounding_mode,
            "prerequisites": prerequisites,
            "study_sequence": study_sequence,
            "common_mistakes": common_mistakes,
            "uncertainty": uncertainty,
            "coverage_summary": {
                "cited_slides": cited_slides,
                "omitted_or_low_confidence_slides": omitted_or_low,
            },
            "tailoring_summary": tailoring_summary,
            "_meta": {
                "grounding_source": self._grounding_source_from_bundle(bundle),
                "student_context": normalized_context,
            },
        }

    # --------------------------
    # Gemini-backed chat
    # --------------------------
    def build_conversation_reply(
        self,
        *,
        bundle: Dict[str, Any],
        message_text: str,
        response_style: str,
        grounding_mode: str,
        previous_messages: Sequence[Dict[str, Any]],
        conversation_id: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        accessor = EvidenceAccessor(bundle)
        lecture_material_ids = accessor.material_ids
        normalized_message = normalize_whitespace(message_text)
        query = self._resolve_query_context(normalized_message, previous_messages, accessor)
        retrieval_query = normalize_whitespace(bundle.get("query_text") or query)
        relevant_items = self._select_chat_evidence(accessor, query, lecture_material_ids, retrieval_query=retrieval_query)

        if self.gemini.configured:
            reply = self._build_conversation_reply_via_gemini(
                bundle=bundle,
                query=query,
                retrieval_query=retrieval_query,
                normalized_message=normalized_message,
                response_style=response_style,
                grounding_mode=grounding_mode,
                previous_messages=previous_messages,
                relevant_items=relevant_items,
            )
            if reply is not None:
                return reply
        gemini_failure = copy.deepcopy(getattr(self.gemini, "last_call_info", {}) or {})
        LOGGER.warning(
            "Falling back to deterministic chat generation for query=%r because Gemini failed with reason=%s detail=%s after models=%s.",
            normalized_message[:120],
            gemini_failure.get("failure_reason") or self._default_chat_fallback_reason(),
            gemini_failure.get("failure_detail"),
            gemini_failure.get("attempted_models") or [],
        )
        return self._with_conversation_answer_source(
            super().build_conversation_reply(
                bundle=bundle,
                message_text=message_text,
                response_style=response_style,
                grounding_mode=grounding_mode,
                previous_messages=previous_messages,
                conversation_id=conversation_id,
            ),
            generation_path="heuristic_fallback",
            query=query,
            relevant_items=relevant_items,
            llm_call_info=gemini_failure,
            fallback_reason=(gemini_failure.get("failure_reason") or self._default_chat_fallback_reason()),
        )

    def _build_conversation_reply_via_gemini(
        self,
        *,
        bundle: Dict[str, Any],
        query: str,
        retrieval_query: str,
        normalized_message: str,
        response_style: str,
        grounding_mode: str,
        previous_messages: Sequence[Dict[str, Any]],
        relevant_items: Sequence[Dict[str, Any]],
    ) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        citation_index = self._citation_index(bundle)
        allowed_citation_ids = [item.get("citation", {}).get("citation_id") for item in relevant_items if (item.get("citation") or {}).get("citation_id")]
        evidence_lines = []
        for item in relevant_items:
            citation = item.get("citation") or {}
            evidence_lines.append(
                f"citation_id={citation.get('citation_id')} | slide={item.get('slide_number')} | title={item.get('material_title')} | text={item.get('text')}"
            )
        if not evidence_lines:
            evidence_lines.append("No matched lecture evidence was retrieved for this question.")
        recent_user_turns = [msg.get("text", "") for msg in previous_messages if msg.get("role") == "user" and msg.get("text")][-2:]
        prompt = (
            f"Grounding mode: {grounding_mode}\n"
            f"Response style: {response_style if response_style in ALLOWED_RESPONSE_STYLES else 'standard'}\n"
            f"Question: {normalized_message}\n"
            f"Retrieval query assist: {retrieval_query}\n"
            f"Recent user turns: {recent_user_turns}\n"
            f"Allowed citation ids: {allowed_citation_ids}\n"
            "Evidence digest:\n"
            + "\n".join(evidence_lines)
            + "\n\nReturn JSON with keys reply_sections and optional clarifying_question.\n"
            "Each reply section must have heading, text, support_status, citation_ids.\n"
            "Grounded sections must cite allowed citation ids. External supplement sections must have no citation ids.\n"
            "strict_lecture_only must not include external_supplement sections.\n"
            "Every grounded section must paraphrase the evidence into fresh wording. Do not quote or closely mirror the evidence digest.\n"
            "CRITICAL: First check whether the evidence actually addresses the question. "
            "If the evidence is genuinely unrelated, return a single section with support_status='insufficient_evidence', empty citation_ids=[], and text explaining the mismatch. "
            "If the evidence is close but only partially answers the question, give the best grounded partial answer first, explicitly name the limit of what the slides show, and use external_supplement only for a short clearly labeled bridge when grounding_mode allows it. "
            "Do NOT summarize unrelated evidence to answer a question it does not address.\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You answer study questions grounded in lecture evidence. Return valid JSON only. "
                "Do not use unsupported lecture claims or unsupported citation ids. "
                "Use insufficient_evidence only when the evidence is truly unrelated or too thin even for a partial lecture-grounded explanation. "
                "When the slides provide a nearby mechanism or intuition, answer with that grounded connection before naming the missing detail. "
                "Always paraphrase the evidence into your own words instead of repeating source phrasing."
            ),
            user_prompt=prompt,
            max_output_tokens=1000,
        )
        llm_call_info = copy.deepcopy(getattr(self.gemini, "last_call_info", {}) or {})
        if not isinstance(payload, dict):
            return self._reject_gemini_output("chat reply", "Expected a top-level JSON object for the chat response.", payload=payload)

        sections = []
        for index, raw in enumerate(payload.get("reply_sections") or [], start=1):
            if not isinstance(raw, dict):
                return self._reject_gemini_output(
                    "chat reply",
                    f"Reply section #{index} was not a JSON object.",
                    payload=payload,
                )
            citations = self._citation_objects(raw.get("citation_ids"), citation_index, allowed_ids=set(allowed_citation_ids))
            raw_status = self._safe_text(raw.get("support_status"))
            if raw_status in {"slide_grounded", "inferred_from_slides"} and not citations:
                return self._reject_gemini_output(
                    "chat reply",
                    f"Reply section #{index} claimed grounded support but did not reference any allowed citation_ids.",
                    payload=payload,
                )
            status = self._normalize_support_status(raw.get("support_status"), citations, grounding_mode, allow_external=True)
            if status == "external_supplement" and grounding_mode == "strict_lecture_only":
                return self._reject_gemini_output(
                    "chat reply",
                    f"Reply section #{index} returned external_supplement in strict_lecture_only mode.",
                    payload=payload,
                )
            section = {
                "heading": self._safe_text(raw.get("heading"), fallback="Grounded answer" if citations else "Answer"),
                "text": self._safe_text(raw.get("text"), fallback="I could not form a reliable grounded answer from the provided evidence."),
                "support_status": status,
                "citations": citations,
            }
            if status in {"slide_grounded", "inferred_from_slides"} and self._repeats_source_material(section["text"], relevant_items):
                LOGGER.warning(
                    "Rejecting Gemini chat reply because it repeated source material directly for query=%r.",
                    normalized_message[:120],
                )
                return self._reject_gemini_output(
                    "chat reply",
                    f"Reply section #{index} repeated source material too closely instead of paraphrasing it.",
                    payload=payload,
                    reason="verbatim_evidence_repetition",
                )
            sections.append(section)

        if not sections:
            return self._reject_gemini_output("chat reply", "No reply_sections were returned.", payload=payload)
        if not any(section["support_status"] in {"slide_grounded", "inferred_from_slides", "insufficient_evidence"} for section in sections):
            return self._reject_gemini_output(
                "chat reply",
                "The chat response did not include any grounded or insufficient_evidence section.",
                payload=payload,
            )
        if len(sections) == 1 and sections[0]["support_status"] == "insufficient_evidence":
            partial_bridge = self._question_specific_partial_answer(normalized_message, relevant_items)
            if partial_bridge:
                return self._reject_gemini_output(
                    "chat reply",
                    "Gemini returned insufficient_evidence even though the retrieved lecture evidence supports a partial grounded explanation.",
                    payload=payload,
                    reason="over_strict_insufficient_evidence",
                )

        if grounding_mode == "lecture_with_fallback" and any(
            section["support_status"] in {"slide_grounded", "inferred_from_slides"} for section in sections
        ) and self._needs_external_supplement(normalized_message, relevant_items) and not any(
            section["support_status"] == "external_supplement" for section in sections
        ):
            grounded_text = " ".join(section["text"] for section in sections if section["support_status"] in {"slide_grounded", "inferred_from_slides"})
            sections.append(
                {
                    "heading": "External supplement",
                    "text": self._external_supplement_text(normalized_message, grounded_text),
                    "support_status": "external_supplement",
                    "citations": [],
                }
            )

        user_message = {
            "message_id": make_id("msg"),
            "role": "user",
            "created_at": utc_now_iso(),
            "text": normalized_message,
        }
        assistant_message: Dict[str, Any] = {
            "message_id": make_id("msg"),
            "role": "assistant",
            "created_at": utc_now_iso(),
            "reply_sections": sections,
        }
        clarifying = payload.get("clarifying_question")
        if isinstance(clarifying, dict) and (clarifying.get("prompt") or clarifying.get("reason")):
            assistant_message["clarifying_question"] = {
                "prompt": self._safe_text(clarifying.get("prompt"), fallback="Please ask a more specific question."),
                "reason": self._safe_text(clarifying.get("reason"), fallback="The question would benefit from a narrower grounded target."),
            }
        LOGGER.info(
            "Gemini chat reply accepted for query=%r using model=%s with %s section(s).",
            normalized_message[:120],
            llm_call_info.get("used_model"),
            len(sections),
        )
        return self._with_conversation_answer_source(
            (user_message, assistant_message),
            generation_path="llm",
            query=query,
            relevant_items=relevant_items,
            llm_call_info=llm_call_info,
        )

    # --------------------------
    # Gemini-backed practice sets
    # --------------------------
    def build_practice_set(
        self,
        *,
        bundle: Dict[str, Any],
        topic_text: Optional[str],
        generation_mode: str,
        template_material_id: Optional[str],
        question_count: int,
        coverage_mode: str,
        difficulty_profile: str,
        include_answer_key: bool,
        include_rubrics: bool,
        grounding_mode: str,
        parent_practice_set_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        artifact = None
        if self.gemini.configured:
            artifact = self._build_practice_set_via_gemini(
                bundle=bundle,
                topic_text=topic_text,
                generation_mode=generation_mode,
                template_material_id=template_material_id,
                question_count=question_count,
                coverage_mode=coverage_mode,
                difficulty_profile=difficulty_profile,
                include_answer_key=include_answer_key,
                include_rubrics=include_rubrics,
                grounding_mode=grounding_mode,
                parent_practice_set_id=parent_practice_set_id,
            )
        if artifact is not None:
            artifact.setdefault("_meta", {})["generation_path"] = "gemini"
            return artifact
        if self.gemini.configured:
            self._log_gemini_fallback("practice set generation", "heuristic fallback")
        artifact = super().build_practice_set(
            bundle=bundle,
            topic_text=topic_text,
            generation_mode=generation_mode,
            template_material_id=template_material_id,
            question_count=question_count,
            coverage_mode=coverage_mode,
            difficulty_profile=difficulty_profile,
            include_answer_key=include_answer_key,
            include_rubrics=include_rubrics,
            grounding_mode=grounding_mode,
            parent_practice_set_id=parent_practice_set_id,
        )
        artifact.setdefault("_meta", {})["generation_path"] = "heuristic_fallback"
        return artifact

    def _build_practice_set_via_gemini(
        self,
        *,
        bundle: Dict[str, Any],
        topic_text: Optional[str],
        generation_mode: str,
        template_material_id: Optional[str],
        question_count: int,
        coverage_mode: str,
        difficulty_profile: str,
        include_answer_key: bool,
        include_rubrics: bool,
        grounding_mode: str,
        parent_practice_set_id: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        accessor = EvidenceAccessor(bundle)
        lecture_material_ids = accessor.lecture_material_ids(template_material_id)
        lecture_records = accessor.concept_records(lecture_material_ids)
        if not lecture_records:
            return None

        template_style_summary = None
        template_verbs: List[str] = []
        if generation_mode == "template_mimic":
            template_items = accessor.filter_items([template_material_id])
            if not template_items:
                return None
            template_style_summary, template_verbs = self._analyze_template_style(template_items)
        selected_records = self._select_records_for_questions(lecture_records, question_count, coverage_mode)
        if not selected_records:
            return None

        citation_index = self._citation_index(bundle)
        evidence_lines = []
        for idx, record in enumerate(selected_records, start=1):
            citation_ids = [citation["citation_id"] for citation in record.get("citations", []) if citation.get("citation_id")]
            evidence_lines.append(
                f"{idx}. concept={record['concept_name']} | slide={record['slide_number']} | citations={citation_ids} | summary={record['summary_text']}"
            )

        prompt = (
            f"Topic focus: {topic_text or 'all ready grounded materials'}\n"
            f"Generation mode: {generation_mode}\n"
            f"Coverage mode: {coverage_mode}\n"
            f"Difficulty profile: {difficulty_profile}\n"
            f"Question count: {question_count}\n"
            f"Grounding mode: {grounding_mode}\n"
            f"Include answer key: {include_answer_key}\n"
            f"Include rubrics: {include_rubrics}\n"
            f"Template style summary: {template_style_summary or ''}\n"
            f"Template verbs: {template_verbs}\n"
            f"Allowed citation ids: {sorted(citation_index)}\n"
            "Evidence digest:\n"
            + "\n".join(evidence_lines)
            + "\n\nReturn JSON with keys template_style_summary and questions.\n"
            "Each question must have question_type, stem, expected_answer, citation_ids, and optional difficulty, rubric, scoring_guide_text, answer_choices, estimated_minutes.\n"
            "Use only allowed citation ids. Every question must cite lecture evidence.\n"
            "Questions must feel like challenging exam questions, not study-guide prompts.\n"
            "Prefer comparison, diagnosis, trade-offs, failure modes, application, or justification over simple definition recall.\n"
            "Do not copy slide or study-guide wording directly into stems, answer choices, or expected answers; paraphrase into a consistent exam voice.\n"
            "For multiple choice, generate four options that are stylistically parallel, similar in length, and equally plausible at a glance.\n"
            "Do not use equations, variable-style symbols, quotation marks, slide headers, or visibly source-like formatting in only one option.\n"
            "Do not make the correct option stand out by specificity, formatting, or length.\n"
            "Expected answers should read like concise evaluator notes, not pasted lecture prose.\n"
            "template_mimic may borrow style only, not template topic content outside the lecture evidence.\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You create grounded practice questions from lecture evidence. Return valid JSON only. "
                "Every question must be supported by allowed citation ids. "
                "Write serious, high-difficulty assessment items when the request asks for harder questions."
            ),
            user_prompt=prompt,
            max_output_tokens=2200,
        )
        if not isinstance(payload, dict):
            return self._reject_gemini_output("practice set", "Expected a top-level JSON object for the practice set response.", payload=payload)

        raw_questions = payload.get("questions") or []
        if len(raw_questions) < question_count:
            return self._reject_gemini_output(
                "practice set",
                f"Expected at least {question_count} questions, but received {len(raw_questions)}.",
                payload=payload,
            )
        expected_types = self._question_types_for_mode(generation_mode, question_count, template_items_present=bool(template_material_id))
        questions = []
        for index, raw in enumerate(raw_questions[:question_count], start=1):
            if not isinstance(raw, dict):
                return self._reject_gemini_output(
                    "practice set",
                    f"Question #{index} was not a JSON object.",
                    payload=payload,
                )
            citations = self._citation_objects(raw.get("citation_ids"), citation_index)
            if not citations:
                return self._reject_gemini_output(
                    "practice set",
                    f"Question #{index} did not reference any allowed citation_ids.",
                    payload=payload,
                )
            question_type = raw.get("question_type")
            if question_type not in ALLOWED_QUESTION_TYPES:
                question_type = expected_types[index - 1]
            elif generation_mode == "short_answer":
                question_type = "short_answer"
            elif generation_mode == "long_answer":
                question_type = "long_answer"
            difficulty = raw.get("difficulty")
            if difficulty not in ALLOWED_DIFFICULTIES:
                difficulty = difficulty_profile
            concept = infer_concept_label(first_sentence(raw.get("expected_answer") or raw.get("stem") or ""), fallback=f"Question {index}")
            rubric = []
            if include_rubrics:
                rubric = self._sanitize_rubric(raw.get("rubric")) or self._rubric_for_question(question_type, concept, True, difficulty)
            scoring_guide = self._safe_text(raw.get("scoring_guide_text"))
            answer_choices: List[str] = []
            expected_answer = self._safe_text(raw.get("expected_answer"), fallback="") if include_answer_key else ""
            if question_type == "multiple_choice":
                answer_choices = self._sanitize_answer_choices(raw.get("answer_choices"))
                if len(answer_choices) != 4:
                    return self._reject_gemini_output(
                        "practice set",
                        f"Multiple-choice question #{index} did not produce exactly four usable answer choices.",
                        payload=payload,
                    )
                if not scoring_guide:
                    scoring_guide = "Award credit only for the option that best matches the lecture-grounded reasoning, not the most generic-sounding statement."
            if question_type == "long_answer" and not scoring_guide:
                scoring_guide = (
                    f"Full credit requires a correct explanation of {concept}, a grounded connection to the lecture context, "
                    "and one clearly stated caution or application detail."
                )
            questions.append(
                {
                    "question_id": make_id("question"),
                    "question_type": question_type,
                    "stem": self._safe_text(raw.get("stem"), fallback=f"Explain {concept} using the grounded lecture evidence."),
                    "expected_answer": expected_answer,
                    "answer_choices": answer_choices,
                    "rubric": rubric,
                    "scoring_guide_text": scoring_guide,
                    "citations": citations,
                    "covered_slides": distinct_slide_numbers(citations),
                    "difficulty": difficulty,
                    "estimated_minutes": self._coerce_estimated_minutes(raw.get("estimated_minutes"), question_type, difficulty),
                }
            )

        cited_slides = sorted({slide for question in questions for slide in question.get("covered_slides", [])})
        considered_slides = accessor.distinct_slide_numbers(lecture_material_ids)
        uncited_or_skipped = sorted(set(considered_slides) - set(cited_slides))
        notes = self._coverage_notes(coverage_mode, considered_slides, cited_slides, question_count, topic_text or "")
        return {
            "practice_set_id": make_id("practice_set"),
            "parent_practice_set_id": parent_practice_set_id,
            "workspace_id": bundle.get("workspace_id"),
            "created_at": utc_now_iso(),
            "topic_text": topic_text or "",
            "generation_mode": generation_mode,
            "template_style_summary": self._safe_text(payload.get("template_style_summary"), fallback=template_style_summary),
            "estimated_duration_minutes": sum(int(question.get("estimated_minutes") or 0) for question in questions),
            "questions": questions,
            "coverage_report": {
                "considered_slide_count": len(considered_slides),
                "cited_slide_count": len(cited_slides),
                "uncited_or_skipped_slides": uncited_or_skipped,
                "notes": notes,
            },
            "human_loop_summary": {
                "used_inputs": [
                    {"label": "Topic focus", "value": topic_text or "All ready grounded materials"},
                    {"label": "Question format", "value": generation_mode.replace("_", " ")},
                    {"label": "Question count", "value": str(question_count)},
                    {"label": "Coverage mode", "value": coverage_mode.replace("_", " ")},
                    {"label": "Difficulty", "value": difficulty_profile},
                    {"label": "Answer key", "value": "Included" if include_answer_key else "Hidden"},
                    {"label": "Rubrics", "value": "Included" if include_rubrics else "Hidden"},
                ],
                "follow_up_actions": [
                    "Lock the strongest questions before revising.",
                    "Regenerate only the questions that feel unclear or off-target.",
                    "Narrow the topic if the current set is too broad.",
                ],
            },
            "_meta": {
                "grounding_source": self._grounding_source_from_bundle(bundle),
                "template_material_id": template_material_id,
                "coverage_mode": coverage_mode,
                "difficulty_profile": difficulty_profile,
                "include_answer_key": include_answer_key,
                "include_rubrics": include_rubrics,
                "grounding_mode": grounding_mode,
            },
        }

    # --------------------------
    # Gemini-backed revisions
    # --------------------------
    def revise_study_plan(
        self,
        *,
        existing_plan: Dict[str, Any],
        instruction_text: str,
        target_section: str,
        locked_item_ids: Sequence[str],
        grounding_mode: str,
    ) -> Dict[str, Any]:
        if self.gemini.configured:
            revised = self._revise_study_plan_via_gemini(
                existing_plan=existing_plan,
                instruction_text=instruction_text,
                target_section=target_section,
                locked_item_ids=locked_item_ids,
                grounding_mode=grounding_mode,
            )
            if revised is not None:
                revised.setdefault("_meta", {})["generation_path"] = "gemini"
                return revised
            self._log_gemini_fallback("study plan revision", "heuristic fallback")
        revised = super().revise_study_plan(
            existing_plan=existing_plan,
            instruction_text=instruction_text,
            target_section=target_section,
            locked_item_ids=locked_item_ids,
            grounding_mode=grounding_mode,
        )
        revised.setdefault("_meta", {})["generation_path"] = "heuristic_fallback"
        return revised

    def _revise_study_plan_via_gemini(
        self,
        *,
        existing_plan: Dict[str, Any],
        instruction_text: str,
        target_section: str,
        locked_item_ids: Sequence[str],
        grounding_mode: str,
    ) -> Optional[Dict[str, Any]]:
        plan = copy.deepcopy(existing_plan)
        locked = set(locked_item_ids)
        all_item_ids = {
            item["item_id"] for item in plan.get("prerequisites", []) + plan.get("common_mistakes", [])
        } | {step["step_id"] for step in plan.get("study_sequence", [])}
        if locked - all_item_ids:
            return None
        target_item_id = None
        if target_section not in {"entire_plan", "prerequisites", "study_sequence", "common_mistakes"}:
            target_item_id = target_section
            if target_item_id not in all_item_ids:
                return None

        editable_prereqs = [
            {"item_id": item["item_id"], "concept_name": item["concept_name"], "why_needed": item["why_needed"]}
            for item in plan.get("prerequisites", [])
            if self._study_item_should_modify("prerequisites", item["item_id"], locked, target_section, target_item_id)
        ]
        editable_steps = [
            {
                "step_id": step["step_id"],
                "title": step["title"],
                "objective": step["objective"],
                "recommended_time_minutes": step["recommended_time_minutes"],
                "tasks": step.get("tasks", []),
            }
            for step in plan.get("study_sequence", [])
            if self._study_item_should_modify("study_sequence", step["step_id"], locked, target_section, target_item_id)
        ]
        editable_mistakes = [
            {
                "item_id": item["item_id"],
                "pattern": item["pattern"],
                "why_it_happens": item["why_it_happens"],
                "prevention_advice": item["prevention_advice"],
            }
            for item in plan.get("common_mistakes", [])
            if self._study_item_should_modify("common_mistakes", item["item_id"], locked, target_section, target_item_id)
        ]
        if not editable_prereqs and not editable_steps and not editable_mistakes:
            return None

        prompt = (
            f"Instruction: {instruction_text}\n"
            f"Grounding mode: {grounding_mode}\n"
            f"Target section: {target_section}\n"
            f"Locked item ids: {sorted(locked)}\n"
            "Return JSON with optional keys prerequisites, study_sequence, common_mistakes.\n"
            "Only include editable items. Preserve the same item ids in your response. Do not include locked items.\n"
            f"Editable prerequisites: {json.dumps(editable_prereqs, ensure_ascii=False)}\n"
            f"Editable study_sequence: {json.dumps(editable_steps, ensure_ascii=False)}\n"
            f"Editable common_mistakes: {json.dumps(editable_mistakes, ensure_ascii=False)}\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You revise grounded study-plan text after user feedback. Return valid JSON only. "
                "Do not change ids, citations, or locked items."
            ),
            user_prompt=prompt,
            max_output_tokens=1400,
        )
        if not isinstance(payload, dict):
            return self._reject_gemini_output("study plan revision", "Expected a top-level JSON object for the study plan revision response.", payload=payload)

        updated = False
        prereq_updates = {item.get("item_id"): item for item in (payload.get("prerequisites") or []) if isinstance(item, dict)}
        step_updates = {item.get("step_id"): item for item in (payload.get("study_sequence") or []) if isinstance(item, dict)}
        mistake_updates = {item.get("item_id"): item for item in (payload.get("common_mistakes") or []) if isinstance(item, dict)}
        if any(item_id in locked for item_id in prereq_updates) or any(item_id in locked for item_id in step_updates) or any(item_id in locked for item_id in mistake_updates):
            return self._reject_gemini_output(
                "study plan revision",
                "The revision response attempted to modify one or more locked study-plan items.",
                payload=payload,
            )

        for item in plan.get("prerequisites", []):
            raw = prereq_updates.get(item["item_id"])
            if raw:
                item["concept_name"] = self._safe_text(raw.get("concept_name"), fallback=item["concept_name"])
                item["why_needed"] = self._safe_text(raw.get("why_needed"), fallback=item["why_needed"])
                updated = True
        for step in plan.get("study_sequence", []):
            raw = step_updates.get(step["step_id"])
            if raw:
                step["title"] = self._safe_text(raw.get("title"), fallback=step["title"])
                step["objective"] = self._safe_text(raw.get("objective"), fallback=step["objective"])
                tasks = [self._safe_text(task) for task in (raw.get("tasks") or []) if self._safe_text(task)]
                if tasks:
                    step["tasks"] = tasks
                step["recommended_time_minutes"] = self._bounded_minutes(raw.get("recommended_time_minutes"), 240, len(plan.get("study_sequence") or [1]))
                updated = True
        for item in plan.get("common_mistakes", []):
            raw = mistake_updates.get(item["item_id"])
            if raw:
                item["pattern"] = self._safe_text(raw.get("pattern"), fallback=item["pattern"])
                item["why_it_happens"] = self._safe_text(raw.get("why_it_happens"), fallback=item["why_it_happens"])
                item["prevention_advice"] = self._safe_text(raw.get("prevention_advice"), fallback=item["prevention_advice"])
                updated = True
        if not updated:
            return self._reject_gemini_output(
                "study plan revision",
                "The revision response did not modify any editable study-plan items.",
                payload=payload,
            )
        revised = {
            **plan,
            "study_plan_id": make_id("study_plan"),
            "parent_study_plan_id": existing_plan.get("study_plan_id"),
            "created_at": utc_now_iso(),
            "grounding_mode": grounding_mode,
        }
        revised.setdefault("uncertainty", []).append(
            {
                "code": "revised_after_feedback",
                "message": "This study plan version was created from a prior version after user feedback.",
            }
        )
        return revised

    def revise_practice_set(
        self,
        *,
        existing_practice_set: Dict[str, Any],
        instruction_text: str,
        target_question_ids: Sequence[str],
        locked_question_ids: Sequence[str],
        maintain_coverage: bool,
    ) -> Dict[str, Any]:
        if self.gemini.configured:
            revised = self._revise_practice_set_via_gemini(
                existing_practice_set=existing_practice_set,
                instruction_text=instruction_text,
                target_question_ids=target_question_ids,
                locked_question_ids=locked_question_ids,
                maintain_coverage=maintain_coverage,
            )
            if revised is not None:
                revised.setdefault("_meta", {})["generation_path"] = "gemini"
                return revised
            self._log_gemini_fallback("practice set revision", "heuristic fallback")
        revised = super().revise_practice_set(
            existing_practice_set=existing_practice_set,
            instruction_text=instruction_text,
            target_question_ids=target_question_ids,
            locked_question_ids=locked_question_ids,
            maintain_coverage=maintain_coverage,
        )
        revised.setdefault("_meta", {})["generation_path"] = "heuristic_fallback"
        return revised

    def _revise_practice_set_via_gemini(
        self,
        *,
        existing_practice_set: Dict[str, Any],
        instruction_text: str,
        target_question_ids: Sequence[str],
        locked_question_ids: Sequence[str],
        maintain_coverage: bool,
    ) -> Optional[Dict[str, Any]]:
        practice_set = copy.deepcopy(existing_practice_set)
        questions = practice_set.get("questions", [])
        question_ids = {question["question_id"] for question in questions}
        locked = set(locked_question_ids)
        if locked - question_ids:
            return None
        targets = set(target_question_ids) if target_question_ids else set(question_ids)
        if targets - question_ids:
            return None
        editable = [
            {
                "question_id": question["question_id"],
                "question_type": question["question_type"],
                "stem": question["stem"],
                "expected_answer": question.get("expected_answer", ""),
                "difficulty": question.get("difficulty"),
                "rubric": question.get("rubric", []),
                "scoring_guide_text": question.get("scoring_guide_text"),
            }
            for question in questions
            if question["question_id"] in targets and question["question_id"] not in locked
        ]
        if not editable:
            return None
        prompt = (
            f"Instruction: {instruction_text}\n"
            f"Maintain coverage: {maintain_coverage}\n"
            f"Locked question ids: {sorted(locked)}\n"
            "Return JSON with key questions. Each question must preserve its question_id. Do not include locked questions.\n"
            f"Editable questions: {json.dumps(editable, ensure_ascii=False)}\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You revise grounded practice questions after user feedback. Return valid JSON only. "
                "Do not change question ids or invent unsupported coverage."
            ),
            user_prompt=prompt,
            max_output_tokens=1600,
        )
        if not isinstance(payload, dict):
            return self._reject_gemini_output("practice set revision", "Expected a top-level JSON object for the practice set revision response.", payload=payload)
        updates = {item.get("question_id"): item for item in (payload.get("questions") or []) if isinstance(item, dict)}
        if any(question_id in locked for question_id in updates):
            return self._reject_gemini_output(
                "practice set revision",
                "The revision response attempted to modify one or more locked questions.",
                payload=payload,
            )
        updated = False
        for question in questions:
            raw = updates.get(question["question_id"])
            if not raw:
                continue
            question["stem"] = self._safe_text(raw.get("stem"), fallback=question["stem"])
            question["expected_answer"] = self._safe_text(raw.get("expected_answer"), fallback=question.get("expected_answer", ""))
            if raw.get("question_type") in ALLOWED_QUESTION_TYPES:
                question["question_type"] = raw["question_type"]
            if raw.get("difficulty") in ALLOWED_DIFFICULTIES:
                question["difficulty"] = raw["difficulty"]
            rubric = self._sanitize_rubric(raw.get("rubric"))
            if rubric:
                question["rubric"] = rubric
            scoring_guide_text = self._safe_text(raw.get("scoring_guide_text"))
            if scoring_guide_text:
                question["scoring_guide_text"] = scoring_guide_text
            if maintain_coverage:
                question["covered_slides"] = list(question.get("covered_slides", []))
                question["citations"] = copy.deepcopy(question.get("citations", []))
            updated = True
        if not updated:
            return self._reject_gemini_output(
                "practice set revision",
                "The revision response did not modify any editable questions.",
                payload=payload,
            )
        revised = {
            **practice_set,
            "practice_set_id": make_id("practice_set"),
            "parent_practice_set_id": existing_practice_set.get("practice_set_id"),
            "created_at": utc_now_iso(),
            "questions": questions,
        }
        notes = revised["coverage_report"].get("notes", "")
        revised["coverage_report"]["notes"] = f"{notes} Revised after feedback while preserving stored history.".strip()
        return revised

    # --------------------------
    # Validation helpers
    # --------------------------
    def _citation_index(self, bundle: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for item in bundle.get("items") or []:
            citation = item.get("citation") or {}
            citation_id = citation.get("citation_id")
            if citation_id and citation_id not in index:
                index[citation_id] = citation
        return index

    def _citation_objects(
        self,
        citation_ids: Any,
        citation_index: Dict[str, Dict[str, Any]],
        allowed_ids: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in citation_ids or []:
            if not isinstance(raw, str):
                continue
            if allowed_ids is not None and raw not in allowed_ids:
                continue
            citation = citation_index.get(raw)
            if not citation or raw in seen:
                continue
            seen.add(raw)
            output.append(copy.deepcopy(citation))
        return output[:3]

    def _normalize_support_status(
        self,
        raw_status: Any,
        citations: Sequence[Dict[str, Any]],
        grounding_mode: str,
        allow_external: bool,
    ) -> str:
        status = str(raw_status or "").strip()
        if status not in ALLOWED_SUPPORT_STATUSES:
            status = ""
        if citations:
            return status if status in {"slide_grounded", "inferred_from_slides"} else ("slide_grounded" if len(citations) == 1 else "inferred_from_slides")
        if grounding_mode == "strict_lecture_only":
            return "insufficient_evidence"
        if allow_external:
            return status if status in {"external_supplement", "insufficient_evidence"} else "external_supplement"
        return "insufficient_evidence"

    def _safe_text(self, value: Any, fallback: str = "") -> str:
        text = normalize_whitespace(str(value or ""))
        return text or fallback

    def _bounded_minutes(self, value: Any, total_minutes: int, step_count: int) -> int:
        try:
            minutes = int(value)
        except Exception:
            minutes = max(5, math.floor(total_minutes / max(1, step_count)))
        return max(5, min(max(total_minutes, 5), minutes))

    def _sanitize_uncertainty(self, value: Any) -> List[Dict[str, str]]:
        output: List[Dict[str, str]] = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            code = self._safe_text(item.get("code"))
            message = self._safe_text(item.get("message"))
            if code and message:
                output.append({"code": code, "message": message})
        return output[:6]

    def _sanitize_rubric(self, value: Any) -> List[Dict[str, Any]]:
        output: List[Dict[str, Any]] = []
        for item in value or []:
            if not isinstance(item, dict):
                continue
            criterion = self._safe_text(item.get("criterion"))
            description = self._safe_text(item.get("description"))
            try:
                points = int(item.get("points", 0))
            except Exception:
                points = 0
            if criterion and description:
                output.append({"criterion": criterion, "description": description, "points": max(0, points)})
        return output

    def _sanitize_answer_choices(self, value: Any) -> List[str]:
        output: List[str] = []
        for item in value or []:
            text = self._safe_text(item)
            text = re.sub(r"^\s*[\(\[]?[A-Da-d1-4][\)\].:-]\s*", "", text)
            text = re.sub(r"^[\"'`]+|[\"'`]+$", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\s+[.;:,]+$", "", text)
            if text:
                text = text[:1].upper() + text[1:]
                if text[-1] not in ".!?":
                    text = f"{text}."
            if text:
                output.append(text)
        deduped: List[str] = []
        seen: set[str] = set()
        for item in output:
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:4]

    def _coerce_estimated_minutes(self, value: Any, question_type: str, difficulty: str) -> int:
        try:
            minutes = int(value)
        except Exception:
            minutes = self._estimated_minutes(question_type, difficulty)
        return max(2, min(25, minutes))

    def _study_item_should_modify(
        self,
        section_name: str,
        item_id: str,
        locked: set[str],
        target_section: str,
        target_item_id: Optional[str],
    ) -> bool:
        if item_id in locked:
            return False
        if target_item_id:
            return item_id == target_item_id
        if target_section == "entire_plan":
            return True
        return target_section == section_name
