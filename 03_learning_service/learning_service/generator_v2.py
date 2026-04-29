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
ALLOWED_QUESTION_TYPES = {"multiple_choice", "short_answer", "long_answer"}
ALLOWED_DIFFICULTIES = {"easier", "mixed", "harder"}

LOGGER = logging.getLogger(__name__)


class GeminiPrimaryClient(OptionalGeminiClient):
    def generate_json(
        self,
        system_instruction: str,
        user_prompt: str,
        max_output_tokens: int = 2048,
        response_json_schema: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        return super().generate_json(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
            response_json_schema=response_json_schema,
        )


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

    def _practice_question_update_schema(
        self,
        *,
        id_field: str,
        include_answer_key: bool,
    ) -> Dict[str, Any]:
        required_fields = [id_field, "stem", "scoring_guide_text"]
        properties: Dict[str, Any] = {
            id_field: {
                "type": "integer" if id_field == "question_index" else "string",
                "description": "Preserve the original question identifier exactly.",
            },
            "stem": {
                "type": "string",
                "description": "A rewritten question stem that stays grounded in the supplied lecture evidence.",
            },
            "scoring_guide_text": {
                "type": "string",
                "description": "A short grading note describing what a strong answer should include.",
            },
        }
        if include_answer_key:
            properties["expected_answer"] = {
                "type": "string",
                "description": "A concise expected answer grounded in the supplied lecture evidence.",
            }
            required_fields.append("expected_answer")
        return {
            "type": "object",
            "properties": properties,
            "required": required_fields,
            "additionalProperties": False,
        }

    def _practice_batch_response_schema(
        self,
        *,
        id_field: str,
        include_answer_key: bool,
        min_items: int,
        max_items: Optional[int] = None,
    ) -> Dict[str, Any]:
        items_schema = self._practice_question_update_schema(
            id_field=id_field,
            include_answer_key=include_answer_key,
        )
        array_schema: Dict[str, Any] = {
            "type": "array",
            "items": items_schema,
            "minItems": min_items,
        }
        if max_items is not None:
            array_schema["maxItems"] = max_items
        return {
            "type": "object",
            "properties": {
                "questions": array_schema,
            },
            "required": ["questions"],
            "additionalProperties": False,
        }

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
        grounded_citations = dedupe_citations(item.get("citation") for item in relevant_items)[:3]
        evidence_lines = []
        for item in relevant_items:
            citation = item.get("citation") or {}
            evidence_lines.append(
                f"slide={item.get('slide_number')} | citation_id={citation.get('citation_id')} | title={item.get('material_title')} | text={item.get('text')}"
            )
        if not evidence_lines:
            evidence_lines.append("No matched lecture evidence was retrieved for this question.")
        recent_user_turns = [msg.get("text", "") for msg in previous_messages if msg.get("role") == "user" and msg.get("text")][-2:]
        partial_bridge = self._question_specific_partial_answer(normalized_message, relevant_items)
        prompt = (
            f"Grounding mode: {grounding_mode}\n"
            f"Response style: {response_style if response_style in ALLOWED_RESPONSE_STYLES else 'standard'}\n"
            f"Question: {normalized_message}\n"
            f"Retrieval query assist: {retrieval_query}\n"
            f"Recent user turns: {recent_user_turns}\n"
            "Evidence digest:\n"
            + "\n".join(evidence_lines)
            + "\n\nWrite the answer as plain text only.\n"
            "Rules:\n"
            "- Be helpful and explanatory, not defensive.\n"
            "- Base the answer on the evidence digest when it is relevant.\n"
            "- If the slides only partially answer the question, give the best grounded partial explanation first, then briefly name the missing detail.\n"
            "- If the evidence is genuinely unrelated, say the materials do not cover it clearly.\n"
            "- Do not return JSON.\n"
            "- Do not include headings, labels, or citation ids in the answer text.\n"
            "- Do not quote or closely mirror the evidence digest; paraphrase it.\n"
        )
        if partial_bridge:
            prompt += f"- Helpful grounded hint: {partial_bridge}\n"
        text = self.gemini.generate_text(
            system_instruction=(
                "You answer study questions grounded in lecture evidence. "
                "Prefer a grounded partial explanation over a refusal when the slides contain nearby mechanism or intuition. "
                "Use plain text only and paraphrase the evidence."
            ),
            user_prompt=prompt,
            max_output_tokens=700,
        )
        llm_call_info = copy.deepcopy(getattr(self.gemini, "last_call_info", {}) or {})
        if not text:
            return None

        answer_text = self._normalize_gemini_chat_text(text, response_style=response_style)
        if not answer_text:
            self._record_gemini_failure("invalid_response_text", "Gemini returned an empty chat answer after normalization.")
            LOGGER.warning("Gemini chat reply normalized to empty text for query=%r.", normalized_message[:120])
            return None

        support_status = self._support_status_for_gemini_chat_text(
            query=query,
            retrieval_query=retrieval_query,
            relevant_items=relevant_items,
            answer_text=answer_text,
            partial_bridge=partial_bridge,
        )
        section_citations = grounded_citations if support_status in {"slide_grounded", "inferred_from_slides"} else []

        if support_status in {"slide_grounded", "inferred_from_slides"} and self._repeats_source_material(answer_text, relevant_items):
            LOGGER.warning(
                "Gemini chat reply repeated source material directly for query=%r; retrying with a paraphrase-only prompt.",
                normalized_message[:120],
            )
            repaired_text = self._repair_gemini_chat_text(
                normalized_message=normalized_message,
                answer_text=answer_text,
                relevant_items=relevant_items,
                response_style=response_style,
            )
            if not repaired_text:
                self._record_gemini_failure(
                    "verbatim_evidence_repetition",
                    "Gemini repeated source material directly and the paraphrase retry did not produce a usable answer.",
                )
                return None
            answer_text = repaired_text
            llm_call_info = copy.deepcopy(getattr(self.gemini, "last_call_info", {}) or llm_call_info)
            if self._repeats_source_material(answer_text, relevant_items):
                self._record_gemini_failure(
                    "verbatim_evidence_repetition",
                    "Gemini repeated source material directly even after the paraphrase retry.",
                )
                return None

        sections = [
            {
                "heading": "Grounded answer" if section_citations else "What the materials show",
                "text": answer_text,
                "support_status": support_status,
                "citations": section_citations,
            }
        ]

        if (
            grounding_mode == "lecture_with_fallback"
            and support_status in {"slide_grounded", "inferred_from_slides"}
            and self._needs_external_supplement(normalized_message, relevant_items)
        ):
            grounded_text = answer_text
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
        LOGGER.info(
            "Gemini chat reply accepted for query=%r using model=%s with support_status=%s and %s section(s).",
            normalized_message[:120],
            llm_call_info.get("used_model"),
            support_status,
            len(sections),
        )
        return self._with_conversation_answer_source(
            (user_message, assistant_message),
            generation_path="llm",
            query=query,
            relevant_items=relevant_items,
            llm_call_info=llm_call_info,
        )

    def _normalize_gemini_chat_text(self, text: str, *, response_style: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^\s*```(?:text|markdown)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"^\s*(?:grounded answer|answer|response|what the materials show)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        if response_style == "step_by_step":
            lines = [line.strip() for line in cleaned.splitlines() if normalize_whitespace(line)]
            return "\n".join(lines)
        return normalize_whitespace(cleaned)

    def _looks_like_insufficient_evidence_chat(self, answer_text: str) -> bool:
        lowered = normalize_whitespace(answer_text).lower()
        insufficient_phrases = [
            "do not cover",
            "does not cover",
            "don't cover",
            "not covered",
            "do not explain",
            "does not explain",
            "don't explain",
            "not enough evidence",
            "insufficient evidence",
            "materials do not",
            "slides do not",
        ]
        return any(phrase in lowered for phrase in insufficient_phrases)

    def _support_status_for_gemini_chat_text(
        self,
        *,
        query: str,
        retrieval_query: str,
        relevant_items: Sequence[Dict[str, Any]],
        answer_text: str,
        partial_bridge: Optional[str],
    ) -> str:
        if not relevant_items:
            return "insufficient_evidence"
        if partial_bridge:
            return "slide_grounded" if len(relevant_items) == 1 else "inferred_from_slides"
        if self._looks_like_insufficient_evidence_chat(answer_text) and self._chat_match_is_weak(
            query,
            relevant_items,
            retrieval_query=retrieval_query,
        ):
            return "insufficient_evidence"
        return "slide_grounded" if len(relevant_items) == 1 else "inferred_from_slides"

    def _repair_gemini_chat_text(
        self,
        *,
        normalized_message: str,
        answer_text: str,
        relevant_items: Sequence[Dict[str, Any]],
        response_style: str,
    ) -> Optional[str]:
        evidence_lines = []
        for item in relevant_items:
            evidence_lines.append(
                f"slide={item.get('slide_number')} | title={item.get('material_title')} | text={item.get('text')}"
            )
        repaired = self.gemini.generate_text(
            system_instruction=(
                "You paraphrase a lecture-grounded answer so it no longer mirrors the source wording. "
                "Use plain text only."
            ),
            user_prompt=(
                f"Student question: {normalized_message}\n"
                f"Current answer draft: {answer_text}\n"
                "Evidence digest:\n"
                + "\n".join(evidence_lines)
                + "\n\nRewrite the answer in fresh wording while keeping the same grounded meaning. "
                "Do not quote or closely mirror the evidence digest."
            ),
            max_output_tokens=500 if response_style != "step_by_step" else 650,
        )
        if not repaired:
            return None
        return self._normalize_gemini_chat_text(repaired, response_style=response_style)

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
        base_artifact = super().build_practice_set(
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
        if self.gemini.configured:
            artifact = self._build_practice_set_via_gemini(
                base_artifact=copy.deepcopy(base_artifact),
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
            self._log_gemini_fallback("practice set generation", "heuristic fallback")
        base_artifact.setdefault("_meta", {})["generation_path"] = "heuristic_fallback"
        return base_artifact

    def _build_practice_set_via_gemini(
        self,
        *,
        base_artifact: Dict[str, Any],
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
        questions = list(base_artifact.get("questions") or [])
        if not questions:
            return None

        prompt_questions: List[Dict[str, Any]] = []
        for index, question in enumerate(questions, start=1):
            evidence_preview = [
                safe_excerpt(citation.get("snippet_text") or "", 180)
                for citation in (question.get("citations") or [])[:2]
                if self._safe_text(citation.get("snippet_text"))
            ]
            prompt_question = {
                "question_index": index,
                "difficulty": question.get("difficulty"),
                "covered_slides": question.get("covered_slides") or [],
                "current_stem": question.get("stem"),
                "current_expected_answer": question.get("expected_answer") if include_answer_key else "",
                "current_scoring_guide_text": question.get("scoring_guide_text") or "",
                "evidence_preview": evidence_preview,
            }
            prompt_questions.append(prompt_question)

        prompt = (
            f"Topic focus: {topic_text or 'all ready grounded materials'}\n"
            f"Coverage mode: {coverage_mode}\n"
            f"Difficulty profile: {difficulty_profile}\n"
            f"Question count: {question_count}\n"
            f"Grounding mode: {grounding_mode}\n"
            f"Include answer key: {include_answer_key}\n"
            "Revise this grounded practice set as one coherent exam-quality batch.\n"
            f"{json.dumps(prompt_questions, ensure_ascii=False)}\n\n"
            "Return one JSON object with key questions.\n"
            "Every returned item must keep the same question_index.\n"
            "Only rewrite stem, expected_answer, and scoring_guide_text.\n"
            "Do not add question formats, answer choices, rubrics, citations, slide coverage, or commentary outside JSON.\n"
            "Use the evidence previews to make the wording more like real course exam questions and less like slide titles or generic placeholders.\n"
            "Make the set feel coherent across questions rather than treating each question independently.\n"
            "Keep each stem concise and exam-like. Keep expected_answer and scoring_guide_text brief.\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You improve a grounded practice set from lecture evidence. "
                "Return a single JSON object whose questions array preserves question_index values exactly."
            ),
            user_prompt=prompt,
            response_json_schema=self._practice_batch_response_schema(
                id_field="question_index",
                include_answer_key=include_answer_key,
                min_items=len(questions),
                max_items=len(questions),
            ),
        )
        if not isinstance(payload, dict):
            return self._reject_gemini_output("practice set", "Expected a top-level JSON object for the practice set response.", payload=payload)

        updates = self._practice_question_updates_by_index(payload.get("questions"), len(questions))
        updated_count = 0
        enhanced_questions: List[Dict[str, Any]] = []
        for index, question in enumerate(questions, start=1):
            enhanced_question, was_updated = self._merge_practice_question_update(
                base_question=question,
                raw_update=updates.get(index),
                include_answer_key=include_answer_key,
                include_rubrics=include_rubrics,
                difficulty_profile=difficulty_profile,
            )
            enhanced_questions.append(enhanced_question)
            if was_updated:
                updated_count += 1

        if updated_count == 0:
            return self._reject_gemini_output(
                "practice set",
                "Gemini did not produce any usable question improvements for the grounded practice draft.",
                reason="no_usable_question_updates",
            )

        artifact = copy.deepcopy(base_artifact)
        artifact["questions"] = enhanced_questions
        artifact["estimated_duration_minutes"] = sum(int(question.get("estimated_minutes") or 0) for question in enhanced_questions)
        artifact.setdefault("_meta", {})["llm_enhanced_questions"] = updated_count
        LOGGER.info(
            "Gemini practice enhancement accepted using model=%s with %s updated question(s).",
            (getattr(self.gemini, "last_call_info", {}) or {}).get("used_model"),
            updated_count,
        )
        artifact.setdefault("_meta", {})["generation_path"] = "gemini"
        return artifact

    def _practice_question_updates_by_index(self, value: Any, question_count: int) -> Dict[int, Dict[str, Any]]:
        updates: Dict[int, Dict[str, Any]] = {}
        fallback_index = 1
        for raw in value or []:
            if not isinstance(raw, dict):
                fallback_index += 1
                continue
            try:
                question_index = int(raw.get("question_index"))
            except Exception:
                question_index = fallback_index
            fallback_index += 1
            if 1 <= question_index <= question_count and question_index not in updates:
                updates[question_index] = raw
        return updates

    def _merge_practice_question_update(
        self,
        *,
        base_question: Dict[str, Any],
        raw_update: Optional[Dict[str, Any]],
        include_answer_key: bool,
        include_rubrics: bool,
        difficulty_profile: str,
    ) -> Tuple[Dict[str, Any], bool]:
        merged = copy.deepcopy(base_question)
        if not isinstance(raw_update, dict):
            return merged, False

        updated = False
        relevant_items = self._practice_relevant_items_from_citations(merged.get("citations") or [])

        stem = self._grounded_practice_text(raw_update.get("stem"), fallback=merged.get("stem", ""), relevant_items=relevant_items)
        if stem and stem != merged.get("stem"):
            merged["stem"] = stem
            updated = True

        if include_answer_key:
            expected_answer = self._grounded_practice_text(
                raw_update.get("expected_answer"),
                fallback=merged.get("expected_answer", ""),
                relevant_items=relevant_items,
            )
            if expected_answer != merged.get("expected_answer", ""):
                merged["expected_answer"] = expected_answer
                updated = True
        else:
            merged["expected_answer"] = ""

        raw_difficulty = raw_update.get("difficulty")
        if raw_difficulty in ALLOWED_DIFFICULTIES and raw_difficulty != merged.get("difficulty"):
            merged["difficulty"] = raw_difficulty
            updated = True
        question_difficulty = merged.get("difficulty") if merged.get("difficulty") in ALLOWED_DIFFICULTIES else difficulty_profile

        if merged.get("question_type") == "multiple_choice":
            answer_choices = self._sanitize_answer_choices(raw_update.get("answer_choices"))
            if len(answer_choices) == 4 and answer_choices != (merged.get("answer_choices") or []):
                merged["answer_choices"] = answer_choices
                updated = True

        if include_rubrics:
            rubric = self._sanitize_rubric(raw_update.get("rubric"))
            if rubric and rubric != (merged.get("rubric") or []):
                merged["rubric"] = rubric
                updated = True
        else:
            merged["rubric"] = []

        scoring_guide_text = self._grounded_practice_text(
            raw_update.get("scoring_guide_text"),
            fallback=merged.get("scoring_guide_text", ""),
            relevant_items=relevant_items,
        )
        if scoring_guide_text != merged.get("scoring_guide_text", ""):
            merged["scoring_guide_text"] = scoring_guide_text
            updated = True

        if raw_update.get("estimated_minutes") is not None:
            estimated_minutes = self._coerce_estimated_minutes(
                raw_update.get("estimated_minutes"),
                merged.get("question_type") or "short_answer",
                question_difficulty,
            )
            if estimated_minutes != merged.get("estimated_minutes"):
                merged["estimated_minutes"] = estimated_minutes
                updated = True

        return merged, updated

    def _practice_relevant_items_from_citations(self, citations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        relevant_items: List[Dict[str, Any]] = []
        for citation in citations:
            snippet_text = self._safe_text(citation.get("snippet_text"))
            if not snippet_text:
                continue
            relevant_items.append(
                {
                    "text": snippet_text,
                    "material_title": citation.get("material_title", ""),
                    "slide_number": citation.get("slide_number"),
                }
            )
        return relevant_items

    def _grounded_practice_text(
        self,
        value: Any,
        *,
        fallback: str,
        relevant_items: Sequence[Dict[str, Any]],
    ) -> str:
        candidate = self._safe_text(value)
        if not candidate:
            return fallback
        if relevant_items and self._repeats_source_material(candidate, relevant_items):
            return fallback
        return candidate

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
                "difficulty": question.get("difficulty"),
                "covered_slides": question.get("covered_slides") or [],
                "stem": question["stem"],
                "expected_answer": question.get("expected_answer", ""),
                "scoring_guide_text": question.get("scoring_guide_text"),
                "evidence_preview": [
                    safe_excerpt(citation.get("snippet_text") or "", 180)
                    for citation in (question.get("citations") or [])[:2]
                    if self._safe_text(citation.get("snippet_text"))
                ],
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
            "Revise these grounded practice questions as one coherent batch.\n"
            f"{json.dumps(editable, ensure_ascii=False)}\n\n"
            "Return one JSON object with key questions.\n"
            "Every returned item must keep the same question_id.\n"
            "Only rewrite stem, expected_answer, and scoring_guide_text.\n"
            "Do not rewrite difficulty, citations, covered_slides, or add commentary outside JSON.\n"
            "Use the evidence previews and user instruction to make the revised questions more exam-like and better grounded.\n"
            "Keep each stem concise and exam-like. Keep expected_answer and scoring_guide_text brief.\n"
        )
        payload = self.gemini.generate_json(
            system_instruction=(
                "You revise grounded practice questions after user feedback. "
                "Return a single JSON object whose questions array preserves question_id values exactly."
            ),
            user_prompt=prompt,
            response_json_schema=self._practice_batch_response_schema(
                id_field="question_id",
                include_answer_key=True,
                min_items=len(editable),
                max_items=len(editable),
            ),
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
                reason="no_usable_question_updates",
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
