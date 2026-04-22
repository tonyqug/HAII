from __future__ import annotations

import copy
import json
import logging
import math
import re
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from .config import Settings
from .utils import (
    dedupe_citations,
    distinct_slide_numbers,
    first_sentence,
    infer_concept_label,
    informative_tokens,
    lexical_overlap_score,
    make_id,
    normalize_whitespace,
    safe_excerpt,
    split_sentences,
    summarize_texts,
    take_sentences,
    top_keywords,
    utc_now_iso,
)


class NeedsUserInputError(RuntimeError):
    def __init__(self, prompt: str, options: Optional[List[str]] = None, kind: str = "clarification"):
        super().__init__(prompt)
        self.prompt = prompt
        self.options = options or []
        self.kind = kind


class ArtifactValidationError(RuntimeError):
    pass


DEFAULT_GEMINI_MODEL_LADDER = (
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
RATE_LIMIT_STATUS_CODE = 429
TRANSIENT_GEMINI_STATUS_CODES = {503}
MODEL_COMPATIBILITY_STATUS_CODES = {400, 403, 404}
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)

LOGGER = logging.getLogger(__name__)


def _json_text_candidates(text: str) -> List[str]:
    stripped = text.strip()
    candidates: List[str] = []
    if stripped:
        candidates.append(stripped)
    for match in JSON_FENCE_RE.finditer(text):
        fenced = match.group(1).strip()
        if fenced:
            candidates.append(fenced)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1].strip())
    deduped: List[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


class OptionalGeminiClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.gemini_api_key.strip()
        self.timeout = settings.request_timeout_seconds
        configured_primary = settings.gemini_model.strip()
        ordered_models = [configured_primary] if configured_primary else []
        ordered_models.extend(DEFAULT_GEMINI_MODEL_LADDER)
        self.model_ladder = tuple(dict.fromkeys(model for model in ordered_models if model))
        self.last_call_info = self._fresh_call_info()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _fresh_call_info(self) -> Dict[str, Any]:
        return {
            "configured": bool(self.api_key),
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": None,
            "reasoning_enabled": False,
            "reasoning_mode": None,
            "attempted_models": [],
            "rate_limited_models": [],
            "failure_reason": None,
            "model_failures": [],
            "raw_response_preview": None,
        }

    def _thinking_config_for_model(self, model: str) -> Dict[str, Any]:
        if model.startswith("gemini-3"):
            return {"thinkingLevel": "high"}
        if model.startswith("gemini-2.5"):
            # Enable dynamic reasoning across the 2.5 fallback ladder, including Flash-Lite.
            return {"thinkingBudget": -1}
        return {}

    def _extract_text(self, payload: Dict[str, Any]) -> Optional[str]:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
        text_parts = [part.get("text", "") for part in parts if part.get("text")]
        combined = "\n".join(text_parts).strip()
        return combined or None

    def _failure_reason_for_response(self, response: requests.Response) -> str:
        if response.status_code == RATE_LIMIT_STATUS_CODE:
            return "rate_limit_exceeded"
        if response.status_code in TRANSIENT_GEMINI_STATUS_CODES:
            return "service_unavailable"
        if response.status_code in {401, 403}:
            return "authentication_failed"
        if response.status_code == 400:
            return "bad_request"
        if response.status_code >= 500:
            return "upstream_error"
        return "http_error"

    def _response_detail_preview(self, response: requests.Response) -> str:
        try:
            payload = response.json()
            return safe_excerpt(json.dumps(payload, ensure_ascii=False), 400)
        except Exception:
            return safe_excerpt(getattr(response, "text", "") or "", 400)

    def _record_model_failure(
        self,
        model: str,
        reason: str,
        *,
        attempt: int,
        status_code: Optional[int] = None,
        used_thinking_config: bool = False,
        response_preview: Optional[str] = None,
        error: Optional[Exception] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "model": model,
            "reason": reason,
            "attempt": attempt,
            "status_code": status_code,
            "used_thinking_config": used_thinking_config,
        }
        if response_preview:
            entry["response_preview"] = response_preview
            if not self.last_call_info.get("raw_response_preview"):
                self.last_call_info["raw_response_preview"] = response_preview
        if error is not None:
            entry["error"] = str(error)
        self.last_call_info["model_failures"].append(entry)

    def _generate_content(
        self,
        *,
        system_instruction: str,
        user_prompt: str,
        max_output_tokens: int,
        response_mime_type: Optional[str] = None,
    ) -> Optional[str]:
        self.last_call_info = self._fresh_call_info()
        if not self.configured:
            self.last_call_info["generation_path"] = "disabled"
            self.last_call_info["failure_reason"] = "gemini_not_configured"
            LOGGER.warning("Gemini generation is disabled because GEMINI_API_KEY is not configured.")
            return None

        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        for model in self.model_ladder:
            self.last_call_info["attempted_models"].append(model)
            base_generation_config: Dict[str, Any] = {
                "temperature": 0.2,
                "maxOutputTokens": max_output_tokens,
            }
            if response_mime_type:
                base_generation_config["responseMimeType"] = response_mime_type
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            configured_thinking = self._thinking_config_for_model(model)
            thinking_variants = [configured_thinking] if not configured_thinking else [configured_thinking, None]
            for thinking_variant in thinking_variants:
                generation_config = dict(base_generation_config)
                if thinking_variant:
                    generation_config["thinkingConfig"] = thinking_variant
                payload = {
                    "systemInstruction": {"parts": [{"text": system_instruction}]},
                    "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                    "generationConfig": generation_config,
                }
                retry_without_thinking = False
                advance_model = False
                for attempt in range(3):
                    try:
                        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                    except requests.RequestException as exc:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        self.last_call_info["failure_reason"] = "request_exception"
                        self._record_model_failure(
                            model,
                            "request_exception",
                            attempt=attempt + 1,
                            used_thinking_config=bool(thinking_variant),
                            error=exc,
                        )
                        LOGGER.warning(
                            "Gemini request exception on model %s (thinking=%s, attempt=%s): %s",
                            model,
                            bool(thinking_variant),
                            attempt + 1,
                            exc,
                        )
                        return None

                    if response.status_code == RATE_LIMIT_STATUS_CODE:
                        preview = self._response_detail_preview(response)
                        self.last_call_info["rate_limited_models"].append(model)
                        self.last_call_info["failure_reason"] = "rate_limit_exceeded"
                        self._record_model_failure(
                            model,
                            "rate_limit_exceeded",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            used_thinking_config=bool(thinking_variant),
                            response_preview=preview,
                        )
                        LOGGER.warning("Gemini rate-limited on model %s; moving to the next model.", model)
                        advance_model = True
                        break

                    if response.status_code in TRANSIENT_GEMINI_STATUS_CODES:
                        if attempt < 2:
                            time.sleep(2 ** attempt)
                            continue
                        preview = self._response_detail_preview(response)
                        self.last_call_info["failure_reason"] = "service_unavailable"
                        self._record_model_failure(
                            model,
                            "service_unavailable",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            used_thinking_config=bool(thinking_variant),
                            response_preview=preview,
                        )
                        LOGGER.warning(
                            "Gemini service remained unavailable on model %s after retries; trying the next model.",
                            model,
                        )
                        advance_model = True
                        break

                    if not response.ok:
                        reason = self._failure_reason_for_response(response)
                        preview = self._response_detail_preview(response)
                        self.last_call_info["failure_reason"] = reason
                        self._record_model_failure(
                            model,
                            reason,
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            used_thinking_config=bool(thinking_variant),
                            response_preview=preview,
                        )
                        if thinking_variant and response.status_code == 400:
                            LOGGER.warning(
                                "Gemini model %s rejected the reasoning config; retrying once without thinkingConfig. Detail: %s",
                                model,
                                preview,
                            )
                            retry_without_thinking = True
                            break
                        if response.status_code in MODEL_COMPATIBILITY_STATUS_CODES:
                            LOGGER.warning(
                                "Gemini model %s is unavailable or incompatible (HTTP %s, reason=%s); trying the next model. Detail: %s",
                                model,
                                response.status_code,
                                reason,
                                preview,
                            )
                            advance_model = True
                            break
                        LOGGER.warning(
                            "Gemini request failed on model %s with HTTP %s (reason=%s). Detail: %s",
                            model,
                            response.status_code,
                            reason,
                            preview,
                        )
                        return None

                    try:
                        data = response.json()
                    except ValueError:
                        preview = self._response_detail_preview(response)
                        self.last_call_info["failure_reason"] = "invalid_response_json"
                        self._record_model_failure(
                            model,
                            "invalid_response_json",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            used_thinking_config=bool(thinking_variant),
                            response_preview=preview,
                        )
                        LOGGER.warning(
                            "Gemini returned invalid HTTP JSON on model %s. Detail: %s",
                            model,
                            preview,
                        )
                        advance_model = True
                        break

                    text = self._extract_text(data)
                    if not text:
                        preview = safe_excerpt(json.dumps(data, ensure_ascii=False), 400)
                        self.last_call_info["failure_reason"] = "empty_response"
                        self._record_model_failure(
                            model,
                            "empty_response",
                            attempt=attempt + 1,
                            status_code=response.status_code,
                            used_thinking_config=bool(thinking_variant),
                            response_preview=preview,
                        )
                        LOGGER.warning("Gemini returned an empty response on model %s. Payload: %s", model, preview)
                        advance_model = True
                        break

                    self.last_call_info.update(
                        {
                            "used_model": model,
                            "reasoning_enabled": bool(thinking_variant),
                            "reasoning_mode": "dynamic" if thinking_variant else None,
                            "failure_reason": None,
                            "raw_response_preview": safe_excerpt(text, 400),
                        }
                    )
                    LOGGER.info(
                        "Gemini request succeeded with model %s (thinking=%s).",
                        model,
                        bool(thinking_variant),
                    )
                    return text

                if retry_without_thinking:
                    continue
                if advance_model:
                    break

        if self.last_call_info["rate_limited_models"]:
            self.last_call_info["failure_reason"] = "rate_limit_exhausted"
        elif not self.last_call_info["failure_reason"]:
            self.last_call_info["failure_reason"] = "llm_generation_failed"
        LOGGER.warning(
            "Gemini generation failed after models %s with reason=%s.",
            self.last_call_info["attempted_models"],
            self.last_call_info["failure_reason"],
        )
        return None

    def generate_text(self, system_instruction: str, user_prompt: str, max_output_tokens: int = 256) -> Optional[str]:
        return self._generate_content(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )

    def external_supplement(self, question_text: str, grounded_answer: str) -> Optional[str]:
        prompt = (
            "Question from a student: "
            f"{question_text}\n\n"
            "Grounded answer already supported by their lecture materials: "
            f"{grounded_answer}\n\n"
            "Write a brief external supplement that is clearly general background knowledge, not a claim about the student's slides. "
            "Do not mention citations. Be concise and explicitly phrase this as supplementary background. "
            "Do not quote or closely mirror the grounded answer."
        )
        return self.generate_text(
            system_instruction=(
                "You provide carefully labeled supplemental background for students. "
                "Never imply that external background came from the user's uploaded materials. "
                "Always paraphrase instead of echoing the lecture wording."
            ),
            user_prompt=prompt,
            max_output_tokens=160,
        )


class EvidenceAccessor:
    def __init__(self, bundle: Dict[str, Any]):
        self.bundle = copy.deepcopy(bundle)
        self.items = sorted(
            list(self.bundle.get("items") or []),
            key=lambda item: (
                item.get("material_title", ""),
                int(item.get("slide_number") or 0),
                item.get("item_id", ""),
            ),
        )
        self.material_ids = list(self.bundle.get("material_ids") or [])
        self.workspace_id = self.bundle.get("workspace_id")
        self.query_text = self.bundle.get("query_text")
        summary = self.bundle.get("summary") or {}
        self.summary = {
            "total_items": int(summary.get("total_items") or len(self.items)),
            "total_slides": int(summary.get("total_slides") or len(self.distinct_slide_numbers())),
            "low_confidence_item_count": int(summary.get("low_confidence_item_count") or self._count_low_confidence()),
        }

    def _count_low_confidence(self) -> int:
        count = 0
        for item in self.items:
            citation = item.get("citation") or {}
            if item.get("extraction_quality") == "low" or citation.get("confidence") == "low":
                count += 1
        return count

    def filter_items(self, material_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        if not material_ids:
            return list(self.items)
        allowed = set(material_ids)
        return [item for item in self.items if item.get("material_id") in allowed]

    def lecture_material_ids(self, template_material_id: Optional[str] = None) -> List[str]:
        if not template_material_id:
            return list(self.material_ids)
        return [material_id for material_id in self.material_ids if material_id != template_material_id]

    def distinct_slide_numbers(self, material_ids: Optional[Sequence[str]] = None) -> List[int]:
        values = {
            int(item.get("slide_number") or 0)
            for item in self.filter_items(material_ids)
            if item.get("slide_number") is not None
        }
        return sorted(values)

    def low_confidence_slides(self, material_ids: Optional[Sequence[str]] = None) -> List[int]:
        values = set()
        for item in self.filter_items(material_ids):
            citation = item.get("citation") or {}
            if item.get("extraction_quality") == "low" or citation.get("confidence") == "low":
                values.add(int(item.get("slide_number") or 0))
        return sorted(values)

    def infer_topic(self, material_ids: Optional[Sequence[str]] = None) -> str:
        items = self.filter_items(material_ids)
        if not items:
            return "selected lecture materials"
        material_titles = [item.get("material_title", "") for item in items if item.get("material_title")]
        keywords = top_keywords(material_titles + [item.get("text", "") for item in items], limit=5)
        if material_titles:
            unique_titles = list(dict.fromkeys(material_titles))
            if len(unique_titles) == 1 and len(unique_titles[0].split()) <= 10:
                base_title = unique_titles[0]
            else:
                base_title = unique_titles[0]
        else:
            base_title = "selected lecture materials"

        concept_names = [record["concept_name"] for record in self.concept_records(material_ids)[:3]]
        if concept_names:
            compact = ", ".join(dict.fromkeys(concept_names[:3]))
            if len(compact) <= 90:
                return compact
        if keywords:
            return ", ".join(word.capitalize() for word in keywords[:3])
        return base_title or "selected lecture materials"

    def search(
        self,
        query: str,
        material_ids: Optional[Sequence[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for item in self.filter_items(material_ids):
            # Score on text content only — material_title would boost every slide from a matching
            # lecture equally, causing unrelated slides to rank ahead of actually relevant ones.
            score = lexical_overlap_score(query, item.get("text", ""))
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1].get("slide_number") or 0), pair[1].get("item_id", "")))
        return [item for _score, item in scored[:top_k]]

    def concept_records(self, material_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
        for item in self.filter_items(material_ids):
            grouped[(item.get("material_id", ""), int(item.get("slide_number") or 0))].append(item)

        records: List[Dict[str, Any]] = []
        for (_material_id, slide_number), items in sorted(grouped.items(), key=lambda pair: (pair[0][1], pair[0][0])):
            texts = [item.get("text", "") for item in items]
            first_text = texts[0] if texts else ""
            concept_name = infer_concept_label(first_sentence(first_text), fallback=f"Slide {slide_number} concept")
            citations = dedupe_citations(item.get("citation") for item in items)
            records.append(
                {
                    "slide_number": slide_number,
                    "slide_id": items[0].get("slide_id"),
                    "material_id": items[0].get("material_id"),
                    "material_title": items[0].get("material_title"),
                    "concept_name": concept_name,
                    "summary_text": summarize_texts(texts, sentence_count=2) or safe_excerpt(first_text),
                    "texts": texts,
                    "citations": citations[:3],
                    "items": items,
                }
            )
        return records

    def options_for_clarification(self, material_ids: Optional[Sequence[str]] = None, limit: int = 5) -> List[str]:
        return [record["concept_name"] for record in self.concept_records(material_ids)[:limit]]


class GroundedGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.gemini = OptionalGeminiClient(settings)

    def _chat_evidence_match(self, query: str, relevant_items: Sequence[Dict[str, Any]]) -> str:
        if not relevant_items:
            return "no_match"
        return "weak_match" if self._chat_match_is_weak(query, relevant_items) else "strong_match"

    def _default_chat_fallback_reason(self) -> str:
        if getattr(self.gemini, "configured", False):
            return "llm_generation_failed"
        return "gemini_not_configured"

    def _conversation_answer_source(
        self,
        *,
        generation_path: str,
        query: str,
        relevant_items: Sequence[Dict[str, Any]],
        llm_call_info: Optional[Dict[str, Any]] = None,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        evidence_match = self._chat_evidence_match(query, relevant_items)
        matched_evidence_count = len(relevant_items)
        if generation_path == "llm":
            info = copy.deepcopy(llm_call_info or getattr(self.gemini, "last_call_info", {}) or {})
            return {
                "path": "llm",
                "provider": "gemini",
                "model": info.get("used_model"),
                "reasoning_enabled": bool(info.get("reasoning_enabled")),
                "reasoning_mode": info.get("reasoning_mode"),
                "matched_evidence_count": matched_evidence_count,
                "evidence_match": evidence_match,
                "rate_limited_models": list(info.get("rate_limited_models") or []),
            }
        return {
            "path": "heuristic_fallback",
            "provider": "deterministic_fallback",
            "model": None,
            "reasoning_enabled": False,
            "reasoning_mode": None,
            "matched_evidence_count": matched_evidence_count,
            "evidence_match": evidence_match,
            "rate_limited_models": list((llm_call_info or {}).get("rate_limited_models") or []),
            "fallback_reason": fallback_reason or self._default_chat_fallback_reason(),
        }

    def _with_conversation_answer_source(
        self,
        result: Tuple[Dict[str, Any], Dict[str, Any]],
        *,
        generation_path: str,
        query: str,
        relevant_items: Sequence[Dict[str, Any]],
        llm_call_info: Optional[Dict[str, Any]] = None,
        fallback_reason: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        user_message, assistant_message = result
        assistant_message["answer_source"] = self._conversation_answer_source(
            generation_path=generation_path,
            query=query,
            relevant_items=relevant_items,
            llm_call_info=llm_call_info,
            fallback_reason=fallback_reason,
        )
        return user_message, assistant_message

    # --------------------------
    # Study plan generation
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
        accessor = EvidenceAccessor(bundle)
        lecture_material_ids = accessor.material_ids
        concept_records = accessor.concept_records(lecture_material_ids)
        if not concept_records:
            raise NeedsUserInputError(
                "I need at least one evidence item to build a study plan.",
                options=[],
            )

        inferred_topic = False
        topic = normalize_whitespace(topic_text or "")
        if not topic:
            topic = accessor.infer_topic(lecture_material_ids)
            inferred_topic = True

        normalized_context = self._normalize_student_context(student_context or {})
        ranked_records = self._rank_concept_records_for_plan(
            concept_records,
            topic,
            normalized_context,
        )
        explicit_topic = bool(normalize_whitespace(topic_text or ""))
        if explicit_topic:
            strongest_topic_score = max(
                int((record.get("_plan_features") or {}).get("topic_score", 0))
                for record in ranked_records
            )
            if strongest_topic_score <= 1:
                raise NeedsUserInputError(
                    f"I could not find strong lecture evidence for '{topic}'. Please name a lecture concept, keyword, or slide number to focus the plan.",
                    options=accessor.options_for_clarification(lecture_material_ids, limit=5),
                )
        selected_sequence_records = self._select_study_sequence_records(
            ranked_records,
            time_budget_minutes,
        )
        prerequisites = self._build_prerequisites(
            accessor,
            concept_records,
            selected_sequence_records,
            topic,
            grounding_mode,
            normalized_context,
        )
        study_sequence = self._build_study_sequence(
            accessor,
            selected_sequence_records,
            time_budget_minutes,
            topic,
            grounding_mode,
            prerequisites,
            normalized_context,
        )
        common_mistakes = self._build_common_mistakes(
            accessor,
            concept_records,
            selected_sequence_records,
            topic,
            grounding_mode,
            normalized_context,
        )

        uncertainty: List[Dict[str, str]] = []
        if inferred_topic:
            uncertainty.append(
                {
                    "code": "topic_inferred",
                    "message": "No explicit topic_text was provided, so the service inferred the study topic from the grounded lecture evidence.",
                }
            )
        if accessor.summary["low_confidence_item_count"] > 0:
            uncertainty.append(
                {
                    "code": "low_confidence_evidence",
                    "message": "Some evidence items were marked low confidence, so treat their related plan items with extra care.",
                }
            )
        if accessor.summary["total_slides"] < 3:
            uncertainty.append(
                {
                    "code": "sparse_evidence",
                    "message": "The study plan was built from a small evidence set, so coverage may be incomplete.",
                }
            )
        if inferred_topic and len(concept_records) >= 6:
            uncertainty.append(
                {
                    "code": "topic_still_broad",
                    "message": "The uploaded materials cover several concepts, so adding a topic focus next time would produce a tighter plan.",
                }
            )

        all_citations = []
        for section in prerequisites + study_sequence + common_mistakes:
            all_citations.extend(section.get("citations") or [])
        cited_slides = distinct_slide_numbers(all_citations)
        omitted_or_low = sorted(
            set(accessor.distinct_slide_numbers()) - set(cited_slides)
            | set(accessor.low_confidence_slides())
        )
        tailoring_summary = self._build_tailoring_summary(
            accessor=accessor,
            topic_text=topic,
            inferred_topic=inferred_topic,
            time_budget_minutes=time_budget_minutes,
            grounding_mode=grounding_mode,
            student_context=normalized_context,
            citations=all_citations,
        )

        artifact = {
            "study_plan_id": make_id("study_plan"),
            "parent_study_plan_id": parent_study_plan_id,
            "workspace_id": bundle.get("workspace_id"),
            "created_at": utc_now_iso(),
            "topic_text": topic,
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
        return artifact

    def revise_study_plan(
        self,
        *,
        existing_plan: Dict[str, Any],
        instruction_text: str,
        target_section: str,
        locked_item_ids: Sequence[str],
        grounding_mode: str,
    ) -> Dict[str, Any]:
        plan = copy.deepcopy(existing_plan)
        locked = set(locked_item_ids)
        all_item_ids = {
            item["item_id"] for item in plan.get("prerequisites", []) + plan.get("common_mistakes", [])
        } | {step["step_id"] for step in plan.get("study_sequence", [])}
        missing = locked - all_item_ids
        if missing:
            raise ArtifactValidationError(f"Unknown locked_item_ids: {sorted(missing)}")

        target_item_id = None
        if target_section not in {"entire_plan", "prerequisites", "study_sequence", "common_mistakes"}:
            target_item_id = target_section
            if target_item_id not in all_item_ids:
                raise ArtifactValidationError(f"Unknown target_section item id: {target_item_id}")

        lowered_instruction = instruction_text.lower()

        def should_modify(section_name: str, item_id: str) -> bool:
            if item_id in locked:
                return False
            if target_item_id:
                return item_id == target_item_id
            if target_section == "entire_plan":
                return True
            return target_section == section_name

        for item in plan.get("prerequisites", []):
            if should_modify("prerequisites", item["item_id"]):
                item["why_needed"] = self._apply_revision_text(item["why_needed"], instruction_text)
                if "focus on " in lowered_instruction:
                    focus = self._extract_focus_phrase(instruction_text)
                    if focus:
                        item["concept_name"] = self._append_focus(item["concept_name"], focus)

        for step in plan.get("study_sequence", []):
            if should_modify("study_sequence", step["step_id"]):
                step["objective"] = self._apply_revision_text(step["objective"], instruction_text)
                step["tasks"] = [self._apply_revision_text(task, instruction_text) for task in step.get("tasks", [])]
                step["recommended_time_minutes"] = self._revise_time_budget(step["recommended_time_minutes"], lowered_instruction)
                if "focus on " in lowered_instruction:
                    focus = self._extract_focus_phrase(instruction_text)
                    if focus:
                        step["title"] = self._append_focus(step["title"], focus)
                        if not any(focus.lower() in task.lower() for task in step["tasks"]):
                            step["tasks"].append(f"Spend extra time connecting this step to {focus}.")

        for item in plan.get("common_mistakes", []):
            if should_modify("common_mistakes", item["item_id"]):
                item["why_it_happens"] = self._apply_revision_text(item["why_it_happens"], instruction_text)
                item["prevention_advice"] = self._apply_revision_text(item["prevention_advice"], instruction_text)

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

    def _build_prerequisites(
        self,
        accessor: EvidenceAccessor,
        concept_records: Sequence[Dict[str, Any]],
        selected_sequence_records: Sequence[Dict[str, Any]],
        topic_text: str,
        grounding_mode: str,
        student_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        prior_knowledge = student_context.get("prior_knowledge", "")
        selected_ids = {record.get("slide_id") for record in selected_sequence_records}
        earliest_selected_slide = min(
            (int(record.get("slide_number") or 0) for record in selected_sequence_records),
            default=0,
        )
        prerequisite_pool = [
            record
            for record in concept_records
            if int(record.get("slide_number") or 0) < earliest_selected_slide
        ]
        if not prerequisite_pool:
            prerequisite_pool = [
                record for record in concept_records
                if record.get("slide_id") not in selected_ids
            ]
        if not prerequisite_pool:
            prerequisite_pool = list(concept_records)

        items: List[Dict[str, Any]] = []
        for record in self._dedupe_records(prerequisite_pool)[:2]:
            summary = take_sentences(record["summary_text"], 1) or record["summary_text"]
            overlap_with_prior = self._record_overlap(record, prior_knowledge) > 0
            overlap_with_topic = self._record_overlap(record, topic_text) > 0
            if overlap_with_prior:
                why_needed = (
                    f"You said you already know {prior_knowledge}, so use {record['concept_name']} as a quick bridge "
                    f"into the lecture's version of the topic: {summary}"
                )
            elif overlap_with_topic:
                why_needed = f"This concept directly supports the requested focus on {topic_text}: {summary}"
            else:
                why_needed = f"This appears early in the grounded materials and anchors later ideas: {summary}"
            items.append(
                {
                    "item_id": make_id("prereq"),
                    "concept_name": record["concept_name"],
                    "why_needed": why_needed,
                    "support_status": "inferred_from_slides",
                    "citations": record["citations"][:2],
                }
            )
        return items

    def _build_study_sequence(
        self,
        accessor: EvidenceAccessor,
        concept_records: Sequence[Dict[str, Any]],
        time_budget_minutes: int,
        topic_text: str,
        grounding_mode: str,
        prerequisites: Sequence[Dict[str, Any]],
        student_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sequence_records = self._dedupe_records(concept_records)[:6]
        weak_areas = normalize_whitespace((student_context or {}).get("weak_areas", ""))
        goals = normalize_whitespace((student_context or {}).get("goals", ""))
        prior_knowledge = normalize_whitespace((student_context or {}).get("prior_knowledge", ""))
        minute_allocation = self._allocate_step_minutes(
            time_budget_minutes,
            [
                self._step_weight_for_record(record, topic_text, student_context)
                for record in sequence_records
            ],
        )
        steps: List[Dict[str, Any]] = []
        for index, record in enumerate(sequence_records, start=1):
            slide_numbers = distinct_slide_numbers(record.get("citations") or []) or [int(record.get("slide_number") or 0)]
            slide_label = ", ".join(str(number) for number in slide_numbers if number)
            quick_bridge = self._record_overlap(record, prior_knowledge) > 0
            previous_concept = sequence_records[index - 2]["concept_name"] if index > 1 else None
            tasks = self._build_step_tasks(
                record=record,
                step_index=index,
                slide_label=slide_label or "?",
                previous_concept=previous_concept,
                quick_bridge=quick_bridge,
                prior_knowledge=prior_knowledge,
                weak_areas=weak_areas,
                goals=goals,
            )
            milestone = self._step_milestone(record["concept_name"], topic_text, goals)
            steps.append(
                {
                    "step_id": make_id("step"),
                    "order_index": index,
                    "title": self._step_title(record, topic_text, weak_areas, quick_bridge),
                    "objective": self._step_objective(record, topic_text, goals),
                    "recommended_time_minutes": minute_allocation[index - 1],
                    "tasks": tasks,
                    "milestone": milestone,
                    "depends_on": [item["item_id"] for item in prerequisites[: min(index, len(prerequisites))]],
                    "support_status": "inferred_from_slides",
                    "citations": record["citations"][:2],
                }
            )
        if not steps:
            steps.append(
                {
                    "step_id": make_id("step"),
                    "order_index": 1,
                    "title": "Clarify the core lecture topic",
                    "objective": "The evidence bundle was too sparse to derive a reliable sequence.",
                    "recommended_time_minutes": time_budget_minutes,
                    "tasks": ["Locate the most relevant slide or provide a narrower topic."],
                    "milestone": "You can identify the single most relevant grounded slide before studying further details.",
                    "depends_on": [],
                    "support_status": "insufficient_evidence" if grounding_mode == "strict_lecture_only" else "external_supplement",
                    "citations": [],
                }
            )
        return steps

    def _build_common_mistakes(
        self,
        accessor: EvidenceAccessor,
        concept_records: Sequence[Dict[str, Any]],
        selected_sequence_records: Sequence[Dict[str, Any]],
        topic_text: str,
        grounding_mode: str,
        student_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        mistakes: List[Dict[str, Any]] = []
        sequence_pool = list(selected_sequence_records) or list(concept_records)
        weak_areas = normalize_whitespace((student_context or {}).get("weak_areas", ""))
        goals = normalize_whitespace((student_context or {}).get("goals", ""))
        if sequence_pool:
            first = sequence_pool[0]
            second = sequence_pool[1] if len(sequence_pool) > 1 else first
            self._append_mistake(
                mistakes,
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Confusing {first['concept_name']} with {second['concept_name']}",
                    "why_it_happens": "The lecture moves through related concepts in sequence, so students can blur their roles if they only memorize isolated phrases.",
                    "prevention_advice": f"After studying, write one sentence that distinguishes {first['concept_name']} from {second['concept_name']}.",
                    "support_status": "inferred_from_slides",
                    "citations": dedupe_citations(first["citations"] + second["citations"])[:3],
                },
            )
        weak_area_candidate = self._find_record_by_text(sequence_pool or concept_records, weak_areas)
        if weak_areas and weak_area_candidate:
            self._append_mistake(
                mistakes,
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Leaving your weak area around {weak_areas} at the recognition level instead of practicing it actively",
                    "why_it_happens": f"The plan needs to do more than reread {weak_area_candidate['concept_name']}; weak areas usually improve only after explanation and self-testing.",
                    "prevention_advice": f"After reviewing the cited slide, answer one no-notes question focused on {weak_areas} before moving on.",
                    "support_status": "inferred_from_slides",
                    "citations": weak_area_candidate["citations"][:2],
                },
            )
        candidate = self._find_record_by_keywords(sequence_pool or concept_records, ["assumption", "condition", "validation", "rate", "penalty", "error", "overfit"])
        if candidate:
            self._append_mistake(
                mistakes,
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Ignoring the conditions or tuning choices around {candidate['concept_name']}",
                    "why_it_happens": "Students often remember the headline idea but skip the conditions, trade-offs, or checks that make it work correctly.",
                    "prevention_advice": "When reviewing the slide, list the condition, tuning choice, or warning next to the main concept before moving on.",
                    "support_status": "inferred_from_slides",
                    "citations": candidate["citations"][:2],
                },
            )
        candidate = self._find_record_by_keywords(sequence_pool or concept_records, ["example", "generalization", "training", "validation", "compare", "procedure"])
        if candidate and len(mistakes) < 3:
            self._append_mistake(
                mistakes,
                {
                    "item_id": make_id("mistake"),
                    "pattern": "Memorizing the wording without practicing when or how to apply it",
                    "why_it_happens": "The evidence suggests procedure-level or evaluation-level reasoning, which students can miss if they only copy definitions.",
                    "prevention_advice": "Turn each major slide into a quick self-test: what is the idea, when do you use it, and what failure mode should you watch for?",
                    "support_status": "inferred_from_slides",
                    "citations": candidate["citations"][:2],
                },
            )
        goal_candidate = self._find_record_by_text(sequence_pool or concept_records, goals)
        if goals and goal_candidate and len(mistakes) < 3:
            self._append_mistake(
                mistakes,
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Studying passively even though your goal is {goals}",
                    "why_it_happens": "Students often collect notes without checking whether they can produce an exam-ready explanation or decision process.",
                    "prevention_advice": f"Use the cited slide to create one retrieval-style checkpoint that matches your goal: {goals}.",
                    "support_status": "inferred_from_slides",
                    "citations": goal_candidate["citations"][:2],
                },
            )
        return mistakes[:3]

    def _padding_item(self, kind: str, grounding_mode: str, hint: str) -> Dict[str, Any]:
        status = "insufficient_evidence" if grounding_mode == "strict_lecture_only" else "external_supplement"
        if kind == "prereq":
            return {
                "item_id": make_id("prereq"),
                "concept_name": "Additional prerequisite knowledge",
                "why_needed": hint,
                "support_status": status,
                "citations": [],
            }
        return {
            "item_id": make_id("mistake"),
            "pattern": "Potential misunderstanding not directly covered by the retrieved evidence",
            "why_it_happens": hint,
            "prevention_advice": "Treat this as a prompt to verify the idea against more grounded lecture evidence before relying on it.",
            "support_status": status,
            "citations": [],
        }

    def _normalize_student_context(self, student_context: Dict[str, Any]) -> Dict[str, str]:
        return {
            "prior_knowledge": normalize_whitespace((student_context or {}).get("prior_knowledge", "")),
            "weak_areas": normalize_whitespace((student_context or {}).get("weak_areas", "")),
            "goals": normalize_whitespace((student_context or {}).get("goals", "")),
        }

    def _record_text(self, record: Dict[str, Any]) -> str:
        return " ".join(
            [
                record.get("concept_name", ""),
                record.get("summary_text", ""),
                " ".join(record.get("texts") or []),
            ]
        )

    def _record_overlap(self, record: Dict[str, Any], text: str) -> int:
        query = normalize_whitespace(text or "")
        if not query:
            return 0
        return lexical_overlap_score(query, self._record_text(record))

    def _rank_concept_records_for_plan(
        self,
        concept_records: Sequence[Dict[str, Any]],
        topic_text: str,
        student_context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        topic = normalize_whitespace(topic_text)
        weak_areas = student_context.get("weak_areas", "")
        goals = student_context.get("goals", "")
        prior_knowledge = student_context.get("prior_knowledge", "")
        ranked: List[Dict[str, Any]] = []
        for record in concept_records:
            annotated = copy.deepcopy(record)
            topic_score = self._record_overlap(record, topic)
            weak_score = self._record_overlap(record, weak_areas)
            goal_score = self._record_overlap(record, goals)
            prior_score = self._record_overlap(record, prior_knowledge)
            annotated["_plan_features"] = {
                "topic_score": topic_score,
                "weak_score": weak_score,
                "goal_score": goal_score,
                "prior_score": prior_score,
            }
            annotated["_plan_score"] = (
                (topic_score * 6)
                + (weak_score * 7)
                + (goal_score * 4)
                + (prior_score * 2)
                + (1 if int(record.get("slide_number") or 0) <= 2 else 0)
            )
            ranked.append(annotated)
        ranked.sort(
            key=lambda record: (
                -int(record.get("_plan_score", 0)),
                int(record.get("slide_number") or 0),
                record.get("concept_name", ""),
            )
        )
        return ranked

    def _select_study_sequence_records(
        self,
        ranked_records: Sequence[Dict[str, Any]],
        time_budget_minutes: int,
    ) -> List[Dict[str, Any]]:
        if not ranked_records:
            return []
        target_steps = max(2, min(6, math.ceil(time_budget_minutes / 25)))
        selected = list(self._dedupe_records(ranked_records)[:target_steps])
        selected.sort(
            key=lambda record: (
                int(record.get("slide_number") or 0),
                -int(record.get("_plan_score", 0)),
                record.get("concept_name", ""),
            )
        )
        return selected

    def _dedupe_records(self, records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[tuple[str, str, int]] = set()
        deduped: List[Dict[str, Any]] = []
        for record in records:
            key = (
                normalize_whitespace(record.get("concept_name", "")).lower(),
                str(record.get("material_id") or ""),
                int(record.get("slide_number") or 0),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(record)
        return deduped

    def _build_step_tasks(
        self,
        *,
        record: Dict[str, Any],
        step_index: int,
        slide_label: str,
        previous_concept: Optional[str],
        quick_bridge: bool,
        prior_knowledge: str,
        weak_areas: str,
        goals: str,
    ) -> List[str]:
        summary = take_sentences(record.get("summary_text", ""), 1) or safe_excerpt(record.get("summary_text", ""), 120)
        task_options = [
            f"Review slide {slide_label} and restate {record['concept_name']} in your own words.",
            f"Write one checkpoint sentence for this idea: {summary}",
            f"Turn {record['concept_name']} into one no-notes self-test question.",
        ]
        if quick_bridge and prior_knowledge:
            task_options[1] = f"Use what you already know about {prior_knowledge} as a bridge, then note what the lecture adds or changes."
        if previous_concept:
            task_options.append(f"Connect {record['concept_name']} to the earlier topic {previous_concept}.")
        if weak_areas and self._record_overlap(record, weak_areas) > 0:
            task_options.append(f"Spend extra practice time here because it overlaps your stated weak area: {weak_areas}.")
        elif goals:
            task_options.append(f"Check how this step supports your goal: {goals}.")
        task_options.append(f"Capture the key condition, trade-off, or example tied to {record['concept_name']} before moving on.")
        return task_options[:4]

    def _append_mistake(self, mistakes: List[Dict[str, Any]], candidate: Dict[str, Any]) -> None:
        normalized_pattern = normalize_whitespace(candidate.get("pattern", "")).lower()
        if not normalized_pattern:
            return
        if any(normalize_whitespace(item.get("pattern", "")).lower() == normalized_pattern for item in mistakes):
            return
        mistakes.append(candidate)

    def _allocate_step_minutes(self, total_minutes: int, weights: Sequence[int]) -> List[int]:
        if not weights:
            return []
        minimum = 5
        allocation = [minimum for _ in weights]
        remaining = max(0, total_minutes - (minimum * len(weights)))
        positive_weights = [max(1, int(weight)) for weight in weights]
        total_weight = sum(positive_weights)
        if total_weight <= 0:
            total_weight = len(positive_weights)
        for index, weight in enumerate(positive_weights):
            share = math.floor((remaining * weight) / total_weight)
            allocation[index] += share
        distributed = sum(allocation)
        pointer = 0
        while distributed < total_minutes:
            allocation[pointer % len(allocation)] += 1
            distributed += 1
            pointer += 1
        return allocation

    def _step_weight_for_record(self, record: Dict[str, Any], topic_text: str, student_context: Dict[str, Any]) -> int:
        features = record.get("_plan_features") or {}
        prior_overlap = int(features.get("prior_score", 0))
        return (
            2
            + (2 if int(features.get("topic_score", 0)) > 0 else 0)
            + (3 if int(features.get("weak_score", 0)) > 0 else 0)
            + (1 if int(features.get("goal_score", 0)) > 0 else 0)
            + (0 if prior_overlap > 0 else 1)
        )

    def _step_title(self, record: Dict[str, Any], topic_text: str, weak_areas: str, quick_bridge: bool) -> str:
        concept = record.get("concept_name", "Key concept")
        if weak_areas and self._record_overlap(record, weak_areas) > 0:
            return f"Milestone: reinforce {concept}"
        if quick_bridge:
            return f"Quick bridge through {concept}"
        if topic_text and self._record_overlap(record, topic_text) > 0:
            return f"Core focus: {concept}"
        return f"Study {concept}"

    def _step_objective(self, record: Dict[str, Any], topic_text: str, goals: str) -> str:
        summary = safe_excerpt(record.get("summary_text", ""), max_length=220)
        if topic_text and self._record_overlap(record, topic_text) > 0:
            return f"Build a grounded explanation of how this record supports {topic_text}: {summary}"
        if goals:
            return f"Use this grounded concept to support your stated goal ({goals}): {summary}"
        return summary

    def _step_milestone(self, concept_name: str, topic_text: str, goals: str) -> str:
        if goals:
            return f"You can explain {concept_name} clearly enough to use it for {goals} without reopening the slides."
        if topic_text:
            return f"You can explain how {concept_name} supports {topic_text} without looking at the cited slide."
        return f"You can explain {concept_name} from memory and connect it to the surrounding lecture content."

    def _find_record_by_text(self, concept_records: Sequence[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        normalized = normalize_whitespace(text)
        if not normalized:
            return None
        ranked = sorted(
            concept_records,
            key=lambda record: (
                -self._record_overlap(record, normalized),
                int(record.get("slide_number") or 0),
            ),
        )
        if not ranked or self._record_overlap(ranked[0], normalized) <= 0:
            return None
        return ranked[0]

    def _build_tailoring_summary(
        self,
        *,
        accessor: EvidenceAccessor,
        topic_text: str,
        inferred_topic: bool,
        time_budget_minutes: int,
        grounding_mode: str,
        student_context: Dict[str, Any],
        citations: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        material_titles = list(
            dict.fromkeys(
                [
                    item.get("material_title")
                    for item in accessor.filter_items(accessor.material_ids)
                    if item.get("material_title")
                ]
            )
        )
        cited_slides = distinct_slide_numbers(citations)
        used_inputs = [
            {
                "key": "topic_text",
                "label": "Topic focus",
                "value": topic_text or "Inferred from uploaded lecture materials",
                "source": "inferred" if inferred_topic else "user",
            },
            {
                "key": "time_budget_minutes",
                "label": "Time budget",
                "value": f"{time_budget_minutes} minutes",
                "source": "user",
            },
            {
                "key": "grounding_mode",
                "label": "Grounding mode",
                "value": grounding_mode.replace("_", " "),
                "source": "user",
            },
        ]
        if material_titles:
            used_inputs.append(
                {
                    "key": "materials",
                    "label": "Lecture materials used",
                    "value": ", ".join(material_titles),
                    "source": "workspace",
                }
            )
        missing_inputs: List[Dict[str, str]] = []
        optional_fields = [
            (
                "prior_knowledge",
                "What you already know",
                "Not provided, so the plan assumes no specific starting point beyond the grounded materials.",
            ),
            (
                "weak_areas",
                "Weak areas",
                "Not provided, so the plan cannot emphasize a specific trouble spot.",
            ),
            (
                "goals",
                "Goals or exam context",
                "Not provided, so the plan optimizes for general review rather than a specific exam or milestone.",
            ),
        ]
        for key, label, fallback_message in optional_fields:
            value = normalize_whitespace(student_context.get(key, ""))
            if value:
                used_inputs.append(
                    {
                        "key": key,
                        "label": label,
                        "value": value,
                        "source": "user",
                    }
                )
            else:
                missing_inputs.append(
                    {
                        "key": key,
                        "label": label,
                        "message": fallback_message,
                    }
                )
        return {
            "used_inputs": used_inputs,
            "missing_inputs": missing_inputs,
            "evidence_scope": {
                "material_count": len(material_titles),
                "material_titles": material_titles,
                "slide_count": len(cited_slides),
                "slide_numbers": cited_slides,
            },
        }

    # --------------------------
    # Conversation generation
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

        relevant_items = self._select_chat_evidence(accessor, query, lecture_material_ids)
        user_message = {
            "message_id": make_id("msg"),
            "role": "user",
            "created_at": utc_now_iso(),
            "text": normalized_message,
        }

        assistant_message = {
            "message_id": make_id("msg"),
            "role": "assistant",
            "created_at": utc_now_iso(),
            "reply_sections": [],
        }

        if not relevant_items or self._chat_match_is_weak(query, relevant_items):
            LOGGER.info(
                "Using deterministic insufficient-evidence chat fallback for query=%r with matched_items=%s.",
                safe_excerpt(query, 120),
                len(relevant_items),
            )
            assistant_message["reply_sections"].append(
                {
                    "heading": "Insufficient evidence from your materials",
                    "text": "I could not find enough grounded lecture evidence to answer this confidently from the current materials.",
                    "support_status": "insufficient_evidence",
                    "citations": [],
                }
            )
            assistant_message["clarifying_question"] = {
                "prompt": "Point me to a lecture concept, keyword, or slide number and I can answer more precisely.",
                "reason": "The current question did not match grounded lecture evidence strongly enough.",
            }
            if grounding_mode == "lecture_with_fallback":
                assistant_message["reply_sections"].append(
                    {
                        "heading": "External supplement",
                        "text": self._external_supplement_text(normalized_message, grounded_answer="")
                        or "General next step: consult a textbook definition or your instructor's notes for this topic, then compare that explanation against the lecture slides before relying on it.",
                        "support_status": "external_supplement",
                        "citations": [],
                    }
                )
            return self._with_conversation_answer_source(
                (user_message, assistant_message),
                generation_path="heuristic_fallback",
                query=query,
                relevant_items=relevant_items,
            )

        grounded_citations = dedupe_citations(item.get("citation") for item in relevant_items)
        grounded_answer = self._compose_grounded_answer(normalized_message, relevant_items, response_style)
        LOGGER.info(
            "Using deterministic grounded chat fallback for query=%r with matched_items=%s.",
            safe_excerpt(query, 120),
            len(relevant_items),
        )
        assistant_message["reply_sections"].append(
            {
                "heading": "Grounded answer",
                "text": grounded_answer,
                "support_status": "slide_grounded" if len(relevant_items) == 1 else "inferred_from_slides",
                "citations": grounded_citations[:3],
            }
        )

        if grounding_mode == "lecture_with_fallback" and self._needs_external_supplement(normalized_message, relevant_items):
            supplement = self._external_supplement_text(normalized_message, grounded_answer)
            assistant_message["reply_sections"].append(
                {
                    "heading": "External supplement",
                    "text": supplement,
                    "support_status": "external_supplement",
                    "citations": [],
                }
            )
        return self._with_conversation_answer_source(
            (user_message, assistant_message),
            generation_path="heuristic_fallback",
            query=query,
            relevant_items=relevant_items,
        )

    def _select_chat_evidence(
        self,
        accessor: EvidenceAccessor,
        query: str,
        material_ids: Optional[Sequence[str]],
    ) -> List[Dict[str, Any]]:
        scored: List[tuple[int, Dict[str, Any]]] = []
        for item in accessor.filter_items(material_ids):
            score = lexical_overlap_score(query, item.get("text", ""))
            if score <= 0:
                continue
            scored.append((score, item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1].get("slide_number") or 0), pair[1].get("item_id", "")))
        deduped: List[Dict[str, Any]] = []
        seen_slide_keys: set[tuple[str, str]] = set()
        for _score, item in scored:
            slide_key = (str(item.get("material_id") or ""), str(item.get("slide_id") or item.get("item_id") or ""))
            if slide_key in seen_slide_keys:
                continue
            seen_slide_keys.add(slide_key)
            deduped.append(item)
            if len(deduped) >= 3:
                break
        return deduped

    def _chat_match_is_weak(self, query: str, relevant_items: Sequence[Dict[str, Any]]) -> bool:
        if not relevant_items:
            return True
        informative_query_tokens = informative_tokens(query)
        if not informative_query_tokens:
            return True
        strongest = max(
            lexical_overlap_score(
                query,
                " ".join([item.get("material_title", ""), item.get("text", "")]),
            )
            for item in relevant_items
        )
        grounded_tokens = set()
        for item in relevant_items:
            grounded_tokens.update(informative_tokens(item.get("text", "")))
        coverage = len(set(informative_query_tokens) & grounded_tokens)
        required_coverage = max(1, len(informative_query_tokens) // 2)
        return strongest <= 1 or coverage < required_coverage

    def _resolve_query_context(
        self,
        message_text: str,
        previous_messages: Sequence[Dict[str, Any]],
        accessor: EvidenceAccessor,
    ) -> str:
        tokens = informative_tokens(message_text)
        vague_pronouns = {"this", "that", "it", "they", "them", "these", "those"}
        if len(tokens) >= 1:
            return message_text
        if any(word in vague_pronouns for word in message_text.lower().split()):
            prior_user_messages = [msg.get("text", "") for msg in previous_messages if msg.get("role") == "user" and msg.get("text")]
            if prior_user_messages:
                return f"{prior_user_messages[-1]} {message_text}"
        if len(tokens) < 2:
            options = accessor.options_for_clarification(limit=5)
            raise NeedsUserInputError(
                "Please mention the concept, term, or slide you want explained.",
                options=options,
            )
        return message_text

    def _compose_grounded_answer(self, question_text: str, relevant_items: Sequence[Dict[str, Any]], response_style: str) -> str:
        if not relevant_items:
            return "I could not form a grounded answer from the current materials."
        points = [self._paraphrased_chat_point(question_text, item) for item in relevant_items]
        points = [point for point in points if point]
        deduped_points: List[str] = []
        seen_points: set[str] = set()
        for point in points:
            key = point.lower()
            if key in seen_points:
                continue
            seen_points.add(key)
            deduped_points.append(point)
        points = deduped_points
        if not points:
            points = [
                f"The cited slide treats {infer_concept_label(first_sentence(item.get('text', '')), fallback='this lecture idea')} as relevant to this question."
                for item in relevant_items
            ]
        if response_style == "concise":
            return normalize_whitespace(" ".join(point for point in points[:2] if point))
        if response_style == "step_by_step":
            steps = []
            for index, item in enumerate(relevant_items, start=1):
                steps.append(f"{index}. {self._paraphrased_chat_point(question_text, item)}")
            return "\n".join(steps)
        if self._is_definition_question(question_text):
            focus = self._question_focus(question_text)
            answer = self._paraphrased_definition_answer(focus or "", question_text, relevant_items)
            if len(points) > 1:
                answer = f"{answer} Related lecture emphasis: {points[1]}"
            return normalize_whitespace(answer)
        return normalize_whitespace(" ".join(points[:3]))

    def _summaries_for_chat(self, question_text: str, relevant_items: Sequence[Dict[str, Any]]) -> List[str]:
        fragments: List[tuple[int, str]] = []
        for item in relevant_items:
            summary = self._best_summary_for_item(question_text, item)
            if not summary:
                continue
            score = lexical_overlap_score(question_text, summary)
            if self._looks_like_slide_metadata(summary):
                score -= 3
            fragments.append((score, summary))
        fragments.sort(key=lambda pair: pair[0], reverse=True)
        ordered: List[str] = []
        seen: set[str] = set()
        for _score, summary in fragments:
            key = summary.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(summary)
        return ordered

    def _best_summary_for_item(self, question_text: str, item: Dict[str, Any]) -> str:
        cleaned = self._clean_evidence_text(item.get("text", ""))
        if not cleaned:
            return ""
        candidates = split_sentences(cleaned) or [cleaned]
        scored: List[tuple[int, str]] = []
        for candidate in candidates[:4]:
            fragment = normalize_whitespace(candidate)
            if len(fragment) < 18:
                continue
            score = lexical_overlap_score(question_text, fragment)
            if self._looks_like_slide_metadata(fragment):
                score -= 4
            scored.append((score, fragment))
        if not scored:
            return safe_excerpt(cleaned, 320)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        best = normalize_whitespace(scored[0][1])
        if len(best) <= 320:
            return best
        return safe_excerpt(best, 320)

    def _is_definition_question(self, question_text: str) -> bool:
        text = normalize_whitespace(question_text).lower()
        return bool(re.match(r"^(what is|what's|define|explain)\b", text))

    def _question_focus(self, question_text: str) -> Optional[str]:
        text = normalize_whitespace(question_text)
        match = re.search(r"(?i)\b(?:what is|what's|define|explain)\s+(?:an?\s+|the\s+)?(.+?)(?:\?|$)", text)
        if not match:
            return None
        focus = normalize_whitespace(match.group(1))
        if not focus:
            return None
        return focus

    def _clean_evidence_text(self, text: str) -> str:
        cleaned = normalize_whitespace(text or "")
        if not cleaned:
            return ""
        cleaned = re.sub(r"\b\d{2,3}-\d{3}/\d{2,3}-\d{3}\b", " ", cleaned)
        cleaned = re.sub(r"\b(?:Pat Virtue|Matt Gormley)\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r"\b(?:Machine Learning Department|School of Computer Science|Carnegie Mellon University)\b",
            " ",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\blecture\s+\d+\b", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+\d{1,3}\s*$", " ", cleaned)
        cleaned = re.sub(r"\s*[|/]\s*", " ", cleaned)
        cleaned = normalize_whitespace(cleaned)
        return cleaned

    def _chat_keywords_for_item(self, question_text: str, item: Dict[str, Any], limit: int = 4) -> List[str]:
        cleaned = self._clean_evidence_text(item.get("text", ""))
        concept = infer_concept_label(first_sentence(cleaned), fallback="")
        concept_tokens = set(informative_tokens(concept))
        question_tokens = set(informative_tokens(question_text))
        chat_noise_tokens = {
            "add",
            "adds",
            "against",
            "describe",
            "describes",
            "help",
            "helps",
            "make",
            "makes",
            "reduce",
            "reduces",
            "term",
            "tune",
            "tunes",
            "use",
            "used",
            "using",
        }
        ranked_keywords = top_keywords([cleaned], limit=limit + 4)
        filtered_keywords = [
            keyword
            for keyword in ranked_keywords
            if keyword not in concept_tokens and keyword not in chat_noise_tokens
        ]
        overlap_keywords = [keyword for keyword in filtered_keywords if keyword in question_tokens]
        supporting_keywords = [keyword for keyword in filtered_keywords if keyword not in overlap_keywords]
        return (overlap_keywords + supporting_keywords)[:limit]

    def _keyword_phrase(self, keywords: Sequence[str]) -> str:
        cleaned = [normalize_whitespace(keyword) for keyword in keywords if normalize_whitespace(keyword)]
        if not cleaned:
            return "the lecture-supported idea in the cited material"
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    def _paraphrased_definition_answer(self, focus: str, question_text: str, relevant_items: Sequence[Dict[str, Any]]) -> str:
        lead_item = relevant_items[0]
        lead_keywords = self._chat_keywords_for_item(question_text, lead_item)
        concept = infer_concept_label(first_sentence(lead_item.get("text", "")), fallback=focus or "this topic")
        if focus and lead_keywords:
            return f"In these slides, {focus} is tied to {self._keyword_phrase(lead_keywords)}."
        if focus:
            return f"In these slides, {focus} is explained through the lecture idea around {concept}."
        if lead_keywords:
            return f"The slides tie this idea to {self._keyword_phrase(lead_keywords)}."
        return f"The slides treat {concept} as the main lecture idea relevant to this question."

    def _paraphrased_chat_point(self, question_text: str, item: Dict[str, Any]) -> str:
        cleaned = self._clean_evidence_text(item.get("text", ""))
        concept = infer_concept_label(first_sentence(cleaned), fallback=f"slide {item.get('slide_number')}")
        patterned_phrase = self._pattern_paraphrase_phrase(cleaned)
        if patterned_phrase:
            return f"The slides connect {concept} to {patterned_phrase}."
        keywords = self._chat_keywords_for_item(question_text, item)
        if keywords:
            phrase = self._keyword_phrase(keywords)
            if concept and concept.lower() not in phrase.lower():
                return f"The slides connect {concept} to {phrase}."
            return f"The slides emphasize {phrase}."
        if concept:
            return f"The cited slide treats {concept} as one of the main ideas relevant to this question."
        return "The cited material contains a relevant lecture-supported point for this question."

    def _pattern_paraphrase_phrase(self, cleaned_text: str) -> Optional[str]:
        lowered = cleaned_text.lower()
        if "regularization" in lowered and "penalty" in lowered:
            if "flexible" in lowered or "overfit" in lowered:
                return "penalty terms and controlling model flexibility"
            return "penalty terms inside the objective"
        if "validation" in lowered and ("tune" in lowered or "generalization" in lowered):
            return "validation-based tuning and generalization checks"
        if "transformer" in lowered and "attention" in lowered:
            return "attention-based sequence modeling"
        if "gradient" in lowered and ("descent" in lowered or "update" in lowered):
            return "gradient-based parameter updates"
        if "probability" in lowered and "distribution" in lowered:
            return "probabilistic modeling of the output distribution"
        return None

    def _token_ngrams(self, text: str, size: int = 8) -> set[str]:
        normalized = normalize_whitespace(text).lower()
        tokens = re.findall(r"[a-z0-9]+", normalized)
        if len(tokens) < size:
            return set()
        return {" ".join(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}

    def _repeats_source_material(
        self,
        response_text: str,
        relevant_items: Sequence[Dict[str, Any]],
        *,
        min_words: int = 8,
    ) -> bool:
        response_ngrams = self._token_ngrams(response_text, size=min_words)
        if not response_ngrams:
            return False
        for item in relevant_items:
            evidence_ngrams = self._token_ngrams(self._clean_evidence_text(item.get("text", "")), size=min_words)
            if response_ngrams & evidence_ngrams:
                return True
        return False

    def _looks_like_slide_metadata(self, text: str) -> bool:
        lowered = normalize_whitespace(text).lower()
        if not lowered:
            return True
        if re.search(r"\b\d{2,3}-\d{3}\b", lowered):
            return True
        if "carnegie mellon university" in lowered:
            return True
        if "school of computer science" in lowered:
            return True
        if "machine learning department" in lowered:
            return True
        token_count = len(informative_tokens(lowered))
        return token_count < 2

    def _needs_external_supplement(self, question_text: str, relevant_items: Sequence[Dict[str, Any]]) -> bool:
        question_tokens = set(informative_tokens(question_text))
        covered_tokens = set()
        for item in relevant_items:
            covered_tokens.update(informative_tokens(item.get("text", "")))
        uncovered = [token for token in question_tokens if token not in covered_tokens]
        broad_cues = {"history", "intuitively", "outside", "broader", "real", "practical", "example"}
        return bool(question_tokens & broad_cues) or (len(uncovered) >= 2 and bool(question_tokens))

    def _external_supplement_text(self, question_text: str, grounded_answer: str) -> str:
        supplement = self.gemini.external_supplement(question_text, grounded_answer)
        if supplement:
            return supplement
        focus = self._extract_focus_phrase(question_text) or normalize_whitespace(question_text.rstrip("?"))
        return (
            f"Supplementary background only: the uploaded materials do not fully cover '{focus}'. "
            "Use external references only as background, and verify any broader explanation against your lecture slides or course textbook before relying on it."
        )

    # --------------------------
    # Practice generation
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
        accessor = EvidenceAccessor(bundle)
        lecture_material_ids = accessor.lecture_material_ids(template_material_id)
        lecture_records = accessor.concept_records(lecture_material_ids)
        if not lecture_records:
            raise NeedsUserInputError("I need lecture evidence before I can generate a practice set.")
        topic = normalize_whitespace(topic_text or "")
        if topic:
            ranked_for_topic = self._rank_records_for_practice_topic(lecture_records, topic)
            strongest_topic_score = max(int(record.get("_topic_score", 0)) for record in ranked_for_topic)
            if strongest_topic_score <= 1:
                raise NeedsUserInputError(
                    f"I could not find strong lecture evidence for '{topic}'. Please confirm a narrower topic before generating a test.",
                    options=accessor.options_for_clarification(lecture_material_ids, limit=5),
                )
            lecture_records = ranked_for_topic

        template_style_summary = None
        template_verbs: List[str] = []
        if generation_mode == "template_mimic":
            template_items = accessor.filter_items([template_material_id])
            if not template_items:
                raise ArtifactValidationError(
                    "template_material_id was provided, but the selected evidence does not include that template material."
                )
            template_style_summary, template_verbs = self._analyze_template_style(template_items)

        selected_records = self._select_records_for_questions(lecture_records, question_count, coverage_mode, topic)
        questions = []
        question_types = self._question_types_for_mode(generation_mode, len(selected_records), template_items_present=bool(template_material_id))
        for index, (record, question_type) in enumerate(zip(selected_records, question_types), start=1):
            comparison_record = self._comparison_record(selected_records, index - 1, record)
            questions.append(
                self._build_question(
                    record=record,
                    comparison_record=comparison_record,
                    question_index=index,
                    question_type=question_type,
                    generation_mode=generation_mode,
                    difficulty_profile=difficulty_profile,
                    include_answer_key=include_answer_key,
                    include_rubrics=include_rubrics,
                    template_verbs=template_verbs,
                )
            )

        cited_slides = sorted({slide for question in questions for slide in question.get("covered_slides", [])})
        considered_slides = accessor.distinct_slide_numbers(lecture_material_ids)
        uncited_or_skipped = sorted(set(considered_slides) - set(cited_slides))
        notes = self._coverage_notes(coverage_mode, considered_slides, cited_slides, question_count, topic)
        estimated_duration = sum(int(question.get("estimated_minutes") or 0) for question in questions)

        artifact = {
            "practice_set_id": make_id("practice_set"),
            "parent_practice_set_id": parent_practice_set_id,
            "workspace_id": bundle.get("workspace_id"),
            "created_at": utc_now_iso(),
            "topic_text": topic,
            "generation_mode": generation_mode,
            "template_style_summary": template_style_summary,
            "estimated_duration_minutes": estimated_duration,
            "questions": questions,
            "coverage_report": {
                "considered_slide_count": len(considered_slides),
                "cited_slide_count": len(cited_slides),
                "uncited_or_skipped_slides": uncited_or_skipped,
                "notes": notes,
            },
            "human_loop_summary": {
                "used_inputs": [
                    {"label": "Topic focus", "value": topic or "All ready grounded materials"},
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
        return artifact

    def revise_practice_set(
        self,
        *,
        existing_practice_set: Dict[str, Any],
        instruction_text: str,
        target_question_ids: Sequence[str],
        locked_question_ids: Sequence[str],
        maintain_coverage: bool,
    ) -> Dict[str, Any]:
        practice_set = copy.deepcopy(existing_practice_set)
        questions = practice_set.get("questions", [])
        question_ids = {question["question_id"] for question in questions}
        locked = set(locked_question_ids)
        missing_locked = locked - question_ids
        if missing_locked:
            raise ArtifactValidationError(f"Unknown locked_question_ids: {sorted(missing_locked)}")
        targets = set(target_question_ids) if target_question_ids else set(question_ids)
        missing_targets = targets - question_ids
        if missing_targets:
            raise ArtifactValidationError(f"Unknown target_question_ids: {sorted(missing_targets)}")

        lowered_instruction = instruction_text.lower()
        revised_questions = []
        for question in questions:
            if question["question_id"] in locked or question["question_id"] not in targets:
                revised_questions.append(question)
                continue
            revised_questions.append(self._revise_question(question, instruction_text, lowered_instruction, maintain_coverage))

        revised = {
            **practice_set,
            "practice_set_id": make_id("practice_set"),
            "parent_practice_set_id": existing_practice_set.get("practice_set_id"),
            "created_at": utc_now_iso(),
            "questions": revised_questions,
        }
        notes = revised["coverage_report"].get("notes", "")
        revised["coverage_report"]["notes"] = f"{notes} Revised after feedback while preserving stored history.".strip()
        return revised

    def _select_records_for_questions(
        self,
        lecture_records: Sequence[Dict[str, Any]],
        question_count: int,
        coverage_mode: str,
        topic_text: str = "",
    ) -> List[Dict[str, Any]]:
        if not lecture_records:
            return []
        ordered = list(lecture_records)
        if topic_text:
            ordered.sort(
                key=lambda record: (
                    -int(record.get("_topic_score", 0)),
                    int(record.get("slide_number") or 0),
                    record.get("concept_name", ""),
                )
            )
        if coverage_mode == "balanced":
            selected: List[Dict[str, Any]] = []
            while len(selected) < question_count:
                for record in ordered:
                    selected.append(record)
                    if len(selected) >= question_count:
                        break
            return selected
        if coverage_mode in {"high_coverage", "exhaustive"}:
            selected = list(ordered[:question_count])
            if len(selected) < question_count:
                idx = 0
                while len(selected) < question_count:
                    selected.append(ordered[idx % len(ordered)])
                    idx += 1
            return selected
        return list(ordered[:question_count])

    def _rank_records_for_practice_topic(
        self,
        lecture_records: Sequence[Dict[str, Any]],
        topic_text: str,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        for record in lecture_records:
            annotated = copy.deepcopy(record)
            annotated["_topic_score"] = self._record_overlap(record, topic_text)
            ranked.append(annotated)
        ranked.sort(
            key=lambda record: (
                -int(record.get("_topic_score", 0)),
                int(record.get("slide_number") or 0),
                record.get("concept_name", ""),
            )
        )
        return ranked

    def _question_types_for_mode(self, generation_mode: str, count: int, template_items_present: bool) -> List[str]:
        if generation_mode == "multiple_choice":
            return ["multiple_choice"] * count
        if generation_mode == "short_answer":
            return ["short_answer"] * count
        if generation_mode == "long_answer":
            return ["long_answer"] * count
        if generation_mode == "mixed":
            sequence = ["multiple_choice", "short_answer", "long_answer"]
            return [sequence[index % len(sequence)] for index in range(count)]
        # template mimic defaults to mixed to preserve style variety.
        return ["short_answer" if index % 2 == 0 else "long_answer" for index in range(count)]

    def _build_question(
        self,
        *,
        record: Dict[str, Any],
        comparison_record: Optional[Dict[str, Any]],
        question_index: int,
        question_type: str,
        generation_mode: str,
        difficulty_profile: str,
        include_answer_key: bool,
        include_rubrics: bool,
        template_verbs: Sequence[str],
    ) -> Dict[str, Any]:
        concept = record["concept_name"]
        summary = take_sentences(record["summary_text"], 2) or record["summary_text"]
        if generation_mode == "template_mimic" and template_verbs:
            lead = template_verbs[(question_index - 1) % len(template_verbs)]
        else:
            lead = self._default_lead_verb(question_type, difficulty_profile, question_index)

        if question_type == "multiple_choice":
            stem = f"{lead} the most defensible lecture-grounded interpretation of {concept}."
            if difficulty_profile == "harder":
                stem = (
                    f"{lead} which interpretation of {concept} is most defensible given the lecture evidence, "
                    "especially when distinguishing it from nearby concepts or common mistakes."
                )
            answer_choices, expected_answer = self._multiple_choice_options(
                concept,
                summary,
                record,
                comparison_record,
                question_index,
                difficulty_profile,
                include_answer_key,
            )
            scoring_guide = "Award credit only for the option that best matches the lecture-grounded reasoning, not the most generic-sounding statement."
        elif question_type == "short_answer":
            if difficulty_profile == "harder":
                if comparison_record:
                    stem = (
                        f"{lead} {concept} in a way that distinguishes it from {comparison_record['concept_name']}, "
                        "then identify one condition, trade-off, or failure mode that matters in practice."
                    )
                else:
                    stem = f"{lead} {concept} and identify one condition, trade-off, or failure mode implied by the lecture materials."
            else:
                stem = f"{lead} {concept} using grounded lecture evidence, then state why it matters in the larger method or evaluation flow."
            answer_choices = []
            expected_answer = self._reference_answer(question_type, concept, summary, comparison_record, difficulty_profile, include_answer_key)
            scoring_guide = None
        else:
            if difficulty_profile == "easier":
                stem = f"{lead} {concept} and summarize the main idea in the lecture in a well-structured paragraph."
            elif comparison_record:
                stem = (
                    f"{lead} a realistic failure case or design decision involving {concept}. "
                    f"Explain how it should be handled, why it should not be confused with {comparison_record['concept_name']}, "
                    "and what evidence or caution from the lecture supports your reasoning."
                )
            else:
                stem = (
                    f"{lead} a realistic failure case or design decision involving {concept}. "
                    "Explain how it should be handled, connect it to the lecture's broader procedure or evaluation logic, "
                    "and state one caution that would prevent a common mistake."
                )
            answer_choices = []
            expected_answer = self._reference_answer(question_type, concept, summary, comparison_record, difficulty_profile, include_answer_key)
            scoring_guide = None
        rubric = self._rubric_for_question(question_type, concept, include_rubrics, difficulty_profile)
        if question_type == "long_answer":
            scoring_guide = (
                f"Full credit requires a correct explanation of {concept}, a grounded connection to the lecture context, and one clearly stated caution or application detail."
            )
        return {
            "question_id": make_id("question"),
            "question_type": question_type,
            "stem": stem,
            "expected_answer": expected_answer,
            "answer_choices": answer_choices,
            "rubric": rubric,
            "scoring_guide_text": scoring_guide,
            "citations": record["citations"][:3],
            "covered_slides": distinct_slide_numbers(record["citations"][:3]),
            "difficulty": difficulty_profile,
            "estimated_minutes": self._estimated_minutes(question_type, difficulty_profile),
        }

    def _rubric_for_question(self, question_type: str, concept: str, include_rubrics: bool, difficulty_profile: str) -> List[Dict[str, Any]]:
        if not include_rubrics:
            return []
        criteria = [
            {
                "criterion": "Core idea",
                "description": f"Accurately states the main lecture-supported idea behind {concept}.",
                "points": 2,
            },
            {
                "criterion": "Grounded detail",
                "description": "Uses a lecture-supported detail, condition, or implication rather than a vague restatement.",
                "points": 2,
            },
        ]
        if question_type == "long_answer" or difficulty_profile == "harder":
            criteria.append(
                {
                    "criterion": "Reasoning / application",
                    "description": "Explains how the concept is used, connected, or checked in context.",
                    "points": 3,
                }
            )
        return criteria

    def _multiple_choice_options(
        self,
        concept: str,
        summary: str,
        record: Dict[str, Any],
        comparison_record: Optional[Dict[str, Any]],
        question_index: int,
        difficulty_profile: str,
        include_answer_key: bool,
    ) -> Tuple[List[str], str]:
        role_statement = self._role_statement(summary, concept)
        comparison_name = comparison_record.get("concept_name") if comparison_record else None
        correct = f"It is mainly used to {role_statement}."
        distractors = self._multiple_choice_distractors(concept, comparison_name, difficulty_profile)
        options = [correct] + distractors
        rotation = (question_index - 1) % len(options)
        rotated = options[rotation:] + options[:rotation]
        expected = ""
        if include_answer_key:
            expected = f"The correct choice is the option that treats {concept} as something mainly used to {role_statement}."
            if comparison_name:
                expected += f" It should not collapse {concept} into {comparison_name}."
        return rotated, expected

    def _reference_answer(
        self,
        question_type: str,
        concept: str,
        summary: str,
        comparison_record: Optional[Dict[str, Any]],
        difficulty_profile: str,
        include_answer_key: bool,
    ) -> str:
        if not include_answer_key:
            return ""
        core = self._summary_fragment(summary)
        comparison_name = comparison_record.get("concept_name") if comparison_record else ""
        answer = f"A strong answer identifies {concept} as {core}."
        if question_type == "short_answer" and comparison_name:
            answer += f" It also distinguishes {concept} from {comparison_name}."
        if question_type == "long_answer" or difficulty_profile == "harder":
            answer += " It should name a concrete condition, trade-off, application detail, or failure mode supported by the lecture."
        return answer

    def _summary_fragment(self, summary: str) -> str:
        fragment = take_sentences(summary, 1) or safe_excerpt(summary, 220)
        fragment = normalize_whitespace(fragment).rstrip(".")
        if not fragment:
            return "the lecture-grounded role described in the cited material"
        first = fragment[:1]
        if first.isupper():
            fragment = first.lower() + fragment[1:]
        return fragment

    def _comparison_record(
        self,
        records: Sequence[Dict[str, Any]],
        current_index: int,
        current_record: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if len(records) < 2:
            return None
        for offset in range(1, len(records)):
            candidate = records[(current_index + offset) % len(records)]
            if candidate.get("concept_name") != current_record.get("concept_name"):
                return candidate
        return None

    def _role_statement(self, summary: str, concept: str) -> str:
        text = normalize_whitespace(summary).lower()
        if any(token in text for token in ["validation", "generalize", "generalization", "holdout", "dev set", "development set", "hyperparameter", "tuning"]):
            return "check whether a modeling choice generalizes before the final evaluation step"
        if any(token in text for token in ["training", "train", "gradient", "optimiz", "update", "backprop"]):
            return "update model behavior during learning rather than only judge the final result"
        if any(token in text for token in ["regularization", "penalty", "overfit", "complexity", "constraint"]):
            return "limit model flexibility so the method does not overfit the available data"
        if any(token in text for token in ["accuracy", "error", "metric", "measure", "loss", "evaluation"]) and "final evaluation" not in text:
            return "evaluate performance against a defined criterion rather than modify the model directly"
        if any(token in text for token in ["feature", "embedding", "representation", "encode"]):
            return "encode useful information so later parts of the method can use it effectively"
        if any(token in text for token in ["assumption", "condition", "bias", "variance", "constraint", "caution"]):
            return "state a condition or caution that determines when the method can be trusted"
        if any(token in text for token in ["compare", "selection", "choose", "baseline", "alternative"]):
            return "compare alternatives and justify a design or modeling decision"
        concept_text = normalize_whitespace(concept).lower()
        if concept_text:
            return f"support the lecture's key reasoning step about {concept_text}"
        return "support the lecture's key reasoning step in context"

    def _multiple_choice_distractors(
        self,
        concept: str,
        comparison_name: Optional[str],
        difficulty_profile: str,
    ) -> List[str]:
        candidates = [
            "It is mainly used to rename parts of the workflow without changing how the main decision is made.",
            "It is mainly used to replace evidence checks with a fixed rule that supposedly works in every setting.",
            "It is mainly used to report the final result after the important design choices are already locked.",
            (
                f"It is mainly used to make {normalize_whitespace(comparison_name).lower()} interchangeable with {normalize_whitespace(concept).lower()}."
                if comparison_name
                else "It is mainly used to make neighboring concepts interchangeable even when the lecture keeps them separate."
            ),
        ]
        if difficulty_profile == "harder":
            candidates[0] = "It is mainly used to emphasize one surface feature while ignoring the lecture's deeper procedural role."
            candidates[1] = "It is mainly used to avoid trade-offs entirely by pretending the method needs no contextual checks."
        return candidates[:3]

    def _estimated_minutes(self, question_type: str, difficulty_profile: str) -> int:
        base = {
            "multiple_choice": 3,
            "short_answer": 6,
            "long_answer": 10,
        }.get(question_type, 6)
        if difficulty_profile == "easier":
            return max(2, base - 1)
        if difficulty_profile == "harder":
            return base + 2
        return base

    def _analyze_template_style(self, template_items: Sequence[Dict[str, Any]]) -> Tuple[str, List[str]]:
        verbs = []
        for item in template_items:
            sentence = first_sentence(item.get("text", ""))
            if not sentence:
                continue
            first_word = sentence.split()[0].strip("():.,").capitalize()
            if re.match(r"^[A-Za-z][A-Za-z\-]+$", first_word):
                verbs.append(first_word)
        if not verbs:
            verbs = ["Explain", "Compare", "Describe"]
        unique_verbs = list(dict.fromkeys(verbs))
        style_bits = []
        if any("compare" == verb.lower() for verb in unique_verbs):
            style_bits.append("comparison prompts")
        if any("explain" == verb.lower() for verb in unique_verbs):
            style_bits.append("directive explanation verbs")
        if any(word in " ".join(item.get("text", "").lower() for item in template_items) for word in ["justify", "state", "describe"]):
            style_bits.append("explicit scoring-oriented wording")
        summary = "Template favors " + ", ".join(style_bits or ["structured question commands"]) + "."
        return summary, unique_verbs

    def _coverage_notes(self, coverage_mode: str, considered_slides: Sequence[int], cited_slides: Sequence[int], question_count: int, topic_text: str) -> str:
        topic_prefix = f"Topic focus '{topic_text}'. " if topic_text else ""
        if coverage_mode == "exhaustive":
            if len(cited_slides) == len(considered_slides):
                return topic_prefix + "Exhaustive mode covered every slide represented in the lecture evidence."
            return (
                topic_prefix + "Exhaustive mode attempted full coverage, but the requested question_count was smaller than the number of distinct grounded lecture slides."
            )
        if coverage_mode == "high_coverage":
            return topic_prefix + "High-coverage mode prioritized breadth across grounded lecture slides before repeating any topic."
        return topic_prefix + "Balanced mode cycled through grounded topics to keep the set representative without forcing full slide-by-slide coverage."

    def _revise_question(
        self,
        question: Dict[str, Any],
        instruction_text: str,
        lowered_instruction: str,
        maintain_coverage: bool,
    ) -> Dict[str, Any]:
        revised = copy.deepcopy(question)
        revised["stem"] = self._apply_revision_text(revised["stem"], instruction_text)
        revised["expected_answer"] = self._apply_revision_text(revised["expected_answer"], instruction_text)
        if "harder" in lowered_instruction:
            revised["difficulty"] = "harder"
            if "justify" not in revised["stem"].lower():
                revised["stem"] += " Justify your reasoning with one grounded lecture detail."
        elif "easier" in lowered_instruction:
            revised["difficulty"] = "easier"
            revised["stem"] = re.sub(r"\s+and.*$", "", revised["stem"]).strip()
        elif "mixed" in lowered_instruction:
            revised["difficulty"] = "mixed"

        if "scenario" in lowered_instruction or "application" in lowered_instruction:
            revised["stem"] = (
                "Consider a realistic study or exam scenario. " + revised["stem"]
            )
        if revised.get("rubric"):
            for criterion in revised["rubric"]:
                criterion["description"] = self._apply_revision_text(criterion["description"], instruction_text)
        if maintain_coverage:
            revised["covered_slides"] = list(question.get("covered_slides", []))
            revised["citations"] = copy.deepcopy(question.get("citations", []))
        return revised

    # --------------------------
    # Shared helpers
    # --------------------------
    def _grounding_source_from_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "workspace_id": bundle.get("workspace_id"),
            "bundle_id": bundle.get("bundle_id"),
            "material_ids": list(bundle.get("material_ids") or []),
            "evidence_bundle": bundle,
        }

    def _find_record_by_keywords(
        self,
        concept_records: Sequence[Dict[str, Any]],
        keywords: Sequence[str],
    ) -> Optional[Dict[str, Any]]:
        keywords_lower = [word.lower() for word in keywords]
        for record in concept_records:
            haystack = " ".join([record.get("concept_name", ""), record.get("summary_text", "")]).lower()
            if any(word in haystack for word in keywords_lower):
                return record
        return concept_records[0] if concept_records else None

    def _apply_revision_text(self, text: str, instruction_text: str) -> str:
        lowered = instruction_text.lower()
        current = normalize_whitespace(text)
        if not current:
            return current
        if any(keyword in lowered for keyword in ["shorter", "condense", "concise"]):
            current = take_sentences(current, 1) or current
        if any(keyword in lowered for keyword in ["detail", "expand", "more detail", "deeper"]):
            detail_sentence = "Add one more explicit reason, condition, or example when studying this point."
            if detail_sentence.lower() not in current.lower():
                current = f"{current} {detail_sentence}".strip()
        if any(keyword in lowered for keyword in ["example", "application", "scenario"]):
            add = "Include a quick example or application check while revising it."
            if add.lower() not in current.lower():
                current = f"{current} {add}".strip()
        focus = self._extract_focus_phrase(instruction_text)
        if focus and focus.lower() not in current.lower() and "focus on" in lowered:
            current = f"{current} Give extra attention to {focus}.".strip()
        return current

    def _revise_time_budget(self, minutes: int, lowered_instruction: str) -> int:
        if "shorter" in lowered_instruction or "condense" in lowered_instruction:
            return max(5, int(minutes * 0.8))
        if "detail" in lowered_instruction or "expand" in lowered_instruction or "longer" in lowered_instruction:
            return max(5, int(minutes * 1.2))
        return minutes

    def _extract_focus_phrase(self, text: str) -> Optional[str]:
        match = re.search(r"focus on ([A-Za-z0-9\- ]+)", text, re.IGNORECASE)
        if match:
            return normalize_whitespace(match.group(1)).rstrip(".?!")
        tokens = informative_tokens(text)
        if len(tokens) >= 1 and len(tokens) <= 4:
            return " ".join(tokens)
        return None

    def _append_focus(self, text: str, focus: str) -> str:
        if focus.lower() in text.lower():
            return text
        return f"{text} ({focus})"

    def _default_lead_verb(self, question_type: str, difficulty_profile: str, index: int) -> str:
        if question_type == "short_answer":
            if difficulty_profile == "harder":
                return "Analyze"
            return "Explain"
        if difficulty_profile == "easier":
            return "Describe"
        if difficulty_profile == "harder":
            return "Justify"
        return "Explain"
