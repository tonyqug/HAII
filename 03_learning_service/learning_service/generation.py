from __future__ import annotations

import copy
import json
import math
import re
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


class OptionalGeminiClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_key = settings.gemini_api_key.strip()
        self.model = settings.gemini_model.strip() or "gemini-2.5-flash"
        self.timeout = settings.request_timeout_seconds

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _generate_text(self, system_instruction: str, user_prompt: str, max_output_tokens: int = 256) -> Optional[str]:
        if not self.configured:
            return None
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": max_output_tokens,
            },
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
            text_parts = [part.get("text", "") for part in parts if part.get("text")]
            combined = "\n".join(text_parts).strip()
            return combined or None
        except Exception:
            return None

    def external_supplement(self, question_text: str, grounded_answer: str) -> Optional[str]:
        prompt = (
            "Question from a student: "
            f"{question_text}\n\n"
            "Grounded answer already supported by their lecture materials: "
            f"{grounded_answer}\n\n"
            "Write a brief external supplement that is clearly general background knowledge, not a claim about the student's slides. "
            "Do not mention citations. Be concise and explicitly phrase this as supplementary background."
        )
        return self._generate_text(
            system_instruction=(
                "You provide carefully labeled supplemental background for students. "
                "Never imply that external background came from the user's uploaded materials."
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
            haystack = " ".join(
                [
                    item.get("material_title", ""),
                    str(item.get("slide_number") or ""),
                    item.get("text", ""),
                ]
            )
            score = lexical_overlap_score(query, haystack)
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
        for record in prerequisite_pool[:3]:
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
        while len(items) < 3:
            items.append(self._padding_item("prereq", grounding_mode, hint="Review baseline background knowledge that the lecture seems to assume before later steps build on it."))
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
        sequence_records = list(concept_records)[:6]
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
            tasks = [
                f"Review slide {slide_label or '?'} and restate {record['concept_name']} in your own words.",
            ]
            if quick_bridge and prior_knowledge:
                tasks.append(
                    f"Use what you already know about {prior_knowledge} as a quick bridge, then note what the lecture adds or changes."
                )
            else:
                tasks.append(
                    f"Write a short checkpoint explaining why this matters: {take_sentences(record['summary_text'], 1)}"
                )
            if index > 1:
                prev = sequence_records[index - 2]["concept_name"]
                tasks.append(f"Connect {record['concept_name']} to the earlier topic {prev}.")
            if weak_areas and self._record_overlap(record, weak_areas) > 0:
                tasks.append(f"Spend extra practice time here because it overlaps your stated weak area: {weak_areas}.")
            elif goals:
                tasks.append(f"Check how this step supports your goal: {goals}.")
            else:
                tasks.append("Turn this step into one self-test question you can answer without reopening the slide.")
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
            mistakes.append(
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Confusing {first['concept_name']} with {second['concept_name']}",
                    "why_it_happens": "The lecture moves through related concepts in sequence, so students can blur their roles if they only memorize isolated phrases.",
                    "prevention_advice": f"After studying, write one sentence that distinguishes {first['concept_name']} from {second['concept_name']}.",
                    "support_status": "inferred_from_slides",
                    "citations": dedupe_citations(first["citations"] + second["citations"])[:3],
                }
            )
        weak_area_candidate = self._find_record_by_text(sequence_pool or concept_records, weak_areas)
        if weak_areas and weak_area_candidate:
            mistakes.append(
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Leaving your weak area around {weak_areas} at the recognition level instead of practicing it actively",
                    "why_it_happens": f"The plan needs to do more than reread {weak_area_candidate['concept_name']}; weak areas usually improve only after explanation and self-testing.",
                    "prevention_advice": f"After reviewing the cited slide, answer one no-notes question focused on {weak_areas} before moving on.",
                    "support_status": "inferred_from_slides",
                    "citations": weak_area_candidate["citations"][:2],
                }
            )
        candidate = self._find_record_by_keywords(sequence_pool or concept_records, ["assumption", "condition", "validation", "rate", "penalty", "error", "overfit"])
        if candidate:
            mistakes.append(
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Ignoring the conditions or tuning choices around {candidate['concept_name']}",
                    "why_it_happens": "Students often remember the headline idea but skip the conditions, trade-offs, or checks that make it work correctly.",
                    "prevention_advice": "When reviewing the slide, list the condition, tuning choice, or warning next to the main concept before moving on.",
                    "support_status": "inferred_from_slides",
                    "citations": candidate["citations"][:2],
                }
            )
        candidate = self._find_record_by_keywords(sequence_pool or concept_records, ["example", "generalization", "training", "validation", "compare", "procedure"])
        if candidate and len(mistakes) < 3:
            mistakes.append(
                {
                    "item_id": make_id("mistake"),
                    "pattern": "Memorizing the wording without practicing when or how to apply it",
                    "why_it_happens": "The evidence suggests procedure-level or evaluation-level reasoning, which students can miss if they only copy definitions.",
                    "prevention_advice": "Turn each major slide into a quick self-test: what is the idea, when do you use it, and what failure mode should you watch for?",
                    "support_status": "inferred_from_slides",
                    "citations": candidate["citations"][:2],
                }
            )
        goal_candidate = self._find_record_by_text(sequence_pool or concept_records, goals)
        if goals and goal_candidate and len(mistakes) < 3:
            mistakes.append(
                {
                    "item_id": make_id("mistake"),
                    "pattern": f"Studying passively even though your goal is {goals}",
                    "why_it_happens": "Students often collect notes without checking whether they can produce an exam-ready explanation or decision process.",
                    "prevention_advice": f"Use the cited slide to create one retrieval-style checkpoint that matches your goal: {goals}.",
                    "support_status": "inferred_from_slides",
                    "citations": goal_candidate["citations"][:2],
                }
            )
        while len(mistakes) < 3:
            mistakes.append(
                self._padding_item(
                    "mistake",
                    grounding_mode,
                    hint="A likely mistake is moving too quickly from a definition to an answer without checking assumptions, examples, or edge cases.",
                )
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
        selected = list(ranked_records[:target_steps])
        selected.sort(
            key=lambda record: (
                int(record.get("slide_number") or 0),
                -int(record.get("_plan_score", 0)),
                record.get("concept_name", ""),
            )
        )
        return selected

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

        relevant_items = accessor.search(query, lecture_material_ids, top_k=4)
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

        if not relevant_items:
            assistant_message["reply_sections"].append(
                {
                    "heading": "Insufficient evidence from your materials",
                    "text": "I could not find grounded slide evidence that directly answers this question from the current materials.",
                    "support_status": "insufficient_evidence",
                    "citations": [],
                }
            )
            assistant_message["clarifying_question"] = {
                "prompt": "Point me to a topic, term, or slide number and I can answer more precisely.",
                "reason": "The current question did not match grounded lecture evidence closely enough.",
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
            return user_message, assistant_message

        grounded_citations = dedupe_citations(item.get("citation") for item in relevant_items)
        grounded_answer = self._compose_grounded_answer(normalized_message, relevant_items, response_style)
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
        return user_message, assistant_message

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
        summaries = self._summaries_for_chat(question_text, relevant_items)
        if not summaries:
            summaries = [safe_excerpt(item.get("text", ""), 180) for item in relevant_items if item.get("text")]
        if response_style == "concise":
            return normalize_whitespace(" ".join(summary for summary in summaries[:2] if summary))
        if response_style == "step_by_step":
            steps = []
            for index, item in enumerate(relevant_items, start=1):
                concept = infer_concept_label(first_sentence(item.get("text", "")), fallback=f"slide {item.get('slide_number')}")
                summary = self._best_summary_for_item(question_text, item) or safe_excerpt(item.get("text", ""), 120)
                steps.append(f"{index}. {concept}: {summary}")
            return "\n".join(steps)
        if self._is_definition_question(question_text):
            focus = self._question_focus(question_text)
            leading = summaries[0]
            if focus:
                answer = f"In these slides, {focus} is described as: {leading}"
            else:
                answer = leading
            if len(summaries) > 1:
                answer = f"{answer} Related detail: {summaries[1]}"
            return normalize_whitespace(answer)
        return normalize_whitespace(" ".join(summaries[:3]))

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
            return safe_excerpt(cleaned, 160)
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return safe_excerpt(scored[0][1], 180)

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
        return bool(uncovered) or bool(question_tokens & broad_cues) or len(relevant_items) < 2

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

        template_style_summary = None
        template_verbs: List[str] = []
        if generation_mode == "template_mimic":
            template_items = accessor.filter_items([template_material_id])
            if not template_items:
                raise ArtifactValidationError(
                    "template_material_id was provided, but the selected evidence does not include that template material."
                )
            template_style_summary, template_verbs = self._analyze_template_style(template_items)

        selected_records = self._select_records_for_questions(lecture_records, question_count, coverage_mode)
        questions = []
        question_types = self._question_types_for_mode(generation_mode, len(selected_records), template_items_present=bool(template_material_id))
        for index, (record, question_type) in enumerate(zip(selected_records, question_types), start=1):
            questions.append(
                self._build_question(
                    record=record,
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
        notes = self._coverage_notes(coverage_mode, considered_slides, cited_slides, question_count)

        artifact = {
            "practice_set_id": make_id("practice_set"),
            "parent_practice_set_id": parent_practice_set_id,
            "workspace_id": bundle.get("workspace_id"),
            "created_at": utc_now_iso(),
            "generation_mode": generation_mode,
            "template_style_summary": template_style_summary,
            "questions": questions,
            "coverage_report": {
                "considered_slide_count": len(considered_slides),
                "cited_slide_count": len(cited_slides),
                "uncited_or_skipped_slides": uncited_or_skipped,
                "notes": notes,
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
    ) -> List[Dict[str, Any]]:
        if not lecture_records:
            return []
        ordered = list(lecture_records)
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

    def _question_types_for_mode(self, generation_mode: str, count: int, template_items_present: bool) -> List[str]:
        if generation_mode == "short_answer":
            return ["short_answer"] * count
        if generation_mode == "long_answer":
            return ["long_answer"] * count
        if generation_mode == "mixed":
            return ["short_answer" if index % 2 == 0 else "long_answer" for index in range(count)]
        # template mimic defaults to mixed to preserve style variety.
        return ["short_answer" if index % 2 == 0 else "long_answer" for index in range(count)]

    def _build_question(
        self,
        *,
        record: Dict[str, Any],
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
        related = concept
        if generation_mode == "template_mimic" and template_verbs:
            lead = template_verbs[(question_index - 1) % len(template_verbs)]
        else:
            lead = self._default_lead_verb(question_type, difficulty_profile, question_index)

        if question_type == "short_answer":
            stem = f"{lead} {concept} as presented in the lecture materials, and state why it matters."
            if difficulty_profile == "harder":
                stem = f"{lead} {concept} and identify one trade-off, condition, or failure mode implied by the lecture materials."
        else:
            stem = (
                f"{lead} {concept}, connect it to the lecture's broader procedure or evaluation logic, "
                "and explain how you would avoid a common mistake when using it."
            )
            if difficulty_profile == "easier":
                stem = f"{lead} {concept} and summarize the main idea in the lecture in a well-structured paragraph."

        expected_answer = summary if include_answer_key else ""
        rubric = self._rubric_for_question(question_type, concept, include_rubrics, difficulty_profile)
        scoring_guide = None
        if question_type == "long_answer":
            scoring_guide = (
                f"Full credit requires a correct explanation of {concept}, a grounded connection to the lecture context, and one clearly stated caution or application detail."
            )
        return {
            "question_id": make_id("question"),
            "question_type": question_type,
            "stem": stem,
            "expected_answer": expected_answer,
            "rubric": rubric,
            "scoring_guide_text": scoring_guide,
            "citations": record["citations"][:3],
            "covered_slides": distinct_slide_numbers(record["citations"][:3]),
            "difficulty": difficulty_profile,
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

    def _coverage_notes(self, coverage_mode: str, considered_slides: Sequence[int], cited_slides: Sequence[int], question_count: int) -> str:
        if coverage_mode == "exhaustive":
            if len(cited_slides) == len(considered_slides):
                return "Exhaustive mode covered every slide represented in the lecture evidence."
            return (
                "Exhaustive mode attempted full coverage, but the requested question_count was smaller than the number of distinct grounded lecture slides."
            )
        if coverage_mode == "high_coverage":
            return "High-coverage mode prioritized breadth across grounded lecture slides before repeating any topic."
        return "Balanced mode cycled through grounded topics to keep the set representative without forcing full slide-by-slide coverage."

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
                return "Explain"
            return "Define"
        if difficulty_profile == "easier":
            return "Describe"
        if difficulty_profile == "harder":
            return "Analyze"
        return "Explain"
