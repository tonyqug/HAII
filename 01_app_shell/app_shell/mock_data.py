from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

from app_shell.utils import deep_copy, make_id, slugify, utc_now_iso


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "mock_workspace.json"


def load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _citation_id(citation: dict) -> str:
    parts = [
        citation.get("material_id", "material"),
        citation.get("slide_id", "slide"),
        str(citation.get("slide_number", "0")),
    ]
    return "citation_" + "_".join(part.replace("/", "_") for part in parts)


def enrich_citation(workspace_id: str, citation: dict, *, fixture_mode: bool) -> dict:
    enriched = deep_copy(citation)
    enriched.setdefault("citation_id", _citation_id(enriched))
    material_id = enriched.get("material_id", "unknown_material")
    slide_id = enriched.get("slide_id", "slide_1")
    base_path = "mock" if fixture_mode else "local"
    preview_url = f"/{base_path}/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/preview"
    source_open_url = f"/{base_path}/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/source"
    enriched.setdefault("preview_url", preview_url)
    enriched.setdefault("source_open_url", source_open_url)
    return enriched


def _enrich_citations_in_object(workspace_id: str, payload: Any, *, fixture_mode: bool) -> Any:
    if isinstance(payload, list):
        return [_enrich_citations_in_object(workspace_id, item, fixture_mode=fixture_mode) for item in payload]
    if isinstance(payload, dict):
        updated = {}
        for key, value in payload.items():
            if key == "citations" and isinstance(value, list):
                updated[key] = [enrich_citation(workspace_id, citation, fixture_mode=fixture_mode) for citation in value]
            else:
                updated[key] = _enrich_citations_in_object(workspace_id, value, fixture_mode=fixture_mode)
        return updated
    return payload


def build_workspace_from_fixture() -> dict:
    raw = load_fixture()
    workspace = raw["workspace"]
    workspace_id = workspace["workspace_id"]
    now = utc_now_iso()
    materials = {material["material_id"]: material for material in deep_copy(raw.get("materials", []))}
    for material in materials.values():
        material.setdefault("workspace_id", workspace_id)
        material.setdefault("source_view_url", f"/mock/workspaces/{workspace_id}/materials/{material['material_id']}/slides/{material['slides'][0]['slide_id']}/source")

    study_plan = _enrich_citations_in_object(workspace_id, raw.get("study_plan"), fixture_mode=True)
    conversation = _enrich_citations_in_object(workspace_id, raw.get("conversation"), fixture_mode=True)
    practice_set = _enrich_citations_in_object(workspace_id, raw.get("practice_set"), fixture_mode=True)

    history = [
        {
            "artifact_type": "study_plan",
            "artifact_id": study_plan["study_plan_id"],
            "created_at": study_plan["created_at"],
            "parent_artifact_id": study_plan.get("parent_study_plan_id"),
            "active": True,
            "title": study_plan.get("topic_text") or "Study plan",
        },
        {
            "artifact_type": "conversation",
            "artifact_id": conversation["conversation_id"],
            "created_at": conversation["created_at"],
            "parent_artifact_id": None,
            "active": True,
            "title": conversation.get("title") or "Conversation",
        },
        {
            "artifact_type": "practice_set",
            "artifact_id": practice_set["practice_set_id"],
            "created_at": practice_set["created_at"],
            "parent_artifact_id": practice_set.get("parent_practice_set_id"),
            "active": True,
            "title": practice_set.get("generation_mode") or "Practice set",
        },
    ]

    return {
        "workspace_id": workspace_id,
        "display_name": workspace.get("display_name") or "Mock workspace",
        "created_at": workspace.get("created_at") or now,
        "last_opened_at": workspace.get("last_opened_at") or now,
        "archived": False,
        "deleted": False,
        "grounding_mode": workspace.get("grounding_mode", "strict_lecture_only"),
        "topic_text": workspace.get("topic_text", ""),
        "time_budget_minutes": workspace.get("time_budget_minutes", 90),
        "student_context": workspace.get("student_context", {"known": "", "weak_areas": "", "goals": ""}),
        "active_material_ids": _group_material_ids(materials),
        "selected_active_conversation_id": conversation["conversation_id"],
        "active_study_plan_id": study_plan["study_plan_id"],
        "active_practice_set_id": practice_set["practice_set_id"],
        "known_study_plan_ids": [study_plan["study_plan_id"]],
        "known_practice_set_ids": [practice_set["practice_set_id"]],
        "known_conversation_ids": [conversation["conversation_id"]],
        "materials": materials,
        "study_plans": {study_plan["study_plan_id"]: study_plan},
        "practice_sets": {practice_set["practice_set_id"]: practice_set},
        "conversations": {conversation["conversation_id"]: conversation},
        "annotations": [],
        "material_preferences": {material_id: "default" for material_id in materials},
        "material_sync_annotations": {},
        "feedback_history": [],
        "ui_preferences": {
            "current_tab": "Overview",
            "sidebar_open": True,
            "source_viewer_open": False,
            "last_active_citation": None,
        },
        "local_edits": {},
        "status": {
            "services": {
                "content": {"available": True, "mode": "mock", "last_checked_at": now},
                "learning": {"available": True, "mode": "mock", "last_checked_at": now},
            },
            "last_successful_sync": {"content": now, "learning": now},
            "warnings": [],
        },
        "history": history,
        "pending_user_actions": [],
    }


def _group_material_ids(materials: Dict[str, dict]) -> Dict[str, List[str]]:
    grouped = {"slides": [], "notes": [], "practice_template": []}
    for material_id, material in materials.items():
        role = material.get("role", "notes")
        grouped.setdefault(role, []).append(material_id)
    return grouped


def build_blank_workspace(workspace_id: str, display_name: str) -> dict:
    now = utc_now_iso()
    return {
        "workspace_id": workspace_id,
        "display_name": display_name or "Untitled workspace",
        "created_at": now,
        "last_opened_at": now,
        "archived": False,
        "deleted": False,
        "grounding_mode": "strict_lecture_only",
        "topic_text": "",
        "time_budget_minutes": None,
        "student_context": {"known": "", "weak_areas": "", "goals": ""},
        "active_material_ids": {"slides": [], "notes": [], "practice_template": []},
        "selected_active_conversation_id": None,
        "active_study_plan_id": None,
        "active_practice_set_id": None,
        "known_study_plan_ids": [],
        "known_practice_set_ids": [],
        "known_conversation_ids": [],
        "materials": {},
        "study_plans": {},
        "practice_sets": {},
        "conversations": {},
        "annotations": [],
        "material_preferences": {},
        "material_sync_annotations": {},
        "feedback_history": [],
        "ui_preferences": {
            "current_tab": "Overview",
            "sidebar_open": True,
            "source_viewer_open": False,
            "last_active_citation": None,
        },
        "local_edits": {},
        "status": {
            "services": {
                "content": {"available": False, "mode": "unknown", "last_checked_at": now},
                "learning": {"available": False, "mode": "unknown", "last_checked_at": now},
            },
            "last_successful_sync": {"content": None, "learning": None},
            "warnings": [],
        },
        "history": [],
        "pending_user_actions": [],
    }


def _primary_material(workspace: dict) -> dict | None:
    materials = list(workspace.get("materials", {}).values())
    ready = [material for material in materials if material.get("processing_status") == "ready"]
    if ready:
        return ready[0]
    return materials[0] if materials else None


def _fallback_slide(material: dict | None) -> dict:
    if not material:
        return {
            "slide_id": "slide_1",
            "slide_number": 1,
            "title": "Uploaded material",
            "snippet_text": "No parsed slide preview is available yet.",
            "bullets": ["Import a lecture source to enable grounded study help."],
        }
    slides = material.get("slides") or []
    if slides:
        return slides[0]
    return {
        "slide_id": f"{slugify(material.get('title', 'slide'))}_1",
        "slide_number": 1,
        "title": material.get("title", "Uploaded material"),
        "snippet_text": material.get("text_body") or material.get("quality_summary", {}).get("notes") or "Preview unavailable.",
        "bullets": [material.get("title", "Uploaded material")],
    }


def _make_citation(workspace_id: str, material: dict | None, slide: dict | None, support_type: str = "explicit") -> dict:
    slide_payload = slide or _fallback_slide(material)
    citation = {
        "material_id": (material or {}).get("material_id", "material_local"),
        "material_title": (material or {}).get("title", "Uploaded material"),
        "slide_id": slide_payload.get("slide_id", "slide_1"),
        "slide_number": int(slide_payload.get("slide_number", 1)),
        "snippet_text": slide_payload.get("snippet_text", "Preview unavailable."),
        "support_type": support_type,
        "confidence": "high",
    }
    return enrich_citation(workspace_id, citation, fixture_mode=False)


def create_mock_material(workspace_id: str, title: str, role: str, *, source_kind: str, text_body: str | None = None, filename: str | None = None) -> dict:
    now = utc_now_iso()
    material_id = make_id("material")
    display_title = title or filename or "Uploaded material"
    kind = "pasted_text" if source_kind == "pasted_text" else (Path(filename).suffix.lstrip(".").lower() if filename else "file")
    snippet_text = (text_body or f"Imported file: {display_title}").strip()
    slide_id = f"slide_{slugify(display_title)}_1"
    return {
        "material_id": material_id,
        "workspace_id": workspace_id,
        "title": display_title,
        "role": role,
        "kind": kind or "other supported import",
        "processing_status": "ready",
        "page_count": 1,
        "created_at": now,
        "quality_summary": {"overall": "medium", "notes": "Locally generated mock material preview."},
        "text_body": text_body or "",
        "slides": [
            {
                "slide_id": slide_id,
                "slide_number": 1,
                "title": display_title,
                "snippet_text": snippet_text,
                "bullets": [snippet_text[:120]],
            }
        ],
        "source_view_url": f"/local/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/source",
    }


def generate_mock_study_plan(workspace: dict, request_payload: dict, *, parent_study_plan_id: str | None = None) -> dict:
    workspace_id = workspace["workspace_id"]
    material = _primary_material(workspace)
    slide = _fallback_slide(material)
    base_citation = _make_citation(workspace_id, material, slide)
    now = utc_now_iso()
    title = request_payload.get("topic_text") or workspace.get("topic_text") or material.get("title", "Uploaded materials") if material else "Uploaded materials"
    prior_knowledge = (workspace.get("student_context", {}) or {}).get("known", "")
    weak_areas = (workspace.get("student_context", {}) or {}).get("weak_areas", "")
    goals = (workspace.get("student_context", {}) or {}).get("goals", "")
    plan_id = make_id("study_plan")
    return {
        "study_plan_id": plan_id,
        "parent_study_plan_id": parent_study_plan_id,
        "workspace_id": workspace_id,
        "created_at": now,
        "topic_text": title,
        "time_budget_minutes": int(request_payload.get("time_budget_minutes") or workspace.get("time_budget_minutes") or 60),
        "grounding_mode": request_payload.get("grounding_mode") or workspace.get("grounding_mode") or "strict_lecture_only",
        "prerequisites": [
            {
                "item_id": make_id("pre"),
                "concept_name": "Core lecture concepts",
                "why_needed": "The generated plan relies on the uploaded lecture source and any notes you provided.",
                "support_status": "slide_grounded",
                "citations": [base_citation],
            }
        ],
        "study_sequence": [
            {
                "step_id": make_id("step"),
                "order_index": 1,
                "title": "Review the grounded source first",
                "objective": "Read the cited slide or note and restate the main concept in your own words.",
                "recommended_time_minutes": max(10, int((request_payload.get("time_budget_minutes") or 60) // 3)),
                "tasks": [
                    "Open the cited source slide",
                    "Write a one-sentence summary",
                    "Check any formulas or definitions against your notes",
                ],
                "milestone": "You can explain the first grounded concept without reopening the source.",
                "depends_on": [],
                "support_status": "slide_grounded",
                "citations": [base_citation],
            },
            {
                "step_id": make_id("step"),
                "order_index": 2,
                "title": "Practice active recall",
                "objective": "Turn the lecture content into a short recall prompt and answer it without looking.",
                "recommended_time_minutes": max(10, int((request_payload.get("time_budget_minutes") or 60) // 3)),
                "tasks": [
                    "Cover the slide",
                    "State the main takeaway aloud",
                    "Compare your answer to the cited evidence",
                ],
                "milestone": "You can answer one self-test prompt from memory and verify it against the cited slide.",
                "depends_on": [],
                "support_status": "slide_grounded",
                "citations": [base_citation],
            },
        ],
        "common_mistakes": [
            {
                "item_id": make_id("mistake"),
                "pattern": "Skipping the cited evidence before memorizing the answer",
                "why_it_happens": "Students often jump to memorization before checking how the lecture phrases the concept.",
                "prevention_advice": "Open the source viewer and anchor your summary in the cited slide or note.",
                "support_status": "slide_grounded",
                "citations": [base_citation],
            },
            {
                "item_id": make_id("mistake"),
                "pattern": "Studying all materials before any are ready",
                "why_it_happens": "Uploads can still be processing while the workspace opens.",
                "prevention_advice": "Proceed with the ready subset and revisit the remaining materials when they finish processing.",
                "support_status": "inferred_from_slides",
                "citations": [base_citation],
            },
            {
                "item_id": make_id("mistake"),
                "pattern": "Ignoring your own weak-area notes",
                "why_it_happens": "Student context is easy to leave blank when working quickly.",
                "prevention_advice": "Add weak areas or goals before regenerating if you want more tailored guidance.",
                "support_status": "supplemental_note",
                "citations": [base_citation],
            },
        ],
        "tailoring_summary": {
            "used_inputs": [
                {"key": "topic_text", "label": "Topic focus", "value": title, "source": "user" if request_payload.get("topic_text") else "inferred"},
                {"key": "time_budget_minutes", "label": "Time budget", "value": f"{int(request_payload.get('time_budget_minutes') or workspace.get('time_budget_minutes') or 60)} minutes", "source": "user"},
                {"key": "grounding_mode", "label": "Grounding mode", "value": request_payload.get("grounding_mode") or workspace.get("grounding_mode") or "strict_lecture_only", "source": "user"},
                {"key": "materials", "label": "Lecture materials used", "value": material.get("title", "Uploaded materials") if material else "Uploaded materials", "source": "workspace"},
            ]
            + ([{"key": "prior_knowledge", "label": "What you already know", "value": prior_knowledge, "source": "user"}] if prior_knowledge else [])
            + ([{"key": "weak_areas", "label": "Weak areas", "value": weak_areas, "source": "user"}] if weak_areas else [])
            + ([{"key": "goals", "label": "Goals or exam context", "value": goals, "source": "user"}] if goals else []),
            "missing_inputs": [
                {"key": "prior_knowledge", "label": "What you already know", "message": "Not provided, so the mock plan assumes no specific starting point."}
                for _ in ([] if prior_knowledge else [1])
            ] + [
                {"key": "weak_areas", "label": "Weak areas", "message": "Not provided, so the mock plan cannot emphasize a specific trouble spot."}
                for _ in ([] if weak_areas else [1])
            ] + [
                {"key": "goals", "label": "Goals or exam context", "message": "Not provided, so the mock plan defaults to general review."}
                for _ in ([] if goals else [1])
            ],
            "evidence_scope": {
                "material_count": 1 if material else 0,
                "material_titles": [material.get("title", "Uploaded materials")] if material else [],
                "slide_count": 1,
                "slide_numbers": [slide.get("slide_number", 1)],
            },
        },
    }


def generate_mock_assistant_message(workspace: dict, question_text: str, grounding_mode: str, response_style: str) -> dict:
    material = _primary_material(workspace)
    slide = _fallback_slide(material)
    citation = _make_citation(workspace["workspace_id"], material, slide)
    return {
        "message_id": make_id("msg_assistant"),
        "role": "assistant",
        "created_at": utc_now_iso(),
        "reply_sections": [
            {
                "heading": "Grounded answer",
                "text": f"Here is a grounded response based on the current workspace materials: {question_text}. The shell preserved {grounding_mode} and response style {response_style} while attaching a citation for inspection.",
                "support_status": "slide_grounded",
                "citations": [citation],
            }
        ],
        "clarifying_question": {"prompt": None, "reason": None},
    }


def generate_mock_practice_set(workspace: dict, request_payload: dict, *, parent_practice_set_id: str | None = None) -> dict:
    material = _primary_material(workspace)
    slide = _fallback_slide(material)
    citation = _make_citation(workspace["workspace_id"], material, slide)
    question_count = int(request_payload.get("question_count") or 3)
    generation_mode = request_payload.get("generation_mode") or "mixed"
    difficulty = request_payload.get("difficulty_profile") or "mixed"
    questions = []
    for index in range(question_count):
        question_type = "short_answer" if generation_mode in {"short_answer", "template_mimic"} else ("long_answer" if generation_mode == "long_answer" else ("short_answer" if index % 2 == 0 else "long_answer"))
        questions.append(
            {
                "question_id": make_id("question"),
                "question_type": question_type,
                "stem": f"Question {index + 1}: explain one grounded idea from {material.get('title', 'the workspace')}.",
                "expected_answer": "A strong answer cites the uploaded source and explains the concept in plain language.",
                "rubric": [
                    {
                        "criterion": "Grounded explanation",
                        "description": "Mentions the cited lecture source and explains the idea accurately.",
                        "points": 2,
                    }
                ] if request_payload.get("include_rubrics", True) else [],
                "scoring_guide_text": "Check that the answer stays anchored in the cited material." if request_payload.get("include_answer_key", True) else "",
                "citations": [citation],
                "covered_slides": [citation.get("slide_number", 1)],
                "difficulty": difficulty,
            }
        )
    return {
        "practice_set_id": make_id("practice"),
        "parent_practice_set_id": parent_practice_set_id,
        "workspace_id": workspace["workspace_id"],
        "created_at": utc_now_iso(),
        "generation_mode": generation_mode,
        "questions": questions,
        "coverage_report": {
            "considered_slide_count": max(1, len(material.get("slides", [])) if material else 1),
            "cited_slide_count": 1,
            "uncited_or_skipped_slides": [],
            "notes": "Mock practice generation uses the primary ready material as grounded evidence.",
        },
    }


def render_slide_preview_svg(workspace_id: str, material: dict, slide: dict) -> str:
    title = html.escape(material.get("title", "Source preview"))
    role = html.escape(material.get("role", "material"))
    slide_title = html.escape(slide.get("title", "Slide"))
    slide_number = int(slide.get("slide_number", 1))
    snippet = html.escape(slide.get("snippet_text", "Preview unavailable."))
    bullets = slide.get("bullets") or []
    bullet_lines = "".join(
        f'<text x="72" y="{190 + idx * 40}" font-size="22">• {html.escape(str(line))}</text>'
        for idx, line in enumerate(bullets[:5])
    )
    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1280\" height=\"720\" viewBox=\"0 0 1280 720\">
  <rect width=\"1280\" height=\"720\" fill=\"#f8fafc\"/>
  <rect x=\"36\" y=\"36\" width=\"1208\" height=\"648\" rx=\"20\" fill=\"white\" stroke=\"#cbd5e1\" stroke-width=\"2\"/>
  <text x=\"72\" y=\"96\" font-size=\"24\" fill=\"#475569\">Study Helper MVP source viewer • {html.escape(workspace_id)}</text>
  <text x=\"72\" y=\"146\" font-size=\"36\" fill=\"#0f172a\">{title}</text>
  <text x=\"72\" y=\"182\" font-size=\"22\" fill=\"#334155\">Role: {role} • Slide {slide_number} • {slide_title}</text>
  <foreignObject x=\"72\" y=\"220\" width=\"1130\" height=\"110\">
    <div xmlns=\"http://www.w3.org/1999/xhtml\" style=\"font-size:24px;color:#1e293b;font-family:Arial, sans-serif;line-height:1.4;\">{snippet}</div>
  </foreignObject>
  {bullet_lines}
</svg>"""
