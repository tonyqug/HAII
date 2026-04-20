from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from app_shell.errors import ShellError


SECTION_TO_KEYS = {
    "prerequisites": "item_id",
    "study_sequence": "step_id",
    "common_mistakes": "item_id",
}

LECTURE_GROUNDING_ROLES = {"slides", "notes"}
CANONICAL_RESPONSE_STYLES = {"standard", "concise", "step_by_step"}



def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value).strip()



def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}



def _role_label(allowed_roles: Sequence[str] | set[str]) -> str:
    roles = list(allowed_roles)
    if set(roles) == LECTURE_GROUNDING_ROLES:
        return "slides or notes"
    return ", ".join(sorted(roles))



def select_material_ids(workspace: dict, *, allowed_roles: Sequence[str] | set[str] | None = None) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    material_preferences = workspace.get("material_preferences", {})
    materials = list(workspace.get("materials", {}).values())
    relevant_roles = set(allowed_roles) if allowed_roles else None
    ready_materials = []
    processing_materials = []
    for material in materials:
        role = material.get("role", "notes")
        if relevant_roles and role not in relevant_roles:
            continue
        preference = material_preferences.get(material["material_id"], "default")
        if preference == "exclude":
            continue
        if material.get("processing_status") == "ready":
            ready_materials.append(material)
        else:
            processing_materials.append(material)
    if not ready_materials:
        role_label = _role_label(relevant_roles or {"materials"})
        if relevant_roles == LECTURE_GROUNDING_ROLES:
            raise ShellError(
                "No ready slides or notes are available yet.",
                status_code=400,
            )
        raise ShellError(
            f"No ready {role_label} are available yet. Wait for at least one upload to finish processing before generating study help.",
            status_code=400,
        )
    if processing_materials:
        titles = ", ".join(material.get("title", material["material_id"]) for material in processing_materials[:3])
        suffix = "" if len(processing_materials) <= 3 else ", ..."
        warnings.append(f"Some materials are still processing, so the request is using the ready subset only: {titles}{suffix}")
    material_ids = [material["material_id"] for material in ready_materials]
    return material_ids, warnings



def normalize_material_import(
    workspace_id: str,
    role: str,
    title: str,
    kind: str | None,
    text: str | None,
    filename: str | None,
) -> Dict[str, Any]:
    role_value = _text(role)
    if role_value not in {"slides", "notes"}:
        raise ShellError("Material role must be slides or notes.")
    title_value = _text(title) or _text(filename) or "Untitled material"
    incoming_kind = _text(kind)
    if filename:
        return {
            "workspace_id": workspace_id,
            "role": role_value,
            "source_kind": "file",
            "title": title_value,
        }
    if incoming_kind and incoming_kind not in {"pasted_text", "text", "notes"}:
        raise ShellError("Only file upload or pasted text import is supported by this shell.")
    text_body = _text(text)
    if not text_body:
        raise ShellError("Pasted text import requires non-empty text.")
    return {
        "workspace_id": workspace_id,
        "role": role_value,
        "source_kind": "pasted_text",
        "title": title_value,
        "text_body": text_body,
    }



def normalize_study_plan_request(workspace: dict, payload: dict) -> Tuple[Dict[str, Any], List[str]]:
    material_ids, warnings = select_material_ids(workspace, allowed_roles=LECTURE_GROUNDING_ROLES)
    student_context_in = payload.get("student_context") or {}
    prior_knowledge = _text(
        student_context_in.get("prior_knowledge")
        or student_context_in.get("known")
        or payload.get("prior_knowledge")
        or payload.get("known")
        or workspace.get("student_context", {}).get("known")
    )
    weak_areas = _text(student_context_in.get("weak_areas") or payload.get("weak_areas") or workspace.get("student_context", {}).get("weak_areas"))
    goals = _text(student_context_in.get("goals") or payload.get("goals") or workspace.get("student_context", {}).get("goals"))

    time_budget = payload.get("time_budget_minutes")
    if time_budget in {None, ""}:
        time_budget = workspace.get("time_budget_minutes")
    if time_budget in {None, ""}:
        raise ShellError("Time budget is required for study plan generation.")

    normalized = {
        "workspace_id": workspace["workspace_id"],
        "material_ids": material_ids,
        "topic_text": _text(payload.get("topic_text") or workspace.get("topic_text")),
        "time_budget_minutes": int(time_budget),
        "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or "strict_lecture_only"),
        "student_context": {
            "prior_knowledge": prior_knowledge,
            "weak_areas": weak_areas,
            "goals": goals,
        },
        "include_annotations": True,
    }
    return normalized, warnings



def normalize_conversation_create(workspace: dict, payload: dict) -> Tuple[Dict[str, Any], List[str]]:
    material_ids, warnings = select_material_ids(workspace, allowed_roles=LECTURE_GROUNDING_ROLES)
    title = _text(payload.get("title")) or "Workspace Q&A"
    return (
        {
            "workspace_id": workspace["workspace_id"],
            "material_ids": material_ids,
            "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or "strict_lecture_only"),
            "title": title,
            "include_annotations": True,
        },
        warnings,
    )



def normalize_response_style(value: Any) -> str:
    style = _text(value).lower().replace(" ", "_")
    if not style:
        return "standard"
    if style == "direct_answer":
        return "standard"
    if style in CANONICAL_RESPONSE_STYLES:
        return style
    return "standard"



def normalize_conversation_message(workspace: dict, payload: dict) -> Dict[str, Any]:
    message_text = _text(payload.get("message_text") or payload.get("text"))
    if not message_text:
        raise ShellError("Message text is required.")
    return {
        "message_text": message_text,
        "response_style": normalize_response_style(payload.get("response_style")),
        "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or "strict_lecture_only"),
        "include_citations": True,
    }



def normalize_practice_request(workspace: dict, payload: dict) -> Tuple[Dict[str, Any], List[str]]:
    generation_mode = _text(payload.get("generation_mode") or "mixed")
    material_ids, warnings = select_material_ids(workspace, allowed_roles=LECTURE_GROUNDING_ROLES)
    question_count = payload.get("question_count") or 3
    normalized = {
        "workspace_id": workspace["workspace_id"],
        "material_ids": material_ids,
        "topic_text": _text(payload.get("topic_text") or workspace.get("practice_preferences", {}).get("topic_text")),
        "generation_mode": generation_mode,
        "question_count": int(question_count),
        "coverage_mode": _text(payload.get("coverage_mode") or "balanced"),
        "difficulty_profile": _text(payload.get("difficulty_profile") or payload.get("difficulty") or "mixed"),
        "include_answer_key": _bool(payload.get("include_answer_key") if "include_answer_key" in payload else payload.get("answer_key"), True),
        "include_rubrics": _bool(payload.get("include_rubrics") if "include_rubrics" in payload else payload.get("rubric"), True),
        "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or "strict_lecture_only"),
        "include_annotations": True,
    }
    if generation_mode == "template_mimic":
        raise ShellError("Template mimic mode is no longer supported. Choose multiple_choice, short_answer, long_answer, or mixed.")
    return normalized, warnings



def _collect_section_item_ids(section_items: Iterable[dict], key: str) -> List[str]:
    return [str(item.get(key)) for item in section_items if item.get(key)]



def map_plan_locked_item_ids(plan: dict, payload: dict) -> List[str]:
    locked_ids = [str(item_id) for item_id in payload.get("locked_item_ids", []) if item_id]
    for section_name in payload.get("locked_sections", []) or []:
        key = SECTION_TO_KEYS.get(section_name)
        if not key:
            continue
        locked_ids.extend(_collect_section_item_ids(plan.get(section_name, []), key))
    seen = set()
    deduped: List[str] = []
    for item_id in locked_ids:
        if item_id not in seen:
            deduped.append(item_id)
            seen.add(item_id)
    return deduped



def synthesize_study_plan_instruction(payload: dict, target_section: str | None, locked_item_ids: List[str]) -> str:
    explicit = _text(payload.get("instruction_text"))
    if explicit:
        return explicit
    feedback = _text(payload.get("feedback_note") or payload.get("correction_note"))
    if feedback:
        return feedback
    if target_section and target_section not in {"", "entire_plan"}:
        return "regenerate this section while preserving all locked items"
    if locked_item_ids:
        return "regenerate the entire plan with a clearer sequence and preserve locked items"
    return "regenerate the entire plan with a clearer sequence"



def normalize_study_plan_revision(workspace: dict, plan: dict, payload: dict) -> Dict[str, Any]:
    target_section = _text(payload.get("target_section") or payload.get("section") or "entire_plan")
    locked_item_ids = map_plan_locked_item_ids(plan, payload)
    return {
        "instruction_text": synthesize_study_plan_instruction(payload, target_section, locked_item_ids),
        "target_section": target_section,
        "locked_item_ids": locked_item_ids,
        "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or plan.get("grounding_mode") or "strict_lecture_only"),
        "include_annotations": True,
    }



def synthesize_practice_instruction(payload: dict, target_question_ids: List[str], locked_question_ids: List[str]) -> str:
    explicit = _text(payload.get("instruction_text"))
    if explicit:
        return explicit
    feedback = _text(payload.get("feedback_note") or payload.get("correction_note"))
    if feedback:
        return feedback
    action = _text(payload.get("action"))
    if action == "create_variant":
        return "create a new variant of this practice set while preserving coverage and locked questions"
    if target_question_ids:
        return "regenerate the selected questions while preserving locked questions"
    if locked_question_ids:
        return "revise this practice set while preserving locked questions"
    return "create a new variant of this practice set while maintaining coverage"



def normalize_practice_revision(practice_set: dict, payload: dict) -> Dict[str, Any]:
    target_question_ids = [str(item) for item in (payload.get("target_question_ids") or payload.get("selected_question_ids") or []) if item]
    locked_question_ids = [str(item) for item in (payload.get("locked_question_ids") or []) if item]
    maintain_coverage = _bool(payload.get("maintain_coverage"), True)
    return {
        "instruction_text": synthesize_practice_instruction(payload, target_question_ids, locked_question_ids),
        "target_question_ids": target_question_ids,
        "locked_question_ids": locked_question_ids,
        "maintain_coverage": maintain_coverage,
    }



def summarize_material_preference(material: dict, preference: str) -> str:
    title = material.get("title") or material.get("material_id") or "material"
    if preference == "focus":
        return f"Prioritize {title} when grounding study help."
    if preference == "exclude":
        return f"Exclude {title} from future grounding requests."
    return f"Restore {title} to the default grounding set."



def build_feedback_annotation(payload: dict) -> Dict[str, Any] | None:
    correction_note = _text(payload.get("correction_note") or payload.get("feedback_note"))
    if not correction_note:
        return None
    annotation_type = _text(payload.get("annotation_type") or "user_correction")
    material_id = payload.get("material_id")
    slide_id = payload.get("slide_id")
    if slide_id and material_id:
        return {
            "annotation_type": annotation_type,
            "scope": "slide",
            "material_id": str(material_id),
            "slide_id": str(slide_id),
            "text": correction_note,
        }
    if material_id:
        return {
            "annotation_type": annotation_type,
            "scope": "material",
            "material_id": str(material_id),
            "text": correction_note,
        }
    return {
        "annotation_type": annotation_type,
        "scope": "workspace",
        "text": correction_note,
    }
