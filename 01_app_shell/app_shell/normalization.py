from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from app_shell.errors import ShellError


LECTURE_GROUNDING_ROLES = {"slides", "notes"}
SUPPORTED_MATERIAL_IMPORT_ROLES = {"slides", "notes", "practice_template"}
CANONICAL_RESPONSE_STYLES = {"standard", "concise", "step_by_step"}
DEFAULT_GROUNDING_MODE = "lecture_with_fallback"
DEFAULT_PRACTICE_QUESTION_COUNT = 6



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
    if role_value not in SUPPORTED_MATERIAL_IMPORT_ROLES:
        raise ShellError("Material role must be slides, notes, or practice_template.")
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



def normalize_conversation_create(workspace: dict, payload: dict) -> Tuple[Dict[str, Any], List[str]]:
    material_ids, warnings = select_material_ids(workspace, allowed_roles=LECTURE_GROUNDING_ROLES)
    title = _text(payload.get("title")) or "Workspace Q&A"
    return (
        {
            "workspace_id": workspace["workspace_id"],
            "material_ids": material_ids,
            "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or DEFAULT_GROUNDING_MODE),
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
        "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or DEFAULT_GROUNDING_MODE),
        "include_citations": True,
    }



def normalize_practice_request(workspace: dict, payload: dict) -> Tuple[Dict[str, Any], List[str]]:
    generation_mode = _text(payload.get("generation_mode") or "mixed")
    material_ids, warnings = select_material_ids(workspace, allowed_roles=LECTURE_GROUNDING_ROLES)
    question_count = payload.get("question_count") or DEFAULT_PRACTICE_QUESTION_COUNT
    template_material_id = _text(payload.get("template_material_id"))
    topic_text = _text(payload.get("topic_text") or workspace.get("practice_preferences", {}).get("topic_text"))
    normalized = {
        "workspace_id": workspace["workspace_id"],
        "material_ids": material_ids,
        "generation_mode": generation_mode,
        "question_count": int(question_count),
        "coverage_mode": _text(payload.get("coverage_mode") or "balanced"),
        "difficulty_profile": _text(payload.get("difficulty_profile") or payload.get("difficulty") or "harder"),
        "include_answer_key": _bool(payload.get("include_answer_key") if "include_answer_key" in payload else payload.get("answer_key"), False),
        "include_rubrics": _bool(payload.get("include_rubrics") if "include_rubrics" in payload else payload.get("rubric"), True),
        "grounding_mode": _text(payload.get("grounding_mode") or workspace.get("grounding_mode") or DEFAULT_GROUNDING_MODE),
        "include_annotations": True,
    }
    if topic_text:
        normalized["topic_text"] = topic_text
    if generation_mode == "template_mimic":
        if not template_material_id:
            raise ShellError("template_material_id is required when generation_mode is template_mimic.")
        normalized["template_material_id"] = template_material_id
    elif template_material_id:
        normalized["template_material_id"] = template_material_id
    return normalized, warnings



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
