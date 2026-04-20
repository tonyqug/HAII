from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypedDict

GroundingMode = Literal["strict_lecture_only", "lecture_with_fallback"]
SupportStatus = Literal[
    "slide_grounded",
    "inferred_from_slides",
    "external_supplement",
    "insufficient_evidence",
]
QuestionType = Literal["multiple_choice", "short_answer", "long_answer"]
ResponseStyle = Literal["concise", "standard", "step_by_step"]
PracticeGenerationMode = Literal["multiple_choice", "short_answer", "long_answer", "mixed", "template_mimic"]
CoverageMode = Literal["balanced", "high_coverage", "exhaustive"]
DifficultyProfile = Literal["easier", "mixed", "harder"]
JobType = Literal["study_plan", "chat_message", "practice_set", "revision"]
JobStatus = Literal["queued", "running", "succeeded", "failed", "needs_user_input"]


class Citation(TypedDict):
    citation_id: str
    material_id: str
    material_title: str
    slide_id: str
    slide_number: int
    snippet_text: str
    support_type: Literal["explicit", "supplemental_note", "inferred"]
    confidence: Literal["high", "medium", "low"]
    preview_url: str
    source_open_url: str


class EvidenceItem(TypedDict):
    item_id: str
    material_id: str
    material_title: str
    slide_id: str
    slide_number: int
    text: str
    extraction_quality: Literal["high", "medium", "low"]
    citation: Citation


class EvidenceSummary(TypedDict):
    total_items: int
    total_slides: int
    low_confidence_item_count: int


class EvidenceBundle(TypedDict):
    bundle_id: str
    workspace_id: str
    material_ids: List[str]
    query_text: NotRequired[str]
    bundle_mode: Literal["precision", "coverage", "full_material"]
    items: List[EvidenceItem]
    summary: EvidenceSummary


class UserAction(TypedDict):
    kind: str | None
    prompt: str | None
    options: List[str]


class JobError(TypedDict):
    code: str | None
    message: str | None
    retryable: bool


class JobObject(TypedDict):
    job_id: str
    job_type: JobType
    status: JobStatus
    progress: int
    stage: str
    message: str
    result_type: str | None
    result_id: str | None
    user_action: UserAction
    error: JobError
    created_at: str
    updated_at: str


JsonObject = Dict[str, Any]
JsonList = List[Any]

__all__ = [
    "CoverageMode",
    "DifficultyProfile",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidenceSummary",
    "GroundingMode",
    "JobError",
    "JobObject",
    "JobStatus",
    "JobType",
    "JsonList",
    "JsonObject",
    "PracticeGenerationMode",
    "QuestionType",
    "ResponseStyle",
    "SupportStatus",
    "Citation",
]
