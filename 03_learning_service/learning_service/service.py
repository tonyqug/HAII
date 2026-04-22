from __future__ import annotations

import copy
import logging
import queue
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import Settings
from .content_client import ContentServiceClient, ContentServiceError
from .generator_v2 import ArtifactValidationError, GroundedGenerator, NeedsUserInputError
from .storage import JsonStore
from .utils import make_id, normalize_whitespace, utc_now_iso

LOGGER = logging.getLogger(__name__)
ALLOWED_GROUNDING_MODES = {"strict_lecture_only", "lecture_with_fallback"}
ALLOWED_RESPONSE_STYLES = {"concise", "standard", "step_by_step"}
RESPONSE_STYLE_ALIASES = {"direct_answer": "standard"}
PRACTICE_GENERATION_SYNTHETIC_QUERY = "full lecture coverage for practice generation"
ALLOWED_GENERATION_MODES = {"multiple_choice", "short_answer", "long_answer", "mixed", "template_mimic"}
ALLOWED_COVERAGE_MODES = {"balanced", "high_coverage", "exhaustive"}
ALLOWED_DIFFICULTIES = {"easier", "mixed", "harder"}
FINAL_JOB_STATUSES = {"succeeded", "failed", "needs_user_input"}


class RequestValidationError(ValueError):
    pass


class BackgroundJobRunner:
    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Any, tuple[Any, ...], dict[str, Any]]] = queue.Queue()
        self._thread = threading.Thread(target=self._worker, name="learning-service-jobs", daemon=True)
        self._thread.start()

    def submit(self, func: Any, *args: Any, **kwargs: Any) -> None:
        self._queue.put((func, args, kwargs))

    def _worker(self) -> None:
        while True:
            func, args, kwargs = self._queue.get()
            try:
                func(*args, **kwargs)
            except Exception:
                LOGGER.exception("Unhandled exception in background job worker")
            finally:
                self._queue.task_done()


class LearningService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = JsonStore(settings.local_data_dir)
        self.content_client = ContentServiceClient(settings)
        self.generator = GroundedGenerator(settings)
        self.job_runner = BackgroundJobRunner()
        LOGGER.info(
            "Learning service initialized with local_data_dir=%s content_service_url=%s gemini_configured=%s.",
            settings.local_data_dir,
            settings.content_service_url,
            bool(settings.gemini_api_key),
        )

    # --------------------------
    # Service metadata
    # --------------------------
    def health_payload(self) -> Dict[str, Any]:
        gemini_configured = bool(self.settings.gemini_api_key)
        content_service_reachable = self.content_client.health_check()
        degraded_reasons: List[str] = []
        if not gemini_configured:
            degraded_reasons.append("GEMINI_API_KEY is not configured for primary Gemini generation")
        if not content_service_reachable:
            degraded_reasons.append("CONTENT_SERVICE_URL is not reachable; integrated mode is currently unavailable")
        return {
            "service_name": self.settings.service_name,
            "ready": gemini_configured,
            "status": "ok",
            "details": {
                "process_up": True,
                "gemini_configured": gemini_configured,
                "content_service_reachable": content_service_reachable,
                "integrated_mode_available": content_service_reachable,
                "standalone_mode_available": True,
                "deterministic_fallback_available": bool(self.settings.use_heuristic_fallback),
                "primary_generation_path": "gemini" if gemini_configured else "heuristic_fallback",
                "problems": degraded_reasons,
            },
        }

    def manifest_payload(self) -> Dict[str, Any]:
        return {
            "service_name": self.settings.service_name,
            "version": self.settings.version,
            "api_base_url": self.settings.api_base_url,
            "capabilities": [
                "conversation_qa",
                "practice_generation",
                "artifact_revision",
                "inline_evidence_mode",
            ],
        }

    # --------------------------
    # Jobs
    # --------------------------
    def create_job(self, job_type: str) -> Dict[str, Any]:
        job = {
            "job_id": make_id("job"),
            "job_type": job_type,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "message": "Job accepted.",
            "result_type": None,
            "result_id": None,
            "user_action": {"kind": None, "prompt": None, "options": []},
            "error": {"code": None, "message": None, "retryable": False},
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.store.save("jobs", job["job_id"], job)
        return job

    def update_job(self, job_id: str, **updates: Any) -> Dict[str, Any]:
        updates.setdefault("updated_at", utc_now_iso())
        return self.store.update("jobs", job_id, **updates)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.store.load("jobs", job_id)

    # --------------------------
    # Study plans
    # --------------------------
    def submit_study_plan(self, request_data: Dict[str, Any]) -> str:
        normalized = self._normalize_study_plan_create(request_data)
        job = self.create_job("study_plan")
        self.job_runner.submit(self._run_study_plan_job, job["job_id"], normalized)
        return job["job_id"]

    def _run_study_plan_job(self, job_id: str, request_data: Dict[str, Any]) -> None:
        try:
            LOGGER.info(
                "Starting study plan job %s for workspace=%s topic=%r.",
                job_id,
                request_data.get("workspace_id"),
                request_data.get("topic_text"),
            )
            self.update_job(job_id, status="running", progress=10, stage="normalize_intent", message="Preparing the study plan request.")
            bundle = self._resolve_grounding_bundle(
                workspace_id=request_data["workspace_id"],
                material_ids=request_data.get("material_ids"),
                evidence_bundle=request_data.get("evidence_bundle"),
                query_text=request_data.get("topic_text"),
                bundle_mode="coverage",
                include_annotations=bool(request_data.get("include_annotations", True)),
            )
            self.update_job(job_id, status="running", progress=45, stage="draft_structured_output", message="Drafting a grounded study plan.")
            artifact = self.generator.build_study_plan(
                bundle=bundle,
                topic_text=request_data.get("topic_text"),
                time_budget_minutes=int(request_data.get("time_budget_minutes") or 90),
                grounding_mode=request_data.get("grounding_mode") or "lecture_with_fallback",
                student_context=request_data.get("student_context") or {},
            )
            self.store.save("study_plans", artifact["study_plan_id"], artifact)
            self.update_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="done",
                message="Study plan created.",
                result_type="study_plan",
                result_id=artifact["study_plan_id"],
            )
            LOGGER.info(
                "Study plan job %s succeeded via %s.",
                job_id,
                (artifact.get("_meta") or {}).get("generation_path"),
            )
        except NeedsUserInputError as exc:
            LOGGER.info("Study plan job %s needs user input: %s", job_id, exc.prompt)
            self.update_job(
                job_id,
                status="needs_user_input",
                progress=100,
                stage="needs_user_input",
                message=exc.prompt,
                user_action={"kind": exc.kind, "prompt": exc.prompt, "options": exc.options},
                error={"code": None, "message": None, "retryable": False},
            )
        except ContentServiceError as exc:
            LOGGER.warning("Study plan job %s failed while fetching evidence: %s", job_id, exc)
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error={"code": "content_service_error", "message": str(exc), "retryable": exc.retryable},
            )
        except ArtifactValidationError as exc:
            LOGGER.warning("Study plan job %s failed validation: %s", job_id, exc)
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error={"code": "artifact_validation_error", "message": str(exc), "retryable": False},
            )
        except Exception as exc:
            LOGGER.exception("Unhandled study plan failure")
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message="Unhandled study plan failure.",
                error={"code": "internal_error", "message": str(exc), "retryable": False},
            )

    def list_study_plans(self, workspace_id: str) -> Dict[str, Any]:
        plans = [
            self._study_plan_list_item(plan)
            for plan in self.store.list_all("study_plans")
            if plan.get("workspace_id") == workspace_id
        ]
        plans.sort(key=lambda item: item["created_at"], reverse=True)
        return {"study_plans": plans}

    def get_study_plan(self, study_plan_id: str) -> Optional[Dict[str, Any]]:
        artifact = self.store.load("study_plans", study_plan_id)
        if artifact is None:
            return None
        return self._public_artifact(artifact)

    def submit_study_plan_revision(self, study_plan_id: str, request_data: Dict[str, Any]) -> str:
        existing = self.store.load("study_plans", study_plan_id)
        if existing is None:
            raise KeyError(study_plan_id)
        normalized = self._normalize_study_plan_revision(existing, request_data)
        job = self.create_job("revision")
        self.job_runner.submit(self._run_study_plan_revision_job, job["job_id"], existing, normalized)
        return job["job_id"]

    def _run_study_plan_revision_job(self, job_id: str, existing: Dict[str, Any], request_data: Dict[str, Any]) -> None:
        try:
            self.update_job(job_id, status="running", progress=20, stage="load_artifact", message="Loading the study plan to revise.")
            revised = self.generator.revise_study_plan(
                existing_plan=existing,
                instruction_text=request_data["instruction_text"],
                target_section=request_data.get("target_section") or "entire_plan",
                locked_item_ids=request_data.get("locked_item_ids") or [],
                grounding_mode=request_data.get("grounding_mode") or existing.get("grounding_mode") or "lecture_with_fallback",
            )
            self.store.save("study_plans", revised["study_plan_id"], revised)
            self.update_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="done",
                message="Study plan revision created.",
                result_type="study_plan",
                result_id=revised["study_plan_id"],
            )
        except ArtifactValidationError as exc:
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error={"code": "artifact_validation_error", "message": str(exc), "retryable": False},
            )
        except Exception as exc:
            LOGGER.exception("Unhandled study plan revision failure")
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message="Unhandled study plan revision failure.",
                error={"code": "internal_error", "message": str(exc), "retryable": False},
            )

    # --------------------------
    # Conversations
    # --------------------------
    def create_conversation(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_conversation_create(request_data)
        grounding_input_mode = "standalone" if normalized.get("evidence_bundle") else "integrated"
        conversation = {
            "conversation_id": make_id("conversation"),
            "workspace_id": normalized["workspace_id"],
            "title": normalized.get("title"),
            "created_at": utc_now_iso(),
            "grounding_mode": normalized.get("grounding_mode") or "lecture_with_fallback",
            "messages": [],
            "_meta": {
                "grounding_input_mode": grounding_input_mode,
                "material_ids": normalized.get("material_ids") or [],
                "evidence_bundle": normalized.get("evidence_bundle"),
                "include_annotations": bool(normalized.get("include_annotations", True)),
            },
        }
        self.store.save("conversations", conversation["conversation_id"], conversation)
        return {"conversation_id": conversation["conversation_id"]}

    def list_conversations(self, workspace_id: str) -> Dict[str, Any]:
        conversations = []
        for artifact in self.store.list_all("conversations"):
            if artifact.get("workspace_id") != workspace_id:
                continue
            conversations.append(
                {
                    "conversation_id": artifact["conversation_id"],
                    "workspace_id": artifact["workspace_id"],
                    "title": artifact.get("title"),
                    "created_at": artifact["created_at"],
                    "message_count": len(artifact.get("messages", [])),
                }
            )
        conversations.sort(key=lambda item: item["created_at"], reverse=True)
        return {"conversations": conversations}

    def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        artifact = self.store.load("conversations", conversation_id)
        if artifact is None:
            return None
        return self._public_artifact(artifact)

    def submit_conversation_message(self, conversation_id: str, request_data: Dict[str, Any]) -> str:
        conversation = self.store.load("conversations", conversation_id)
        if conversation is None:
            raise KeyError(conversation_id)
        normalized = self._normalize_conversation_message(request_data, conversation)
        job = self.create_job("chat_message")
        self.job_runner.submit(self._run_conversation_message_job, job["job_id"], conversation, normalized)
        return job["job_id"]

    def _run_conversation_message_job(self, job_id: str, conversation: Dict[str, Any], request_data: Dict[str, Any]) -> None:
        try:
            LOGGER.info(
                "Starting conversation job %s for conversation=%s workspace=%s message=%r.",
                job_id,
                conversation.get("conversation_id"),
                conversation.get("workspace_id"),
                normalize_whitespace(request_data.get("message_text", ""))[:160],
            )
            self.update_job(job_id, status="running", progress=15, stage="acquire_evidence", message="Loading grounded evidence for the message.")
            bundle = self._resolve_conversation_bundle(conversation, request_data.get("message_text", ""))
            self.update_job(job_id, status="running", progress=50, stage="draft_structured_output", message="Drafting a grounded answer.")
            user_message, assistant_message = self.generator.build_conversation_reply(
                bundle=bundle,
                message_text=request_data["message_text"],
                response_style=request_data.get("response_style") or "standard",
                grounding_mode=request_data.get("grounding_mode") or conversation.get("grounding_mode") or "lecture_with_fallback",
                previous_messages=conversation.get("messages", []),
                conversation_id=conversation["conversation_id"],
            )
            updated = copy.deepcopy(conversation)
            updated["grounding_mode"] = request_data.get("grounding_mode") or updated.get("grounding_mode")
            updated.setdefault("messages", []).extend([user_message, assistant_message])
            self.store.save("conversations", updated["conversation_id"], updated)
            self.update_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="done",
                message="Conversation reply created.",
                result_type="message",
                result_id=assistant_message["message_id"],
            )
            answer_source = assistant_message.get("answer_source") or {}
            LOGGER.info(
                "Conversation job %s succeeded with path=%s provider=%s fallback_reason=%s.",
                job_id,
                answer_source.get("path"),
                answer_source.get("provider"),
                answer_source.get("fallback_reason"),
            )
        except NeedsUserInputError as exc:
            LOGGER.info("Conversation job %s needs user input: %s", job_id, exc.prompt)
            self.update_job(
                job_id,
                status="needs_user_input",
                progress=100,
                stage="needs_user_input",
                message=exc.prompt,
                user_action={"kind": exc.kind, "prompt": exc.prompt, "options": exc.options},
                error={"code": None, "message": None, "retryable": False},
            )
        except ContentServiceError as exc:
            LOGGER.warning("Conversation job %s failed while fetching evidence: %s", job_id, exc)
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error={"code": "content_service_error", "message": str(exc), "retryable": exc.retryable},
            )
        except Exception as exc:
            LOGGER.exception("Unhandled conversation failure")
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message="Unhandled conversation failure.",
                error={"code": "internal_error", "message": str(exc), "retryable": False},
            )

    def clear_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        conversation = self.store.load("conversations", conversation_id)
        if conversation is None:
            return None
        conversation["messages"] = []
        self.store.save("conversations", conversation_id, conversation)
        return {"conversation_id": conversation_id, "cleared": True}

    # --------------------------
    # Practice sets
    # --------------------------
    def submit_practice_set(self, request_data: Dict[str, Any]) -> str:
        normalized = self._normalize_practice_set_create(request_data)
        job = self.create_job("practice_set")
        self.job_runner.submit(self._run_practice_set_job, job["job_id"], normalized)
        return job["job_id"]

    def _run_practice_set_job(self, job_id: str, request_data: Dict[str, Any]) -> None:
        try:
            LOGGER.info(
                "Starting practice set job %s for workspace=%s topic=%r mode=%s count=%s.",
                job_id,
                request_data.get("workspace_id"),
                request_data.get("topic_text"),
                request_data.get("generation_mode"),
                request_data.get("question_count"),
            )
            self.update_job(job_id, status="running", progress=15, stage="acquire_evidence", message="Loading grounded evidence for practice generation.")
            bundle = self._resolve_practice_bundle(request_data)
            self.update_job(job_id, status="running", progress=55, stage="draft_structured_output", message="Drafting the grounded practice set.")
            artifact = self.generator.build_practice_set(
                bundle=bundle,
                topic_text=request_data.get("topic_text"),
                generation_mode=request_data["generation_mode"],
                template_material_id=request_data.get("template_material_id"),
                question_count=int(request_data.get("question_count") or 8),
                coverage_mode=request_data.get("coverage_mode") or "balanced",
                difficulty_profile=request_data.get("difficulty_profile") or "mixed",
                include_answer_key=bool(request_data.get("include_answer_key", True)),
                include_rubrics=bool(request_data.get("include_rubrics", True)),
                grounding_mode=request_data.get("grounding_mode") or "lecture_with_fallback",
            )
            self.store.save("practice_sets", artifact["practice_set_id"], artifact)
            self.update_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="done",
                message="Practice set created.",
                result_type="practice_set",
                result_id=artifact["practice_set_id"],
            )
            LOGGER.info(
                "Practice set job %s succeeded via %s.",
                job_id,
                (artifact.get("_meta") or {}).get("generation_path"),
            )
        except NeedsUserInputError as exc:
            LOGGER.info("Practice set job %s needs user input: %s", job_id, exc.prompt)
            self.update_job(
                job_id,
                status="needs_user_input",
                progress=100,
                stage="needs_user_input",
                message=exc.prompt,
                user_action={"kind": exc.kind, "prompt": exc.prompt, "options": exc.options},
                error={"code": None, "message": None, "retryable": False},
            )
        except (ArtifactValidationError, ContentServiceError) as exc:
            LOGGER.warning("Practice set job %s failed: %s", job_id, exc)
            retryable = getattr(exc, "retryable", False)
            code = "content_service_error" if isinstance(exc, ContentServiceError) else "artifact_validation_error"
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error={"code": code, "message": str(exc), "retryable": retryable},
            )
        except Exception as exc:
            LOGGER.exception("Unhandled practice set failure")
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message="Unhandled practice set failure.",
                error={"code": "internal_error", "message": str(exc), "retryable": False},
            )

    def list_practice_sets(self, workspace_id: str) -> Dict[str, Any]:
        practice_sets = []
        for artifact in self.store.list_all("practice_sets"):
            if artifact.get("workspace_id") != workspace_id:
                continue
            practice_sets.append(
                {
                    "practice_set_id": artifact["practice_set_id"],
                    "parent_practice_set_id": artifact.get("parent_practice_set_id"),
                    "workspace_id": artifact["workspace_id"],
                    "created_at": artifact["created_at"],
                    "generation_mode": artifact["generation_mode"],
                }
            )
        practice_sets.sort(key=lambda item: item["created_at"], reverse=True)
        return {"practice_sets": practice_sets}

    def get_practice_set(self, practice_set_id: str) -> Optional[Dict[str, Any]]:
        artifact = self.store.load("practice_sets", practice_set_id)
        if artifact is None:
            return None
        return self._public_artifact(artifact)

    def submit_practice_set_revision(self, practice_set_id: str, request_data: Dict[str, Any]) -> str:
        existing = self.store.load("practice_sets", practice_set_id)
        if existing is None:
            raise KeyError(practice_set_id)
        normalized = self._normalize_practice_set_revision(existing, request_data)
        job = self.create_job("revision")
        self.job_runner.submit(self._run_practice_set_revision_job, job["job_id"], existing, normalized)
        return job["job_id"]

    def _run_practice_set_revision_job(self, job_id: str, existing: Dict[str, Any], request_data: Dict[str, Any]) -> None:
        try:
            self.update_job(job_id, status="running", progress=20, stage="load_artifact", message="Loading the practice set to revise.")
            revised = self.generator.revise_practice_set(
                existing_practice_set=existing,
                instruction_text=request_data["instruction_text"],
                target_question_ids=request_data.get("target_question_ids") or [],
                locked_question_ids=request_data.get("locked_question_ids") or [],
                maintain_coverage=bool(request_data.get("maintain_coverage", True)),
            )
            self.store.save("practice_sets", revised["practice_set_id"], revised)
            self.update_job(
                job_id,
                status="succeeded",
                progress=100,
                stage="done",
                message="Practice set revision created.",
                result_type="practice_set",
                result_id=revised["practice_set_id"],
            )
        except ArtifactValidationError as exc:
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message=str(exc),
                error={"code": "artifact_validation_error", "message": str(exc), "retryable": False},
            )
        except Exception as exc:
            LOGGER.exception("Unhandled practice set revision failure")
            self.update_job(
                job_id,
                status="failed",
                progress=100,
                stage="failed",
                message="Unhandled practice set revision failure.",
                error={"code": "internal_error", "message": str(exc), "retryable": False},
            )

    # --------------------------
    # Bundle resolution
    # --------------------------
    def _resolve_grounding_bundle(
        self,
        *,
        workspace_id: str,
        material_ids: Optional[List[str]],
        evidence_bundle: Optional[Dict[str, Any]],
        query_text: Optional[str],
        bundle_mode: str,
        include_annotations: bool,
    ) -> Dict[str, Any]:
        if material_ids and evidence_bundle:
            raise ArtifactValidationError("Provide exactly one of material_ids or evidence_bundle, not both.")
        if not material_ids and not evidence_bundle:
            raise ArtifactValidationError("Provide exactly one of material_ids or evidence_bundle.")
        if evidence_bundle:
            bundle = copy.deepcopy(evidence_bundle)
            if bundle.get("workspace_id") and bundle.get("workspace_id") != workspace_id:
                raise ArtifactValidationError("workspace_id does not match evidence_bundle.workspace_id")
            bundle["workspace_id"] = workspace_id
            return bundle
        return self.content_client.fetch_evidence_bundle(
            workspace_id=workspace_id,
            material_ids=material_ids or [],
            query_text=query_text,
            bundle_mode=bundle_mode,
            include_annotations=include_annotations,
        )

    def _resolve_conversation_bundle(self, conversation: Dict[str, Any], query_text: str) -> Dict[str, Any]:
        meta = conversation.get("_meta") or {}
        if meta.get("grounding_input_mode") == "standalone":
            bundle = copy.deepcopy(meta.get("evidence_bundle") or {})
            bundle["workspace_id"] = conversation["workspace_id"]
            bundle["query_text"] = query_text
            return bundle
        return self.content_client.fetch_evidence_bundle(
            workspace_id=conversation["workspace_id"],
            material_ids=meta.get("material_ids") or [],
            query_text=query_text,
            bundle_mode="precision",
            include_annotations=bool(meta.get("include_annotations", True)),
        )

    def _resolve_practice_bundle(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        if request_data.get("evidence_bundle"):
            return self._resolve_grounding_bundle(
                workspace_id=request_data["workspace_id"],
                material_ids=None,
                evidence_bundle=request_data.get("evidence_bundle"),
                query_text=None,
                bundle_mode="full_material" if request_data.get("coverage_mode") == "exhaustive" else "coverage",
                include_annotations=bool(request_data.get("include_annotations", True)),
            )

        lecture_bundle = self.content_client.fetch_evidence_bundle(
            workspace_id=request_data["workspace_id"],
            material_ids=request_data.get("material_ids") or [],
            query_text=PRACTICE_GENERATION_SYNTHETIC_QUERY,
            bundle_mode="full_material" if request_data.get("coverage_mode") == "exhaustive" else "coverage",
            include_annotations=bool(request_data.get("include_annotations", True)),
        )

        template_material_id = request_data.get("template_material_id")
        if not template_material_id:
            return lecture_bundle
        template_bundle = self.content_client.fetch_evidence_bundle(
            workspace_id=request_data["workspace_id"],
            material_ids=[template_material_id],
            query_text="template style analysis",
            bundle_mode="full_material",
            include_annotations=False,
        )
        combined = copy.deepcopy(lecture_bundle)
        combined["material_ids"] = list(dict.fromkeys((lecture_bundle.get("material_ids") or []) + (template_bundle.get("material_ids") or [])))
        combined["items"] = list(lecture_bundle.get("items") or []) + list(template_bundle.get("items") or [])
        combined["summary"] = {
            "total_items": len(combined["items"]),
            "total_slides": len({int(item.get("slide_number") or 0) for item in combined["items"]}),
            "low_confidence_item_count": sum(
                1
                for item in combined["items"]
                if item.get("extraction_quality") == "low" or (item.get("citation") or {}).get("confidence") == "low"
            ),
        }
        return combined

    # --------------------------
    # Normalization helpers
    # --------------------------
    def _normalize_study_plan_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._ensure_mapping(payload)
        workspace_id = self._required_text(data.get("workspace_id"), "workspace_id")
        material_ids, evidence_bundle = self._normalize_grounding_input(data)
        student_context = self._normalize_student_context(data.get("student_context"))
        topic_text = self._optional_text(data.get("topic_text"))
        return {
            "workspace_id": workspace_id,
            "material_ids": material_ids,
            "evidence_bundle": evidence_bundle,
            "topic_text": topic_text,
            "time_budget_minutes": self._positive_int(data.get("time_budget_minutes"), "time_budget_minutes", default=90),
            "grounding_mode": self._grounding_mode(data.get("grounding_mode")),
            "student_context": student_context,
            "include_annotations": self._bool_value(data.get("include_annotations"), True),
        }

    def _normalize_study_plan_revision(self, existing: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._ensure_mapping(payload)
        locked_item_ids = set(self._string_list(data.get("locked_item_ids")))
        locked_item_ids.update(self._expand_locked_sections(existing, data.get("locked_sections")))
        instruction_text = self._optional_text(data.get("instruction_text")) or self._optional_text(data.get("feedback_note"))
        target_section = self._optional_text(data.get("target_section")) or "entire_plan"
        if not instruction_text:
            instruction_text = self._synthesize_study_plan_revision_instruction(target_section, bool(locked_item_ids))
        return {
            "instruction_text": instruction_text,
            "target_section": target_section,
            "locked_item_ids": sorted(locked_item_ids),
            "grounding_mode": self._grounding_mode(data.get("grounding_mode"), default=existing.get("grounding_mode") or "lecture_with_fallback"),
            "include_annotations": self._bool_value(data.get("include_annotations"), True),
        }

    def _normalize_conversation_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._ensure_mapping(payload)
        workspace_id = self._required_text(data.get("workspace_id"), "workspace_id")
        material_ids, evidence_bundle = self._normalize_grounding_input(data)
        return {
            "workspace_id": workspace_id,
            "material_ids": material_ids,
            "evidence_bundle": evidence_bundle,
            "grounding_mode": self._grounding_mode(data.get("grounding_mode")),
            "title": self._optional_text(data.get("title")),
            "include_annotations": self._bool_value(data.get("include_annotations"), True),
        }

    def _normalize_conversation_message(self, payload: Dict[str, Any], conversation: Dict[str, Any]) -> Dict[str, Any]:
        data = self._ensure_mapping(payload)
        message_text = self._optional_text(data.get("message_text")) or self._optional_text(data.get("text"))
        if not message_text:
            raise RequestValidationError("message_text is required")
        response_style = self._optional_text(data.get("response_style")) or "standard"
        response_style = RESPONSE_STYLE_ALIASES.get(response_style, response_style)
        if response_style not in ALLOWED_RESPONSE_STYLES:
            allowed = sorted(ALLOWED_RESPONSE_STYLES | set(RESPONSE_STYLE_ALIASES))
            raise RequestValidationError(f"response_style must be one of {allowed}")
        return {
            "message_text": message_text,
            "response_style": response_style,
            "grounding_mode": self._grounding_mode(data.get("grounding_mode"), default=conversation.get("grounding_mode") or "lecture_with_fallback"),
            "include_citations": self._bool_value(data.get("include_citations"), True),
        }

    def _normalize_practice_set_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._ensure_mapping(payload)
        workspace_id = self._required_text(data.get("workspace_id"), "workspace_id")
        material_ids, evidence_bundle = self._normalize_grounding_input(data)
        generation_mode = self._required_text(data.get("generation_mode"), "generation_mode")
        if generation_mode not in ALLOWED_GENERATION_MODES:
            raise RequestValidationError(f"generation_mode must be one of {sorted(ALLOWED_GENERATION_MODES)}")
        coverage_mode = self._optional_text(data.get("coverage_mode")) or "balanced"
        if coverage_mode not in ALLOWED_COVERAGE_MODES:
            raise RequestValidationError(f"coverage_mode must be one of {sorted(ALLOWED_COVERAGE_MODES)}")
        difficulty_profile = self._optional_text(data.get("difficulty_profile")) or self._optional_text(data.get("difficulty")) or "mixed"
        if difficulty_profile not in ALLOWED_DIFFICULTIES:
            raise RequestValidationError(f"difficulty_profile must be one of {sorted(ALLOWED_DIFFICULTIES)}")
        template_material_id = self._optional_text(data.get("template_material_id"))
        if generation_mode == "template_mimic" and not template_material_id:
            raise RequestValidationError("template_material_id is required when generation_mode is template_mimic")
        return {
            "workspace_id": workspace_id,
            "material_ids": material_ids,
            "evidence_bundle": evidence_bundle,
            "topic_text": self._optional_text(data.get("topic_text")),
            "generation_mode": generation_mode,
            "template_material_id": template_material_id,
            "question_count": self._positive_int(data.get("question_count"), "question_count", default=8),
            "coverage_mode": coverage_mode,
            "difficulty_profile": difficulty_profile,
            "include_answer_key": self._bool_value(data.get("include_answer_key"), self._bool_value(data.get("answer_key"), True)),
            "include_rubrics": self._bool_value(data.get("include_rubrics"), self._bool_value(data.get("rubric"), True)),
            "grounding_mode": self._grounding_mode(data.get("grounding_mode")),
            "include_annotations": self._bool_value(data.get("include_annotations"), True),
        }

    def _normalize_practice_set_revision(self, existing: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        data = self._ensure_mapping(payload)
        target_question_ids = self._string_list(data.get("target_question_ids")) or self._string_list(data.get("selected_question_ids"))
        locked_question_ids = self._string_list(data.get("locked_question_ids"))
        maintain_coverage = self._bool_value(data.get("maintain_coverage"), True)
        instruction_text = self._optional_text(data.get("instruction_text"))
        if not instruction_text:
            instruction_text = self._synthesize_practice_revision_instruction(bool(target_question_ids), bool(locked_question_ids), maintain_coverage)
        return {
            "instruction_text": instruction_text,
            "target_question_ids": target_question_ids,
            "locked_question_ids": locked_question_ids,
            "maintain_coverage": maintain_coverage,
        }

    def _normalize_grounding_input(self, data: Mapping[str, Any]) -> tuple[Optional[List[str]], Optional[Dict[str, Any]]]:
        evidence_bundle = data.get("evidence_bundle")
        if evidence_bundle is not None and not isinstance(evidence_bundle, dict):
            raise RequestValidationError("evidence_bundle must be an object when provided")

        material_ids = self._string_list(data.get("material_ids"))
        included_material_ids = self._string_list(data.get("included_material_ids"))
        excluded_material_ids = set(self._string_list(data.get("excluded_material_ids")))
        focused_material_ids = self._string_list(data.get("focused_material_ids"))
        focus_only = any(
            self._bool_value(data.get(flag_name), False)
            for flag_name in ("focus_only", "focused_materials_only", "focused_material_ids_only")
        )
        if not material_ids:
            if focus_only and focused_material_ids:
                material_ids = focused_material_ids
            elif included_material_ids:
                material_ids = [material_id for material_id in included_material_ids if material_id not in excluded_material_ids]
        elif excluded_material_ids:
            material_ids = [material_id for material_id in material_ids if material_id not in excluded_material_ids]
        if material_ids:
            material_ids = list(dict.fromkeys(material_ids))
        if evidence_bundle and material_ids:
            raise RequestValidationError("Provide exactly one of material_ids or evidence_bundle")
        if not evidence_bundle and not material_ids:
            raise RequestValidationError("Provide exactly one of material_ids or evidence_bundle")
        return (material_ids or None, copy.deepcopy(evidence_bundle) if evidence_bundle else None)

    def _normalize_student_context(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise RequestValidationError("student_context must be an object when provided")
        return {
            "prior_knowledge": self._optional_text(value.get("prior_knowledge")) or self._optional_text(value.get("known")),
            "weak_areas": self._optional_text(value.get("weak_areas")),
            "goals": self._optional_text(value.get("goals")),
        }

    def _expand_locked_sections(self, existing: Dict[str, Any], value: Any) -> List[str]:
        expanded: List[str] = []
        for entry in self._string_list(value):
            if entry == "entire_plan":
                expanded.extend(item["item_id"] for item in existing.get("prerequisites", []))
                expanded.extend(step["step_id"] for step in existing.get("study_sequence", []))
                expanded.extend(item["item_id"] for item in existing.get("common_mistakes", []))
            elif entry == "prerequisites":
                expanded.extend(item["item_id"] for item in existing.get("prerequisites", []))
            elif entry == "study_sequence":
                expanded.extend(step["step_id"] for step in existing.get("study_sequence", []))
            elif entry == "common_mistakes":
                expanded.extend(item["item_id"] for item in existing.get("common_mistakes", []))
            else:
                expanded.append(entry)
        return list(dict.fromkeys(expanded))

    def _synthesize_study_plan_revision_instruction(self, target_section: str, has_locks: bool) -> str:
        if target_section == "study_sequence":
            return "revise the study sequence for clarity while preserving locked sections" if has_locks else "revise the study sequence for clarity"
        if target_section == "prerequisites":
            return "revise the prerequisites for clarity while preserving locked sections" if has_locks else "revise the prerequisites for clarity"
        if target_section == "common_mistakes":
            return "revise the common mistakes for clarity while preserving locked sections" if has_locks else "revise the common mistakes for clarity"
        if target_section not in {"entire_plan", "prerequisites", "study_sequence", "common_mistakes"}:
            return "revise the selected study-plan item for clarity while preserving locked sections" if has_locks else "revise the selected study-plan item for clarity"
        return "revise the study sequence for clarity while preserving locked sections" if has_locks else "revise the study sequence for clarity"

    def _synthesize_practice_revision_instruction(self, has_targets: bool, has_locks: bool, maintain_coverage: bool) -> str:
        if has_targets:
            return "regenerate the selected questions while preserving the locked questions and overall coverage"
        if maintain_coverage:
            return "create a new variant with different wording but similar topic coverage"
        return "revise the practice set while preserving locked questions when possible" if has_locks else "revise the practice set for clarity"

    # --------------------------
    # Public shaping helpers
    # --------------------------
    def _public_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(artifact)
        payload.pop("_meta", None)
        return payload

    def _study_plan_list_item(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "study_plan_id": plan["study_plan_id"],
            "parent_study_plan_id": plan.get("parent_study_plan_id"),
            "workspace_id": plan["workspace_id"],
            "created_at": plan["created_at"],
            "topic_text": plan["topic_text"],
            "time_budget_minutes": plan["time_budget_minutes"],
        }

    # --------------------------
    # Basic coercion helpers
    # --------------------------
    def _ensure_mapping(self, value: Any) -> Dict[str, Any]:
        if not isinstance(value, Mapping):
            raise RequestValidationError("Request body must be a JSON object")
        return dict(value)

    def _required_text(self, value: Any, field_name: str) -> str:
        text = self._optional_text(value)
        if not text:
            raise RequestValidationError(f"{field_name} is required")
        return text

    def _optional_text(self, value: Any) -> Optional[str]:
        text = normalize_whitespace(str(value or ""))
        return text or None

    def _string_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            text = normalize_whitespace(value)
            return [text] if text else []
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
            items = []
            for item in value:
                text = self._optional_text(item)
                if text:
                    items.append(text)
            return items
        text = self._optional_text(value)
        return [text] if text else []

    def _positive_int(self, value: Any, field_name: str, default: Optional[int] = None) -> int:
        if value is None:
            if default is None:
                raise RequestValidationError(f"{field_name} is required")
            return default
        try:
            coerced = int(value)
        except Exception as exc:
            raise RequestValidationError(f"{field_name} must be a positive integer") from exc
        if coerced <= 0:
            raise RequestValidationError(f"{field_name} must be a positive integer")
        return coerced

    def _bool_value(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)

    def _grounding_mode(self, value: Any, default: str = "lecture_with_fallback") -> str:
        mode = self._optional_text(value) or default
        if mode not in ALLOWED_GROUNDING_MODES:
            raise RequestValidationError(f"grounding_mode must be one of {sorted(ALLOWED_GROUNDING_MODES)}")
        return mode
