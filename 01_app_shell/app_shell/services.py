from __future__ import annotations

import atexit
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import httpx

from app_shell.config import AppConfig
from app_shell.errors import ShellError
from app_shell.mock_data import (
    build_workspace_from_fixture,
    create_mock_material,
    enrich_citation,
    generate_mock_assistant_message,
    generate_mock_practice_set,
    generate_mock_study_plan,
    render_slide_preview_svg,
)
from app_shell.normalization import (
    build_feedback_annotation,
    normalize_conversation_create,
    normalize_conversation_message,
    normalize_material_import,
    normalize_practice_request,
    normalize_practice_revision,
    normalize_study_plan_request,
    normalize_study_plan_revision,
    summarize_material_preference,
)
from app_shell.storage import LocalStorage
from app_shell.utils import absolutize_url, deep_copy, make_id, utc_now_iso


class ServiceLauncher:
    def __init__(self, config: AppConfig):
        self.config = config
        self.processes: Dict[str, subprocess.Popen[str]] = {}
        self._registered = False

    def _health_url(self, base_url: str) -> str:
        return f"{base_url.rstrip('/')}/healthz"

    def check_health(self, base_url: str) -> bool:
        try:
            response = httpx.get(self._health_url(base_url), timeout=0.8)
            if response.status_code != 200:
                return False
            payload = response.json()
            return bool(payload.get("ready", False))
        except Exception:
            return False

    def _maybe_register_cleanup(self) -> None:
        if self._registered:
            return
        atexit.register(self.shutdown)
        self._registered = True

    def start_if_needed(self, service_name: str, folder_name: str, base_url: str) -> None:
        if self.config.testing:
            return
        if self.check_health(base_url):
            return
        if service_name in self.processes and self.processes[service_name].poll() is None:
            return
        folder = self.config.project_root / folder_name
        run_local = folder / "run_local.py"
        if not run_local.exists():
            return
        env = os.environ.copy()
        process = subprocess.Popen(
            [sys.executable, str(run_local)],
            cwd=str(folder),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.processes[service_name] = process
        self._maybe_register_cleanup()
        deadline = time.time() + 12
        while time.time() < deadline:
            if self.check_health(base_url):
                return
            if process.poll() is not None:
                return
            time.sleep(0.3)

    def shutdown(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.processes.clear()


class ShellService:
    def __init__(self, config: AppConfig, storage: LocalStorage):
        self.config = config
        self.storage = storage
        self.launcher = ServiceLauncher(config)
        self.effective_mode = "mock" if config.mode == "mock" else "integrated"
        self._last_status_snapshot: dict | None = None
        self._last_status_time = 0.0

    def startup(self) -> None:
        if self.config.mode == "mock":
            self.effective_mode = "mock"
            self.storage.maybe_seed_mock_fixture()
            self._last_status_snapshot = self._build_mock_status()
            return
        if not self.config.testing:
            self.launcher.start_if_needed("content", "02_content_service", self.config.content_service_url)
            self.launcher.start_if_needed("learning", "03_learning_service", self.config.learning_service_url)
        status = self.refresh_status(force=True)
        if self.config.mode == "auto" and not any(service["available"] for service in status["services"].values()):
            self.effective_mode = "mock"
            self.storage.maybe_seed_mock_fixture()
            self._last_status_snapshot = self._build_mock_status()
        else:
            self.effective_mode = "integrated"
            self._last_status_snapshot = status

    def shutdown(self) -> None:
        self.launcher.shutdown()

    def _build_mock_status(self) -> dict:
        now = utc_now_iso()
        return {
            "effective_mode": "mock",
            "services": {
                "content": {
                    "available": True,
                    "mode": "mock",
                    "base_url": self.config.content_service_url,
                    "last_checked_at": now,
                },
                "learning": {
                    "available": True,
                    "mode": "mock",
                    "base_url": self.config.learning_service_url,
                    "last_checked_at": now,
                },
            },
        }

    def refresh_status(self, *, force: bool = False) -> dict:
        if self.effective_mode == "mock" and self.config.mode == "mock":
            self._last_status_snapshot = self._build_mock_status()
            self._last_status_time = time.time()
            return deep_copy(self._last_status_snapshot)
        if not force and self._last_status_snapshot is not None and (time.time() - self._last_status_time) < 0.8:
            return deep_copy(self._last_status_snapshot)
        now = utc_now_iso()
        status = {
            "effective_mode": self.effective_mode,
            "services": {
                "content": {
                    "available": self.launcher.check_health(self.config.content_service_url),
                    "mode": self.effective_mode,
                    "base_url": self.config.content_service_url,
                    "last_checked_at": now,
                },
                "learning": {
                    "available": self.launcher.check_health(self.config.learning_service_url),
                    "mode": self.effective_mode,
                    "base_url": self.config.learning_service_url,
                    "last_checked_at": now,
                },
            },
        }
        if self.config.mode == "auto" and self.effective_mode != "mock" and not any(service["available"] for service in status["services"].values()):
            self.effective_mode = "mock"
            self.storage.maybe_seed_mock_fixture()
            status = self._build_mock_status()
        else:
            status["effective_mode"] = self.effective_mode
        self._last_status_snapshot = status
        self._last_status_time = time.time()
        return deep_copy(status)

    def status_snapshot(self) -> dict:
        return self.refresh_status(force=True)

    def _workspace_or_error(self, workspace_id: str) -> dict:
        workspace = self.storage.get_workspace(workspace_id)
        if workspace is None:
            raise ShellError(f"Workspace {workspace_id} was not found.", status_code=404)
        return workspace

    def _touch_status(self, workspace: dict) -> dict:
        snapshot = self.refresh_status(force=True)
        workspace.setdefault("status", {})["services"] = deep_copy(snapshot["services"])
        workspace["status"].setdefault("last_successful_sync", {"content": None, "learning": None})
        workspace["status"].setdefault("warnings", [])
        return workspace

    def _summarize_workspace(self, workspace: dict) -> dict:
        materials = list(workspace.get("materials", {}).values())
        summary = {
            "workspace_id": workspace["workspace_id"],
            "display_name": workspace.get("display_name", "Untitled workspace"),
            "created_at": workspace.get("created_at"),
            "last_opened_at": workspace.get("last_opened_at"),
            "archived": workspace.get("archived", False),
            "grounding_mode": workspace.get("grounding_mode", "strict_lecture_only"),
            "material_counts": {
                "total": len(materials),
                "ready": len([material for material in materials if material.get("processing_status") == "ready"]),
                "processing": len([material for material in materials if material.get("processing_status") not in {"ready", "failed"}]),
                "failed": len([material for material in materials if material.get("processing_status") == "failed"]),
            },
            "artifact_counts": {
                "study_plans": len(workspace.get("known_study_plan_ids", [])),
                "practice_sets": len(workspace.get("known_practice_set_ids", [])),
                "conversations": len(workspace.get("known_conversation_ids", [])),
            },
            "has_chat_history": bool(workspace.get("known_conversation_ids")),
            "has_study_plans": bool(workspace.get("known_study_plan_ids")),
            "has_practice_sets": bool(workspace.get("known_practice_set_ids")),
            "has_slides": bool(workspace.get("active_material_ids", {}).get("slides")),
            "has_notes": bool(workspace.get("active_material_ids", {}).get("notes")),
            "services": deep_copy(workspace.get("status", {}).get("services", self.refresh_status(force=True)["services"])),
        }
        return summary

    def list_workspaces(self) -> dict:
        workspaces = [self._touch_status(workspace) for workspace in self.storage.list_workspaces()]
        for workspace in workspaces:
            self.storage.save_workspace(workspace)
        return {
            "workspaces": [self._summarize_workspace(workspace) for workspace in workspaces],
            "status": self.refresh_status(force=True),
        }

    def create_workspace(self, display_name: str) -> dict:
        workspace = self.storage.create_workspace(display_name)
        workspace = self._touch_status(workspace)
        self.storage.save_workspace(workspace)
        return self.serialize_workspace(workspace)

    def duplicate_workspace(self, workspace_id: str) -> dict:
        workspace = self.storage.duplicate_workspace(workspace_id)
        workspace = self._touch_status(workspace)
        self.storage.save_workspace(workspace)
        return self.serialize_workspace(workspace)

    def archive_workspace(self, workspace_id: str) -> dict:
        workspace = self.storage.archive_workspace(workspace_id)
        workspace = self._touch_status(workspace)
        return self.serialize_workspace(workspace)

    def delete_workspace(self, workspace_id: str) -> None:
        self.storage.delete_workspace(workspace_id)

    def serialize_workspace(self, workspace: dict) -> dict:
        workspace = deep_copy(workspace)
        materials = list(workspace.get("materials", {}).values())
        materials.sort(key=lambda item: item.get("created_at") or "")
        active_study_plan = None
        if workspace.get("active_study_plan_id"):
            active_study_plan = workspace.get("study_plans", {}).get(workspace["active_study_plan_id"])
        active_practice_set = None
        if workspace.get("active_practice_set_id"):
            active_practice_set = workspace.get("practice_sets", {}).get(workspace["active_practice_set_id"])
        active_conversation = None
        if workspace.get("selected_active_conversation_id"):
            active_conversation = workspace.get("conversations", {}).get(workspace["selected_active_conversation_id"])
        payload = {
            **workspace,
            "materials": materials,
            "active_study_plan": active_study_plan,
            "active_practice_set": active_practice_set,
            "active_conversation": active_conversation,
            "jobs": self.storage.list_jobs_for_workspace(workspace["workspace_id"]),
        }
        return payload

    def get_workspace(self, workspace_id: str, *, refresh: bool = True) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        workspace = self._touch_status(workspace)
        if refresh and self.effective_mode == "integrated":
            workspace = self.hydrate_workspace(workspace)
        self.storage.save_workspace(workspace)
        return self.serialize_workspace(workspace)

    def get_history(self, workspace_id: str) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        history = deep_copy(workspace.get("history", []))
        history.sort(key=lambda entry: entry.get("created_at") or "", reverse=True)
        return {"history": history}

    def activate_artifact(self, workspace_id: str, artifact_type: str, artifact_id: str) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        mapping = {
            "study_plan": ("study_plans", "active_study_plan_id"),
            "practice_set": ("practice_sets", "active_practice_set_id"),
            "conversation": ("conversations", "selected_active_conversation_id"),
        }
        if artifact_type not in mapping:
            raise ShellError("Artifact type must be study_plan, practice_set, or conversation.")
        store_name, active_field = mapping[artifact_type]
        if artifact_id not in workspace.get(store_name, {}):
            raise ShellError(f"{artifact_type} {artifact_id} was not found in this workspace.", status_code=404)
        workspace[active_field] = artifact_id
        self._sync_history_flags(workspace)
        self.storage.save_workspace(workspace)
        return self.serialize_workspace(workspace)

    def _history_active_id(self, workspace: dict, artifact_type: str) -> str | None:
        if artifact_type == "study_plan":
            return workspace.get("active_study_plan_id")
        if artifact_type == "practice_set":
            return workspace.get("active_practice_set_id")
        if artifact_type == "conversation":
            return workspace.get("selected_active_conversation_id")
        return None

    def _sync_history_flags(self, workspace: dict) -> None:
        for entry in workspace.get("history", []):
            entry["active"] = entry.get("artifact_id") == self._history_active_id(workspace, entry.get("artifact_type", ""))

    def _choose_active_artifact_id(self, current_id: str | None, artifacts: dict, id_field: str) -> str | None:
        if current_id and current_id in artifacts:
            return current_id
        if not artifacts:
            return None
        newest = max(
            artifacts.values(),
            key=lambda item: ((item.get("created_at") or ""), str(item.get(id_field) or "")),
        )
        return newest.get(id_field)

    def _material_slide_map(self, material: dict) -> dict[str, dict]:
        return {slide.get("slide_id"): slide for slide in material.get("slides", []) if slide.get("slide_id")}

    def _normalize_slide_summary(self, material: dict, slide: dict) -> dict:
        normalized = deep_copy(slide)
        normalized.setdefault("material_id", material.get("material_id"))
        normalized.setdefault("material_title", material.get("title"))
        normalized["preview_url"] = absolutize_url(self.config.content_service_url, normalized.get("preview_url"))
        normalized["source_open_url"] = absolutize_url(
            self.config.content_service_url,
            normalized.get("source_open_url") or material.get("source_view_url"),
        )
        return normalized

    def _normalize_material_urls(self, material: dict) -> dict:
        normalized = deep_copy(material)
        normalized["source_view_url"] = absolutize_url(self.config.content_service_url, normalized.get("source_view_url"))
        slides = [self._normalize_slide_summary(normalized, slide) for slide in normalized.get("slides", []) if isinstance(slide, dict)]
        if slides:
            normalized["slides"] = slides
        return normalized

    def _normalize_citation(self, workspace: dict, citation: dict) -> dict:
        normalized = deep_copy(citation)
        material_id = normalized.get("material_id")
        material = workspace.get("materials", {}).get(material_id)
        slide = None
        if material and normalized.get("slide_id"):
            slide = self._material_slide_map(material).get(normalized.get("slide_id"))
        if material:
            normalized.setdefault("material_title", material.get("title"))
        if slide:
            normalized.setdefault("preview_url", slide.get("preview_url"))
            normalized.setdefault("source_open_url", slide.get("source_open_url"))
            normalized.setdefault("slide_number", slide.get("slide_number"))
            normalized.setdefault("snippet_text", slide.get("snippet_text"))
        if material and not normalized.get("source_open_url"):
            normalized["source_open_url"] = material.get("source_view_url")
        normalized["preview_url"] = absolutize_url(self.config.content_service_url, normalized.get("preview_url"))
        normalized["source_open_url"] = absolutize_url(self.config.content_service_url, normalized.get("source_open_url"))
        return normalized

    def _normalize_citation_payload(self, workspace: dict, payload: Any) -> Any:
        if isinstance(payload, list):
            return [self._normalize_citation_payload(workspace, item) for item in payload]
        if isinstance(payload, dict):
            updated: dict[str, Any] = {}
            for key, value in payload.items():
                if key == "citations" and isinstance(value, list):
                    updated[key] = [self._normalize_citation(workspace, citation) for citation in value if isinstance(citation, dict)]
                else:
                    updated[key] = self._normalize_citation_payload(workspace, value)
            return updated
        return payload

    def _annotation_key(self, annotation: dict) -> str:
        annotation_id = annotation.get("annotation_id")
        if annotation_id:
            return f"id:{annotation_id}"
        return "semantic:" + "|".join(
            [
                str(annotation.get("annotation_type", "")),
                str(annotation.get("scope", "")),
                str(annotation.get("material_id", "")),
                str(annotation.get("slide_id", "")),
                str(annotation.get("text", "")),
            ]
        )

    def _merge_annotations(self, workspace: dict, remote_annotations: Iterable[dict]) -> dict:
        existing_map = {self._annotation_key(annotation): deep_copy(annotation) for annotation in workspace.get("annotations", [])}
        for annotation in remote_annotations:
            existing_map[self._annotation_key(annotation)] = deep_copy(annotation)
        annotations = list(existing_map.values())
        annotations.sort(key=lambda item: item.get("created_at") or item.get("annotation_id") or "")
        workspace["annotations"] = annotations
        sync_map: Dict[str, Dict[str, str]] = workspace.get("material_sync_annotations", {})
        preferences = workspace.get("material_preferences", {})
        for material_id in list(preferences.keys()):
            if material_id not in sync_map:
                sync_map[material_id] = {}
        for annotation in annotations:
            scope = annotation.get("scope")
            material_id = annotation.get("material_id")
            annotation_type = annotation.get("annotation_type")
            if scope == "material" and material_id and annotation_type in {"focus", "exclude_from_grounding"}:
                sync_map.setdefault(material_id, {})[annotation_type] = annotation.get("annotation_id")
        for material_id, mapping in sync_map.items():
            if mapping.get("exclude_from_grounding"):
                preferences[material_id] = "exclude"
            elif mapping.get("focus"):
                preferences[material_id] = "focus"
            else:
                preferences.setdefault(material_id, "default")
        workspace["material_sync_annotations"] = sync_map
        workspace["material_preferences"] = preferences
        return workspace

    def _update_history_entry(self, workspace: dict, artifact_type: str, artifact_id: str, created_at: str, parent_id: str | None, title: str) -> None:
        history = workspace.setdefault("history", [])
        entry = next(
            (
                existing
                for existing in history
                if existing.get("artifact_type") == artifact_type and existing.get("artifact_id") == artifact_id
            ),
            None,
        )
        payload = {
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "created_at": created_at,
            "parent_artifact_id": parent_id,
            "title": title,
        }
        if entry is None:
            history.append({**payload, "active": False})
        else:
            entry.update(payload)
        history.sort(key=lambda entry: ((entry.get("created_at") or ""), str(entry.get("artifact_id") or "")))
        self._sync_history_flags(workspace)

    def _upsert_material(self, workspace: dict, material: dict) -> None:
        material_id = material["material_id"]
        existing = workspace.setdefault("materials", {}).get(material_id, {})
        merged = deep_copy(existing)
        merged.update(self._normalize_material_urls(material))
        if not merged.get("slides") and existing.get("slides"):
            merged["slides"] = deep_copy(existing["slides"])
        if not merged.get("source_view_url") and existing.get("source_view_url"):
            merged["source_view_url"] = existing["source_view_url"]
        workspace.setdefault("materials", {})[material_id] = merged
        workspace.setdefault("material_preferences", {}).setdefault(material_id, "default")
        grouped = {"slides": [], "notes": [], "practice_template": []}
        for existing_material in workspace.get("materials", {}).values():
            role = existing_material.get("role", "notes")
            grouped.setdefault(role, []).append(existing_material["material_id"])
        workspace["active_material_ids"] = grouped

    def _upsert_study_plan(self, workspace: dict, study_plan: dict, *, set_active: bool = True) -> None:
        study_plan = self._normalize_citation_payload(workspace, study_plan)
        study_plan_id = study_plan["study_plan_id"]
        workspace.setdefault("study_plans", {})[study_plan_id] = deep_copy(study_plan)
        if study_plan_id not in workspace.setdefault("known_study_plan_ids", []):
            workspace["known_study_plan_ids"].append(study_plan_id)
        if set_active:
            workspace["active_study_plan_id"] = study_plan_id
        self._update_history_entry(
            workspace,
            "study_plan",
            study_plan_id,
            study_plan.get("created_at") or utc_now_iso(),
            study_plan.get("parent_study_plan_id"),
            study_plan.get("topic_text") or "Study plan",
        )

    def _upsert_practice_set(self, workspace: dict, practice_set: dict, *, set_active: bool = True) -> None:
        practice_set = self._normalize_citation_payload(workspace, practice_set)
        practice_id = practice_set["practice_set_id"]
        workspace.setdefault("practice_sets", {})[practice_id] = deep_copy(practice_set)
        if practice_id not in workspace.setdefault("known_practice_set_ids", []):
            workspace["known_practice_set_ids"].append(practice_id)
        if set_active:
            workspace["active_practice_set_id"] = practice_id
        self._update_history_entry(
            workspace,
            "practice_set",
            practice_id,
            practice_set.get("created_at") or utc_now_iso(),
            practice_set.get("parent_practice_set_id"),
            practice_set.get("generation_mode") or "Practice set",
        )

    def _merge_conversation_messages(self, existing: dict | None, remote: dict) -> dict:
        merged = deep_copy(remote)
        existing_messages = (existing or {}).get("messages", [])
        remote_messages = merged.get("messages", [])
        pending_messages = [message for message in existing_messages if message.get("pending")]
        remote_user_texts = {message.get("text") for message in remote_messages if message.get("role") == "user"}
        for pending_message in pending_messages:
            if pending_message.get("role") == "user" and pending_message.get("text") not in remote_user_texts:
                remote_messages.append(pending_message)
        remote_messages.sort(key=lambda item: item.get("created_at") or "")
        merged["messages"] = remote_messages
        return merged

    def _upsert_conversation(self, workspace: dict, conversation: dict, *, set_active: bool = True) -> None:
        conversation_id = conversation["conversation_id"]
        existing = workspace.setdefault("conversations", {}).get(conversation_id)
        merged = self._merge_conversation_messages(existing, self._normalize_citation_payload(workspace, conversation))
        workspace["conversations"][conversation_id] = merged
        if conversation_id not in workspace.setdefault("known_conversation_ids", []):
            workspace["known_conversation_ids"].append(conversation_id)
        if set_active:
            workspace["selected_active_conversation_id"] = conversation_id
        self._update_history_entry(
            workspace,
            "conversation",
            conversation_id,
            conversation.get("created_at") or utc_now_iso(),
            None,
            conversation.get("title") or "Conversation",
        )

    def _extract_items(self, payload: Any, plural_key: str, singular_key: str | None = None) -> List[dict]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        for key in [plural_key, singular_key, "items", "results", "data"]:
            if key and isinstance(payload.get(key), list):
                return [item for item in payload[key] if isinstance(item, dict)]
        if singular_key and isinstance(payload.get(singular_key), dict):
            return [payload[singular_key]]
        return []

    def _extract_detail(self, payload: Any, singular_key: str) -> dict:
        if isinstance(payload, dict) and isinstance(payload.get(singular_key), dict):
            return payload[singular_key]
        if isinstance(payload, dict):
            return payload
        raise ShellError("Remote service returned an unexpected response shape.", status_code=502)

    def _remote_json(self, service: str, method: str, path: str, *, params: dict | None = None, json_body: dict | None = None, data: dict | None = None, files: dict | None = None) -> dict:
        base_url = self.config.content_service_url if service == "content" else self.config.learning_service_url
        try:
            with httpx.Client(timeout=6.0) as client:
                response = client.request(method, f"{base_url}{path}", params=params, json=json_body, data=data, files=files)
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                payload = response.json()
            else:
                payload = {"message": response.text}
            if response.status_code >= 400:
                message = None
                if isinstance(payload, dict):
                    message = (
                        payload.get("error", {}).get("message")
                        if isinstance(payload.get("error"), dict)
                        else payload.get("message")
                    )
                raise ShellError(message or f"{service.title()} service request failed with HTTP {response.status_code}.", status_code=502, details={"remote_status_code": response.status_code, "service": service})
            if isinstance(payload, dict):
                return payload
            raise ShellError("Remote service returned a non-JSON response.", status_code=502)
        except ShellError:
            raise
        except Exception as exc:
            raise ShellError(f"Could not reach the {service} service: {exc}", status_code=503)

    def hydrate_workspace(self, workspace: dict) -> dict:
        workspace = deep_copy(workspace)
        current_active_study_plan_id = workspace.get("active_study_plan_id")
        current_active_practice_set_id = workspace.get("active_practice_set_id")
        current_active_conversation_id = workspace.get("selected_active_conversation_id")
        snapshot = self.refresh_status(force=True)
        workspace["status"]["services"] = deep_copy(snapshot["services"])
        if snapshot["services"]["content"]["available"]:
            materials_payload = self._remote_json("content", "GET", "/v1/materials", params={"workspace_id": workspace["workspace_id"]})
            remote_materials = self._extract_items(materials_payload, "materials", "material")
            for material in remote_materials:
                material.setdefault("workspace_id", workspace["workspace_id"])
                material = self._normalize_material_urls(material)
                if material.get("processing_status") == "ready":
                    try:
                        slides_payload = self._remote_json("content", "GET", f"/v1/materials/{material['material_id']}/slides")
                        slides = self._extract_items(slides_payload, "slides", "slide")
                        if slides:
                            material["slides"] = [self._normalize_slide_summary(material, slide) for slide in slides]
                            if not material.get("source_view_url"):
                                material["source_view_url"] = material["slides"][0].get("source_open_url")
                    except ShellError:
                        pass
                self._upsert_material(workspace, material)
            annotations_payload = self._remote_json("content", "GET", f"/v1/workspaces/{workspace['workspace_id']}/annotations")
            remote_annotations = self._extract_items(annotations_payload, "annotations", "annotation")
            workspace = self._merge_annotations(workspace, remote_annotations)
            workspace["status"].setdefault("last_successful_sync", {})["content"] = utc_now_iso()
        if snapshot["services"]["learning"]["available"]:
            plan_payload = self._remote_json("learning", "GET", "/v1/study-plans", params={"workspace_id": workspace["workspace_id"]})
            for plan in self._extract_items(plan_payload, "study_plans", "study_plan"):
                if "prerequisites" not in plan:
                    detail = self._remote_json("learning", "GET", f"/v1/study-plans/{plan['study_plan_id']}")
                    plan = self._extract_detail(detail, "study_plan")
                self._upsert_study_plan(workspace, plan, set_active=False)
            practice_payload = self._remote_json("learning", "GET", "/v1/practice-sets", params={"workspace_id": workspace["workspace_id"]})
            for practice_set in self._extract_items(practice_payload, "practice_sets", "practice_set"):
                if "questions" not in practice_set:
                    detail = self._remote_json("learning", "GET", f"/v1/practice-sets/{practice_set['practice_set_id']}")
                    practice_set = self._extract_detail(detail, "practice_set")
                self._upsert_practice_set(workspace, practice_set, set_active=False)
            conversations_payload = self._remote_json("learning", "GET", "/v1/conversations", params={"workspace_id": workspace["workspace_id"]})
            for conversation in self._extract_items(conversations_payload, "conversations", "conversation"):
                if "messages" not in conversation:
                    detail = self._remote_json("learning", "GET", f"/v1/conversations/{conversation['conversation_id']}")
                    conversation = self._extract_detail(detail, "conversation")
                self._upsert_conversation(workspace, conversation, set_active=False)
            workspace["status"].setdefault("last_successful_sync", {})["learning"] = utc_now_iso()
        workspace["active_study_plan_id"] = self._choose_active_artifact_id(
            current_active_study_plan_id,
            workspace.get("study_plans", {}),
            "study_plan_id",
        )
        workspace["active_practice_set_id"] = self._choose_active_artifact_id(
            current_active_practice_set_id,
            workspace.get("practice_sets", {}),
            "practice_set_id",
        )
        workspace["selected_active_conversation_id"] = self._choose_active_artifact_id(
            current_active_conversation_id,
            workspace.get("conversations", {}),
            "conversation_id",
        )
        self._sync_history_flags(workspace)
        return workspace

    def _create_job(self, workspace_id: str, operation: str, *, service: str, context: dict, remote_job_id: str | None = None, status: str = "queued", message: str = "Queued", stage: str = "queued", result_type: str | None = None, result_id: str | None = None, error: dict | None = None, user_action: dict | None = None, finalized: bool = False) -> dict:
        job = {
            "job_id": make_id("job"),
            "workspace_id": workspace_id,
            "operation": operation,
            "service": service,
            "remote_job_id": remote_job_id,
            "status": status,
            "progress": 0 if status == "queued" else (100 if status == "succeeded" else 0),
            "stage": stage,
            "message": message,
            "result_type": result_type,
            "result_id": result_id,
            "user_action": user_action,
            "error": error,
            "context": deep_copy(context),
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "poll_count": 0,
            "finalized": finalized,
        }
        return self.storage.create_job(job)

    def _fail_job(self, workspace_id: str, operation: str, message: str, *, service: str) -> dict:
        return self._create_job(
            workspace_id,
            operation,
            service=service,
            context={},
            status="failed",
            message=message,
            stage="failed",
            error={"message": message, "retryable": True},
        )

    def _file_dedup_key(self, *, role: str, file_payload: dict) -> str:
        digest = hashlib.sha256(file_payload.get("content") or b"").hexdigest()
        return f"{role}:{digest}"

    def _find_duplicate_material_id(self, workspace: dict, dedup_key: str) -> str | None:
        dedup_map = workspace.setdefault("file_import_dedup", {})
        existing_id = dedup_map.get(dedup_key)
        if existing_id and existing_id in workspace.get("materials", {}):
            return str(existing_id)
        return None

    def _remember_file_dedup_material(self, workspace: dict, dedup_key: str | None, material_id: str | None) -> None:
        if not dedup_key or not material_id:
            return
        workspace.setdefault("file_import_dedup", {})[dedup_key] = material_id

    def import_material(self, workspace_id: str, form: dict, file_payload: dict | None = None) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        normalized = normalize_material_import(
            workspace_id,
            form.get("role", "notes"),
            form.get("title", ""),
            form.get("kind"),
            form.get("text") or form.get("source_text") or form.get("text_body"),
            file_payload.get("filename") if file_payload else None,
        )
        dedup_key = None
        if file_payload:
            dedup_key = self._file_dedup_key(role=normalized["role"], file_payload=file_payload)
            duplicate_material_id = self._find_duplicate_material_id(workspace, dedup_key)
            if duplicate_material_id:
                self.storage.save_workspace(workspace)
                return self._create_job(
                    workspace_id,
                    "import_material",
                    service="local_dedup",
                    context={
                        "normalized_request": normalized,
                        "deduplication": {"skipped": True, "material_id": duplicate_material_id},
                    },
                    status="succeeded",
                    stage="completed",
                    message="Duplicate file detected. Reusing the existing imported material.",
                    result_type="material",
                    result_id=duplicate_material_id,
                    finalized=True,
                )
        if self.effective_mode == "mock":
            return self._create_job(
                workspace_id,
                "import_material",
                service="local_mock",
                context={
                    "normalized_request": normalized,
                    "dedup_key": dedup_key,
                    "file_payload": {
                        "filename": file_payload.get("filename") if file_payload else None,
                    },
                },
                message="Material import queued in mock mode.",
                stage="queued",
            )
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["content"]["available"]:
            return self._fail_job(workspace_id, "import_material", "The content service is unavailable, so material import cannot proceed right now.", service="content")
        files = None
        if file_payload:
            files = {
                "file": (
                    file_payload.get("filename") or "upload.bin",
                    file_payload.get("content") or b"",
                    file_payload.get("content_type") or "application/octet-stream",
                )
            }
        payload = self._remote_json("content", "POST", "/v1/materials/import", data=normalized, files=files)
        remote_job_id = payload.get("job_id") or payload.get("job", {}).get("job_id")
        if not remote_job_id:
            raise ShellError("The content service did not return a job_id for material import.", status_code=502)
        return self._create_job(
            workspace_id,
            "import_material",
            service="content",
            remote_job_id=remote_job_id,
            context={
                "normalized_request": normalized,
                "dedup_key": dedup_key,
            },
            message="Material import submitted to the content service.",
            stage="submitted",
        )

    def _finalize_mock_job(self, job: dict) -> dict:
        workspace = self._workspace_or_error(job["workspace_id"])
        operation = job["operation"]
        context = job.get("context", {})
        if operation == "import_material":
            request_payload = context.get("normalized_request", {})
            file_name = (context.get("file_payload") or {}).get("filename")
            material = create_mock_material(
                workspace["workspace_id"],
                request_payload.get("title", "Uploaded material"),
                request_payload.get("role", "notes"),
                source_kind=request_payload.get("source_kind", "pasted_text"),
                text_body=request_payload.get("text_body"),
                filename=file_name,
            )
            self._upsert_material(workspace, material)
            self._remember_file_dedup_material(workspace, context.get("dedup_key"), material.get("material_id"))
            self.storage.save_workspace(workspace)
            job.update(
                {
                    "status": "succeeded",
                    "progress": 100,
                    "stage": "completed",
                    "message": "Material import completed.",
                    "result_type": "material",
                    "result_id": material["material_id"],
                    "finalized": True,
                    "updated_at": utc_now_iso(),
                }
            )
            return job
        if operation == "study_plan_generate":
            plan = generate_mock_study_plan(workspace, context["normalized_request"])
            self._upsert_study_plan(workspace, plan)
            self.storage.save_workspace(workspace)
            job.update({"status": "succeeded", "progress": 100, "stage": "completed", "message": "Study plan ready.", "result_type": "study_plan", "result_id": plan["study_plan_id"], "finalized": True, "updated_at": utc_now_iso()})
            return job
        if operation == "study_plan_revise":
            parent_plan_id = context.get("study_plan_id")
            plan = generate_mock_study_plan(workspace, {"topic_text": workspace.get("topic_text"), "time_budget_minutes": workspace.get("time_budget_minutes") or 60, "grounding_mode": workspace.get("grounding_mode")}, parent_study_plan_id=parent_plan_id)
            self._upsert_study_plan(workspace, plan)
            self.storage.save_workspace(workspace)
            job.update({"status": "succeeded", "progress": 100, "stage": "completed", "message": "Study plan revision ready.", "result_type": "study_plan", "result_id": plan["study_plan_id"], "finalized": True, "updated_at": utc_now_iso()})
            return job
        if operation == "conversation_message":
            conversation = workspace.get("conversations", {}).get(context["conversation_id"])
            if not conversation:
                raise ShellError("Conversation was not found while finalizing the local chat job.", status_code=404)
            for message in conversation.get("messages", []):
                if message.get("pending") and message.get("job_id") == job["job_id"]:
                    message["pending"] = False
            assistant_message = generate_mock_assistant_message(workspace, context["message_text"], context["grounding_mode"], context.get("response_style", "standard"))
            conversation.setdefault("messages", []).append(assistant_message)
            self._upsert_conversation(workspace, conversation)
            self.storage.save_workspace(workspace)
            job.update({"status": "succeeded", "progress": 100, "stage": "completed", "message": "Grounded answer ready.", "result_type": "assistant_message", "result_id": assistant_message["message_id"], "finalized": True, "updated_at": utc_now_iso()})
            return job
        if operation == "practice_generate":
            practice_set = generate_mock_practice_set(workspace, context["normalized_request"])
            self._upsert_practice_set(workspace, practice_set)
            self.storage.save_workspace(workspace)
            job.update({"status": "succeeded", "progress": 100, "stage": "completed", "message": "Practice set ready.", "result_type": "practice_set", "result_id": practice_set["practice_set_id"], "finalized": True, "updated_at": utc_now_iso()})
            return job
        if operation == "practice_revise":
            parent_id = context.get("practice_set_id")
            practice_set = generate_mock_practice_set(workspace, {"generation_mode": workspace.get("practice_sets", {}).get(parent_id, {}).get("generation_mode", "mixed"), "question_count": len(workspace.get("practice_sets", {}).get(parent_id, {}).get("questions", [])) or 3, "difficulty_profile": "mixed", "include_rubrics": True, "include_answer_key": True}, parent_practice_set_id=parent_id)
            self._upsert_practice_set(workspace, practice_set)
            self.storage.save_workspace(workspace)
            job.update({"status": "succeeded", "progress": 100, "stage": "completed", "message": "Practice revision ready.", "result_type": "practice_set", "result_id": practice_set["practice_set_id"], "finalized": True, "updated_at": utc_now_iso()})
            return job
        raise ShellError(f"Unsupported mock job operation: {operation}", status_code=500)

    def _finalize_remote_job(self, job: dict) -> dict:
        workspace = self._workspace_or_error(job["workspace_id"])
        workspace = self.hydrate_workspace(workspace)
        operation = job["operation"]
        context = job.get("context", {})
        result_id = job.get("result_id")
        if operation in {"study_plan_generate", "study_plan_revise"}:
            if result_id and result_id in workspace.get("study_plans", {}):
                workspace["active_study_plan_id"] = result_id
        elif operation in {"practice_generate", "practice_revise"}:
            if result_id and result_id in workspace.get("practice_sets", {}):
                workspace["active_practice_set_id"] = result_id
        elif operation == "conversation_message":
            conversation_id = context.get("conversation_id")
            if conversation_id and conversation_id in workspace.get("conversations", {}):
                workspace["selected_active_conversation_id"] = conversation_id
                conversation = workspace["conversations"][conversation_id]
                for message in conversation.get("messages", []):
                    if message.get("pending") and message.get("job_id") == job["job_id"]:
                        message["pending"] = False
                workspace["conversations"][conversation_id] = conversation
        elif operation == "import_material":
            self._remember_file_dedup_material(workspace, context.get("dedup_key"), result_id)
        self._sync_history_flags(workspace)
        self.storage.save_workspace(workspace)
        job["finalized"] = True
        job["updated_at"] = utc_now_iso()
        return job

    def poll_job(self, workspace_id: str, job_id: str) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        job = self.storage.get_job(job_id)
        if job is None or job.get("workspace_id") != workspace_id:
            raise ShellError(f"Job {job_id} was not found for workspace {workspace_id}.", status_code=404)
        if job["service"] == "local_mock":
            def _advance(current: dict) -> dict:
                if current.get("status") in {"failed", "needs_user_input", "succeeded"} and current.get("finalized"):
                    return current
                poll_count = int(current.get("poll_count", 0)) + 1
                current["poll_count"] = poll_count
                current["updated_at"] = utc_now_iso()
                if poll_count == 1:
                    current.update({"status": "running", "progress": 55, "stage": "running", "message": "Working in mock mode."})
                    return current
                return self._finalize_mock_job(current)
            return self.storage.update_job(job_id, _advance)
        if job["service"] in {"content", "learning"}:
            if job.get("status") in {"failed", "needs_user_input", "succeeded"} and job.get("finalized"):
                return job
            service = job["service"]
            snapshot = self.refresh_status(force=True)
            if not snapshot["services"][service]["available"]:
                def _mark_unavailable(current: dict) -> dict:
                    current["updated_at"] = utc_now_iso()
                    current["message"] = f"Waiting for the {service} service to return."
                    current["stage"] = "waiting_for_service"
                    return current
                return self.storage.update_job(job_id, _mark_unavailable)
            remote_payload = self._remote_json(service, "GET", f"/v1/jobs/{job['remote_job_id']}")
            remote_job = remote_payload.get("job", remote_payload)
            def _merge_remote(current: dict) -> dict:
                current["status"] = remote_job.get("status", current.get("status"))
                current["progress"] = remote_job.get("progress", current.get("progress", 0))
                current["stage"] = remote_job.get("stage", current.get("stage"))
                current["message"] = remote_job.get("message", current.get("message"))
                current["result_type"] = remote_job.get("result_type", current.get("result_type"))
                current["result_id"] = remote_job.get("result_id", current.get("result_id"))
                current["user_action"] = remote_job.get("user_action")
                current["error"] = remote_job.get("error")
                current["updated_at"] = utc_now_iso()
                if current["status"] == "needs_user_input":
                    return current
                if current["status"] == "succeeded":
                    return self._finalize_remote_job(current)
                if current["status"] == "failed":
                    current["finalized"] = True
                return current
            updated = self.storage.update_job(job_id, _merge_remote)
            if updated["status"] == "needs_user_input":
                workspace = self._workspace_or_error(workspace_id)
                if updated["operation"] == "conversation_message":
                    conversation_id = updated.get("context", {}).get("conversation_id")
                    if conversation_id and conversation_id in workspace.get("conversations", {}):
                        conversation = workspace["conversations"][conversation_id]
                        conversation.setdefault("messages", []).append(
                            {
                                "message_id": make_id("msg_assistant"),
                                "role": "assistant",
                                "created_at": utc_now_iso(),
                                "reply_sections": [],
                                "clarifying_question": {
                                    "prompt": updated.get("user_action", {}).get("prompt"),
                                    "reason": updated.get("user_action", {}).get("kind"),
                                },
                            }
                        )
                        workspace["conversations"][conversation_id] = conversation
                workspace.setdefault("pending_user_actions", []).append({"job_id": updated["job_id"], "user_action": updated.get("user_action")})
                self.storage.save_workspace(workspace)
            return updated
        return job

    def generate_study_plan(self, workspace_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        normalized, warnings = normalize_study_plan_request(workspace, payload)
        workspace["topic_text"] = normalized.get("topic_text") or workspace.get("topic_text")
        workspace["time_budget_minutes"] = normalized["time_budget_minutes"]
        workspace["grounding_mode"] = normalized["grounding_mode"]
        workspace["student_context"] = {
            "known": normalized["student_context"].get("prior_knowledge", ""),
            "weak_areas": normalized["student_context"].get("weak_areas", ""),
            "goals": normalized["student_context"].get("goals", ""),
        }
        self.storage.save_workspace(workspace)
        if self.effective_mode == "mock":
            return self._create_job(workspace_id, "study_plan_generate", service="local_mock", context={"normalized_request": normalized, "warnings": warnings}, message="Study plan generation queued in mock mode.")
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["learning"]["available"]:
            return self._fail_job(workspace_id, "study_plan_generate", "The learning service is unavailable, so study-plan generation cannot proceed right now.", service="learning")
        payload_out = self._remote_json("learning", "POST", "/v1/study-plans", json_body=normalized)
        remote_job_id = payload_out.get("job_id") or payload_out.get("job", {}).get("job_id")
        if not remote_job_id:
            raise ShellError("The learning service did not return a job_id for study-plan generation.", status_code=502)
        return self._create_job(workspace_id, "study_plan_generate", service="learning", remote_job_id=remote_job_id, context={"normalized_request": normalized, "warnings": warnings}, message="Study plan generation submitted to the learning service.")

    def revise_study_plan(self, workspace_id: str, study_plan_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        study_plan = workspace.get("study_plans", {}).get(study_plan_id)
        if not study_plan:
            raise ShellError(f"Study plan {study_plan_id} was not found in this workspace.", status_code=404)
        normalized = normalize_study_plan_revision(workspace, study_plan, payload)
        if self.effective_mode == "mock":
            return self._create_job(workspace_id, "study_plan_revise", service="local_mock", context={"normalized_request": normalized, "study_plan_id": study_plan_id}, message="Study plan revision queued in mock mode.")
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["learning"]["available"]:
            return self._fail_job(workspace_id, "study_plan_revise", "The learning service is unavailable, so study-plan revision cannot proceed right now.", service="learning")
        payload_out = self._remote_json("learning", "POST", f"/v1/study-plans/{study_plan_id}/revise", json_body=normalized)
        remote_job_id = payload_out.get("job_id") or payload_out.get("job", {}).get("job_id")
        if not remote_job_id:
            raise ShellError("The learning service did not return a job_id for study-plan revision.", status_code=502)
        return self._create_job(workspace_id, "study_plan_revise", service="learning", remote_job_id=remote_job_id, context={"normalized_request": normalized, "study_plan_id": study_plan_id}, message="Study plan revision submitted to the learning service.")

    def create_conversation(self, workspace_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        normalized, warnings = normalize_conversation_create(workspace, payload)
        if self.effective_mode == "mock":
            conversation_id = make_id("conversation")
            conversation = {
                "conversation_id": conversation_id,
                "workspace_id": workspace_id,
                "created_at": utc_now_iso(),
                "title": normalized.get("title") or "Workspace Q&A",
                "messages": [],
                "warnings": warnings,
            }
            self._upsert_conversation(workspace, conversation)
            self.storage.save_workspace(workspace)
            return {"conversation": conversation, "warnings": warnings}
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["learning"]["available"]:
            raise ShellError("The learning service is unavailable, so a real conversation cannot be created right now.", status_code=503)
        payload_out = self._remote_json("learning", "POST", "/v1/conversations", json_body=normalized)
        conversation = self._extract_detail(payload_out, "conversation")
        if not conversation.get("conversation_id"):
            raise ShellError("The learning service did not return a valid conversation object.", status_code=502)
        self._upsert_conversation(workspace, conversation)
        self.storage.save_workspace(workspace)
        return {"conversation": conversation, "warnings": warnings}

    def send_conversation_message(self, workspace_id: str, conversation_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        conversation = workspace.get("conversations", {}).get(conversation_id)
        if not conversation:
            raise ShellError(f"Conversation {conversation_id} was not found in this workspace.", status_code=404)
        normalized = normalize_conversation_message(workspace, payload)
        user_message = {
            "message_id": make_id("msg_user"),
            "role": "user",
            "created_at": utc_now_iso(),
            "text": normalized["message_text"],
            "pending": True,
        }
        conversation.setdefault("messages", []).append(user_message)
        self._upsert_conversation(workspace, conversation)
        self.storage.save_workspace(workspace)

        def rollback_pending_user() -> None:
            latest_workspace = self._workspace_or_error(workspace_id)
            latest_conversation = latest_workspace.get("conversations", {}).get(conversation_id)
            if not latest_conversation:
                return
            latest_conversation["messages"] = [
                message
                for message in latest_conversation.get("messages", [])
                if message.get("message_id") != user_message["message_id"]
            ]
            latest_workspace["conversations"][conversation_id] = latest_conversation
            self.storage.save_workspace(latest_workspace)

        if self.effective_mode == "mock":
            job = self._create_job(
                workspace_id,
                "conversation_message",
                service="local_mock",
                context={
                    "conversation_id": conversation_id,
                    "message_text": normalized["message_text"],
                    "grounding_mode": normalized["grounding_mode"],
                    "response_style": normalized["response_style"],
                },
                message="Chat response queued in mock mode.",
            )
            workspace = self._workspace_or_error(workspace_id)
            conversation = workspace.get("conversations", {}).get(conversation_id)
            if conversation and conversation.get("messages"):
                conversation["messages"][-1]["job_id"] = job["job_id"]
                workspace["conversations"][conversation_id] = conversation
                self.storage.save_workspace(workspace)
            return {"job": job, "conversation": self.serialize_workspace(workspace).get("active_conversation")}
        try:
            snapshot = self.refresh_status(force=True)
            if not snapshot["services"]["learning"]["available"]:
                raise ShellError("The learning service is unavailable, so the chat message could not be sent.", status_code=503)
            payload_out = self._remote_json("learning", "POST", f"/v1/conversations/{conversation_id}/messages", json_body=normalized)
            remote_job_id = payload_out.get("job_id") or payload_out.get("job", {}).get("job_id")
            if not remote_job_id:
                raise ShellError("The learning service did not return a job_id for the conversation message.", status_code=502)
            job = self._create_job(
                workspace_id,
                "conversation_message",
                service="learning",
                remote_job_id=remote_job_id,
                context={
                    "conversation_id": conversation_id,
                    "message_text": normalized["message_text"],
                    "grounding_mode": normalized["grounding_mode"],
                    "response_style": normalized["response_style"],
                },
                message="Chat message submitted to the learning service.",
            )
            workspace = self._workspace_or_error(workspace_id)
            conversation = workspace.get("conversations", {}).get(conversation_id)
            if conversation and conversation.get("messages"):
                conversation["messages"][-1]["job_id"] = job["job_id"]
                workspace["conversations"][conversation_id] = conversation
                self.storage.save_workspace(workspace)
            return {"job": job, "conversation": self.serialize_workspace(workspace).get("active_conversation")}
        except Exception:
            rollback_pending_user()
            raise

    def clear_conversation(self, workspace_id: str, conversation_id: str) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        conversation = workspace.get("conversations", {}).get(conversation_id)
        if not conversation:
            raise ShellError(f"Conversation {conversation_id} was not found in this workspace.", status_code=404)
        if self.effective_mode == "mock":
            conversation["messages"] = []
            self._upsert_conversation(workspace, conversation)
            self.storage.save_workspace(workspace)
            return {"conversation": conversation}
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["learning"]["available"]:
            raise ShellError("The learning service is unavailable, so the conversation could not be cleared safely.", status_code=503)
        payload_out = self._remote_json("learning", "POST", f"/v1/conversations/{conversation_id}/clear", json_body={})
        cleared = self._extract_detail(payload_out, "conversation")
        self._upsert_conversation(workspace, cleared)
        self.storage.save_workspace(workspace)
        return {"conversation": cleared}

    def generate_practice_set(self, workspace_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        normalized, warnings = normalize_practice_request(workspace, payload)
        if self.effective_mode == "mock":
            return self._create_job(workspace_id, "practice_generate", service="local_mock", context={"normalized_request": normalized, "warnings": warnings}, message="Practice generation queued in mock mode.")
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["learning"]["available"]:
            return self._fail_job(workspace_id, "practice_generate", "The learning service is unavailable, so practice generation cannot proceed right now.", service="learning")
        payload_out = self._remote_json("learning", "POST", "/v1/practice-sets", json_body=normalized)
        remote_job_id = payload_out.get("job_id") or payload_out.get("job", {}).get("job_id")
        if not remote_job_id:
            raise ShellError("The learning service did not return a job_id for practice generation.", status_code=502)
        return self._create_job(workspace_id, "practice_generate", service="learning", remote_job_id=remote_job_id, context={"normalized_request": normalized, "warnings": warnings}, message="Practice generation submitted to the learning service.")

    def revise_practice_set(self, workspace_id: str, practice_set_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        practice_set = workspace.get("practice_sets", {}).get(practice_set_id)
        if not practice_set:
            raise ShellError(f"Practice set {practice_set_id} was not found in this workspace.", status_code=404)
        normalized = normalize_practice_revision(practice_set, payload)
        if self.effective_mode == "mock":
            return self._create_job(workspace_id, "practice_revise", service="local_mock", context={"normalized_request": normalized, "practice_set_id": practice_set_id}, message="Practice revision queued in mock mode.")
        snapshot = self.refresh_status(force=True)
        if not snapshot["services"]["learning"]["available"]:
            return self._fail_job(workspace_id, "practice_revise", "The learning service is unavailable, so practice revision cannot proceed right now.", service="learning")
        payload_out = self._remote_json("learning", "POST", f"/v1/practice-sets/{practice_set_id}/revise", json_body=normalized)
        remote_job_id = payload_out.get("job_id") or payload_out.get("job", {}).get("job_id")
        if not remote_job_id:
            raise ShellError("The learning service did not return a job_id for practice revision.", status_code=502)
        return self._create_job(workspace_id, "practice_revise", service="learning", remote_job_id=remote_job_id, context={"normalized_request": normalized, "practice_set_id": practice_set_id}, message="Practice revision submitted to the learning service.")

    def delete_material(self, workspace_id: str, material_id: str) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        if material_id not in workspace.get("materials", {}):
            raise ShellError(f"Material {material_id} was not found in this workspace.", status_code=404)
        if self.effective_mode == "integrated":
            snapshot = self.refresh_status(force=False)
            if snapshot["services"]["content"]["available"]:
                try:
                    self._remote_json("content", "DELETE", f"/v1/materials/{material_id}")
                except Exception:
                    pass
        workspace.get("materials", {}).pop(material_id, None)
        workspace.get("material_preferences", {}).pop(material_id, None)
        workspace.get("material_sync_annotations", {}).pop(material_id, None)
        for role_list in workspace.get("active_material_ids", {}).values():
            if material_id in role_list:
                role_list.remove(material_id)
        self.storage.save_workspace(workspace)
        return self.serialize_workspace(workspace)

    def set_material_preference(self, workspace_id: str, material_id: str, preference: str) -> dict:
        if preference not in {"default", "focus", "exclude"}:
            raise ShellError("Material preference must be default, focus, or exclude.")
        workspace = self._workspace_or_error(workspace_id)
        material = workspace.get("materials", {}).get(material_id)
        if not material:
            raise ShellError(f"Material {material_id} was not found in this workspace.", status_code=404)
        workspace.setdefault("material_preferences", {})[material_id] = preference
        workspace.setdefault("material_sync_annotations", {}).setdefault(material_id, {})
        sync_map = workspace["material_sync_annotations"][material_id]
        warning = None
        if self.effective_mode == "integrated":
            snapshot = self.refresh_status(force=True)
            if snapshot["services"]["content"]["available"]:
                try:
                    for annotation_type in ["focus", "exclude_from_grounding"]:
                        if preference == "focus" and annotation_type == "focus":
                            continue
                        if preference == "exclude" and annotation_type == "exclude_from_grounding":
                            continue
                        existing_id = sync_map.get(annotation_type)
                        if existing_id:
                            self._remote_json("content", "DELETE", f"/v1/workspaces/{workspace_id}/annotations/{existing_id}")
                            sync_map.pop(annotation_type, None)
                    if preference in {"focus", "exclude"}:
                        annotation_type = "focus" if preference == "focus" else "exclude_from_grounding"
                        if not sync_map.get(annotation_type):
                            annotation_payload = {
                                "annotation_type": annotation_type,
                                "scope": "material",
                                "material_id": material_id,
                                "text": summarize_material_preference(material, preference),
                            }
                            created = self._remote_json("content", "POST", f"/v1/workspaces/{workspace_id}/annotations", json_body=annotation_payload)
                            annotation = self._extract_detail(created, "annotation")
                            sync_map[annotation_type] = annotation.get("annotation_id")
                            workspace = self._merge_annotations(workspace, [annotation])
                    elif preference == "default":
                        sync_map.pop("focus", None)
                        sync_map.pop("exclude_from_grounding", None)
                        remaining = []
                        for annotation in workspace.get("annotations", []):
                            if annotation.get("scope") == "material" and annotation.get("material_id") == material_id and annotation.get("annotation_type") in {"focus", "exclude_from_grounding"}:
                                continue
                            remaining.append(annotation)
                        workspace["annotations"] = remaining
                except ShellError as exc:
                    warning = f"Saved locally, but Team 2 synchronization failed: {exc.message}"
            else:
                warning = "Saved locally, but the content service is unavailable so the preference has not been synchronized for retrieval yet."
        self.storage.save_workspace(workspace)
        return {"workspace": self.serialize_workspace(workspace), "warning": warning}

    def record_feedback(self, workspace_id: str, payload: dict) -> dict:
        workspace = self._workspace_or_error(workspace_id)
        feedback_entry = {
            "feedback_id": make_id("feedback"),
            "created_at": utc_now_iso(),
            "target_type": payload.get("target_type") or "workspace",
            "target_id": payload.get("target_id"),
            "kind": payload.get("kind") or "note",
            "correction_note": payload.get("correction_note") or payload.get("feedback_note") or "",
            "material_id": payload.get("material_id"),
            "slide_id": payload.get("slide_id"),
        }
        workspace.setdefault("feedback_history", []).append(feedback_entry)
        annotation_warning = None
        annotation = build_feedback_annotation(payload)
        if annotation:
            if self.effective_mode == "integrated" and self.refresh_status(force=True)["services"]["content"]["available"]:
                try:
                    created = self._remote_json("content", "POST", f"/v1/workspaces/{workspace_id}/annotations", json_body=annotation)
                    created_annotation = self._extract_detail(created, "annotation")
                    workspace = self._merge_annotations(workspace, [created_annotation])
                    feedback_entry["remote_annotation_id"] = created_annotation.get("annotation_id")
                except ShellError as exc:
                    annotation_warning = f"The correction note was saved locally, but Team 2 synchronization failed: {exc.message}"
            else:
                annotation_warning = "The correction note was saved locally, but the content service is unavailable so it has not been synchronized for future grounding yet."
        self.storage.save_workspace(workspace)
        return {"workspace": self.serialize_workspace(workspace), "feedback": feedback_entry, "warning": annotation_warning}

    def resolve_citation(self, workspace_id: str, citation: dict) -> dict:
        if self.effective_mode == "integrated" and self.refresh_status(force=True)["services"]["content"]["available"]:
            try:
                resolved = self._remote_json("content", "POST", "/v1/citations/resolve", json_body=citation)
                return resolved
            except ShellError:
                pass
        enriched = enrich_citation(workspace_id, citation, fixture_mode=workspace_id == build_workspace_from_fixture()["workspace_id"])
        return {"citation": enriched, "resolved": False, "message": "Citation metadata is shown even though the preview could not be resolved through the content service."}

    def get_slide_preview_svg(self, workspace_id: str, material_id: str, slide_id: str) -> str:
        workspace = self._workspace_or_error(workspace_id)
        material = workspace.get("materials", {}).get(material_id)
        if not material:
            raise ShellError(f"Material {material_id} was not found.", status_code=404)
        slides = material.get("slides") or []
        slide = next((item for item in slides if item.get("slide_id") == slide_id), None)
        if slide is None:
            raise ShellError(f"Slide {slide_id} was not found.", status_code=404)
        return render_slide_preview_svg(workspace_id, material, slide)

    def get_slide_source_html(self, workspace_id: str, material_id: str, slide_id: str) -> str:
        svg_url = f"/local/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/preview"
        if workspace_id == build_workspace_from_fixture()["workspace_id"]:
            svg_url = f"/mock/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/preview"
        return f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>Source Viewer</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #f8fafc; }}
      .frame {{ background: white; border: 1px solid #cbd5e1; border-radius: 16px; padding: 16px; }}
      img {{ max-width: 100%; height: auto; border: 1px solid #e2e8f0; border-radius: 12px; }}
    </style>
  </head>
  <body>
    <div class=\"frame\">
      <h1>Source Viewer</h1>
      <img src=\"{svg_url}\" alt=\"Source preview\" />
    </div>
  </body>
</html>"""
