from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app_shell.main import create_app
from app_shell.errors import ShellError


@pytest.fixture()
def mock_client(tmp_path: Path) -> TestClient:
    env = {
        "APP_SHELL_MODE": "mock",
        "LOCAL_DATA_DIR": str(tmp_path / "local_data"),
        "AUTO_OPEN_BROWSER": "false",
        "APP_SHELL_TESTING": "true",
    }
    app = create_app(env_override=env)
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def degraded_client(tmp_path: Path) -> TestClient:
    env = {
        "APP_SHELL_MODE": "integrated",
        "LOCAL_DATA_DIR": str(tmp_path / "local_data"),
        "AUTO_OPEN_BROWSER": "false",
        "APP_SHELL_TESTING": "true",
        "CONTENT_SERVICE_URL": "http://127.0.0.1:39910",
        "LEARNING_SERVICE_URL": "http://127.0.0.1:39920",
    }
    app = create_app(env_override=env)
    with TestClient(app) as client:
        yield client


def poll_job(client: TestClient, workspace_id: str, job_id: str, *, timeout_seconds: float = 8.0) -> dict:
    deadline = time.time() + timeout_seconds
    last_job = None
    while time.time() < deadline:
        last_job = client.get(f"/api/jobs/{job_id}", params={"workspace_id": workspace_id}).json()["job"]
        if last_job["status"] in {"succeeded", "failed", "needs_user_input"}:
            return last_job
        time.sleep(0.25)
    raise AssertionError(f"Job {job_id} did not finish in time; last job payload={last_job}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class UvicornThread:
    def __init__(self, app: FastAPI, port: int):
        self.port = port
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def start(self) -> None:
        self.thread.start()
        deadline = time.time() + 8
        while time.time() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{self.port}/healthz", timeout=0.5) as response:
                    if response.status == 200:
                        return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError(f"Timed out starting test server on port {self.port}")

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)


@contextmanager
def content_stub_server():
    state: dict = {
        "materials": {},
        "annotations": {},
        "jobs": {},
        "received_imports": [],
        "annotation_posts": [],
        "annotation_deletes": [],
        "slide_requests": [],
    }
    app = FastAPI()

    def material_payload(workspace_id: str, title: str, role: str, kind: str) -> dict:
        material_id = f"mat_{len(state['received_imports']) + len(state['materials'].get(workspace_id, {})) + 1}"
        slide_id = f"slide_{material_id}_1"
        return {
            "material_id": material_id,
            "workspace_id": workspace_id,
            "title": title,
            "role": role,
            "kind": kind,
            "processing_status": "ready",
            "page_count": 1,
            "created_at": "2026-03-28T00:00:00Z",
            "quality_summary": {"overall": "high", "notes": "Stub-imported material"},
            "slides": [
                {
                    "slide_id": slide_id,
                    "slide_number": 1,
                    "title": title,
                    "snippet_text": f"Evidence from {title}",
                    "bullets": [f"Role: {role}", f"Kind: {kind}"],
                    "preview_url": f"/preview/{material_id}/{slide_id}",
                    "source_open_url": f"/source/{material_id}/{slide_id}",
                }
            ],
            "source_view_url": f"/source/{material_id}/{slide_id}",
        }

    @app.get("/healthz")
    def healthz() -> dict:
        return {"service_name": "content_stub", "ready": True, "status": "ok"}

    @app.post("/v1/materials/import")
    async def import_material(request: Request) -> dict:
        form = await request.form()
        keys = sorted(form.keys())
        fields = {key: form.get(key) for key in keys if key != "file"}
        upload = form.get("file")
        state["received_imports"].append({"keys": keys, "fields": fields, "has_file": upload is not None})
        source_kind = fields.get("source_kind")
        if source_kind == "file":
            if set(keys) != {"file", "role", "source_kind", "title", "workspace_id"}:
                return JSONResponse(status_code=400, content={"error": {"message": f"Unexpected file import keys: {keys}"}})
            if upload is None:
                return JSONResponse(status_code=400, content={"error": {"message": "Missing file payload"}})
            kind = Path(upload.filename).suffix.lstrip(".").lower() or "file"
            material = material_payload(fields["workspace_id"], fields["title"], fields["role"], kind)
        elif source_kind == "pasted_text":
            if set(keys) != {"role", "source_kind", "text_body", "title", "workspace_id"}:
                return JSONResponse(status_code=400, content={"error": {"message": f"Unexpected pasted-text keys: {keys}"}})
            if not fields.get("text_body"):
                return JSONResponse(status_code=400, content={"error": {"message": "Missing text_body"}})
            material = material_payload(fields["workspace_id"], fields["title"], fields["role"], "pasted_text")
        else:
            return JSONResponse(status_code=400, content={"error": {"message": f"Unexpected source_kind: {source_kind}"}})
        job_id = f"content_job_{len(state['jobs']) + 1}"
        state["jobs"][job_id] = {"polls": 0, "material": material}
        return {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = state["jobs"][job_id]
        job["polls"] += 1
        material = job["material"]
        if job["polls"] >= 1:
            state["materials"].setdefault(material["workspace_id"], {})[material["material_id"]] = material
            return {
                "job": {
                    "job_id": job_id,
                    "status": "succeeded",
                    "progress": 100,
                    "stage": "completed",
                    "message": "imported",
                    "result_type": "material",
                    "result_id": material["material_id"],
                    "user_action": None,
                    "error": None,
                }
            }
        return {"job": {"job_id": job_id, "status": "running", "progress": 50, "stage": "parsing", "message": "running", "result_type": None, "result_id": None, "user_action": None, "error": None}}

    @app.get("/v1/materials")
    def get_materials(workspace_id: str) -> dict:
        items = []
        for material in state["materials"].get(workspace_id, {}).values():
            item = dict(material)
            item.pop("slides", None)
            items.append(item)
        return {"materials": items}

    @app.get("/v1/materials/{material_id}")
    def get_material(material_id: str) -> dict:
        for materials in state["materials"].values():
            if material_id in materials:
                return {"material": materials[material_id]}
        return JSONResponse(status_code=404, content={"error": {"message": "material not found"}})

    @app.get("/v1/materials/{material_id}/slides")
    def get_material_slides(material_id: str) -> dict:
        state["slide_requests"].append(material_id)
        for materials in state["materials"].values():
            if material_id in materials:
                return {"slides": materials[material_id].get("slides", [])}
        return JSONResponse(status_code=404, content={"error": {"message": "material not found"}})

    @app.get("/preview/{material_id}/{slide_id}")
    def preview_slide(material_id: str, slide_id: str):
        return JSONResponse({"material_id": material_id, "slide_id": slide_id, "preview": True})

    @app.get("/source/{material_id}/{slide_id}")
    def source_slide(material_id: str, slide_id: str):
        return JSONResponse({"material_id": material_id, "slide_id": slide_id, "source": True})

    @app.get("/v1/workspaces/{workspace_id}/annotations")
    def get_annotations(workspace_id: str) -> dict:
        return {"annotations": list(state["annotations"].get(workspace_id, []))}

    @app.post("/v1/workspaces/{workspace_id}/annotations")
    async def post_annotations(workspace_id: str, request: Request) -> dict:
        payload = await request.json()
        state["annotation_posts"].append(payload)
        annotation = {
            "annotation_id": f"ann_{len(state['annotation_posts'])}",
            "created_at": "2026-03-28T00:00:00Z",
            **payload,
        }
        state["annotations"].setdefault(workspace_id, []).append(annotation)
        return {"annotation": annotation}

    @app.delete("/v1/workspaces/{workspace_id}/annotations/{annotation_id}")
    def delete_annotation(workspace_id: str, annotation_id: str) -> dict:
        state["annotation_deletes"].append(annotation_id)
        state["annotations"][workspace_id] = [annotation for annotation in state["annotations"].get(workspace_id, []) if annotation.get("annotation_id") != annotation_id]
        return {"deleted": True}

    @app.post("/v1/citations/resolve")
    async def resolve_citation(request: Request) -> dict:
        citation = await request.json()
        return {"citation": citation, "resolved": True}

    port = free_port()
    server = UvicornThread(app, port)
    server.start()
    try:
        yield port, state
    finally:
        server.stop()


@contextmanager
def learning_stub_server():
    state: dict = {
        "jobs": {},
        "study_plans": {},
        "conversations": {},
        "practice_sets": {},
        "received_study_plan_posts": [],
        "received_study_plan_revisions": [],
        "received_conversation_posts": [],
        "received_message_posts": [],
        "received_practice_posts": [],
        "received_practice_revisions": [],
        "conversation_sequence": 0,
    }
    app = FastAPI()

    def citation(material_id: str) -> dict:
        slide_id = f"slide_{material_id}_1"
        return {
            "citation_id": f"cit_{material_id}",
            "material_id": material_id,
            "material_title": f"Material {material_id}",
            "slide_id": slide_id,
            "slide_number": 1,
            "snippet_text": f"Evidence from {material_id}",
            "support_type": "explicit",
            "confidence": "high",
            "preview_url": f"/preview/{material_id}/{slide_id}",
            "source_open_url": f"/source/{material_id}/{slide_id}",
        }

    def make_plan(workspace_id: str, payload: dict, parent_study_plan_id: str | None = None, sequence: int = 1) -> dict:
        material_id = payload["material_ids"][0]
        return {
            "study_plan_id": f"plan_{workspace_id}_{sequence}",
            "parent_study_plan_id": parent_study_plan_id,
            "workspace_id": workspace_id,
            "created_at": f"2026-03-28T00:00:0{sequence}Z",
            "topic_text": payload.get("topic_text") or "Inferred topic",
            "time_budget_minutes": payload.get("time_budget_minutes", 60),
            "grounding_mode": payload.get("grounding_mode", "strict_lecture_only"),
            "prerequisites": [
                {"item_id": f"pre_{sequence}", "concept_name": "Prereq", "why_needed": "Needed", "support_status": "slide_grounded", "citations": [citation(material_id)]}
            ],
            "study_sequence": [
                {"step_id": f"step_{sequence}", "order_index": 1, "title": "Review", "objective": "Read the source", "recommended_time_minutes": 20, "tasks": ["Review"], "depends_on": [f"pre_{sequence}"], "support_status": "slide_grounded", "citations": [citation(material_id)]}
            ],
            "common_mistakes": [
                {"item_id": f"mistake_{sequence}", "pattern": "Skipping evidence", "why_it_happens": "Haste", "prevention_advice": "Use the citation", "support_status": "slide_grounded", "citations": [citation(material_id)]}
            ],
        }

    def make_practice(workspace_id: str, payload: dict, parent_practice_set_id: str | None = None, sequence: int = 1) -> dict:
        material_id = payload.get("material_ids", ["material_x"])[0]
        question_type = "short_answer" if payload.get("generation_mode") in {"short_answer", "template_mimic"} else "long_answer"
        return {
            "practice_set_id": f"practice_{workspace_id}_{sequence}",
            "parent_practice_set_id": parent_practice_set_id,
            "workspace_id": workspace_id,
            "created_at": f"2026-03-28T00:10:0{sequence}Z",
            "generation_mode": payload.get("generation_mode", "mixed"),
            "questions": [
                {
                    "question_id": f"question_{sequence}_1",
                    "question_type": question_type,
                    "stem": "Explain the grounded concept.",
                    "expected_answer": "Use the cited material.",
                    "rubric": [{"criterion": "Grounded", "description": "Uses evidence", "points": 2}],
                    "scoring_guide_text": "Look for evidence use.",
                    "citations": [citation(material_id)],
                    "covered_slides": [1],
                    "difficulty": payload.get("difficulty_profile", "mixed"),
                }
            ],
            "coverage_report": {"considered_slide_count": 1, "cited_slide_count": 1, "uncited_or_skipped_slides": [], "notes": "stub coverage"},
        }

    def make_conversation(workspace_id: str, title: str, material_ids: list[str]) -> dict:
        state["conversation_sequence"] += 1
        sequence = state["conversation_sequence"]
        conversation_id = f"conv_{workspace_id}_{sequence}"
        conversation = {
            "conversation_id": conversation_id,
            "workspace_id": workspace_id,
            "created_at": f"2026-03-28T00:20:0{sequence}Z",
            "title": title,
            "messages": [],
            "material_ids": list(material_ids),
        }
        state["conversations"].setdefault(workspace_id, {})[conversation_id] = conversation
        return conversation

    def reject_unexpected_keys(payload: dict, expected_keys: set[str]) -> JSONResponse | None:
        if set(payload.keys()) != expected_keys:
            return JSONResponse(status_code=400, content={"error": {"message": f"Unexpected keys: {sorted(payload.keys())}"}})
        return None

    @app.get("/healthz")
    def healthz() -> dict:
        return {"service_name": "learning_stub", "ready": True, "status": "ok"}

    @app.post("/v1/study-plans")
    async def create_study_plan(request: Request):
        payload = await request.json()
        rejection = reject_unexpected_keys(payload, {"workspace_id", "material_ids", "topic_text", "time_budget_minutes", "grounding_mode", "student_context", "include_annotations"})
        if rejection:
            return rejection
        if set(payload["student_context"].keys()) != {"prior_knowledge", "weak_areas", "goals"}:
            return JSONResponse(status_code=400, content={"error": {"message": "Unexpected student_context shape"}})
        state["received_study_plan_posts"].append(payload)
        workspace_id = payload["workspace_id"]
        sequence = len(state["study_plans"].get(workspace_id, {})) + 1
        plan = make_plan(workspace_id, payload, sequence=sequence)
        job_id = f"learning_job_plan_{workspace_id}_{sequence}"
        state["jobs"][job_id] = {"kind": "study_plan", "workspace_id": workspace_id, "artifact": plan, "polls": 0}
        return {"job_id": job_id}

    @app.get("/v1/study-plans")
    def list_study_plans(workspace_id: str) -> dict:
        return {"study_plans": list(reversed(list(state["study_plans"].get(workspace_id, {}).values())))}

    @app.get("/v1/study-plans/{study_plan_id}")
    def get_study_plan(study_plan_id: str) -> dict:
        for plans in state["study_plans"].values():
            if study_plan_id in plans:
                return {"study_plan": plans[study_plan_id]}
        return JSONResponse(status_code=404, content={"error": {"message": "plan not found"}})

    @app.post("/v1/study-plans/{study_plan_id}/revise")
    async def revise_study_plan(study_plan_id: str, request: Request):
        payload = await request.json()
        rejection = reject_unexpected_keys(payload, {"instruction_text", "target_section", "locked_item_ids", "grounding_mode", "include_annotations"})
        if rejection:
            return rejection
        state["received_study_plan_revisions"].append(payload)
        existing = None
        workspace_id = None
        for ws_id, plans in state["study_plans"].items():
            if study_plan_id in plans:
                existing = plans[study_plan_id]
                workspace_id = ws_id
                break
        if existing is None or workspace_id is None:
            return JSONResponse(status_code=404, content={"error": {"message": "parent plan not found"}})
        sequence = len(state["study_plans"].get(workspace_id, {})) + 1
        parent_payload = {
            "material_ids": [existing["prerequisites"][0]["citations"][0]["material_id"]],
            "topic_text": existing["topic_text"],
            "time_budget_minutes": existing["time_budget_minutes"],
            "grounding_mode": payload["grounding_mode"],
        }
        plan = make_plan(workspace_id, parent_payload, parent_study_plan_id=study_plan_id, sequence=sequence)
        job_id = f"learning_job_plan_revise_{workspace_id}_{sequence}"
        state["jobs"][job_id] = {"kind": "study_plan", "workspace_id": workspace_id, "artifact": plan, "polls": 0}
        return {"job_id": job_id}

    @app.post("/v1/conversations")
    async def create_conversation(request: Request):
        payload = await request.json()
        rejection = reject_unexpected_keys(payload, {"workspace_id", "material_ids", "grounding_mode", "title", "include_annotations"})
        if rejection:
            return rejection
        state["received_conversation_posts"].append(payload)
        conversation = make_conversation(payload["workspace_id"], payload["title"], payload["material_ids"])
        return {"conversation": conversation}

    @app.get("/v1/conversations")
    def list_conversations(workspace_id: str) -> dict:
        return {"conversations": list(reversed(list(state["conversations"].get(workspace_id, {}).values())))}

    @app.get("/v1/conversations/{conversation_id}")
    def get_conversation(conversation_id: str) -> dict:
        for conversations in state["conversations"].values():
            if conversation_id in conversations:
                return {"conversation": conversations[conversation_id]}
        return JSONResponse(status_code=404, content={"error": {"message": "conversation not found"}})

    @app.post("/v1/conversations/{conversation_id}/messages")
    async def post_message(conversation_id: str, request: Request):
        payload = await request.json()
        rejection = reject_unexpected_keys(payload, {"message_text", "response_style", "grounding_mode", "include_citations"})
        if rejection:
            return rejection
        if payload.get("response_style") not in {"standard", "concise", "step_by_step"}:
            return JSONResponse(status_code=422, content={"error": {"message": "Unsupported response_style"}})
        state["received_message_posts"].append(payload)
        workspace_id = None
        for ws_id, conversations in state["conversations"].items():
            if conversation_id in conversations:
                workspace_id = ws_id
                break
        if workspace_id is None:
            return JSONResponse(status_code=404, content={"error": {"message": "conversation not found"}})
        job_id = f"learning_job_message_{conversation_id}_{len(state['received_message_posts'])}"
        state["jobs"][job_id] = {"kind": "message", "workspace_id": workspace_id, "conversation_id": conversation_id, "payload": payload, "polls": 0, "finalized": False}
        return {"job_id": job_id}

    @app.post("/v1/conversations/{conversation_id}/clear")
    def clear_conversation(conversation_id: str):
        for conversations in state["conversations"].values():
            if conversation_id in conversations:
                conversations[conversation_id]["messages"] = []
                return {"conversation": conversations[conversation_id]}
        return JSONResponse(status_code=404, content={"error": {"message": "conversation not found"}})

    @app.post("/v1/practice-sets")
    async def create_practice(request: Request):
        payload = await request.json()
        expected = {"workspace_id", "material_ids", "generation_mode", "question_count", "coverage_mode", "difficulty_profile", "include_answer_key", "include_rubrics", "grounding_mode", "include_annotations"}
        if payload.get("generation_mode") == "template_mimic":
            expected.add("template_material_id")
        rejection = reject_unexpected_keys(payload, expected)
        if rejection:
            return rejection
        state["received_practice_posts"].append(payload)
        workspace_id = payload["workspace_id"]
        sequence = len(state["practice_sets"].get(workspace_id, {})) + 1
        practice_set = make_practice(workspace_id, payload, sequence=sequence)
        job_id = f"learning_job_practice_{workspace_id}_{sequence}"
        state["jobs"][job_id] = {"kind": "practice", "workspace_id": workspace_id, "artifact": practice_set, "polls": 0}
        return {"job_id": job_id}

    @app.get("/v1/practice-sets")
    def list_practice_sets(workspace_id: str) -> dict:
        return {"practice_sets": list(reversed(list(state["practice_sets"].get(workspace_id, {}).values())))}

    @app.get("/v1/practice-sets/{practice_set_id}")
    def get_practice_set(practice_set_id: str) -> dict:
        for practice_sets in state["practice_sets"].values():
            if practice_set_id in practice_sets:
                return {"practice_set": practice_sets[practice_set_id]}
        return JSONResponse(status_code=404, content={"error": {"message": "practice set not found"}})

    @app.post("/v1/practice-sets/{practice_set_id}/revise")
    async def revise_practice(practice_set_id: str, request: Request):
        payload = await request.json()
        rejection = reject_unexpected_keys(payload, {"instruction_text", "target_question_ids", "locked_question_ids", "maintain_coverage"})
        if rejection:
            return rejection
        state["received_practice_revisions"].append(payload)
        existing = None
        workspace_id = None
        for ws_id, practice_sets in state["practice_sets"].items():
            if practice_set_id in practice_sets:
                existing = practice_sets[practice_set_id]
                workspace_id = ws_id
                break
        if existing is None or workspace_id is None:
            return JSONResponse(status_code=404, content={"error": {"message": "parent practice set not found"}})
        sequence = len(state["practice_sets"].get(workspace_id, {})) + 1
        parent_payload = {
            "workspace_id": workspace_id,
            "material_ids": [existing["questions"][0]["citations"][0]["material_id"]],
            "generation_mode": existing["generation_mode"],
            "question_count": len(existing["questions"]),
            "difficulty_profile": existing["questions"][0]["difficulty"],
        }
        practice_set = make_practice(workspace_id, parent_payload, parent_practice_set_id=practice_set_id, sequence=sequence)
        job_id = f"learning_job_practice_revise_{workspace_id}_{sequence}"
        state["jobs"][job_id] = {"kind": "practice", "workspace_id": workspace_id, "artifact": practice_set, "polls": 0}
        return {"job_id": job_id}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = state["jobs"][job_id]
        job["polls"] += 1
        if job["kind"] == "study_plan":
            plan = job["artifact"]
            state["study_plans"].setdefault(job["workspace_id"], {})[plan["study_plan_id"]] = plan
            return {"job": {"job_id": job_id, "status": "succeeded", "progress": 100, "stage": "completed", "message": "study plan ready", "result_type": "study_plan", "result_id": plan["study_plan_id"], "user_action": None, "error": None}}
        if job["kind"] == "practice":
            practice_set = job["artifact"]
            state["practice_sets"].setdefault(job["workspace_id"], {})[practice_set["practice_set_id"]] = practice_set
            return {"job": {"job_id": job_id, "status": "succeeded", "progress": 100, "stage": "completed", "message": "practice ready", "result_type": "practice_set", "result_id": practice_set["practice_set_id"], "user_action": None, "error": None}}
        if job["kind"] == "message":
            conversation = state["conversations"][job["workspace_id"]][job["conversation_id"]]
            if not job["finalized"]:
                message_text = job["payload"]["message_text"]
                conversation["messages"].append({"message_id": f"user_{len(conversation['messages'])+1}", "role": "user", "created_at": "2026-03-28T00:20:10Z", "text": message_text})
                material_id = conversation.get("material_ids", [f"material_for_{job['conversation_id']}"])[0]
                conversation["messages"].append({
                    "message_id": f"assistant_{len(conversation['messages'])+1}",
                    "role": "assistant",
                    "created_at": "2026-03-28T00:20:20Z",
                    "answer_source": {
                        "path": "llm",
                        "provider": "gemini",
                        "model": "gemini-3-flash-preview",
                        "reasoning_enabled": True,
                        "reasoning_mode": "dynamic",
                        "matched_evidence_count": 1,
                        "evidence_match": "strong_match",
                        "rate_limited_models": [],
                    },
                    "reply_sections": [{"heading": "Grounded reply", "text": f"Answer for: {message_text}", "support_status": "slide_grounded", "citations": [citation(material_id)]}],
                    "clarifying_question": {"prompt": None, "reason": None},
                })
                job["finalized"] = True
            return {"job": {"job_id": job_id, "status": "succeeded", "progress": 100, "stage": "completed", "message": "message ready", "result_type": "assistant_message", "result_id": conversation['messages'][-1]['message_id'], "user_action": None, "error": None}}
        return {"job": {"job_id": job_id, "status": "failed", "progress": 100, "stage": "failed", "message": "unknown job", "result_type": None, "result_id": None, "user_action": None, "error": {"message": "unknown job", "retryable": False}}}

    port = free_port()
    server = UvicornThread(app, port)
    server.start()
    try:
        yield port, state
    finally:
        server.stop()


@contextmanager
def integrated_client(tmp_path: Path):
    with content_stub_server() as (content_port, content_state), learning_stub_server() as (learning_port, learning_state):
        env = {
            "APP_SHELL_MODE": "integrated",
            "LOCAL_DATA_DIR": str(tmp_path / "local_data"),
            "AUTO_OPEN_BROWSER": "false",
            "APP_SHELL_TESTING": "true",
            "CONTENT_SERVICE_URL": f"http://127.0.0.1:{content_port}",
            "LEARNING_SERVICE_URL": f"http://127.0.0.1:{learning_port}",
        }
        app = create_app(env_override=env)
        with TestClient(app) as client:
            yield client, content_state, learning_state, env


def test_health_manifest_and_root_ui(mock_client: TestClient) -> None:
    health = mock_client.get("/healthz")
    manifest = mock_client.get("/manifest")
    root = mock_client.get("/")

    assert health.status_code == 200
    assert health.json() == {"service_name": "app_shell", "ready": True, "status": "ok"}
    assert manifest.status_code == 200
    assert manifest.json()["ui_base_url"].startswith("http://127.0.0.1:")
    assert "workspace_library" in manifest.json()["capabilities"]
    assert root.status_code == 200
    assert "Study Helper MVP" in root.text
    assert "source-viewer" in root.text
    assert '<option value="standard" selected>standard</option>' in root.text
    assert 'value="direct_answer"' not in root.text

def test_mock_fixture_workspace_is_bootstrapped_with_source_viewer_assets(mock_client: TestClient) -> None:
    workspaces = mock_client.get("/api/workspaces").json()["workspaces"]
    assert workspaces
    workspace_id = workspaces[0]["workspace_id"]
    payload = mock_client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]

    assert payload["display_name"] == "Backpropagation Midterm Prep"
    assert len(payload["materials"]) >= 3
    assert payload["active_study_plan"]["prerequisites"]
    assert payload["active_conversation"]["messages"]
    assert payload["active_practice_set"]["questions"]

    preview = mock_client.get("/mock/workspaces/workspace_fixture_demo/materials/material_fixture_slides/slides/slide_2/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/svg+xml")
    assert "Chain Rule Intuition" in preview.text


def test_create_workspace_import_material_and_persist_after_restart(tmp_path: Path) -> None:
    env = {
        "APP_SHELL_MODE": "mock",
        "LOCAL_DATA_DIR": str(tmp_path / "local_data"),
        "AUTO_OPEN_BROWSER": "false",
        "APP_SHELL_TESTING": "true",
    }
    app = create_app(env_override=env)
    with TestClient(app) as client:
        created = client.post("/api/workspaces", json={"display_name": "Persistence demo"}).json()["workspace"]
        workspace_id = created["workspace_id"]
        job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Quick notes", "role": "notes", "kind": "pasted_text", "text": "Backpropagation relies on the chain rule and cached activations."},
        ).json()["job"]
        finished = poll_job(client, workspace_id, job["job_id"])
        assert finished["status"] == "succeeded"
        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assert any(material["title"] == "Quick notes" for material in workspace["materials"])

    restarted = create_app(env_override=env)
    with TestClient(restarted) as restarted_client:
        workspace = restarted_client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assert workspace["display_name"] == "Persistence demo"
        assert any(material["title"] == "Quick notes" for material in workspace["materials"])


def test_integrated_material_import_normalization_and_hydration(tmp_path: Path) -> None:
    with integrated_client(tmp_path) as (client, content_state, _learning_state, env):
        workspace = client.post("/api/workspaces", json={"display_name": "Integrated import workspace"}).json()["workspace"]
        workspace_id = workspace["workspace_id"]

        pasted_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Pasted notes", "role": "notes", "kind": "pasted_text", "text": "These notes should normalize to text_body."},
        ).json()["job"]
        assert poll_job(client, workspace_id, pasted_job["job_id"])["status"] == "succeeded"

        file_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            files={"file": ("lecture.pdf", io.BytesIO(b"%PDF-1.4 stub"), "application/pdf")},
            data={"title": "Lecture slides", "role": "slides"},
        ).json()["job"]
        assert poll_job(client, workspace_id, file_job["job_id"])["status"] == "succeeded"
        assert len(content_state["received_imports"]) == 2

        duplicate_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            files={"file": ("lecture.pdf", io.BytesIO(b"%PDF-1.4 stub"), "application/pdf")},
            data={"title": "Lecture slides duplicate", "role": "slides"},
        ).json()["job"]
        finished_duplicate = poll_job(client, workspace_id, duplicate_job["job_id"])
        assert finished_duplicate["status"] == "succeeded"
        assert "Duplicate file detected" in finished_duplicate["message"]
        # Deduplication happens in app-shell, so the content service should not receive a second copy.
        assert len(content_state["received_imports"]) == 2

        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assert len(workspace["materials"]) == 2

        pasted_request = content_state["received_imports"][0]
        assert pasted_request["fields"]["source_kind"] == "pasted_text"
        assert pasted_request["fields"]["text_body"] == "These notes should normalize to text_body."
        assert "kind" not in pasted_request["keys"]
        assert "text" not in pasted_request["keys"]

        file_request = content_state["received_imports"][1]
        assert file_request["fields"]["source_kind"] == "file"
        assert file_request["has_file"] is True
        assert "text_body" not in file_request["keys"]

        material_ids = {material["material_id"] for material in workspace["materials"]}
        assert material_ids.issubset(set(content_state["slide_requests"]))
        for material in workspace["materials"]:
            assert material["slides"]
            first_slide = material["slides"][0]
            assert first_slide["slide_id"]
            assert first_slide["slide_number"] == 1
            assert first_slide["preview_url"].startswith(env["CONTENT_SERVICE_URL"])
            assert first_slide["source_open_url"].startswith(env["CONTENT_SERVICE_URL"])
            with urlopen(first_slide["preview_url"], timeout=1.0) as response:
                preview_body = json.loads(response.read().decode("utf-8"))
            assert preview_body["preview"] is True
            with urlopen(first_slide["source_open_url"], timeout=1.0) as response:
                source_body = json.loads(response.read().decode("utf-8"))
            assert source_body["source"] is True

        restarted = create_app(env_override=env)
        with TestClient(restarted) as restarted_client:
            restarted_workspace = restarted_client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
            assert len(restarted_workspace["materials"]) == 2
            for material in restarted_workspace["materials"]:
                assert material["slides"]
                assert material["slides"][0]["preview_url"].startswith(env["CONTENT_SERVICE_URL"])
                assert material["slides"][0]["source_open_url"].startswith(env["CONTENT_SERVICE_URL"])

def test_integrated_study_plan_generation_and_revision_normalization(tmp_path: Path) -> None:
    with integrated_client(tmp_path) as (client, content_state, learning_state, env):
        workspace = client.post("/api/workspaces", json={"display_name": "Planning workspace"}).json()["workspace"]
        workspace_id = workspace["workspace_id"]

        slides_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            files={"file": ("deck.pdf", io.BytesIO(b"pdf"), "application/pdf")},
            data={"title": "Ready slides", "role": "slides"},
        ).json()["job"]
        notes_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Ready notes", "role": "notes", "kind": "pasted_text", "text": "Useful notes"},
        ).json()["job"]
        template_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Practice template", "role": "practice_template", "kind": "pasted_text", "text": "Template wording"},
        ).json()["job"]
        poll_job(client, workspace_id, slides_job["job_id"])
        poll_job(client, workspace_id, notes_job["job_id"])
        poll_job(client, workspace_id, template_job["job_id"])

        content_state["materials"].setdefault(workspace_id, {})["mat_processing"] = {
            "material_id": "mat_processing",
            "workspace_id": workspace_id,
            "title": "Still processing",
            "role": "slides",
            "kind": "pdf",
            "processing_status": "running",
            "page_count": 0,
            "created_at": "2026-03-28T00:00:00Z",
            "quality_summary": {"overall": "medium", "notes": "still parsing"},
            "slides": [],
            "source_view_url": "",
        }

        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        lecture_ready_ids = [material["material_id"] for material in workspace["materials"] if material["processing_status"] == "ready" and material["role"] in {"slides", "notes"}]
        template_id = next(material["material_id"] for material in workspace["materials"] if material["role"] == "practice_template")
        excluded_material_id = lecture_ready_ids[0]
        included_material_id = lecture_ready_ids[1]
        client.post(
            f"/api/workspaces/{workspace_id}/materials/{excluded_material_id}/preference",
            json={"preference": "exclude"},
        )

        job = client.post(
            f"/api/workspaces/{workspace_id}/study-plans/generate",
            json={
                "topic_text": "Backpropagation",
                "time_budget_minutes": 75,
                "grounding_mode": "strict_lecture_only",
                "student_context": {"known": "derivatives", "weak_areas": "sign errors", "goals": "midterm prep"},
            },
        ).json()["job"]
        finished = poll_job(client, workspace_id, job["job_id"])
        assert finished["status"] == "succeeded"

        outbound = learning_state["received_study_plan_posts"][-1]
        assert set(outbound.keys()) == {"workspace_id", "material_ids", "topic_text", "time_budget_minutes", "grounding_mode", "student_context", "include_annotations"}
        assert outbound["student_context"]["prior_knowledge"] == "derivatives"
        assert outbound["material_ids"] == [included_material_id]
        assert template_id not in outbound["material_ids"]
        assert "mat_processing" not in outbound["material_ids"]
        assert outbound["include_annotations"] is True

        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        first_plan_id = workspace["active_study_plan"]["study_plan_id"]
        locked_prereq_ids = [item["item_id"] for item in workspace["active_study_plan"]["prerequisites"]]

        revise_job = client.post(
            f"/api/workspaces/{workspace_id}/study-plans/{first_plan_id}/revise",
            json={
                "target_section": "study_sequence",
                "feedback_note": "Emphasize checking the update sign before memorizing formulas.",
                "locked_sections": ["prerequisites"],
            },
        ).json()["job"]
        assert poll_job(client, workspace_id, revise_job["job_id"])["status"] == "succeeded"
        revision = learning_state["received_study_plan_revisions"][-1]
        assert set(revision.keys()) == {"instruction_text", "target_section", "locked_item_ids", "grounding_mode", "include_annotations"}
        assert revision["instruction_text"] == "Emphasize checking the update sign before memorizing formulas."
        assert revision["target_section"] == "study_sequence"
        assert revision["locked_item_ids"] == locked_prereq_ids
        assert revision["include_annotations"] is True

        refreshed_workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        revised_plan_id = refreshed_workspace["active_study_plan"]["study_plan_id"]
        assert revised_plan_id != first_plan_id
        study_history = [entry for entry in refreshed_workspace["history"] if entry["artifact_type"] == "study_plan"]
        assert len([entry for entry in study_history if entry["active"]]) == 1
        assert [entry for entry in study_history if entry["active"]][0]["artifact_id"] == revised_plan_id

def test_integrated_conversation_job_flow_clear_and_reuse_materials(tmp_path: Path) -> None:
    with integrated_client(tmp_path) as (client, _content_state, learning_state, env):
        workspace = client.post("/api/workspaces", json={"display_name": "Chat workspace"}).json()["workspace"]
        workspace_id = workspace["workspace_id"]

        notes_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Chat notes", "role": "notes", "kind": "pasted_text", "text": "The minus sign matters."},
        ).json()["job"]
        template_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Practice template", "role": "practice_template", "kind": "pasted_text", "text": "Template style"},
        ).json()["job"]
        poll_job(client, workspace_id, notes_job["job_id"])
        poll_job(client, workspace_id, template_job["job_id"])

        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        template_id = next(material["material_id"] for material in workspace["materials"] if material["role"] == "practice_template")

        created = client.post(
            f"/api/workspaces/{workspace_id}/conversations",
            json={"title": "Question thread", "grounding_mode": "strict_lecture_only"},
        ).json()["conversation"]
        conversation_id = created["conversation_id"]
        create_payload = learning_state["received_conversation_posts"][-1]
        assert set(create_payload.keys()) == {"workspace_id", "material_ids", "grounding_mode", "title", "include_annotations"}
        assert create_payload["material_ids"]
        assert template_id not in create_payload["material_ids"]
        assert create_payload["include_annotations"] is True

        send_response = client.post(
            f"/api/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
            json={"text": "Why does the update subtract the gradient?", "grounding_mode": "strict_lecture_only"},
        ).json()
        assert "job" in send_response
        assert "assistant_message" not in send_response
        outbound = learning_state["received_message_posts"][-1]
        assert set(outbound.keys()) == {"message_text", "response_style", "grounding_mode", "include_citations"}
        assert outbound["message_text"] == "Why does the update subtract the gradient?"
        assert outbound["response_style"] == "standard"
        assert outbound["include_citations"] is True

        workspace_before = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        pending_users = [message for message in workspace_before["active_conversation"]["messages"] if message["role"] == "user" and message.get("pending")]
        assert pending_users
        finished = poll_job(client, workspace_id, send_response["job"]["job_id"])
        assert finished["status"] == "succeeded"

        workspace_after = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assistant_messages = [message for message in workspace_after["active_conversation"]["messages"] if message["role"] == "assistant"]
        assert assistant_messages
        assert assistant_messages[-1]["reply_sections"][0]["citations"]
        assert assistant_messages[-1]["answer_source"]["path"] == "llm"
        assert assistant_messages[-1]["answer_source"]["model"] == "gemini-3-flash-preview"

        cleared = client.post(f"/api/workspaces/{workspace_id}/conversations/{conversation_id}/clear", json={}).json()["conversation"]
        assert cleared["messages"] == []

        legacy_send = client.post(
            f"/api/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
            json={"text": "Ask again after clearing.", "grounding_mode": "strict_lecture_only", "response_style": "direct_answer"},
        ).json()
        assert poll_job(client, workspace_id, legacy_send["job"]["job_id"])["status"] == "succeeded"
        legacy_outbound = learning_state["received_message_posts"][-1]
        assert legacy_outbound["response_style"] == "standard"

        second_conversation = client.post(
            f"/api/workspaces/{workspace_id}/conversations",
            json={"title": "Second thread", "grounding_mode": "strict_lecture_only"},
        ).json()["conversation"]
        second_id = second_conversation["conversation_id"]
        refreshed = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assert refreshed["active_conversation"]["conversation_id"] == second_id

        activated = client.post(
            f"/api/workspaces/{workspace_id}/history/conversation/{conversation_id}/activate"
        ).json()["workspace"]
        assert activated["active_conversation"]["conversation_id"] == conversation_id
        reloaded = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assert reloaded["active_conversation"]["conversation_id"] == conversation_id
        conversation_history = [entry for entry in reloaded["history"] if entry["artifact_type"] == "conversation"]
        assert len([entry for entry in conversation_history if entry["active"]]) == 1
        assert [entry for entry in conversation_history if entry["active"]][0]["artifact_id"] == conversation_id


def test_failed_message_submit_rolls_back_pending_user_message(mock_client: TestClient, monkeypatch) -> None:
    workspace = mock_client.post("/api/workspaces", json={"display_name": "Rollback workspace"}).json()["workspace"]
    workspace_id = workspace["workspace_id"]
    import_job = mock_client.post(
        f"/api/workspaces/{workspace_id}/materials/import",
        data={"title": "Rollback notes", "role": "notes", "kind": "pasted_text", "text": "Grounded notes"},
    ).json()["job"]
    assert poll_job(mock_client, workspace_id, import_job["job_id"])["status"] == "succeeded"
    conversation = mock_client.post(
        f"/api/workspaces/{workspace_id}/conversations",
        json={"title": "Rollback chat", "grounding_mode": "strict_lecture_only"},
    ).json()["conversation"]
    conversation_id = conversation["conversation_id"]

    service = mock_client.app.state.shell_service
    service.effective_mode = "integrated"

    def fake_status(*, force: bool = False):
        return {
            "effective_mode": "integrated",
            "services": {
                "content": {"available": True, "mode": "integrated", "base_url": "http://127.0.0.1:39910"},
                "learning": {"available": True, "mode": "integrated", "base_url": "http://127.0.0.1:39920"},
            },
        }

    def fake_remote(service_name, method, path, **kwargs):
        if service_name == "learning" and method == "POST" and path.endswith(f"/v1/conversations/{conversation_id}/messages"):
            raise ShellError("simulated learning send failure", status_code=503)
        return {}

    monkeypatch.setattr(service, "refresh_status", fake_status)
    monkeypatch.setattr(service, "_remote_json", fake_remote)

    response = mock_client.post(
        f"/api/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
        json={"text": "this should fail", "grounding_mode": "strict_lecture_only"},
    )
    assert response.status_code == 503

    refreshed = mock_client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
    active = refreshed["active_conversation"]
    assert all(message.get("text") != "this should fail" for message in active.get("messages", []))

def test_integrated_practice_generation_and_revision_normalization(tmp_path: Path) -> None:
    with integrated_client(tmp_path) as (client, _content_state, learning_state, env):
        workspace = client.post("/api/workspaces", json={"display_name": "Practice workspace"}).json()["workspace"]
        workspace_id = workspace["workspace_id"]

        template_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Template notes", "role": "practice_template", "kind": "pasted_text", "text": "Template style"},
        ).json()["job"]
        slides_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            files={"file": ("lecture.pdf", io.BytesIO(b"pdf"), "application/pdf")},
            data={"title": "Lecture", "role": "slides"},
        ).json()["job"]
        notes_job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Lecture notes", "role": "notes", "kind": "pasted_text", "text": "Lecture notes"},
        ).json()["job"]
        poll_job(client, workspace_id, template_job["job_id"])
        poll_job(client, workspace_id, slides_job["job_id"])
        poll_job(client, workspace_id, notes_job["job_id"])

        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        template_id = next(material["material_id"] for material in workspace["materials"] if material["role"] == "practice_template")
        lecture_ids = {material["material_id"] for material in workspace["materials"] if material["role"] in {"slides", "notes"}}

        short_job = client.post(
            f"/api/workspaces/{workspace_id}/practice-sets/generate",
            json={
                "generation_mode": "short_answer",
                "question_count": 2,
                "difficulty": "mixed",
                "answer_key": True,
                "rubric": True,
                "grounding_mode": "strict_lecture_only",
            },
        ).json()["job"]
        assert poll_job(client, workspace_id, short_job["job_id"])["status"] == "succeeded"
        short_outbound = learning_state["received_practice_posts"][-1]
        assert set(short_outbound.keys()) == {"workspace_id", "material_ids", "generation_mode", "question_count", "coverage_mode", "difficulty_profile", "include_answer_key", "include_rubrics", "grounding_mode", "include_annotations"}
        assert short_outbound["generation_mode"] == "short_answer"
        assert set(short_outbound["material_ids"]) == lecture_ids
        assert template_id not in short_outbound["material_ids"]

        mimic_job = client.post(
            f"/api/workspaces/{workspace_id}/practice-sets/generate",
            json={
                "generation_mode": "template_mimic",
                "question_count": 3,
                "difficulty": "mixed",
                "answer_key": True,
                "rubric": True,
                "template_material_id": template_id,
                "grounding_mode": "strict_lecture_only",
            },
        ).json()["job"]
        assert poll_job(client, workspace_id, mimic_job["job_id"])["status"] == "succeeded"
        outbound = learning_state["received_practice_posts"][-1]
        assert set(outbound.keys()) == {"workspace_id", "material_ids", "generation_mode", "question_count", "coverage_mode", "difficulty_profile", "include_answer_key", "include_rubrics", "grounding_mode", "include_annotations", "template_material_id"}
        assert outbound["difficulty_profile"] == "mixed"
        assert outbound["include_answer_key"] is True
        assert outbound["include_rubrics"] is True
        assert outbound["template_material_id"] == template_id
        assert set(outbound["material_ids"]) == lecture_ids
        assert template_id not in outbound["material_ids"]

        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        practice = workspace["active_practice_set"]
        selected_question_id = practice["questions"][0]["question_id"]
        prior_practice_id = practice["practice_set_id"]
        revise_job = client.post(
            f"/api/workspaces/{workspace_id}/practice-sets/{practice['practice_set_id']}/revise",
            json={
                "selected_question_ids": [selected_question_id],
                "locked_question_ids": [selected_question_id],
            },
        ).json()["job"]
        assert poll_job(client, workspace_id, revise_job["job_id"])["status"] == "succeeded"
        revision = learning_state["received_practice_revisions"][-1]
        assert set(revision.keys()) == {"instruction_text", "target_question_ids", "locked_question_ids", "maintain_coverage"}
        assert revision["target_question_ids"] == [selected_question_id]
        assert revision["locked_question_ids"] == [selected_question_id]
        assert revision["instruction_text"] == "regenerate the selected questions while preserving locked questions"
        assert revision["maintain_coverage"] is True

        refreshed_workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        revised_practice_id = refreshed_workspace["active_practice_set"]["practice_set_id"]
        assert revised_practice_id != prior_practice_id
        practice_history = [entry for entry in refreshed_workspace["history"] if entry["artifact_type"] == "practice_set"]
        assert len([entry for entry in practice_history if entry["active"]]) == 1
        assert [entry for entry in practice_history if entry["active"]][0]["artifact_id"] == revised_practice_id

def test_annotation_hydration_preference_sync_and_feedback_annotation(tmp_path: Path) -> None:
    with integrated_client(tmp_path) as (client, content_state, _learning_state, env):
        workspace = client.post("/api/workspaces", json={"display_name": "Annotation workspace"}).json()["workspace"]
        workspace_id = workspace["workspace_id"]
        job = client.post(
            f"/api/workspaces/{workspace_id}/materials/import",
            data={"title": "Annotated notes", "role": "notes", "kind": "pasted_text", "text": "Notes for annotation tests."},
        ).json()["job"]
        poll_job(client, workspace_id, job["job_id"])
        workspace = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        material_id = workspace["materials"][0]["material_id"]

        content_state["annotations"].setdefault(workspace_id, []).append(
            {
                "annotation_id": "preloaded_ann",
                "created_at": "2026-03-28T00:00:00Z",
                "annotation_type": "user_correction",
                "scope": "workspace",
                "text": "Preloaded correction",
            }
        )
        first = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        second = client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
        assert len([annotation for annotation in second["annotations"] if annotation["annotation_id"] == "preloaded_ann"]) == 1
        assert len(second["annotations"]) >= len(first["annotations"])

        client.post(f"/api/workspaces/{workspace_id}/materials/{material_id}/preference", json={"preference": "focus"})
        client.post(f"/api/workspaces/{workspace_id}/materials/{material_id}/preference", json={"preference": "focus"})
        focus_posts = [payload for payload in content_state["annotation_posts"] if payload.get("annotation_type") == "focus" and payload.get("material_id") == material_id]
        assert len(focus_posts) == 1

        client.post(f"/api/workspaces/{workspace_id}/materials/{material_id}/preference", json={"preference": "default"})
        assert content_state["annotation_deletes"]

        client.post(f"/api/workspaces/{workspace_id}/materials/{material_id}/preference", json={"preference": "exclude"})
        exclude_posts = [payload for payload in content_state["annotation_posts"] if payload.get("annotation_type") == "exclude_from_grounding" and payload.get("material_id") == material_id]
        assert len(exclude_posts) == 1

        feedback_response = client.post(
            f"/api/workspaces/{workspace_id}/feedback",
            json={
                "target_type": "study_plan_item",
                "target_id": "step_1",
                "correction_note": "The plan should mention sign checks.",
                "kind": "wrong",
            },
        ).json()
        assert feedback_response["feedback"]["correction_note"] == "The plan should mention sign checks."
        last_annotation = content_state["annotation_posts"][-1]
        assert last_annotation["annotation_type"] == "user_correction"
        assert last_annotation["scope"] == "workspace"

        client.post(
            f"/api/workspaces/{workspace_id}/feedback",
            json={
                "target_type": "slide",
                "target_id": "slide_1",
                "material_id": material_id,
                "slide_id": workspace["materials"][0]["slides"][0]["slide_id"],
                "correction_note": "This specific slide needs more emphasis.",
            },
        )
        last_annotation = content_state["annotation_posts"][-1]
        assert last_annotation["scope"] == "slide"
        assert last_annotation["material_id"] == material_id


def test_degraded_integrated_mode_opens_workspace_and_never_fakes_success(degraded_client: TestClient) -> None:
    status = degraded_client.get("/api/status").json()
    assert status["effective_mode"] == "integrated"
    assert status["services"]["content"]["available"] is False
    assert status["services"]["learning"]["available"] is False

    workspace = degraded_client.post("/api/workspaces", json={"display_name": "Degraded mode workspace"}).json()["workspace"]
    workspace_id = workspace["workspace_id"]
    loaded = degraded_client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
    assert loaded["display_name"] == "Degraded mode workspace"
    assert loaded["status"]["services"]["content"]["available"] is False
    assert loaded["status"]["services"]["learning"]["available"] is False

    failed_import_job = degraded_client.post(
        f"/api/workspaces/{workspace_id}/materials/import",
        data={"title": "Will fail", "role": "slides", "kind": "pasted_text", "text": "content"},
    ).json()["job"]
    assert failed_import_job["status"] == "failed"

    conversation_response = degraded_client.post(
        f"/api/workspaces/{workspace_id}/conversations",
        json={"title": "Should not exist"},
    )
    assert conversation_response.status_code >= 400
    reloaded = degraded_client.get(f"/api/workspaces/{workspace_id}").json()["workspace"]
    assert reloaded["known_conversation_ids"] == []


def test_run_local_starts_server_with_one_command(tmp_path: Path) -> None:
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "APP_SHELL_MODE": "mock",
            "LOCAL_DATA_DIR": str(tmp_path / "local_data"),
            "AUTO_OPEN_BROWSER": "false",
            "APP_SHELL_TESTING": "true",
            "APP_SHELL_PORT": str(port),
        }
    )
    process = subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "run_local.py")],
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1.0) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    assert body["service_name"] == "app_shell"
                    break
            except Exception:
                time.sleep(0.25)
        else:
            output = process.stdout.read() if process.stdout else ""
            raise AssertionError(f"run_local.py did not start the server in time. Output so far:\n{output}")

        with urlopen(f"http://127.0.0.1:{port}/") as response:
            html_text = response.read().decode("utf-8")
        assert "Study Helper MVP" in html_text
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
