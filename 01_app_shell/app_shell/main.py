from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app_shell.config import AppConfig
from app_shell.errors import ShellError
from app_shell.services import ShellService
from app_shell.storage import LocalStorage


THIS_DIR = Path(__file__).resolve().parent
STATIC_DIR = THIS_DIR / "static"


def create_app(*, env_override: dict[str, str] | None = None) -> FastAPI:
    config = AppConfig.load(THIS_DIR.parent, env_override=env_override)
    storage = LocalStorage(config.local_data_dir)
    shell_service = ShellService(config, storage)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        shell_service.startup()
        try:
            yield
        finally:
            shell_service.shutdown()

    app = FastAPI(title="Study Helper MVP - App Shell", lifespan=lifespan)
    app.state.config = config
    app.state.storage = storage
    app.state.shell_service = shell_service

    @app.exception_handler(ShellError)
    def _handle_shell_error(_: Request, exc: ShellError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.as_payload())

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        return {"service_name": "app_shell", "ready": True, "status": "ok"}

    @app.get("/manifest")
    def manifest() -> dict[str, Any]:
        return {
            "service_name": "app_shell",
            "version": config.version,
            "ui_base_url": config.ui_base_url,
            "api_base_url": config.api_base_url,
            "capabilities": [
                "workspace_library",
                "material_upload",
                "study_plan_ui",
                "chat_ui",
                "practice_ui",
                "source_viewer",
                "history",
                "mock_mode",
                "integrated_launcher",
            ],
        }

    @app.get("/api/status")
    def api_status() -> dict[str, Any]:
        return shell_service.status_snapshot()

    @app.get("/api/workspaces")
    def api_list_workspaces() -> dict[str, Any]:
        return shell_service.list_workspaces()

    @app.post("/api/workspaces")
    async def api_create_workspace(request: Request) -> dict[str, Any]:
        payload = await request.json()
        workspace = shell_service.create_workspace(payload.get("display_name", "Untitled workspace"))
        return {"workspace": workspace}

    @app.post("/api/workspaces/{workspace_id}/duplicate")
    def api_duplicate_workspace(workspace_id: str) -> dict[str, Any]:
        return {"workspace": shell_service.duplicate_workspace(workspace_id)}

    @app.post("/api/workspaces/{workspace_id}/archive")
    def api_archive_workspace(workspace_id: str) -> dict[str, Any]:
        return {"workspace": shell_service.archive_workspace(workspace_id)}

    @app.delete("/api/workspaces/{workspace_id}")
    def api_delete_workspace(workspace_id: str) -> dict[str, Any]:
        shell_service.delete_workspace(workspace_id)
        return {"deleted": True}

    @app.get("/api/workspaces/{workspace_id}")
    def api_get_workspace(workspace_id: str, refresh: bool = Query(default=True)) -> dict[str, Any]:
        return {"workspace": shell_service.get_workspace(workspace_id, refresh=refresh)}

    @app.get("/api/workspaces/{workspace_id}/history")
    def api_history(workspace_id: str) -> dict[str, Any]:
        return shell_service.get_history(workspace_id)

    @app.post("/api/workspaces/{workspace_id}/history/{artifact_type}/{artifact_id}/activate")
    def api_activate_artifact(workspace_id: str, artifact_type: str, artifact_id: str) -> dict[str, Any]:
        return {"workspace": shell_service.activate_artifact(workspace_id, artifact_type, artifact_id)}

    @app.post("/api/workspaces/{workspace_id}/materials/import")
    async def api_import_material(
        workspace_id: str,
        title: str = Form(default=""),
        role: str = Form(default="notes"),
        kind: Optional[str] = Form(default=None),
        text: Optional[str] = Form(default=None),
        source_text: Optional[str] = Form(default=None),
        text_body: Optional[str] = Form(default=None),
        file: Optional[UploadFile] = File(default=None),
    ) -> dict[str, Any]:
        file_payload = None
        if file is not None:
            file_payload = {
                "filename": file.filename or "upload.bin",
                "content_type": file.content_type or "application/octet-stream",
                "content": await file.read(),
            }
        job = shell_service.import_material(
            workspace_id,
            {
                "title": title,
                "role": role,
                "kind": kind,
                "text": text or source_text or text_body,
            },
            file_payload=file_payload,
        )
        return {"job": job}

    @app.get("/api/jobs/{job_id}")
    def api_poll_job(job_id: str, workspace_id: str = Query(...)) -> dict[str, Any]:
        return {"job": shell_service.poll_job(workspace_id, job_id)}

    @app.post("/api/workspaces/{workspace_id}/study-plans/generate")
    async def api_generate_study_plan(workspace_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return {"job": shell_service.generate_study_plan(workspace_id, payload)}

    @app.post("/api/workspaces/{workspace_id}/study-plans/{study_plan_id}/revise")
    async def api_revise_study_plan(workspace_id: str, study_plan_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return {"job": shell_service.revise_study_plan(workspace_id, study_plan_id, payload)}

    @app.post("/api/workspaces/{workspace_id}/conversations")
    async def api_create_conversation(workspace_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return shell_service.create_conversation(workspace_id, payload)

    @app.post("/api/workspaces/{workspace_id}/conversations/{conversation_id}/messages")
    async def api_send_message(workspace_id: str, conversation_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return shell_service.send_conversation_message(workspace_id, conversation_id, payload)

    @app.post("/api/workspaces/{workspace_id}/conversations/{conversation_id}/clear")
    async def api_clear_conversation(workspace_id: str, conversation_id: str) -> dict[str, Any]:
        return shell_service.clear_conversation(workspace_id, conversation_id)

    @app.post("/api/workspaces/{workspace_id}/practice-sets/generate")
    async def api_generate_practice(workspace_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return {"job": shell_service.generate_practice_set(workspace_id, payload)}

    @app.post("/api/workspaces/{workspace_id}/practice-sets/{practice_set_id}/revise")
    async def api_revise_practice(workspace_id: str, practice_set_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return {"job": shell_service.revise_practice_set(workspace_id, practice_set_id, payload)}

    @app.delete("/api/workspaces/{workspace_id}/materials/{material_id}")
    async def api_delete_material(workspace_id: str, material_id: str) -> dict[str, Any]:
        return {"workspace": shell_service.delete_material(workspace_id, material_id)}

    @app.post("/api/workspaces/{workspace_id}/materials/{material_id}/preference")
    async def api_set_material_preference(workspace_id: str, material_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return shell_service.set_material_preference(workspace_id, material_id, payload.get("preference", "default"))

    @app.post("/api/workspaces/{workspace_id}/feedback")
    async def api_feedback(workspace_id: str, request: Request) -> dict[str, Any]:
        payload = await request.json()
        return shell_service.record_feedback(workspace_id, payload)

    @app.post("/api/citations/resolve")
    async def api_resolve_citation(request: Request) -> dict[str, Any]:
        payload = await request.json()
        workspace_id = payload.get("workspace_id")
        citation = payload.get("citation") or payload
        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required")
        return shell_service.resolve_citation(workspace_id, citation)

    @app.get("/mock/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/preview")
    def mock_preview(workspace_id: str, material_id: str, slide_id: str) -> Response:
        svg = shell_service.get_slide_preview_svg(workspace_id, material_id, slide_id)
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/mock/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/source", response_class=HTMLResponse)
    def mock_source(workspace_id: str, material_id: str, slide_id: str) -> HTMLResponse:
        return HTMLResponse(shell_service.get_slide_source_html(workspace_id, material_id, slide_id))

    @app.get("/local/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/preview")
    def local_preview(workspace_id: str, material_id: str, slide_id: str) -> Response:
        svg = shell_service.get_slide_preview_svg(workspace_id, material_id, slide_id)
        return Response(content=svg, media_type="image/svg+xml")

    @app.get("/local/workspaces/{workspace_id}/materials/{material_id}/slides/{slide_id}/source", response_class=HTMLResponse)
    def local_source(workspace_id: str, material_id: str, slide_id: str) -> HTMLResponse:
        return HTMLResponse(shell_service.get_slide_source_html(workspace_id, material_id, slide_id))

    return app
