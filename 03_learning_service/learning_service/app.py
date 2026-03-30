from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .service import LearningService, RequestValidationError


def _request_error_to_http(exc: RequestValidationError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    service_settings = settings or Settings.from_env()
    service = LearningService(service_settings)

    app = FastAPI(title="Learning Intelligence Service", version=service_settings.version)
    app.state.learning_service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return service.health_payload()

    @app.get("/manifest")
    def manifest() -> Dict[str, Any]:
        return service.manifest_payload()

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> Dict[str, Any]:
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    @app.post("/v1/study-plans")
    def create_study_plan(request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        try:
            job_id = service.submit_study_plan(request)
        except RequestValidationError as exc:
            raise _request_error_to_http(exc)
        return {"job_id": job_id}

    @app.get("/v1/study-plans")
    def list_study_plans(workspace_id: str = Query(...)) -> Dict[str, Any]:
        return service.list_study_plans(workspace_id)

    @app.get("/v1/study-plans/{study_plan_id}")
    def read_study_plan(study_plan_id: str) -> Dict[str, Any]:
        artifact = service.get_study_plan(study_plan_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="study_plan not found")
        return artifact

    @app.post("/v1/study-plans/{study_plan_id}/revise")
    def revise_study_plan(study_plan_id: str, request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        try:
            job_id = service.submit_study_plan_revision(study_plan_id, request)
        except KeyError:
            raise HTTPException(status_code=404, detail="study_plan not found")
        except RequestValidationError as exc:
            raise _request_error_to_http(exc)
        return {"job_id": job_id}

    @app.post("/v1/conversations")
    def create_conversation(request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        try:
            return service.create_conversation(request)
        except RequestValidationError as exc:
            raise _request_error_to_http(exc)

    @app.get("/v1/conversations")
    def list_conversations(workspace_id: str = Query(...)) -> Dict[str, Any]:
        return service.list_conversations(workspace_id)

    @app.get("/v1/conversations/{conversation_id}")
    def read_conversation(conversation_id: str) -> Dict[str, Any]:
        artifact = service.get_conversation(conversation_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return artifact

    @app.post("/v1/conversations/{conversation_id}/messages")
    def send_message(conversation_id: str, request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        try:
            job_id = service.submit_conversation_message(conversation_id, request)
        except KeyError:
            raise HTTPException(status_code=404, detail="conversation not found")
        except RequestValidationError as exc:
            raise _request_error_to_http(exc)
        return {"job_id": job_id}

    @app.post("/v1/conversations/{conversation_id}/clear")
    def clear_conversation(conversation_id: str) -> Dict[str, Any]:
        result = service.clear_conversation(conversation_id)
        if result is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return result

    @app.post("/v1/practice-sets")
    def create_practice_set(request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        try:
            job_id = service.submit_practice_set(request)
        except RequestValidationError as exc:
            raise _request_error_to_http(exc)
        return {"job_id": job_id}

    @app.get("/v1/practice-sets")
    def list_practice_sets(workspace_id: str = Query(...)) -> Dict[str, Any]:
        return service.list_practice_sets(workspace_id)

    @app.get("/v1/practice-sets/{practice_set_id}")
    def read_practice_set(practice_set_id: str) -> Dict[str, Any]:
        artifact = service.get_practice_set(practice_set_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="practice_set not found")
        return artifact

    @app.post("/v1/practice-sets/{practice_set_id}/revise")
    def revise_practice_set(practice_set_id: str, request: Dict[str, Any] = Body(...)) -> Dict[str, str]:
        try:
            job_id = service.submit_practice_set_revision(practice_set_id, request)
        except KeyError:
            raise HTTPException(status_code=404, detail="practice_set not found")
        except RequestValidationError as exc:
            raise _request_error_to_http(exc)
        return {"job_id": job_id}

    return app
