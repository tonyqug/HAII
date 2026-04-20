from __future__ import annotations

import mimetypes
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .db import init_db
from .importers import material_base_dir, process_material_import, source_file_path
from .rendering import html_page, relative_path, render_text_image
from .repository import ANNOTATION_TYPES, ROLE_VALUES, SCOPE_VALUES, Repository
from .retrieval import (
    RetrievalEngine,
    annotation_preview_url,
    annotation_source_open_url,
    material_source_view_url,
    public_annotation,
    public_material,
    public_material_detail,
    public_slide,
    public_slide_detail,
)
from .utils import extract_title_guess, new_id


class MaterialPatchRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)


class RetrievalSearchRequest(BaseModel):
    workspace_id: str
    material_ids: list[str] = Field(default_factory=list)
    query_text: str
    top_k: int = Field(default=6, ge=1, le=100)
    retrieval_mode: str = Field(default="precision")
    include_annotations: bool = True
    min_extraction_quality: str = Field(default="low")


class RetrievalBundleRequest(BaseModel):
    workspace_id: str
    material_ids: list[str] = Field(default_factory=list)
    query_text: Optional[str] = None
    bundle_mode: str = Field(default="precision")
    token_budget: int = Field(default=0, ge=0)
    max_items: int = Field(default=0, ge=0, le=500)
    include_annotations: bool = True


class EvidenceBundleRequest(BaseModel):
    workspace_id: str
    material_ids: list[str] = Field(default_factory=list)
    query_text: Optional[str] = None
    bundle_mode: str
    include_annotations: bool
    token_budget: int = Field(default=0, ge=0)
    max_items: int = Field(default=0, ge=0, le=500)


class CitationResolveRequest(BaseModel):
    citation_ids: list[str]


class AnnotationCreateRequest(BaseModel):
    annotation_type: str
    scope: str
    material_id: Optional[str] = None
    slide_id: Optional[str] = None
    text: str = Field(min_length=1)


VALID_RETRIEVAL_MODES = {"precision", "broad", "coverage"}
VALID_BUNDLE_MODES = {"precision", "coverage", "full_material"}
VALID_QUALITY_VALUES = {"low", "medium", "high"}
ANNOTATION_TYPE_ALIASES = {
    "user_correction": "user_correction",
    "user-correction": "user_correction",
    "correction": "user_correction",
    "study_note": "study_note",
    "study-note": "study_note",
    "note": "study_note",
    "focus": "focus",
    "focus_boost": "focus",
    "focus-boost": "focus",
    "exclude_from_grounding": "exclude_from_grounding",
    "exclude-from-grounding": "exclude_from_grounding",
    "exclude": "exclude_from_grounding",
}
SCOPE_ALIASES = {
    "workspace": "workspace",
    "material": "material",
    "slide": "slide",
}
LEGACY_SCOPE_ERROR = "Legacy annotation mapping supports target_type values workspace, material, or slide only."
LEGACY_SLIDE_MAPPING_ERROR = (
    "Legacy slide-target annotations must provide a valid slide target_id or explicit material_id and slide_id. "
    "Supported Team 2 scopes are workspace, material, and slide."
)
LEGACY_ANNOTATION_TYPE_ERROR = (
    "Unsupported annotation kind. Supported Team 2 annotation types are user_correction, study_note, focus, and exclude_from_grounding."
)


def _clean_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_token(value: Any) -> Optional[str]:
    text = _clean_string(value)
    if text is None:
        return None
    return text.lower().replace(" ", "_")


async def _read_json_object(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return data


def create_app(settings: Settings) -> FastAPI:
    repo = Repository(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings)
        app.state.executor = ThreadPoolExecutor(max_workers=settings.import_workers)
        try:
            yield
        finally:
            executor: Optional[ThreadPoolExecutor] = getattr(app.state, "executor", None)
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(title="Content Grounding Service", version=settings.version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def retrieval_engine() -> RetrievalEngine:
        return RetrievalEngine(settings, repo)

    def storage_available() -> bool:
        try:
            settings.local_data_dir.mkdir(parents=True, exist_ok=True)
            settings.storage_dir.mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            return False

    def material_file_system_path(material: dict[str, Any]) -> Optional[Path]:
        rel = material.get("source_relpath")
        if not rel:
            return None
        path = settings.local_data_dir / rel
        return path if path.exists() else None

    def converted_pdf_path(material: dict[str, Any]) -> Optional[Path]:
        rel = material.get("converted_pdf_relpath")
        if not rel:
            return None
        path = settings.local_data_dir / rel
        return path if path.exists() else None

    def job_response(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "job_type": job["job_type"],
            "status": job["status"],
            "progress": int(job["progress"] or 0),
            "stage": job["stage"],
            "message": job["message"],
            "result_type": job.get("result_type"),
            "result_id": job.get("result_id"),
            "error": {
                "code": job.get("error_code"),
                "message": job.get("error_message"),
                "retryable": bool(job["error_retryable"]) if job.get("error_retryable") is not None else None,
            },
        }

    def validate_retrieval_mode(mode: str, *, bundle: bool = False) -> None:
        valid = VALID_BUNDLE_MODES if bundle else VALID_RETRIEVAL_MODES
        if mode not in valid:
            raise HTTPException(status_code=400, detail=f"Unsupported mode '{mode}'. Expected one of {sorted(valid)}.")

    def validate_quality(value: str) -> None:
        if value not in VALID_QUALITY_VALUES:
            raise HTTPException(status_code=400, detail=f"Unsupported extraction quality '{value}'.")

    def validate_annotation_payload(payload: AnnotationCreateRequest, workspace_id: str) -> None:
        if payload.annotation_type not in ANNOTATION_TYPES:
            raise HTTPException(status_code=400, detail="Invalid annotation_type.")
        if payload.scope not in SCOPE_VALUES:
            raise HTTPException(status_code=400, detail="Invalid scope.")
        if payload.scope == "workspace":
            if payload.material_id or payload.slide_id:
                raise HTTPException(status_code=400, detail="Workspace-scoped annotations cannot include material_id or slide_id.")
        elif payload.scope == "material":
            if not payload.material_id or payload.slide_id:
                raise HTTPException(status_code=400, detail="Material-scoped annotations require material_id and must omit slide_id.")
            material = repo.get_material(payload.material_id)
            if not material or material["workspace_id"] != workspace_id:
                raise HTTPException(status_code=404, detail="Referenced material was not found in this workspace.")
        elif payload.scope == "slide":
            if not payload.material_id or not payload.slide_id:
                raise HTTPException(status_code=400, detail="Slide-scoped annotations require both material_id and slide_id.")
            material = repo.get_material(payload.material_id)
            if not material or material["workspace_id"] != workspace_id:
                raise HTTPException(status_code=404, detail="Referenced material was not found in this workspace.")
            slide = repo.get_slide(payload.material_id, payload.slide_id)
            if not slide:
                raise HTTPException(status_code=404, detail="Referenced slide was not found.")

    def annotation_dir(workspace_id: str, annotation_id: str) -> Path:
        return settings.storage_dir / "workspaces" / workspace_id / "annotations" / annotation_id

    def build_annotation_preview(annotation_type: str, scope: str, text: str, workspace_id: str, annotation_id: str) -> str:
        adir = annotation_dir(workspace_id, annotation_id)
        adir.mkdir(parents=True, exist_ok=True)
        preview_path = adir / "preview.png"
        title = f"{annotation_type.replace('_', ' ').title()} ({scope})"
        render_text_image(text, preview_path, title=title, footer=f"Annotation {annotation_id}")
        return relative_path(settings.local_data_dir, preview_path)

    def schedule_import(material_id: str, job_id: str) -> None:
        executor: ThreadPoolExecutor = app.state.executor
        executor.submit(process_material_import, settings, repo, material_id, job_id)

    def normalize_import_payload(fields: Mapping[str, Any], upload: Any) -> dict[str, Any]:
        workspace_id = _clean_string(fields.get("workspace_id"))
        role = _normalized_token(fields.get("role"))
        title = _clean_string(fields.get("title"))
        source_kind = _normalized_token(fields.get("source_kind"))
        legacy_kind = _normalized_token(fields.get("kind"))
        text_body = fields.get("text_body")
        if text_body is None:
            text_body = fields.get("text")
        if text_body is None:
            text_body = fields.get("source_text")
        text_body = None if text_body is None else str(text_body)

        has_upload = upload is not None and getattr(upload, "filename", None) not in {None, ""}
        if source_kind is None:
            if has_upload:
                source_kind = "file"
            elif legacy_kind == "pasted_text":
                source_kind = "pasted_text"
            elif _clean_string(text_body):
                source_kind = "pasted_text"

        return {
            "workspace_id": workspace_id,
            "role": role,
            "source_kind": source_kind,
            "title": title,
            "text_body": text_body,
            "upload": upload if has_upload else None,
        }

    def normalize_annotation_payload(data: Mapping[str, Any], workspace_id: str) -> AnnotationCreateRequest:
        annotation_type = _normalized_token(data.get("annotation_type"))
        if annotation_type is None:
            legacy_kind = _normalized_token(data.get("kind"))
            if legacy_kind is not None:
                annotation_type = ANNOTATION_TYPE_ALIASES.get(legacy_kind)
                if annotation_type is None:
                    raise HTTPException(status_code=400, detail=LEGACY_ANNOTATION_TYPE_ERROR)

        scope = _normalized_token(data.get("scope"))
        if scope is None:
            target_type = _normalized_token(data.get("target_type"))
            if target_type is not None:
                scope = SCOPE_ALIASES.get(target_type)
                if scope is None:
                    raise HTTPException(status_code=400, detail=LEGACY_SCOPE_ERROR)

        material_id = _clean_string(data.get("material_id"))
        slide_id = _clean_string(data.get("slide_id"))
        target_id = _clean_string(data.get("target_id"))

        if scope == "material" and material_id is None and target_id is not None:
            material_id = target_id
        elif scope == "slide" and slide_id is None and target_id is not None:
            slide_id = target_id

        if scope == "slide" and slide_id and material_id is None:
            resolved = repo.resolve_slide_scope(workspace_id, slide_id)
            if resolved is None:
                raise HTTPException(status_code=400, detail=LEGACY_SLIDE_MAPPING_ERROR)
            material_id, slide_id = resolved

        text = data.get("text")
        try:
            return AnnotationCreateRequest(
                annotation_type=annotation_type,
                scope=scope,
                material_id=material_id,
                slide_id=slide_id,
                text="" if text is None else str(text),
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=exc.errors()) from exc

    def bundle_response(
        *,
        workspace_id: str,
        material_ids: list[str],
        query_text: Optional[str],
        bundle_mode: str,
        token_budget: int,
        max_items: int,
        include_annotations: bool,
    ) -> dict[str, Any]:
        validate_retrieval_mode(bundle_mode, bundle=True)
        engine = retrieval_engine()
        return engine.bundle(
            workspace_id=workspace_id,
            material_ids=material_ids,
            query_text=query_text,
            bundle_mode=bundle_mode,
            token_budget=token_budget,
            max_items=max_items,
            include_annotations=include_annotations,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "service_name": settings.service_name,
            "ready": True,
            "status": "ok",
            "details": {"storage_available": storage_available()},
        }

    @app.get("/manifest")
    async def manifest() -> dict[str, Any]:
        return {
            "service_name": settings.service_name,
            "version": settings.version,
            "api_base_url": settings.api_base_url,
            "capabilities": [
                "material_import",
                "slide_preview",
                "retrieval_search",
                "retrieval_bundle",
                "evidence_bundle",
                "citations",
                "annotations",
            ],
        }

    @app.get("/")
    async def root() -> HTMLResponse:
        body = f"""
        <h1>Content Grounding Service</h1>
        <p class=\"meta\">Local API base: <code>{settings.api_base_url}</code></p>
        <ul>
          <li><a href=\"/docs\">Interactive API docs</a></li>
          <li><a href=\"/openapi.json\">OpenAPI JSON</a></li>
          <li><a href=\"/healthz\">Health check</a></li>
          <li><a href=\"/manifest\">Manifest</a></li>
        </ul>
        <p>Use the documented HTTP contract for import, retrieval, evidence bundles, citation resolution, slide preview, and annotations.</p>
        """
        return HTMLResponse(html_page("Content Grounding Service", body))

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> dict[str, Any]:
        job = repo.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job_response(job)

    @app.post("/v1/materials/import")
    async def import_material(request: Request) -> JSONResponse:
        ctype = request.headers.get("content-type", "")
        raw_fields: Mapping[str, Any]
        upload = None

        if "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
            form = await request.form()
            raw_fields = dict(form)
            upload = form.get("file")
        else:
            raw_fields = await _read_json_object(request)

        normalized = normalize_import_payload(raw_fields, upload)
        workspace_id = normalized["workspace_id"]
        role = normalized["role"]
        source_kind = normalized["source_kind"]
        title = normalized["title"]
        text_body = normalized["text_body"]
        upload = normalized["upload"]

        if not workspace_id:
            raise HTTPException(status_code=400, detail="workspace_id is required.")
        if role not in ROLE_VALUES:
            raise HTTPException(status_code=400, detail="role must be one of slides, notes, practice_template.")
        if source_kind not in {"file", "pasted_text"}:
            raise HTTPException(status_code=400, detail="source_kind must be file or pasted_text.")
        if source_kind == "file" and upload is None:
            raise HTTPException(status_code=400, detail="file is required when source_kind=file.")
        if source_kind == "pasted_text" and not (text_body or "").strip():
            raise HTTPException(status_code=400, detail="text_body is required when source_kind=pasted_text.")

        material_id = new_id("mat")
        job_id = new_id("job")
        kind = "pasted_text"
        original_filename = None

        if source_kind == "file":
            filename = getattr(upload, "filename", None) or (title or "upload.bin")
            original_filename = filename
            suffix = Path(filename).suffix.lower()
            if suffix == ".pdf":
                kind = "pdf"
            elif suffix == ".pptx":
                kind = "pptx"
            elif suffix in {".txt", ".md", ".markdown", ".text", ".rst"}:
                kind = "text"
            else:
                kind = "unsupported"
            display_title = title or Path(filename).stem or filename
        else:
            inferred_title = extract_title_guess(text_body or "")
            display_title = title or inferred_title or "Pasted Notes"
            kind = "pasted_text"

        repo.reserve_material(
            material_id=material_id,
            workspace_id=workspace_id,
            title=display_title,
            original_filename=original_filename,
            role=role,
            kind=kind,
            source_kind=source_kind,
        )
        repo.reserve_job(
            job_id=job_id,
            job_type="material_import",
            stage="queued",
            message="Material import queued.",
            result_type="material",
            result_id=material_id,
        )

        base_dir = material_base_dir(settings, workspace_id, material_id)
        base_dir.mkdir(parents=True, exist_ok=True)

        if source_kind == "file":
            filename = getattr(upload, "filename", None) or "upload.bin"
            destination = source_file_path(settings, workspace_id, material_id, filename)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as f:
                while True:
                    chunk = await upload.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            repo.update_material(material_id, source_relpath=relative_path(settings.local_data_dir, destination))
        else:
            destination = base_dir / "source" / "pasted_text.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text_body or "", encoding="utf-8")
            repo.update_material(material_id, source_relpath=relative_path(settings.local_data_dir, destination))

        schedule_import(material_id, job_id)
        return JSONResponse({"material_id": material_id, "job_id": job_id, "processing_status": "queued"})

    @app.get("/v1/materials")
    async def list_materials(workspace_id: str) -> dict[str, Any]:
        materials = [public_material(material, settings) for material in repo.list_materials(workspace_id)]
        return {"materials": materials}

    @app.get("/v1/materials/{material_id}")
    async def read_material(material_id: str) -> dict[str, Any]:
        material = repo.get_material(material_id)
        if not material or material.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        return public_material_detail(material, settings)

    @app.patch("/v1/materials/{material_id}")
    async def patch_material(material_id: str, payload: MaterialPatchRequest) -> dict[str, Any]:
        material = repo.get_material(material_id)
        if not material or material.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        repo.update_material(material_id, title=payload.title.strip())
        updated = repo.get_material(material_id)
        assert updated is not None
        return public_material_detail(updated, settings)

    @app.delete("/v1/materials/{material_id}")
    async def delete_material(material_id: str) -> dict[str, Any]:
        existing = repo.get_material(material_id)
        if existing is None or existing.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        material = repo.delete_material(material_id)
        assert material is not None
        material_dir = material_base_dir(settings, material["workspace_id"], material_id)
        if material_dir.exists():
            shutil.rmtree(material_dir, ignore_errors=True)
        return {"deleted": True, "material_id": material_id}

    @app.get("/v1/materials/{material_id}/slides")
    async def list_slides(material_id: str) -> dict[str, Any]:
        material = repo.get_material(material_id)
        if not material or material.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        slides = [public_slide(material_id, slide, settings) for slide in repo.list_slides(material_id)]
        return {"material_id": material_id, "slides": slides}

    @app.get("/v1/materials/{material_id}/slides/{slide_id}")
    async def read_slide(material_id: str, slide_id: str) -> dict[str, Any]:
        slide = repo.get_slide(material_id, slide_id)
        if not slide:
            raise HTTPException(status_code=404, detail="Slide not found.")
        return public_slide_detail(material_id, slide, settings)

    @app.get("/v1/materials/{material_id}/slides/{slide_id}/preview")
    async def slide_preview(material_id: str, slide_id: str) -> FileResponse:
        slide = repo.get_slide(material_id, slide_id)
        if not slide:
            raise HTTPException(status_code=404, detail="Slide not found.")
        rel = slide.get("preview_relpath")
        if not rel:
            raise HTTPException(status_code=404, detail="Preview not available for this slide.")
        path = settings.local_data_dir / rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="Preview file is missing.")
        return FileResponse(path)

    @app.get("/v1/materials/{material_id}/slides/{slide_id}/source")
    async def slide_source(material_id: str, slide_id: str) -> HTMLResponse:
        material = repo.get_material(material_id)
        slide = repo.get_slide(material_id, slide_id)
        if not material or not slide:
            raise HTTPException(status_code=404, detail="Slide not found.")
        preview = f'<p><img src="{slide_preview_url(material_id=material_id, slide_id=slide_id)}" alt="Slide preview" /></p>' if slide.get("preview_relpath") else "<p><em>No preview available.</em></p>"
        source_file_link = f'<p><a href="{settings.api_base_url}/v1/materials/{material_id}/file">Open original source file</a></p>'
        body = f"""
        <h1>{material['title']} — Slide {slide['slide_number']}</h1>
        <div class=\"meta\">Extraction quality: {slide['extraction_quality']}</div>
        {preview}
        {source_file_link}
        <h2>Extracted text</h2>
        <pre>{(slide.get('extracted_text') or '').replace('<', '&lt;').replace('>', '&gt;')}</pre>
        <p><a href=\"{material_source_view_url(settings, material_id)}\">Back to material view</a></p>
        """
        return HTMLResponse(html_page(f"{material['title']} Slide {slide['slide_number']}", body))

    @app.get("/v1/materials/{material_id}/source")
    async def material_source(material_id: str) -> HTMLResponse:
        material = repo.get_material(material_id)
        if not material or material.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        slides = repo.list_slides(material_id)
        cards = []
        for slide in slides:
            preview_html = (
                f'<img src="{settings.api_base_url}/v1/materials/{material_id}/slides/{slide["slide_id"]}/preview" alt="Preview" />'
                if slide.get("preview_relpath")
                else "<em>No preview available.</em>"
            )
            cards.append(
                f"""
                <div class=\"card\">
                  <h3>Slide {slide['slide_number']}</h3>
                  <div class=\"meta\">{slide['extraction_quality']} quality</div>
                  <a href=\"{settings.api_base_url}/v1/materials/{material_id}/slides/{slide['slide_id']}/source\">{preview_html}</a>
                </div>
                """
            )
        original_links = []
        original_links.append(f'<a href="{settings.api_base_url}/v1/materials/{material_id}/file">Open original file</a>')
        if material.get("converted_pdf_relpath"):
            original_links.append(f'<a href="{settings.api_base_url}/v1/materials/{material_id}/converted-pdf">Open rendered PDF</a>')
        body = f"""
        <h1>{material['title']}</h1>
        <div class=\"meta\">Role: {material['role']} · Kind: {material['kind']} · Status: {material['processing_status']}</div>
        <p>{' · '.join(original_links)}</p>
        <div class=\"thumb-grid\">{''.join(cards) or '<p>No slide metadata is available yet.</p>'}</div>
        """
        return HTMLResponse(html_page(material["title"], body))

    @app.get("/v1/materials/{material_id}/file")
    async def material_file(material_id: str) -> FileResponse:
        material = repo.get_material(material_id)
        if not material or material.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        path = material_file_system_path(material)
        if not path:
            raise HTTPException(status_code=404, detail="Source file not found.")
        media_type, _ = mimetypes.guess_type(path.name)
        return FileResponse(path, media_type=media_type, filename=material.get("original_filename") or path.name)

    @app.get("/v1/materials/{material_id}/converted-pdf")
    async def material_converted_pdf(material_id: str) -> FileResponse:
        material = repo.get_material(material_id)
        if not material or material.get("is_hidden"):
            raise HTTPException(status_code=404, detail="Material not found.")
        path = converted_pdf_path(material)
        if not path:
            raise HTTPException(status_code=404, detail="Converted PDF not available for this material.")
        return FileResponse(path, media_type="application/pdf", filename=f"{material['title']}.pdf")

    @app.post("/v1/retrieval/search")
    async def retrieval_search(payload: RetrievalSearchRequest) -> dict[str, Any]:
        validate_retrieval_mode(payload.retrieval_mode)
        validate_quality(payload.min_extraction_quality)
        engine = retrieval_engine()
        return engine.search(
            workspace_id=payload.workspace_id,
            material_ids=payload.material_ids,
            query_text=payload.query_text,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            include_annotations=payload.include_annotations,
            min_extraction_quality=payload.min_extraction_quality,
        )

    @app.post("/v1/retrieval/bundle")
    async def retrieval_bundle(payload: RetrievalBundleRequest) -> dict[str, Any]:
        return bundle_response(
            workspace_id=payload.workspace_id,
            material_ids=payload.material_ids,
            query_text=payload.query_text,
            bundle_mode=payload.bundle_mode,
            token_budget=payload.token_budget,
            max_items=payload.max_items,
            include_annotations=payload.include_annotations,
        )

    @app.post("/v1/evidence-bundles")
    @app.post("/v1/evidence-bundle")
    async def evidence_bundles(payload: EvidenceBundleRequest) -> dict[str, Any]:
        return bundle_response(
            workspace_id=payload.workspace_id,
            material_ids=payload.material_ids,
            query_text=payload.query_text,
            bundle_mode=payload.bundle_mode,
            token_budget=payload.token_budget,
            max_items=payload.max_items,
            include_annotations=payload.include_annotations,
        )

    @app.post("/v1/citations/resolve")
    async def citations_resolve(payload: CitationResolveRequest) -> dict[str, Any]:
        engine = retrieval_engine()
        return {"citations": engine.resolve_citations(payload.citation_ids)}

    @app.get("/v1/workspaces/{workspace_id}/annotations")
    async def list_annotations(workspace_id: str) -> dict[str, Any]:
        annotations = [public_annotation(annotation) for annotation in repo.list_annotations(workspace_id)]
        return {"workspace_id": workspace_id, "annotations": annotations}

    @app.post("/v1/workspaces/{workspace_id}/annotations")
    async def create_annotation(workspace_id: str, request: Request) -> dict[str, Any]:
        raw_payload = await _read_json_object(request)
        payload = normalize_annotation_payload(raw_payload, workspace_id)
        validate_annotation_payload(payload, workspace_id)
        annotation_id = new_id("ann")
        preview_relpath = build_annotation_preview(payload.annotation_type, payload.scope, payload.text, workspace_id, annotation_id)
        virtual_slide_number = repo.next_annotation_slide_number(workspace_id)
        repo.create_annotation(
            annotation_id=annotation_id,
            workspace_id=workspace_id,
            annotation_type=payload.annotation_type,
            scope=payload.scope,
            material_id=payload.material_id,
            slide_id=payload.slide_id,
            text=payload.text,
            virtual_slide_number=virtual_slide_number,
            preview_relpath=preview_relpath,
        )
        created = repo.get_annotation(workspace_id, annotation_id)
        assert created is not None
        return {
            "annotation_id": created["annotation_id"],
            "workspace_id": workspace_id,
            "annotation_type": created["annotation_type"],
            "scope": created["scope"],
            "material_id": created.get("material_id"),
            "slide_id": created.get("slide_id"),
            "text": created["text"],
            "created_at": created["created_at"],
        }

    @app.delete("/v1/workspaces/{workspace_id}/annotations/{annotation_id}")
    async def delete_annotation(workspace_id: str, annotation_id: str) -> dict[str, Any]:
        annotation = repo.delete_annotation(workspace_id, annotation_id)
        if annotation is None:
            raise HTTPException(status_code=404, detail="Annotation not found.")
        adir = annotation_dir(workspace_id, annotation_id)
        if adir.exists():
            shutil.rmtree(adir, ignore_errors=True)
        return {"deleted": True, "annotation_id": annotation_id}

    @app.get("/v1/workspaces/{workspace_id}/annotations/{annotation_id}/preview")
    async def annotation_preview(workspace_id: str, annotation_id: str) -> FileResponse:
        annotation = repo.get_annotation(workspace_id, annotation_id)
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found.")
        rel = annotation.get("preview_relpath")
        if not rel:
            raise HTTPException(status_code=404, detail="Annotation preview not available.")
        path = settings.local_data_dir / rel
        if not path.exists():
            raise HTTPException(status_code=404, detail="Annotation preview file is missing.")
        return FileResponse(path)

    @app.get("/v1/workspaces/{workspace_id}/annotations/{annotation_id}/source")
    async def annotation_source(workspace_id: str, annotation_id: str) -> HTMLResponse:
        annotation = repo.get_annotation(workspace_id, annotation_id)
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found.")
        scope_html = f"<div class=\"meta\">Type: {annotation['annotation_type']} · Scope: {annotation['scope']}</div>"
        preview_html = f'<p><img src="{annotation_preview_url(settings, workspace_id, annotation_id)}" alt="Annotation preview" /></p>'
        source_context = ""
        if annotation.get("material_id") and annotation.get("slide_id"):
            source_context = (
                f'<p><a href="{settings.api_base_url}/v1/materials/{annotation["material_id"]}/slides/{annotation["slide_id"]}/source">'
                "Open linked source slide</a></p>"
            )
        elif annotation.get("material_id"):
            source_context = f'<p><a href="{settings.api_base_url}/v1/materials/{annotation["material_id"]}/source">Open linked material</a></p>'
        body = f"""
        <h1>Annotation {annotation_id}</h1>
        {scope_html}
        {preview_html}
        {source_context}
        <h2>Stored text</h2>
        <pre>{annotation['text'].replace('<', '&lt;').replace('>', '&gt;')}</pre>
        """
        return HTMLResponse(html_page(f"Annotation {annotation_id}", body))

    def slide_preview_url(material_id: str, slide_id: str) -> str:
        return f"{settings.api_base_url}/v1/materials/{material_id}/slides/{slide_id}/preview"

    return app
