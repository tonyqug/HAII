from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi.testclient import TestClient

from content_service.app import create_app
from content_service.config import Settings, load_settings


SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
PDF_SAMPLE = SAMPLE_DIR / "sample_slides.pdf"
PPTX_SAMPLE = SAMPLE_DIR / "sample_deck.pptx"
NOTES_SAMPLE = SAMPLE_DIR / "sample_notes.md"


def make_client(local_data_dir: Path, port: int = 38410) -> TestClient:
    settings = Settings(port=port, local_data_dir=local_data_dir, import_workers=2)
    app = create_app(settings)
    return TestClient(app)


def wait_for_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        last = response.json()
        if last["status"] in {"succeeded", "failed", "needs_user_input"}:
            return last
        time.sleep(0.25)
    raise AssertionError(f"Job {job_id} did not finish in time. Last state: {last}")


def url_path(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path + (f"?{parsed.query}" if parsed.query else "")


def import_file(client: TestClient, workspace_id: str, role: str, path: Path, title: str | None = None) -> tuple[str, str]:
    with path.open("rb") as fh:
        response = client.post(
            "/v1/materials/import",
            data={
                "workspace_id": workspace_id,
                "role": role,
                "source_kind": "file",
                **({"title": title} if title else {}),
            },
            files={"file": (path.name, fh, "application/octet-stream")},
        )
    response.raise_for_status()
    payload = response.json()
    return payload["material_id"], payload["job_id"]


def import_text(client: TestClient, workspace_id: str, role: str, text: str, title: str | None = None) -> tuple[str, str]:
    response = client.post(
        "/v1/materials/import",
        json={
            "workspace_id": workspace_id,
            "role": role,
            "source_kind": "pasted_text",
            "title": title,
            "text_body": text,
        },
    )
    response.raise_for_status()
    payload = response.json()
    return payload["material_id"], payload["job_id"]


def test_health_and_manifest(tmp_path: Path) -> None:
    with make_client(tmp_path / "local_data") as client:
        health = client.get("/healthz")
        assert health.status_code == 200
        assert health.json()["service_name"] == "content_service"

        manifest = client.get("/manifest")
        assert manifest.status_code == 200
        payload = manifest.json()
        assert payload["api_base_url"] == "http://127.0.0.1:38410"
        assert "retrieval_search" in payload["capabilities"]
        assert "retrieval_bundle" in payload["capabilities"]
        assert "evidence_bundle" in payload["capabilities"]


def test_full_import_retrieval_citations_annotations_evidence_bundles_and_restart(tmp_path: Path) -> None:
    local_data = tmp_path / "local_data"
    workspace_id = "ws_demo"

    with make_client(local_data) as client:
        pdf_material_id, pdf_job_id = import_file(client, workspace_id, "slides", PDF_SAMPLE)
        pptx_material_id, pptx_job_id = import_file(client, workspace_id, "slides", PPTX_SAMPLE)
        notes_material_id, notes_job_id = import_text(
            client,
            workspace_id,
            "notes",
            NOTES_SAMPLE.read_text(encoding="utf-8") + "\nATP is produced during cellular respiration.",
            title="Class notes",
        )

        for job_id in (pdf_job_id, pptx_job_id, notes_job_id):
            job = wait_for_job(client, job_id)
            assert job["status"] == "succeeded", job

        materials = client.get(f"/v1/materials?workspace_id={workspace_id}")
        materials.raise_for_status()
        material_items = materials.json()["materials"]
        assert len(material_items) == 3
        assert {item["processing_status"] for item in material_items} == {"ready"}

        pdf_detail = client.get(f"/v1/materials/{pdf_material_id}")
        pdf_detail.raise_for_status()
        pdf_payload = pdf_detail.json()
        assert pdf_payload["page_count"] >= 2
        assert pdf_payload["ready_for_retrieval"] is True
        assert pdf_payload["previews_available"] is True

        pdf_slides = client.get(f"/v1/materials/{pdf_material_id}/slides")
        pdf_slides.raise_for_status()
        slides = pdf_slides.json()["slides"]
        assert len(slides) >= 2
        first_slide = slides[0]

        preview_resp = client.get(url_path(first_slide["preview_url"]))
        assert preview_resp.status_code == 200
        assert preview_resp.headers["content-type"].startswith("image/")
        source_resp = client.get(url_path(first_slide["source_open_url"]))
        assert source_resp.status_code == 200
        assert "Extracted text" in source_resp.text

        slide_detail = client.get(f"/v1/materials/{pdf_material_id}/slides/{first_slide['slide_id']}")
        slide_detail.raise_for_status()
        assert slide_detail.json()["extracted_text"]

        study_note = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "annotation_type": "study_note",
                "scope": "workspace",
                "text": "ATP is the cell's main energy currency.",
            },
        )
        study_note.raise_for_status()
        study_note_id = study_note.json()["annotation_id"]

        focus_note = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "annotation_type": "focus",
                "scope": "material",
                "material_id": notes_material_id,
                "text": "Prioritize ATP and respiration concepts.",
            },
        )
        focus_note.raise_for_status()

        exclude_note = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "annotation_type": "exclude_from_grounding",
                "scope": "slide",
                "material_id": pdf_material_id,
                "slide_id": first_slide["slide_id"],
                "text": "Exclude this introductory slide from retrieval.",
            },
        )
        exclude_note.raise_for_status()

        annotations = client.get(f"/v1/workspaces/{workspace_id}/annotations")
        annotations.raise_for_status()
        assert len(annotations.json()["annotations"]) == 3

        search = client.post(
            "/v1/retrieval/search",
            json={
                "workspace_id": workspace_id,
                "material_ids": [pdf_material_id, pptx_material_id, notes_material_id],
                "query_text": "Where does ATP come from?",
                "top_k": 6,
                "retrieval_mode": "coverage",
                "include_annotations": True,
                "min_extraction_quality": "low",
            },
        )
        search.raise_for_status()
        search_payload = search.json()
        evidence_items = search_payload["evidence_items"]
        assert evidence_items, search_payload
        assert any(item["citation"]["support_type"] == "supplemental_note" for item in evidence_items)
        assert all(item["citation"]["slide_id"] != first_slide["slide_id"] for item in evidence_items if item["citation"]["support_type"] == "explicit")

        citation_ids = [item["citation"]["citation_id"] for item in evidence_items]
        resolved = client.post("/v1/citations/resolve", json={"citation_ids": citation_ids})
        resolved.raise_for_status()
        resolved_payload = resolved.json()["citations"]
        assert len(resolved_payload) == len(citation_ids)
        assert all("preview_url" in citation for citation in resolved_payload)
        assert all("source_open_url" in citation for citation in resolved_payload)

        bundle_request = {
            "workspace_id": workspace_id,
            "material_ids": [pdf_material_id],
            "query_text": "cell structure",
            "bundle_mode": "full_material",
            "token_budget": 0,
            "max_items": 10,
            "include_annotations": True,
        }
        retrieval_bundle = client.post("/v1/retrieval/bundle", json=bundle_request)
        retrieval_bundle.raise_for_status()
        bundle_payload = retrieval_bundle.json()
        assert bundle_payload["bundle_id"]
        assert bundle_payload["workspace_id"] == workspace_id
        assert bundle_payload["material_ids"] == [pdf_material_id]
        assert bundle_payload["bundle_mode"] == "full_material"
        assert bundle_payload["items"]
        assert set(bundle_payload["summary"].keys()) == {"total_items", "total_slides", "low_confidence_item_count"}
        ordered_numbers = [item["slide_number"] for item in bundle_payload["items"]]
        assert ordered_numbers == sorted(ordered_numbers)
        assert all(item["citation"]["preview_url"] for item in bundle_payload["items"])
        assert all(item["citation"]["source_open_url"] for item in bundle_payload["items"])

        evidence_bundle = client.post("/v1/evidence-bundles", json=bundle_request)
        evidence_bundle.raise_for_status()
        evidence_payload = evidence_bundle.json()
        assert evidence_payload == bundle_payload

        ann_preview = client.get(url_path(f"http://127.0.0.1:38410/v1/workspaces/{workspace_id}/annotations/{study_note_id}/preview"))
        assert ann_preview.status_code == 200
        ann_source = client.get(url_path(f"http://127.0.0.1:38410/v1/workspaces/{workspace_id}/annotations/{study_note_id}/source"))
        assert ann_source.status_code == 200
        assert "Stored text" in ann_source.text

        deleted_annotation = client.delete(f"/v1/workspaces/{workspace_id}/annotations/{study_note_id}")
        deleted_annotation.raise_for_status()
        remaining_annotations = client.get(f"/v1/workspaces/{workspace_id}/annotations")
        assert len(remaining_annotations.json()["annotations"]) == 2

    with make_client(local_data) as restarted_client:
        materials_after_restart = restarted_client.get(f"/v1/materials?workspace_id={workspace_id}")
        materials_after_restart.raise_for_status()
        assert len(materials_after_restart.json()["materials"]) == 3

        search_after_restart = restarted_client.post(
            "/v1/retrieval/search",
            json={
                "workspace_id": workspace_id,
                "material_ids": [pdf_material_id, pptx_material_id, notes_material_id],
                "query_text": "photosynthesis chloroplasts",
                "top_k": 4,
                "retrieval_mode": "precision",
                "include_annotations": False,
                "min_extraction_quality": "low",
            },
        )
        search_after_restart.raise_for_status()
        assert search_after_restart.json()["evidence_items"]


def test_evidence_bundle_endpoints_accept_omitted_and_null_query_text_without_regressing_retrieval_bundle(tmp_path: Path) -> None:
    workspace_id = "ws_evidence_null_query"
    with make_client(tmp_path / "local_data") as client:
        material_id, job_id = import_text(
            client,
            workspace_id,
            "notes",
            "Lecture overview\n\nCellular respiration includes glycolysis, the citric acid cycle, and oxidative phosphorylation.",
            title="Respiration Notes",
        )
        job = wait_for_job(client, job_id)
        assert job["status"] == "succeeded"

        omitted_request = {
            "workspace_id": workspace_id,
            "material_ids": [material_id],
            "bundle_mode": "coverage",
            "token_budget": 0,
            "max_items": 5,
            "include_annotations": True,
        }
        retrieval_omitted = client.post("/v1/retrieval/bundle", json=omitted_request)
        retrieval_omitted.raise_for_status()
        retrieval_omitted_payload = retrieval_omitted.json()
        assert retrieval_omitted_payload["query_text"] is None
        assert retrieval_omitted_payload["items"]
        assert all(item["citation"]["preview_url"] for item in retrieval_omitted_payload["items"])
        assert all(item["citation"]["source_open_url"] for item in retrieval_omitted_payload["items"])

        plural_omitted = client.post("/v1/evidence-bundles", json=omitted_request)
        plural_omitted.raise_for_status()
        singular_omitted = client.post("/v1/evidence-bundle", json=omitted_request)
        singular_omitted.raise_for_status()
        assert plural_omitted.json() == retrieval_omitted_payload
        assert singular_omitted.json() == retrieval_omitted_payload

        null_request = {
            "workspace_id": workspace_id,
            "material_ids": [material_id],
            "query_text": None,
            "bundle_mode": "full_material",
            "token_budget": 0,
            "max_items": 5,
            "include_annotations": True,
        }
        retrieval_null = client.post("/v1/retrieval/bundle", json=null_request)
        retrieval_null.raise_for_status()
        retrieval_null_payload = retrieval_null.json()
        assert retrieval_null_payload["query_text"] is None
        assert retrieval_null_payload["items"]
        assert all(item["citation"]["preview_url"] for item in retrieval_null_payload["items"])
        assert all(item["citation"]["source_open_url"] for item in retrieval_null_payload["items"])

        plural_null = client.post("/v1/evidence-bundles", json=null_request)
        plural_null.raise_for_status()
        singular_null = client.post("/v1/evidence-bundle", json=null_request)
        singular_null.raise_for_status()
        assert plural_null.json() == retrieval_null_payload
        assert singular_null.json() == retrieval_null_payload


def test_legacy_import_aliases_are_accepted_and_normalized(tmp_path: Path) -> None:
    workspace_id = "ws_legacy_import"
    with make_client(tmp_path / "local_data") as client:
        with PDF_SAMPLE.open("rb") as fh:
            file_response = client.post(
                "/v1/materials/import",
                data={
                    "workspace_id": workspace_id,
                    "role": "slides",
                    "title": "Legacy PDF Import",
                },
                files={"file": (PDF_SAMPLE.name, fh, "application/pdf")},
            )
        file_response.raise_for_status()
        file_payload = file_response.json()

        text_response = client.post(
            "/v1/materials/import",
            json={
                "workspace_id": workspace_id,
                "role": "notes",
                "kind": "pasted_text",
                "title": "Legacy Text Alias",
                "text": "Electron transport chain notes. ATP synthase produces ATP.",
            },
        )
        text_response.raise_for_status()
        text_payload = text_response.json()

        source_text_response = client.post(
            "/v1/materials/import",
            json={
                "workspace_id": workspace_id,
                "role": "practice_template",
                "kind": "pasted_text",
                "title": "Legacy Source Text Alias",
                "source_text": "Example practice prompt about membrane transport and osmosis.",
            },
        )
        source_text_response.raise_for_status()
        source_text_payload = source_text_response.json()

        for job_id in (file_payload["job_id"], text_payload["job_id"], source_text_payload["job_id"]):
            job = wait_for_job(client, job_id)
            assert job["status"] == "succeeded", job

        materials = client.get(f"/v1/materials?workspace_id={workspace_id}")
        materials.raise_for_status()
        by_title = {item["title"]: item for item in materials.json()["materials"]}
        assert by_title["Legacy PDF Import"]["kind"] == "pdf"
        assert by_title["Legacy Text Alias"]["kind"] == "pasted_text"
        assert by_title["Legacy Source Text Alias"]["kind"] == "pasted_text"

        practice_material = client.get(f"/v1/materials/{source_text_payload['material_id']}")
        practice_material.raise_for_status()
        practice_payload = practice_material.json()
        assert practice_payload["page_count"] >= 1
        assert practice_payload["ready_for_retrieval"] is True


def test_legacy_annotation_aliases_work_when_mapping_is_clear_and_fail_otherwise(tmp_path: Path) -> None:
    workspace_id = "ws_legacy_annotations"
    with make_client(tmp_path / "local_data") as client:
        material_id, job_id = import_text(
            client,
            workspace_id,
            "notes",
            "Glycolysis overview\n\nGlycolysis happens in the cytoplasm and produces ATP.",
            title="Bio Notes",
        )
        job = wait_for_job(client, job_id)
        assert job["status"] == "succeeded"

        slides = client.get(f"/v1/materials/{material_id}/slides").json()["slides"]
        slide_id = slides[0]["slide_id"]

        canonical = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "annotation_type": "study_note",
                "scope": "workspace",
                "text": "Canonical annotation still works.",
            },
        )
        canonical.raise_for_status()
        assert canonical.json()["annotation_type"] == "study_note"

        legacy_focus = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "kind": "focus",
                "target_type": "material",
                "target_id": material_id,
                "text": "Prioritize glycolysis details.",
            },
        )
        legacy_focus.raise_for_status()
        legacy_focus_payload = legacy_focus.json()
        assert legacy_focus_payload["annotation_type"] == "focus"
        assert legacy_focus_payload["scope"] == "material"
        assert legacy_focus_payload["material_id"] == material_id
        assert legacy_focus_payload["slide_id"] is None

        legacy_exclude = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "kind": "exclude",
                "target_type": "slide",
                "target_id": slide_id,
                "text": "Do not ground against this slide.",
            },
        )
        legacy_exclude.raise_for_status()
        legacy_exclude_payload = legacy_exclude.json()
        assert legacy_exclude_payload["annotation_type"] == "exclude_from_grounding"
        assert legacy_exclude_payload["scope"] == "slide"
        assert legacy_exclude_payload["material_id"] == material_id
        assert legacy_exclude_payload["slide_id"] == slide_id

        exclusion_search = client.post(
            "/v1/retrieval/search",
            json={
                "workspace_id": workspace_id,
                "material_ids": [material_id],
                "query_text": "Where does glycolysis happen?",
                "top_k": 3,
                "retrieval_mode": "precision",
                "include_annotations": False,
                "min_extraction_quality": "low",
            },
        )
        exclusion_search.raise_for_status()
        assert exclusion_search.json()["evidence_items"] == []

        unsupported_target = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "kind": "focus",
                "target_type": "practice_question",
                "target_id": "pq_1",
                "text": "This should fail cleanly.",
            },
        )
        assert unsupported_target.status_code == 400
        assert "workspace, material, or slide" in unsupported_target.json()["detail"]

        unknown_kind = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "kind": "unsupported_alias",
                "target_type": "material",
                "target_id": material_id,
                "text": "This should also fail.",
            },
        )
        assert unknown_kind.status_code == 400
        assert "Supported Team 2 annotation types" in unknown_kind.json()["detail"]


def test_load_settings_uses_root_env_and_shared_local_data_defaults(tmp_path: Path) -> None:
    project_root = tmp_path / "study_helper_root"
    service_root = project_root / "02_content_service"
    (project_root / "01_app_shell").mkdir(parents=True)
    service_root.mkdir(parents=True)
    (project_root / "03_learning_service").mkdir(parents=True)

    env_file = project_root / ".env"
    env_file.write_text("CONTENT_SERVICE_PORT=38555\n", encoding="utf-8")

    defaults_from_root = load_settings(service_root=service_root, cwd=service_root, environ={})
    assert defaults_from_root.port == 38555
    assert defaults_from_root.local_data_dir == (project_root / "local_data").resolve()

    env_file.write_text(
        "CONTENT_SERVICE_PORT=38555\nLOCAL_DATA_DIR=./shared_local_data\nCONTENT_SERVICE_VERSION=2.0.0\n",
        encoding="utf-8",
    )
    from_root_env = load_settings(service_root=service_root, cwd=service_root, environ={})
    assert from_root_env.port == 38555
    assert from_root_env.version == "2.0.0"
    assert from_root_env.local_data_dir == (project_root / "shared_local_data").resolve()

    overridden = load_settings(
        service_root=service_root,
        cwd=service_root,
        environ={
            "CONTENT_SERVICE_PORT": "39999",
            "LOCAL_DATA_DIR": "./process_override_data",
            "CONTENT_SERVICE_VERSION": "9.9.9",
        },
    )
    assert overridden.port == 39999
    assert overridden.version == "9.9.9"
    assert overridden.local_data_dir == (service_root / "process_override_data").resolve()


def test_delete_material_preserves_workspace_scoped_annotations_only(tmp_path: Path) -> None:
    workspace_id = "ws_delete"
    with make_client(tmp_path / "local_data") as client:
        material_id, job_id = import_text(client, workspace_id, "notes", "Topic A\n\nDetails about Topic A.", title="Topic A Notes")
        job = wait_for_job(client, job_id)
        assert job["status"] == "succeeded"

        slides = client.get(f"/v1/materials/{material_id}/slides").json()["slides"]
        slide_id = slides[0]["slide_id"]

        ws_ann = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={"annotation_type": "study_note", "scope": "workspace", "text": "Global study reminder."},
        )
        ws_ann.raise_for_status()
        mat_ann = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={"annotation_type": "study_note", "scope": "material", "material_id": material_id, "text": "Material-specific note."},
        )
        mat_ann.raise_for_status()
        slide_ann = client.post(
            f"/v1/workspaces/{workspace_id}/annotations",
            json={
                "annotation_type": "user_correction",
                "scope": "slide",
                "material_id": material_id,
                "slide_id": slide_id,
                "text": "This slide needs careful wording.",
            },
        )
        slide_ann.raise_for_status()

        delete_resp = client.delete(f"/v1/materials/{material_id}")
        delete_resp.raise_for_status()
        assert delete_resp.json()["deleted"] is True

        remaining_materials = client.get(f"/v1/materials?workspace_id={workspace_id}")
        assert remaining_materials.json()["materials"] == []
        remaining_annotations = client.get(f"/v1/workspaces/{workspace_id}/annotations")
        remaining = remaining_annotations.json()["annotations"]
        assert len(remaining) == 1
        assert remaining[0]["scope"] == "workspace"


def test_bad_upload_fails_as_job_not_service_crash(tmp_path: Path) -> None:
    workspace_id = "ws_bad"
    bad_file = tmp_path / "broken.pdf"
    bad_file.write_bytes(b"not a real pdf")
    with make_client(tmp_path / "local_data") as client:
        material_id, job_id = import_file(client, workspace_id, "slides", bad_file)
        job = wait_for_job(client, job_id)
        assert job["status"] == "failed"
        material = client.get(f"/v1/materials/{material_id}")
        material.raise_for_status()
        assert material.json()["processing_status"] == "failed"
