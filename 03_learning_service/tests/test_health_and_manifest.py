from __future__ import annotations

from .conftest import wait_for_job


def test_service_imports_and_launches_in_shared_runtime(app_factory):
    client = app_factory()
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["service_name"] == "learning_service"


def test_health_and_manifest(client):
    health = client.get("/healthz")
    assert health.status_code == 200
    payload = health.json()
    assert payload["service_name"] == "learning_service"
    assert payload["ready"] is False
    assert payload["status"] == "ok"
    assert payload["details"]["process_up"] is True
    assert payload["details"]["gemini_configured"] is False
    assert payload["details"]["deterministic_fallback_available"] is True
    assert payload["details"]["primary_generation_path"] == "heuristic_fallback"
    assert "content_service_reachable" in payload["details"]

    manifest = client.get("/manifest")
    assert manifest.status_code == 200
    manifest_payload = manifest.json()
    assert manifest_payload["service_name"] == "learning_service"
    assert "inline_evidence_mode" in manifest_payload["capabilities"]
    assert manifest_payload["api_base_url"].startswith("http://127.0.0.1:38420")


def test_integrated_mode_fails_cleanly_when_content_service_unreachable(client):
    response = client.post(
        "/v1/study-plans",
        json={
            "workspace_id": "ws1",
            "material_ids": ["mat_1"],
            "evidence_bundle": None,
            "topic_text": "test",
            "time_budget_minutes": 30,
            "grounding_mode": "strict_lecture_only",
            "student_context": {},
            "include_annotations": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "content_service_error"
