from __future__ import annotations

from typing import Any, Dict, List

import pytest

from .conftest import wait_for_job


class FakeResponse:
    def __init__(self, status_code: int, payload: Any):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload



def test_integrated_mode_fetches_bundle_from_canonical_team2_endpoint(client, bundle, monkeypatch):
    paths: List[str] = []

    def fake_post(url, json=None, timeout=None):
        paths.append(url)
        if url.endswith("/v1/evidence-bundles"):
            return FakeResponse(200, bundle)
        return FakeResponse(404, {"detail": "not found"})

    monkeypatch.setattr("learning_service.content_client.requests.post", fake_post)

    response = client.post(
        "/v1/study-plans",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": list(bundle["material_ids"]),
            "topic_text": "regularization",
            "time_budget_minutes": 60,
            "grounding_mode": "lecture_with_fallback",
            "student_context": {},
            "include_annotations": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert any(path.endswith("/v1/evidence-bundles") for path in paths)



def test_integrated_conversation_flow_fetches_bundle_from_retrieval_bundle_compat_endpoint(client, bundle, monkeypatch):
    paths: List[str] = []

    def fake_post(url, json=None, timeout=None):
        paths.append(url)
        if url.endswith("/v1/evidence-bundles"):
            return FakeResponse(404, {"detail": "not found"})
        if url.endswith("/v1/retrieval/bundle"):
            return FakeResponse(200, bundle)
        return FakeResponse(404, {"detail": "not found"})

    monkeypatch.setattr("learning_service.content_client.requests.post", fake_post)

    create = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": list(bundle["material_ids"]),
            "grounding_mode": "lecture_with_fallback",
            "title": "Integrated flow",
            "include_annotations": True,
        },
    )
    assert create.status_code == 200
    conversation_id = create.json()["conversation_id"]

    first = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What does regularization do?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert first.status_code == 200
    first_job = wait_for_job(client, first.json()["job_id"])
    assert first_job["status"] == "succeeded"

    conversation = client.get(f"/v1/conversations/{conversation_id}")
    assert conversation.status_code == 200
    assert len(conversation.json()["messages"]) == 2

    cleared = client.post(f"/v1/conversations/{conversation_id}/clear")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True

    second = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "How is validation used?",
            "response_style": "concise",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert second.status_code == 200
    second_job = wait_for_job(client, second.json()["job_id"])
    assert second_job["status"] == "succeeded"

    assert any(path.endswith("/v1/evidence-bundles") for path in paths)
    assert any(path.endswith("/v1/retrieval/bundle") for path in paths)


@pytest.mark.parametrize(
    ("generation_mode", "template_material_id"),
    [
        ("short_answer", None),
        ("long_answer", None),
        ("template_mimic", "mat_template_midterm_1"),
    ],
)
def test_integrated_practice_generation_is_robust_to_bundle_endpoint_compatibility(
    client,
    template_bundle,
    generation_mode,
    template_material_id,
    monkeypatch,
):
    observed_requests: list[dict[str, Any]] = []

    def fake_post(url, json=None, timeout=None):
        observed_requests.append({"url": url, "json": dict(json or {})})
        if url.endswith("/v1/evidence-bundles"):
            return FakeResponse(404, {"detail": "compatibility fallback"})
        if url.endswith("/v1/retrieval/bundle"):
            if json is not None and not json.get("query_text"):
                return FakeResponse(422, {"detail": "query_text must not be null"})
            return FakeResponse(200, template_bundle)
        return FakeResponse(404, {"detail": "not found"})

    monkeypatch.setattr("learning_service.content_client.requests.post", fake_post)

    response = client.post(
        "/v1/practice-sets",
        json={
            "workspace_id": template_bundle["workspace_id"],
            "material_ids": ["mat_lecture_1", "mat_lecture_2"],
            "generation_mode": generation_mode,
            "template_material_id": template_material_id,
            "question_count": 3,
            "coverage_mode": "balanced",
            "difficulty_profile": "mixed",
            "include_answer_key": True,
            "include_rubrics": True,
            "grounding_mode": "lecture_with_fallback",
            "include_annotations": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job

    compat_requests = [request for request in observed_requests if request["url"].endswith("/v1/retrieval/bundle")]
    assert compat_requests
    lecture_requests = [request for request in compat_requests if request["json"].get("material_ids") == ["mat_lecture_1", "mat_lecture_2"]]
    assert lecture_requests
    assert all(request["json"].get("query_text") == "full lecture coverage for practice generation" for request in lecture_requests)

    if generation_mode == "template_mimic":
        template_requests = [
            request
            for request in compat_requests
            if request["json"].get("material_ids") == ["mat_template_midterm_1"]
        ]
        assert template_requests
        assert all(request["json"].get("query_text") == "template style analysis" for request in template_requests)
