from __future__ import annotations

import time

from .conftest import fetch_job, wait_for_job



def _delayed_bound_method(monkeypatch, obj, method_name: str, delay_seconds: float):
    original = getattr(obj, method_name)

    def delayed(*args, **kwargs):
        time.sleep(delay_seconds)
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, method_name, delayed)



def test_every_job_endpoint_returns_quickly_and_persists_final_artifacts(client, bundle, template_bundle, monkeypatch):
    service = client.app.state.learning_service
    delay = 0.30

    _delayed_bound_method(monkeypatch, service.generator, "build_study_plan", delay)
    started = time.perf_counter()
    response = client.post(
        "/v1/study-plans",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "topic_text": "regularization",
            "time_budget_minutes": 60,
            "grounding_mode": "lecture_with_fallback",
            "student_context": {},
            "include_annotations": True,
        },
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < delay / 2
    study_job_id = response.json()["job_id"]
    assert fetch_job(client, study_job_id)["status"] in {"queued", "running"}
    study_job = wait_for_job(client, study_job_id)
    assert study_job["status"] == "succeeded"
    study_plan_id = study_job["result_id"]

    _delayed_bound_method(monkeypatch, service.generator, "revise_study_plan", delay)
    started = time.perf_counter()
    response = client.post(
        f"/v1/study-plans/{study_plan_id}/revise",
        json={
            "instruction_text": "make the plan shorter",
            "target_section": "study_sequence",
            "locked_item_ids": [],
            "grounding_mode": "lecture_with_fallback",
        },
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < delay / 2
    revision_job = wait_for_job(client, response.json()["job_id"])
    assert revision_job["status"] == "succeeded"
    assert revision_job["result_id"]

    conversation_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Background jobs",
        },
    )
    conversation_id = conversation_response.json()["conversation_id"]

    _delayed_bound_method(monkeypatch, service.generator, "build_conversation_reply", delay)
    started = time.perf_counter()
    response = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What does regularization do?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < delay / 2
    chat_job = wait_for_job(client, response.json()["job_id"])
    assert chat_job["status"] == "succeeded"
    assert chat_job["result_type"] == "message"

    _delayed_bound_method(monkeypatch, service.generator, "build_practice_set", delay)
    started = time.perf_counter()
    response = client.post(
        "/v1/practice-sets",
        json={
            "workspace_id": template_bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": template_bundle,
            "generation_mode": "template_mimic",
            "template_material_id": "mat_template_midterm_1",
            "question_count": 3,
            "coverage_mode": "balanced",
            "difficulty_profile": "mixed",
            "include_answer_key": True,
            "include_rubrics": True,
            "grounding_mode": "lecture_with_fallback",
            "include_annotations": True,
        },
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < delay / 2
    practice_job = wait_for_job(client, response.json()["job_id"])
    assert practice_job["status"] == "succeeded"
    practice_set_id = practice_job["result_id"]

    _delayed_bound_method(monkeypatch, service.generator, "revise_practice_set", delay)
    started = time.perf_counter()
    response = client.post(
        f"/v1/practice-sets/{practice_set_id}/revise",
        json={
            "instruction_text": "make the set easier",
            "target_question_ids": [],
            "locked_question_ids": [],
            "maintain_coverage": True,
        },
    )
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    assert elapsed < delay / 2
    practice_revision_job = wait_for_job(client, response.json()["job_id"])
    assert practice_revision_job["status"] == "succeeded"
    assert practice_revision_job["result_id"]



def test_alias_inputs_are_normalized_without_brittle_validation(client, bundle, template_bundle, monkeypatch):
    service = client.app.state.learning_service
    captured: dict[str, object] = {}

    def fake_fetch_evidence_bundle(*, workspace_id, material_ids, query_text, bundle_mode, include_annotations):
        captured["workspace_id"] = workspace_id
        captured["material_ids"] = list(material_ids)
        captured["query_text"] = query_text
        captured["bundle_mode"] = bundle_mode
        return bundle

    monkeypatch.setattr(service.content_client, "fetch_evidence_bundle", fake_fetch_evidence_bundle)

    response = client.post(
        "/v1/study-plans",
        json={
            "workspace_id": bundle["workspace_id"],
            "included_material_ids": list(bundle["material_ids"]),
            "excluded_material_ids": [],
            "focused_material_ids": [bundle["material_ids"][0]],
            "topic_text": "validation",
            "time_budget_minutes": 45,
            "student_context": {"known": "basic algebra"},
            "include_annotations": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert captured["material_ids"] == bundle["material_ids"]

    original = client.get(f"/v1/study-plans/{job['result_id']}").json()
    revise = client.post(
        f"/v1/study-plans/{original['study_plan_id']}/revise",
        json={
            "feedback_note": "Please clarify the unlocked parts.",
            "locked_sections": ["prerequisites"],
            "target_section": "entire_plan",
        },
    )
    assert revise.status_code == 200
    revised_job = wait_for_job(client, revise.json()["job_id"])
    assert revised_job["status"] == "succeeded"
    revised = client.get(f"/v1/study-plans/{revised_job['result_id']}").json()
    assert revised["prerequisites"] == original["prerequisites"]

    conversation = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
        },
    ).json()["conversation_id"]
    send = client.post(
        f"/v1/conversations/{conversation}/messages",
        json={
            "text": "What is validation used for?",
            "response_style": "concise",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    assert wait_for_job(client, send.json()["job_id"])["status"] == "succeeded"

    practice = client.post(
        "/v1/practice-sets",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "generation_mode": "short_answer",
            "question_count": 3,
            "coverage_mode": "balanced",
            "difficulty": "easier",
            "answer_key": False,
            "rubric": False,
            "grounding_mode": "lecture_with_fallback",
        },
    )
    assert practice.status_code == 200
    practice_job = wait_for_job(client, practice.json()["job_id"])
    assert practice_job["status"] == "succeeded"
    practice_artifact = client.get(f"/v1/practice-sets/{practice_job['result_id']}").json()
    assert all(question["expected_answer"] == "" for question in practice_artifact["questions"])
    assert all(question["rubric"] == [] for question in practice_artifact["questions"])

    revise_practice = client.post(
        f"/v1/practice-sets/{practice_artifact['practice_set_id']}/revise",
        json={
            "selected_question_ids": [practice_artifact["questions"][0]["question_id"]],
            "locked_question_ids": [practice_artifact["questions"][1]["question_id"]],
            "maintain_coverage": True,
        },
    )
    assert revise_practice.status_code == 200
    practice_revision = wait_for_job(client, revise_practice.json()["job_id"])
    assert practice_revision["status"] == "succeeded"



def test_response_style_direct_answer_alias_maps_to_standard(client, bundle, monkeypatch):
    service = client.app.state.learning_service
    captured: dict[str, object] = {}
    original = service.generator.build_conversation_reply

    def wrapped(*args, **kwargs):
        captured["response_style"] = kwargs.get("response_style")
        return original(*args, **kwargs)

    monkeypatch.setattr(service.generator, "build_conversation_reply", wrapped)

    conversation_id = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "include_annotations": True,
        },
    ).json()["conversation_id"]

    response = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "Summarize the lecture evidence.",
            "response_style": "direct_answer",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    assert captured["response_style"] == "standard"
