from __future__ import annotations

from pathlib import Path

from .conftest import wait_for_job



def test_artifacts_survive_restart(app_factory, bundle, template_bundle, tmp_path: Path):
    shared_data_dir = tmp_path / "restart_data"
    client_a = app_factory(shared_data_dir)

    study_plan_response = client_a.post(
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
    plan_job = wait_for_job(client_a, study_plan_response.json()["job_id"])
    assert plan_job["status"] == "succeeded"
    study_plan_id = plan_job["result_id"]

    conversation = client_a.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Persist me",
            "include_annotations": True,
        },
    ).json()
    conversation_id = conversation["conversation_id"]
    message_job = client_a.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What is gradient descent?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert wait_for_job(client_a, message_job.json()["job_id"])["status"] == "succeeded"

    practice_response = client_a.post(
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
    practice_job = wait_for_job(client_a, practice_response.json()["job_id"])
    assert practice_job["status"] == "succeeded"
    practice_set_id = practice_job["result_id"]

    client_b = app_factory(shared_data_dir)

    study_plan = client_b.get(f"/v1/study-plans/{study_plan_id}")
    assert study_plan.status_code == 200
    assert study_plan.json()["study_plan_id"] == study_plan_id

    conversation_after_restart = client_b.get(f"/v1/conversations/{conversation_id}")
    assert conversation_after_restart.status_code == 200
    assert len(conversation_after_restart.json()["messages"]) == 2

    practice_after_restart = client_b.get(f"/v1/practice-sets/{practice_set_id}")
    assert practice_after_restart.status_code == 200
    assert practice_after_restart.json()["practice_set_id"] == practice_set_id
