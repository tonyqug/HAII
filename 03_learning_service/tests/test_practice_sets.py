from __future__ import annotations

from .conftest import wait_for_job



def create_practice_set(client, bundle, generation_mode, **overrides):
    payload = {
        "workspace_id": bundle["workspace_id"],
        "material_ids": None,
        "evidence_bundle": bundle,
        "generation_mode": generation_mode,
        "template_material_id": overrides.get("template_material_id"),
        "question_count": overrides.get("question_count", 4),
        "coverage_mode": overrides.get("coverage_mode", "balanced"),
        "difficulty_profile": overrides.get("difficulty_profile", "mixed"),
        "include_answer_key": overrides.get("include_answer_key", True),
        "include_rubrics": overrides.get("include_rubrics", True),
        "grounding_mode": overrides.get("grounding_mode", "lecture_with_fallback"),
        "include_annotations": True,
    }
    response = client.post("/v1/practice-sets", json=payload)
    assert response.status_code == 200, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    artifact = client.get(f"/v1/practice-sets/{job['result_id']}")
    assert artifact.status_code == 200
    return artifact.json(), job



def test_practice_set_short_and_long_answer_modes(client, bundle):
    short_set, _job = create_practice_set(client, bundle, "short_answer", question_count=3)
    assert short_set["generation_mode"] == "short_answer"
    assert all(question["question_type"] == "short_answer" for question in short_set["questions"])
    assert all(question["citations"] for question in short_set["questions"])
    assert short_set["coverage_report"]["considered_slide_count"] >= short_set["coverage_report"]["cited_slide_count"]

    long_set, _job = create_practice_set(client, bundle, "long_answer", question_count=3, difficulty_profile="harder")
    assert long_set["generation_mode"] == "long_answer"
    assert all(question["question_type"] == "long_answer" for question in long_set["questions"])
    assert all(question["rubric"] for question in long_set["questions"])



def test_template_mimic_generation_and_revision(client, template_bundle):
    practice_set, _job = create_practice_set(
        client,
        template_bundle,
        "template_mimic",
        template_material_id="mat_template_midterm_1",
        question_count=4,
        coverage_mode="exhaustive",
    )
    assert practice_set["generation_mode"] == "template_mimic"
    assert practice_set["template_style_summary"]
    assert practice_set["coverage_report"]["notes"]
    first_question = practice_set["questions"][0]
    locked_id = first_question["question_id"]

    response = client.post(
        f"/v1/practice-sets/{practice_set['practice_set_id']}/revise",
        json={
            "instruction_text": "Make the unlocked questions harder and more application-focused.",
            "target_question_ids": [],
            "locked_question_ids": [locked_id],
            "maintain_coverage": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job

    revised = client.get(f"/v1/practice-sets/{job['result_id']}").json()
    assert revised["practice_set_id"] != practice_set["practice_set_id"]
    assert revised["parent_practice_set_id"] == practice_set["practice_set_id"]
    revised_locked = next(question for question in revised["questions"] if question["question_id"] == locked_id)
    assert revised_locked == first_question
