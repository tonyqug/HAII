from __future__ import annotations

from .conftest import wait_for_job


def create_plan(client, bundle, topic_text=None, grounding_mode="lecture_with_fallback"):
    response = client.post(
        "/v1/study-plans",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "topic_text": topic_text,
            "time_budget_minutes": 90,
            "grounding_mode": grounding_mode,
            "student_context": {
                "prior_knowledge": "basic algebra",
                "weak_areas": "validation",
                "goals": "prepare for the midterm",
            },
            "include_annotations": True,
        },
    )
    assert response.status_code == 200, response.text
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded", job
    return client.get(f"/v1/study-plans/{job['result_id']}").json(), job



def test_study_plan_structure_and_honesty(client, bundle):
    plan, job = create_plan(client, bundle, topic_text=None)
    assert plan["study_plan_id"] == job["result_id"]
    assert plan["workspace_id"] == bundle["workspace_id"]
    assert len(plan["prerequisites"]) >= 3
    assert len(plan["study_sequence"]) >= 1
    assert len(plan["common_mistakes"]) == 3
    assert any(note["code"] == "topic_inferred" for note in plan["uncertainty"])

    for item in plan["prerequisites"]:
        assert item["support_status"] in {
            "slide_grounded",
            "inferred_from_slides",
            "external_supplement",
            "insufficient_evidence",
        }
        if item["support_status"] in {"slide_grounded", "inferred_from_slides"}:
            assert item["citations"]

    assert plan["coverage_summary"]["cited_slides"]



def test_study_plan_revision_preserves_locked_items(client, bundle):
    original, _job = create_plan(client, bundle, topic_text="regularization and validation")
    locked_item_id = original["prerequisites"][0]["item_id"]
    original_locked = original["prerequisites"][0]

    response = client.post(
        f"/v1/study-plans/{original['study_plan_id']}/revise",
        json={
            "instruction_text": "Make the study sequence shorter and focus on validation.",
            "target_section": "entire_plan",
            "locked_item_ids": [locked_item_id],
            "grounding_mode": "lecture_with_fallback",
            "include_annotations": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"

    revised = client.get(f"/v1/study-plans/{job['result_id']}").json()
    assert revised["study_plan_id"] != original["study_plan_id"]
    assert revised["parent_study_plan_id"] == original["study_plan_id"]
    revised_locked = next(item for item in revised["prerequisites"] if item["item_id"] == locked_item_id)
    assert revised_locked == original_locked

    listing = client.get(f"/v1/study-plans?workspace_id={bundle['workspace_id']}")
    assert listing.status_code == 200
    assert len(listing.json()["study_plans"]) >= 2
