from __future__ import annotations

from .conftest import wait_for_job



def test_gemini_is_primary_generation_path_when_configured(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
        return {
            "topic_text": "Regularization",
            "prerequisites": [
                {
                    "concept_name": "Loss functions",
                    "why_needed": "The lecture connects regularization to optimizing an objective that includes penalties.",
                    "support_status": "slide_grounded",
                    "citation_ids": ["cit_001"],
                },
                {
                    "concept_name": "Validation",
                    "why_needed": "The lecture compares training and validation behavior when tuning regularization.",
                    "support_status": "slide_grounded",
                    "citation_ids": ["cit_004"],
                },
                {
                    "concept_name": "Model complexity",
                    "why_needed": "The lecture frames regularization as a way to control overly flexible models.",
                    "support_status": "inferred_from_slides",
                    "citation_ids": ["cit_001", "cit_004"],
                },
            ],
            "study_sequence": [
                {
                    "title": "Review the penalty term",
                    "objective": "Understand how the lecture introduces the regularized objective and what the penalty changes.",
                    "recommended_time_minutes": 20,
                    "tasks": [
                        "Read the cited slide and restate the objective in your own words.",
                        "Note what behavior the penalty is trying to discourage.",
                    ],
                    "depends_on_prereq_indexes": [1, 3],
                    "support_status": "slide_grounded",
                    "citation_ids": ["cit_001"],
                },
                {
                    "title": "Compare training and validation behavior",
                    "objective": "Use the lecture evidence to connect regularization to generalization and model selection.",
                    "recommended_time_minutes": 20,
                    "tasks": [
                        "Compare the lecture's training and validation discussion.",
                        "Summarize when stronger or weaker regularization could help.",
                    ],
                    "depends_on_prereq_indexes": [2, 3],
                    "support_status": "inferred_from_slides",
                    "citation_ids": ["cit_004", "cit_005"],
                },
            ],
            "common_mistakes": [
                {
                    "pattern": "Treating regularization as a free accuracy boost.",
                    "why_it_happens": "Students remember the headline goal but forget the tuning trade-off described in the lecture.",
                    "prevention_advice": "Pair the penalty idea with the lecture's validation-based tuning advice.",
                    "support_status": "slide_grounded",
                    "citation_ids": ["cit_004"],
                },
                {
                    "pattern": "Ignoring what the penalty is acting on.",
                    "why_it_happens": "The lecture distinguishes the penalty from the base objective, and that distinction is easy to blur.",
                    "prevention_advice": "Restate the objective and mark which part is the regularizer.",
                    "support_status": "slide_grounded",
                    "citation_ids": ["cit_001"],
                },
                {
                    "pattern": "Tuning without checking validation behavior.",
                    "why_it_happens": "The lecture emphasizes validation signals, but students can skip that step when memorizing definitions.",
                    "prevention_advice": "Check the cited validation slide before finalizing your study note.",
                    "support_status": "slide_grounded",
                    "citation_ids": ["cit_005"],
                },
            ],
            "uncertainty": [],
        }

    monkeypatch.setattr(service.generator.gemini, "generate_json", fake_generate_json)

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
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    stored = service.store.load("study_plans", job["result_id"])
    assert stored["_meta"]["generation_path"] == "gemini"
    assert stored["tailoring_summary"]["used_inputs"]
    assert all("milestone" in step for step in stored["study_sequence"])
    bundle_citation_ids = {item["citation"]["citation_id"] for item in bundle["items"]}
    artifact_citation_ids = {
        citation["citation_id"]
        for section in stored["prerequisites"] + stored["study_sequence"] + stored["common_mistakes"]
        for citation in section.get("citations", [])
    }
    assert artifact_citation_ids <= bundle_citation_ids



def test_gemini_validation_blocks_unsupported_citations_and_falls_back(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
        return {
            "template_style_summary": None,
            "questions": [
                {
                    "question_type": "short_answer",
                    "stem": "Explain regularization.",
                    "expected_answer": "Unsupported answer.",
                    "citation_ids": ["not_in_bundle"],
                },
                {
                    "question_type": "short_answer",
                    "stem": "Explain validation.",
                    "expected_answer": "Unsupported answer.",
                    "citation_ids": ["not_in_bundle"],
                },
                {
                    "question_type": "short_answer",
                    "stem": "Explain gradient descent.",
                    "expected_answer": "Unsupported answer.",
                    "citation_ids": ["not_in_bundle"],
                },
            ],
        }

    monkeypatch.setattr(service.generator.gemini, "generate_json", fake_generate_json)

    response = client.post(
        "/v1/practice-sets",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "generation_mode": "short_answer",
            "question_count": 3,
            "coverage_mode": "balanced",
            "difficulty_profile": "mixed",
            "include_answer_key": True,
            "include_rubrics": True,
            "grounding_mode": "lecture_with_fallback",
            "include_annotations": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "succeeded"
    stored = service.store.load("practice_sets", job["result_id"])
    assert stored["_meta"]["generation_path"] == "heuristic_fallback"
    bundle_citation_ids = {item["citation"]["citation_id"] for item in bundle["items"]}
    practice_citation_ids = {citation["citation_id"] for question in stored["questions"] for citation in question["citations"]}
    assert practice_citation_ids <= bundle_citation_ids
