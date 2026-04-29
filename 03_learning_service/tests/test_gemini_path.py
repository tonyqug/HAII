from __future__ import annotations

import json
import logging

import requests

from .conftest import wait_for_job
from learning_service.config import Settings
from learning_service.generator_v2 import GeminiPrimaryClient



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



def test_gemini_practice_enhancement_preserves_local_grounding_when_llm_metadata_is_bad(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
        return {
            "template_style_summary": "Mechanism-focused exam prompts with short evaluator notes.",
            "questions": [
                {
                    "question_index": 1,
                    "stem": "Explain how regularization changes the learning objective and why that matters for generalization.",
                    "expected_answer": "A strong answer explains that the lecture adds a penalty term to discourage overly flexible fits and connects that penalty to better generalization when tuned well.",
                    "citation_ids": ["not_in_bundle"],
                },
                {
                    "question_index": 2,
                    "stem": "Describe how the lecture uses validation behavior when deciding whether regularization is too weak or too strong.",
                    "expected_answer": "A strong answer ties validation behavior to model selection rather than treating regularization as a fixed free improvement.",
                    "citation_ids": ["not_in_bundle"],
                },
                {
                    "question_index": 3,
                    "stem": "Explain why regularization should be tuned rather than assumed to help automatically.",
                    "expected_answer": "A strong answer mentions the trade-off between flexibility and generalization and uses validation evidence to justify the chosen strength.",
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
    assert stored["_meta"]["generation_path"] == "gemini"
    assert stored["_meta"]["llm_enhanced_questions"] >= 1
    bundle_citation_ids = {item["citation"]["citation_id"] for item in bundle["items"]}
    practice_citation_ids = {citation["citation_id"] for question in stored["questions"] for citation in question["citations"]}
    assert practice_citation_ids <= bundle_citation_ids
    assert "generalization" in stored["questions"][0]["stem"].lower()


def test_gemini_practice_enhancement_salvages_partial_updates_without_fallback(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
        return {
            "questions": [
                {
                    "question_index": 1,
                    "stem": "Choose the lecture-grounded explanation that best connects regularization to controlling model flexibility.",
                    "answer_choices": ["Too short", "Still too short"],
                }
            ]
        }

    monkeypatch.setattr(service.generator.gemini, "generate_json", fake_generate_json)

    response = client.post(
        "/v1/practice-sets",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "generation_mode": "multiple_choice",
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
    assert stored["_meta"]["generation_path"] == "gemini"
    assert len(stored["questions"]) == 3
    assert all(len(question["answer_choices"]) == 4 for question in stored["questions"])
    assert "model flexibility" in stored["questions"][0]["stem"].lower()


def test_practice_job_uses_structured_output_schema_and_stays_on_gemini_path(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service
    captured_requests = []

    class FakeResponse:
        status_code = 200
        text = "ok"

        @property
        def ok(self):
            return True

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        captured_requests.append(json)
        return FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json_module.dumps(
                                        {
                                            "questions": [
                                                {
                                                    "question_index": 1,
                                                    "stem": "Explain how the lecture connects regularization to controlling model flexibility.",
                                                    "expected_answer": "A strong answer explains that the lecture treats regularization as a way to limit overly flexible models so they generalize better.",
                                                    "estimated_minutes": 7,
                                                }
                                            ]
                                        }
                                    )
                                },
                            ]
                        }
                    }
                ]
            }
        )

    json_module = json
    monkeypatch.setattr("learning_service.generation.requests.post", fake_post)

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
    assert stored["_meta"]["generation_path"] == "gemini"
    assert len(captured_requests) == 1
    generation_config = captured_requests[0]["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseJsonSchema"]["required"] == ["questions"]
    assert "model flexibility" in stored["questions"][0]["stem"].lower()


def test_gemini_client_uses_rate_limit_model_ladder_and_reasoning(monkeypatch):
    client = GeminiPrimaryClient(Settings(gemini_api_key="test-key"))
    requests_seen = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        @property
        def ok(self):
            return self.status_code < 400

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        requests_seen.append((url, json, timeout))
        if "gemini-3-flash-preview" in url:
            return FakeResponse(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
        if "gemini-2.5-flash:generateContent" in url:
            return FakeResponse(429, {"error": {"status": "RESOURCE_EXHAUSTED"}})
        return FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": json_module.dumps({"reply_sections": []})},
                            ]
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("learning_service.generation.requests.post", fake_post)

    payload = client.generate_json("system", "user prompt")

    assert payload == {"reply_sections": []}
    assert [url.split("/models/")[1].split(":generateContent")[0] for url, _body, _timeout in requests_seen] == [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]
    assert requests_seen[0][1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}
    assert requests_seen[1][1]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}
    assert requests_seen[2][1]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}
    assert client.last_call_info["used_model"] == "gemini-2.5-flash-lite"
    assert client.last_call_info["rate_limited_models"] == ["gemini-3-flash-preview", "gemini-2.5-flash"]
    assert client.last_call_info["reasoning_enabled"] is True


def test_gemini_client_uses_timeout_ladder_before_falling_back(monkeypatch):
    client = GeminiPrimaryClient(Settings(gemini_api_key="test-key"))
    requests_seen = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        @property
        def ok(self):
            return self.status_code < 400

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        model = url.split("/models/")[1].split(":generateContent")[0]
        thinking_config = (json or {}).get("generationConfig", {}).get("thinkingConfig")
        requests_seen.append((model, thinking_config))
        if model == "gemini-3-flash-preview":
            raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=15)")
        return FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": json_module.dumps({"reply_sections": []})},
                            ]
                        }
                    }
                ]
            },
        )

    json_module = json
    monkeypatch.setattr("learning_service.generation.requests.post", fake_post)
    monkeypatch.setattr("learning_service.generation.time.sleep", lambda *_args, **_kwargs: None)

    payload = client.generate_json("system", "user prompt")

    assert payload == {"reply_sections": []}
    assert requests_seen == [
        ("gemini-3-flash-preview", {"thinkingLevel": "high"}),
        ("gemini-3-flash-preview", {"thinkingLevel": "high"}),
        ("gemini-3-flash-preview", {"thinkingLevel": "high"}),
        ("gemini-3-flash-preview", None),
        ("gemini-3-flash-preview", None),
        ("gemini-3-flash-preview", None),
        ("gemini-2.5-flash", {"thinkingBudget": -1}),
    ]
    assert client.last_call_info["attempted_models"] == ["gemini-3-flash-preview", "gemini-2.5-flash"]
    assert client.last_call_info["used_model"] == "gemini-2.5-flash"
    assert client.last_call_info["reasoning_enabled"] is True
    assert client.last_call_info["failure_reason"] is None
    assert [failure["reason"] for failure in client.last_call_info["model_failures"]] == [
        "request_timeout",
        "request_timeout",
    ]


def test_gemini_client_includes_response_json_schema_when_requested(monkeypatch):
    client = GeminiPrimaryClient(Settings(gemini_api_key="test-key"))
    requests_seen = []

    class FakeResponse:
        status_code = 200
        text = "plain text"

        @property
        def ok(self):
            return True

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        requests_seen.append(json)
        return FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": json_module.dumps({"reply_sections": []})},
                            ]
                        }
                    }
                ]
            }
        )

    json_module = json
    monkeypatch.setattr("learning_service.generation.requests.post", fake_post)

    payload = client.generate_json(
        "system",
        "user prompt",
        response_json_schema={
            "type": "object",
            "properties": {
                "reply_sections": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
            "required": ["reply_sections"],
        },
    )

    assert payload == {"reply_sections": []}
    assert len(requests_seen) == 1
    assert requests_seen[0]["generationConfig"]["responseMimeType"] == "application/json"
    assert requests_seen[0]["generationConfig"]["responseJsonSchema"]["required"] == ["reply_sections"]
    assert client.last_call_info["failure_reason"] is None


def test_chat_uses_gemini_path_even_without_strong_local_match(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_text(system_instruction, user_prompt, max_output_tokens=256):
        service.generator.gemini.last_call_info = {
            "configured": True,
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": "gemini-3-flash-preview",
            "reasoning_enabled": True,
            "reasoning_mode": "dynamic",
            "attempted_models": ["gemini-3-flash-preview"],
            "rate_limited_models": [],
            "failure_reason": None,
        }
        return "The current lecture materials do not cover that topic directly."

    monkeypatch.setattr(service.generator.gemini, "generate_text", fake_generate_text)
    monkeypatch.setattr(
        service.generator.gemini,
        "generate_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chat should not call generate_json")),
    )

    create_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Gemini first chat",
            "include_annotations": True,
        },
    )
    assert create_response.status_code == 200
    conversation_id = create_response.json()["conversation_id"]

    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What is the history of neural networks?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    job = wait_for_job(client, send.json()["job_id"])
    assert job["status"] == "succeeded"

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assistant = conversation["messages"][-1]
    assert assistant["answer_source"]["path"] == "llm"
    assert assistant["answer_source"]["model"] == "gemini-3-flash-preview"
    assert assistant["answer_source"]["reasoning_enabled"] is True
    assert assistant["reply_sections"][0]["support_status"] == "insufficient_evidence"


def test_chat_rejects_verbatim_gemini_reply_and_fails_instead_of_saving_fallback(app_factory, bundle, monkeypatch):
    adjusted_bundle = json.loads(json.dumps(bundle))
    adjusted_bundle["items"][0]["text"] = "Regularization adds a penalty term to discourage overly flexible models and reduce overfitting."
    adjusted_bundle["items"][1]["text"] = "Validation is used to tune the strength of regularization against generalization performance."
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service
    verbatim_text = adjusted_bundle["items"][0]["text"]
    responses = [verbatim_text, None]

    def fake_generate_text(system_instruction, user_prompt, max_output_tokens=256):
        service.generator.gemini.last_call_info = {
            "configured": True,
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": "gemini-3-flash-preview",
            "reasoning_enabled": True,
            "reasoning_mode": "dynamic",
            "attempted_models": ["gemini-3-flash-preview"],
            "rate_limited_models": [],
            "failure_reason": None,
        }
        return responses.pop(0)

    monkeypatch.setattr(service.generator.gemini, "generate_text", fake_generate_text)

    create_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": adjusted_bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": adjusted_bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Gemini repetition guard",
            "include_annotations": True,
        },
    )
    assert create_response.status_code == 200
    conversation_id = create_response.json()["conversation_id"]

    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What does regularization do in these lectures?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    job = wait_for_job(client, send.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "primary_generation_fallback_blocked"
    assert job["error"]["retryable"] is False
    assert "verbatim_evidence_repetition" in job["message"]

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assert conversation["messages"] == []


def test_practice_job_fails_when_configured_gemini_would_store_timeout_fallback(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_build_practice_set(**kwargs):
        service.generator.gemini.last_call_info = {
            "configured": True,
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": None,
            "reasoning_enabled": True,
            "reasoning_mode": "dynamic",
            "attempted_models": ["gemini-3-flash-preview", "gemini-2.5-flash"],
            "rate_limited_models": [],
            "failure_reason": "request_timeout",
            "failure_detail": "Read timed out.",
        }
        return {
            "practice_set_id": "practice_fallback_should_not_save",
            "workspace_id": bundle["workspace_id"],
            "created_at": "2026-04-24T12:00:00Z",
            "generation_mode": kwargs["generation_mode"],
            "questions": [],
            "coverage_report": {"considered_slide_count": 0, "cited_slide_count": 0, "uncited_or_skipped_slides": [], "notes": ""},
            "_meta": {"generation_path": "heuristic_fallback"},
        }

    monkeypatch.setattr(service.generator, "build_practice_set", fake_build_practice_set)

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
    assert job["status"] == "failed"
    assert job["error"]["code"] == "primary_generation_fallback_blocked"
    assert job["error"]["retryable"] is True
    assert "request_timeout" in job["message"]
    assert service.store.list_all("practice_sets") == []


def test_practice_revision_fails_when_configured_gemini_would_store_fallback_revision(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service
    seeded = {
        "practice_set_id": "practice_seed",
        "parent_practice_set_id": None,
        "workspace_id": bundle["workspace_id"],
        "created_at": "2026-04-24T12:00:00Z",
        "generation_mode": "short_answer",
        "questions": [
            {
                "question_id": "question_seed_1",
                "question_type": "short_answer",
                "stem": "Explain regularization.",
                "expected_answer": "It adds a penalty term.",
                "rubric": [{"criterion": "Grounded", "description": "Uses lecture evidence", "points": 2}],
                "scoring_guide_text": "Look for the penalty term and overfitting trade-off.",
                "citations": [bundle["items"][0]["citation"]],
                "covered_slides": [bundle["items"][0]["slide_number"]],
                "difficulty": "mixed",
                "estimated_minutes": 6,
            }
        ],
        "coverage_report": {"considered_slide_count": 1, "cited_slide_count": 1, "uncited_or_skipped_slides": [], "notes": "seeded"},
        "_meta": {"generation_path": "gemini"},
    }
    service.store.save("practice_sets", seeded["practice_set_id"], seeded)

    def fake_revise_practice_set(**kwargs):
        service.generator.gemini.last_call_info = {
            "configured": True,
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": "gemini-3-flash-preview",
            "reasoning_enabled": True,
            "reasoning_mode": "dynamic",
            "attempted_models": ["gemini-3-flash-preview"],
            "rate_limited_models": [],
            "failure_reason": "no_usable_question_updates",
            "failure_detail": "Gemini revision did not modify any editable questions.",
        }
        revised = json.loads(json.dumps(kwargs["existing_practice_set"]))
        revised["practice_set_id"] = "practice_seed_revision"
        revised["parent_practice_set_id"] = kwargs["existing_practice_set"]["practice_set_id"]
        revised["_meta"] = {"generation_path": "heuristic_fallback"}
        return revised

    monkeypatch.setattr(service.generator, "revise_practice_set", fake_revise_practice_set)

    response = client.post(
        f"/v1/practice-sets/{seeded['practice_set_id']}/revise",
        json={
            "instruction_text": "Make the question harder.",
            "target_question_ids": [],
            "locked_question_ids": [],
            "maintain_coverage": True,
        },
    )
    assert response.status_code == 200
    job = wait_for_job(client, response.json()["job_id"])
    assert job["status"] == "failed"
    assert job["error"]["code"] == "primary_generation_fallback_blocked"
    assert job["error"]["retryable"] is False
    assert "no_usable_question_updates" in job["message"]
    assert service.store.load("practice_sets", "practice_seed_revision") is None
    assert service.store.load("practice_sets", seeded["practice_set_id"]) is not None


def test_chat_uses_text_path_and_does_not_fall_back_for_normal_llm_reply(app_factory, bundle, monkeypatch, caplog):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_text(system_instruction, user_prompt, max_output_tokens=256):
        service.generator.gemini.last_call_info = {
            "configured": True,
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": "gemini-3-flash-preview",
            "reasoning_enabled": True,
            "reasoning_mode": "dynamic",
            "attempted_models": ["gemini-3-flash-preview"],
            "rate_limited_models": [],
            "failure_reason": None,
            "failure_detail": None,
        }
        return "Regularization discourages overly flexible weights and the lecture ties it to controlling overfitting."

    monkeypatch.setattr(service.generator.gemini, "generate_text", fake_generate_text)
    monkeypatch.setattr(
        service.generator.gemini,
        "generate_json",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("chat should not call generate_json")),
    )
    caplog.set_level(logging.WARNING)

    create_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Gemini invalid payload logging",
            "include_annotations": True,
        },
    )
    assert create_response.status_code == 200
    conversation_id = create_response.json()["conversation_id"]

    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What does regularization do in these lectures?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    job = wait_for_job(client, send.json()["job_id"])
    assert job["status"] == "succeeded"

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assistant = conversation["messages"][-1]
    assert assistant["answer_source"]["path"] == "llm"
    assert assistant["answer_source"]["model"] == "gemini-3-flash-preview"
    assert assistant["reply_sections"][0]["support_status"] in {"slide_grounded", "inferred_from_slides"}
    assert not any("Falling back to deterministic chat generation" in record.getMessage() for record in caplog.records)


def test_chat_keeps_llm_path_for_partial_grounded_answer(app_factory, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service
    bundle = {
        "bundle_id": "bundle_backprop_demo",
        "workspace_id": "ws_backprop_demo",
        "material_ids": ["mat_backprop"],
        "query_text": None,
        "bundle_mode": "precision",
        "items": [
            {
                "item_id": "item_backprop_1",
                "material_id": "mat_backprop",
                "material_title": "Neural Networks Backprop",
                "slide_id": "slide_10",
                "slide_number": 10,
                "text": "Gradient descent updates weights by moving in the direction of the negative gradient after backpropagation computes the needed partial derivatives.",
                "extraction_quality": "high",
                "citation": {
                    "citation_id": "cit_backprop_1",
                    "material_id": "mat_backprop",
                    "material_title": "Neural Networks Backprop",
                    "slide_id": "slide_10",
                    "slide_number": 10,
                    "snippet_text": "Gradient descent updates weights by moving in the direction of the negative gradient.",
                    "support_type": "explicit",
                    "confidence": "high",
                    "preview_url": "http://example.test/preview/10",
                    "source_open_url": "http://example.test/source/10",
                },
            }
        ],
        "summary": {"total_items": 1, "total_slides": 1, "low_confidence_item_count": 0},
    }

    def fake_generate_text(system_instruction, user_prompt, max_output_tokens=256):
        service.generator.gemini.last_call_info = {
            "configured": True,
            "provider": "gemini",
            "generation_path": "llm",
            "used_model": "gemini-3-flash-preview",
            "reasoning_enabled": True,
            "reasoning_mode": "dynamic",
            "attempted_models": ["gemini-3-flash-preview"],
            "rate_limited_models": [],
            "failure_reason": None,
            "failure_detail": None,
        }
        return (
            "The slides tie the update step to moving opposite the gradient. "
            "So the subtraction sign reflects stepping in the loss-reducing direction, even though the lecture emphasizes computing gradients more than unpacking the sign itself."
        )

    monkeypatch.setattr(service.generator.gemini, "generate_text", fake_generate_text)
    monkeypatch.setattr(
        service.generator,
        "expand_chat_retrieval_query",
        lambda message_text, previous_messages=(): f"{message_text} negative gradient gradient descent backpropagation",
    )

    create_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Gemini over-strict insufficiency",
            "include_annotations": True,
        },
    )
    assert create_response.status_code == 200
    conversation_id = create_response.json()["conversation_id"]

    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "Why does the update rule use subtraction?",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    job = wait_for_job(client, send.json()["job_id"])
    assert job["status"] == "succeeded"

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assistant = conversation["messages"][-1]
    assert assistant["answer_source"]["path"] == "llm"
    assert assistant["reply_sections"][0]["support_status"] in {"slide_grounded", "inferred_from_slides"}
    assert "opposite the gradient" in assistant["reply_sections"][0]["text"].lower()


def test_gemini_client_handles_incompatible_models_and_wrapped_json(monkeypatch):
    client = GeminiPrimaryClient(Settings(gemini_api_key="test-key"))
    requests_seen = []

    class FakeResponse:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        @property
        def ok(self):
            return self.status_code < 400

        def json(self):
            return self._payload

    def fake_post(url, headers=None, json=None, timeout=None):
        model = url.split("/models/")[1].split(":generateContent")[0]
        thinking_config = (json or {}).get("generationConfig", {}).get("thinkingConfig")
        requests_seen.append((model, thinking_config))
        if model == "gemini-3-flash-preview":
            return FakeResponse(404, {"error": {"message": "model not found"}})
        if model == "gemini-2.5-flash" and thinking_config:
            return FakeResponse(400, {"error": {"message": "thinkingConfig is not supported"}})
        return FakeResponse(
            200,
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "```json\n{\"reply_sections\": []}\n```"},
                            ]
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("learning_service.generation.requests.post", fake_post)

    payload = client.generate_json("system", "user prompt")

    assert payload == {"reply_sections": []}
    assert requests_seen == [
        ("gemini-3-flash-preview", {"thinkingLevel": "high"}),
        ("gemini-2.5-flash", {"thinkingBudget": -1}),
        ("gemini-2.5-flash", None),
    ]
    assert client.last_call_info["used_model"] == "gemini-2.5-flash"
    assert client.last_call_info["reasoning_enabled"] is False


def test_gemini_client_records_failure_detail_for_non_json_text(monkeypatch):
    client = GeminiPrimaryClient(Settings(gemini_api_key="test-key"))

    class FakeResponse:
        status_code = 200
        text = "plain text"

        @property
        def ok(self):
            return True

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "This is not JSON"},
                            ]
                        }
                    }
                ]
            }

    monkeypatch.setattr("learning_service.generation.requests.post", lambda *args, **kwargs: FakeResponse())

    payload = client.generate_json("system", "user prompt")

    assert payload is None
    assert client.last_call_info["failure_reason"] == "invalid_response_json"
    assert client.last_call_info["failure_detail"] == "Gemini returned text that could not be parsed as JSON."
