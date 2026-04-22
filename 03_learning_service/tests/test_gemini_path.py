from __future__ import annotations

import json
import logging

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


def test_chat_uses_gemini_path_even_without_strong_local_match(app_factory, bundle, monkeypatch):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
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
        return {
            "reply_sections": [
                {
                    "heading": "Not covered here",
                    "text": "The current lecture materials do not cover that topic directly.",
                    "support_status": "insufficient_evidence",
                    "citation_ids": [],
                }
            ]
        }

    monkeypatch.setattr(service.generator.gemini, "generate_json", fake_generate_json)

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


def test_chat_rejects_verbatim_gemini_reply_and_falls_back(app_factory, bundle, monkeypatch):
    adjusted_bundle = json.loads(json.dumps(bundle))
    adjusted_bundle["items"][0]["text"] = "Regularization adds a penalty term to discourage overly flexible models and reduce overfitting."
    adjusted_bundle["items"][1]["text"] = "Validation is used to tune the strength of regularization against generalization performance."
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service
    verbatim_text = adjusted_bundle["items"][0]["text"]
    citation_id = adjusted_bundle["items"][0]["citation"]["citation_id"]

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
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
        return {
            "reply_sections": [
                {
                    "heading": "Grounded answer",
                    "text": verbatim_text,
                    "support_status": "slide_grounded",
                    "citation_ids": [citation_id],
                }
            ]
        }

    monkeypatch.setattr(service.generator.gemini, "generate_json", fake_generate_json)

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
    assert job["status"] == "succeeded"

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assistant = conversation["messages"][-1]
    assert assistant["answer_source"]["path"] == "heuristic_fallback"
    assert assistant["answer_source"]["fallback_reason"] == "verbatim_evidence_repetition"
    assert assistant["reply_sections"][0]["text"] != verbatim_text


def test_chat_invalid_structured_output_logs_detail_and_surfaces_debug_info(app_factory, bundle, monkeypatch, caplog):
    client = app_factory(gemini_api_key="test-key")
    service = client.app.state.learning_service

    def fake_generate_json(system_instruction, user_prompt, max_output_tokens=2048):
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
        return {
            "reply_sections": [
                {
                    "heading": "Grounded answer",
                    "text": "Regularization adds a penalty.",
                    "support_status": "slide_grounded",
                    "citation_ids": ["not_in_bundle"],
                }
            ]
        }

    monkeypatch.setattr(service.generator.gemini, "generate_json", fake_generate_json)
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
    assert assistant["answer_source"]["path"] == "heuristic_fallback"
    assert assistant["answer_source"]["fallback_reason"] == "invalid_response_json_payload"
    assert "did not reference any allowed citation_ids" in assistant["answer_source"]["fallback_detail"]
    assert assistant["answer_source"]["attempted_models"] == ["gemini-3-flash-preview"]
    assert any("Rejecting Gemini chat reply output" in record.getMessage() for record in caplog.records)
    assert any("Falling back to deterministic chat generation" in record.getMessage() for record in caplog.records)


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
