from __future__ import annotations

import copy

from .conftest import wait_for_job



def create_conversation(client, bundle, grounding_mode="lecture_with_fallback", **overrides):
    response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": bundle["workspace_id"],
            "material_ids": overrides.get("material_ids"),
            "evidence_bundle": overrides.get("evidence_bundle", bundle),
            "grounding_mode": grounding_mode,
            "title": "Linear models QA",
            "include_annotations": True,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["conversation_id"]



def test_conversation_grounded_answer_clear_and_reuse(client, bundle):
    conversation_id = create_conversation(client, bundle)

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
    assert job["status"] == "succeeded", job

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assert len(conversation["messages"]) == 2
    assistant_message = conversation["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["reply_sections"][0]["citations"]
    assert assistant_message["answer_source"]["path"] == "heuristic_fallback"
    assert assistant_message["answer_source"]["provider"] == "deterministic_fallback"

    cleared = client.post(f"/v1/conversations/{conversation_id}/clear")
    assert cleared.status_code == 200
    assert cleared.json()["cleared"] is True

    after_clear = client.get(f"/v1/conversations/{conversation_id}").json()
    assert after_clear["messages"] == []

    resend = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "How is validation used?",
            "response_style": "concise",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert resend.status_code == 200
    resend_job = wait_for_job(client, resend.json()["job_id"])
    assert resend_job["status"] == "succeeded"



def test_conversation_strict_mode_returns_insufficient_evidence_without_fabrication(client, bundle):
    conversation_id = create_conversation(client, bundle, grounding_mode="strict_lecture_only")
    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "What is the history of neural networks?",
            "response_style": "standard",
            "grounding_mode": "strict_lecture_only",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    job = wait_for_job(client, send.json()["job_id"])
    assert job["status"] == "succeeded"

    conversation = client.get(f"/v1/conversations/{conversation_id}").json()
    assistant = conversation["messages"][-1]
    first_section = assistant["reply_sections"][0]
    assert first_section["support_status"] == "insufficient_evidence"
    assert first_section["citations"] == []
    assert assistant["answer_source"]["path"] == "heuristic_fallback"
    assert assistant["answer_source"]["fallback_reason"] == "gemini_not_configured"



def test_conversation_vague_question_can_request_clarification(client, bundle):
    conversation_id = create_conversation(client, bundle)
    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "this",
            "response_style": "standard",
            "grounding_mode": "lecture_with_fallback",
            "include_citations": True,
        },
    )
    assert send.status_code == 200
    job = wait_for_job(client, send.json()["job_id"])
    assert job["status"] == "needs_user_input"
    assert job["user_action"]["kind"] == "clarification"
    assert job["user_action"]["options"]


def test_conversation_definition_answer_avoids_slide_header_noise(client, bundle):
    noisy_bundle = copy.deepcopy(bundle)
    noisy_bundle["items"][0]["text"] = (
        "RNN LMs + Transformer LMs 1 10-301/10-601 Introduction to Machine Learning "
        "Pat Virtue & Matt Gormley Lecture 18 Mar. 18, 2026 Machine Learning Department "
        "School of Computer Science Carnegie Mellon University"
    )
    noisy_bundle["items"][1]["text"] = "ATTENTION The key building block for Transformer language models 82"
    noisy_bundle["items"][2]["text"] = "TRANSFORMER LANGUAGE MODELS Generative Pretrained Transformers (GPT) 103"
    conversation_id = create_conversation(client, noisy_bundle, evidence_bundle=noisy_bundle)

    send = client.post(
        f"/v1/conversations/{conversation_id}/messages",
        json={
            "message_text": "what is a transformer?",
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
    grounded = assistant["reply_sections"][0]["text"].lower()
    assert "transformer" in grounded
    assert "10-301" not in grounded
    assert "carnegie mellon university" not in grounded


def test_conversation_fallback_paraphrases_rather_than_copying_slide_text(client, bundle):
    paraphrase_bundle = copy.deepcopy(bundle)
    verbatim_sentence = "Regularization adds a penalty term to discourage overly flexible models and reduce overfitting."
    paraphrase_bundle["items"][0]["text"] = verbatim_sentence
    paraphrase_bundle["items"][1]["text"] = "Validation is used to tune the strength of regularization against generalization performance."
    conversation_id = create_conversation(client, paraphrase_bundle, evidence_bundle=paraphrase_bundle)

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
    grounded = conversation["messages"][-1]["reply_sections"][0]["text"].lower()
    assert "regularization" in grounded
    assert verbatim_sentence.lower() not in grounded


def test_conversation_expands_retrieval_query_for_update_rule_questions(app_factory, monkeypatch):
    client = app_factory(gemini_api_key="")
    service = client.app.state.learning_service
    captured = {}
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

    def fake_fetch_evidence_bundle(*, workspace_id, material_ids, query_text, bundle_mode, include_annotations):
        captured["query_text"] = query_text
        return copy.deepcopy(bundle)

    monkeypatch.setattr(service.content_client, "fetch_evidence_bundle", fake_fetch_evidence_bundle)

    create_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": "ws_backprop_demo",
            "material_ids": ["mat_backprop"],
            "grounding_mode": "lecture_with_fallback",
            "title": "Backprop chat",
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
    assert "negative gradient" in captured["query_text"]


def test_conversation_can_give_grounded_partial_answer_for_update_rule_subtraction(app_factory, monkeypatch):
    client = app_factory(gemini_api_key="")
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
            },
            {
                "item_id": "item_backprop_2",
                "material_id": "mat_backprop",
                "material_title": "Neural Networks Backprop",
                "slide_id": "slide_11",
                "slide_number": 11,
                "text": "Backpropagation computes the partial derivatives for each weight so the network can apply the update step layer by layer.",
                "extraction_quality": "high",
                "citation": {
                    "citation_id": "cit_backprop_2",
                    "material_id": "mat_backprop",
                    "material_title": "Neural Networks Backprop",
                    "slide_id": "slide_11",
                    "slide_number": 11,
                    "snippet_text": "Backpropagation computes the partial derivatives for each weight.",
                    "support_type": "explicit",
                    "confidence": "high",
                    "preview_url": "http://example.test/preview/11",
                    "source_open_url": "http://example.test/source/11",
                },
            },
        ],
        "summary": {"total_items": 2, "total_slides": 2, "low_confidence_item_count": 0},
    }
    monkeypatch.setattr(
        service.generator,
        "expand_chat_retrieval_query",
        lambda message_text, previous_messages=(): f"{message_text} negative gradient gradient descent backpropagation",
    )

    create_response = client.post(
        "/v1/conversations",
        json={
            "workspace_id": "ws_backprop_demo",
            "material_ids": None,
            "evidence_bundle": bundle,
            "grounding_mode": "lecture_with_fallback",
            "title": "Backprop chat",
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
    first_section = assistant["reply_sections"][0]
    assert first_section["support_status"] in {"slide_grounded", "inferred_from_slides"}
    assert "opposite the gradient" in first_section["text"].lower()
    assert "subtraction sign" in first_section["text"].lower()
