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
