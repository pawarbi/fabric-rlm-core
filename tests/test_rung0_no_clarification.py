"""Prompt hardening: rung-0 must not submit clarification requests.

Trajectory analysis of the 2026-05-03 6/25 Fabric run found rung-0 attempts
submitting "Acknowledged. Please confirm..." as the answer, and the
reflection turn marked it REFLECTION_OK. Both the system prompt and the
reflection prompt now explicitly call this out.

These guards are universal — no template names, no domain-specific terms.
"""

from __future__ import annotations

import pytest

from fabric_rlm.prompts import build_reflection_prompt, build_system_prompt


# -------- system prompt --------


def test_system_prompt_contains_no_clarification_rule():
    sp = build_system_prompt(
        inline_task="solve the problem",
        inline_outputs=["answer"],
        inputs={"question": "Q1: a  Q2: b  Q3: c"},
    )
    assert "Answering rules" in sp
    assert "Do NOT submit clarification requests" in sp
    # Mentions canonical opener phrases
    assert "Acknowledged" in sp
    assert "Please confirm" in sp
    assert "I need more information" in sp


def test_system_prompt_contains_multipart_shape_rule():
    sp = build_system_prompt(
        inline_task="solve",
        inline_outputs=["answer"],
        inputs={"q": "anything"},
    )
    assert "ONE answer per sub-question" in sp
    assert "Partial answers" in sp
    assert "rejected" in sp.lower()


def test_system_prompt_still_renders_basics():
    """Backward compat: existing required content still present."""
    sp = build_system_prompt(
        inline_task="solve",
        inline_outputs=["answer", "report"],
        inputs={"q": "Q?"},
    )
    assert "RLM" in sp
    assert "SUBMIT" in sp
    assert "answer" in sp
    assert "report" in sp


# -------- reflection prompt --------


def test_reflection_prompt_includes_clarification_check():
    prompt = build_reflection_prompt(
        submitted_payload={"answer": "ok"},
        original_question="Q1: a Q2: b Q3: c",
    )
    assert "clarification request" in prompt
    assert "Acknowledged" in prompt
    assert "Please confirm" in prompt
    assert "INVALID — re-SUBMIT" in prompt


def test_reflection_prompt_includes_shape_check():
    prompt = build_reflection_prompt(
        submitted_payload={"answer": [1, 2, 3]},
        original_question="Q1..Q50",
    )
    assert "enumerates N sub-questions" in prompt
    assert "exactly N items" in prompt
    assert "Short or partial answers" in prompt


def test_reflection_prompt_keeps_existing_structure():
    """Backward compat: existing instructions still rendered."""
    prompt = build_reflection_prompt(
        submitted_payload={"answer": "x"},
        original_question="solve",
    )
    assert "REFLECTION_OK" in prompt
    assert "<payload>" in prompt
    assert "invariants" in prompt
    # Numbered steps preserved
    assert "1." in prompt
    assert "5." in prompt or "6." in prompt


def test_reflection_prompt_repro_canonical_failure_case():
    """The failure case: rung-0 'Acknowledged...' and reflection said OK.

    With the new prompt the model is explicitly told this is invalid.
    """
    prompt = build_reflection_prompt(
        submitted_payload={"answer": "Acknowledged. Please confirm the target."},
        original_question="solve a 50-part problem",
    )
    # Guidance to recognise this exact pattern
    assert "Acknowledged" in prompt
    assert "Please confirm" in prompt
    # And what to do about it
    assert "concrete attempt" in prompt
