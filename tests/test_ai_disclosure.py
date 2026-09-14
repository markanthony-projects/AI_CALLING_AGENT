"""The prospect is told they are talking to software, in the first sentence, every call.

TRAI's February 2025 amendment to the TCCCPR makes the disclosure mandatory on automated
commercial calls, and a channel partner placing promotional calls is a sender under it.
It was the oldest open item on the production plan, blocked on wording; it is now a
setting with a default, so the wording can change without a release and the disclosure
cannot be forgotten by one.

It lives in the introduction sentence rather than in the prompt, because the greeting is
played by the system and a rule in the prompt is a suggestion. And the prompt is told the
same sentence, from the same function, so the model's own introduction — used when the
prospect speaks first and the greeting is cancelled — carries it too.
"""

from datetime import datetime

import pytest

from app.core.config import settings
from app.prompts.agent_prompts import AGENT_NAME, get_system_prompt, introduction
from app.services.agent import build_opening_line, build_reintroduction
from app.utils.sentences import sentences

MORNING = datetime(2026, 9, 14, 9, 30)
DEFAULT = "an AI assistant"


def test_the_default_wording_is_set():
    assert settings.AI_DISCLOSURE == DEFAULT


def test_the_greeting_discloses_before_it_asks_anything():
    line = build_opening_line("Abhee New Dimension", "Rahul", MORNING)
    assert f"My name is {AGENT_NAME}, {DEFAULT}, and I am calling you from" in line
    assert line.index(DEFAULT) < line.index("Am I speaking with")


def test_the_reintroduction_discloses_too():
    """Said when their first words are "Hello?" — the disclosure they missed is the one
    thing the re-introduction exists to repeat."""
    assert DEFAULT in build_reintroduction("X", "Rahul", "Abhee Ventures")


def test_the_prompt_tells_the_model_the_same_sentence():
    prompt = get_system_prompt("Project Name: X", "Rahul")
    assert f"My name is {AGENT_NAME}, {DEFAULT}, and I am calling you from" in prompt


def test_a_custom_wording_reaches_the_greeting_and_the_prompt(monkeypatch):
    monkeypatch.setattr(settings, "AI_DISCLOSURE", "a virtual assistant")
    assert "Priya, a virtual assistant, and I am calling" in build_opening_line("X", "Rahul", MORNING)
    assert "Priya, a virtual assistant, and I am calling" in get_system_prompt("Project Name: X")


def test_blank_omits_the_clause_cleanly(monkeypatch):
    """Blank is allowed and logged loudly at startup; what it must not do is leave a
    double comma in the one sentence the prospect uses to decide whether this is a
    person."""
    monkeypatch.setattr(settings, "AI_DISCLOSURE", "")
    line = build_opening_line("X", "Rahul", MORNING)
    assert "My name is Priya, and I am calling you from X." in line
    assert ",," not in line and ", ," not in line


def test_the_explicit_argument_wins_over_the_setting():
    assert introduction("Priya", "X", "an AI caller") == (
        "My name is Priya, an AI caller, and I am calling you from X."
    )
    assert introduction("Priya", "X", "") == "My name is Priya, and I am calling you from X."


def test_the_disclosure_does_not_add_a_sentence_to_the_greeting():
    """One clause inside the introduction, not a fourth sentence. A separate "Please note
    this is an AI call" is the line people hang up on; a clause is said and moved past."""
    line = build_opening_line("Abhee New Dimension", "Rahul", MORNING, developer_name="Abhee")
    assert len(sentences(line)) == 3
