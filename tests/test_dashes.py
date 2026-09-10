"""Dashes the voice engine misreads, turned into punctuation it honours.

Live call 2eeb48a0, 10 Sep 2026: "Varthur – Sarjapur Road. It is called Abhee Codename New
Dimension – Bengaluru's first Scotland‑themed residential…" — two en-dashes and a
non-breaking hyphen, straight out of the campaign context, and the prospect said "Sorry I
didn't catch that."
"""

import asyncio
import inspect

import pytest

from app.utils.dashes import DashFilter, spoken_punctuation

LIVE = (
    "We are launching a new project in Varthur – Sarjapur Road. "
    "It is called Abhee Codename New Dimension – Bengaluru's first Scotland‑themed residential township."
)


def test_the_live_line_comes_out_in_commas_and_hyphens():
    assert spoken_punctuation(LIVE) == (
        "We are launching a new project in Varthur, Sarjapur Road. "
        "It is called Abhee Codename New Dimension, Bengaluru's first Scotland-themed residential township."
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A – B", "A, B"),
        ("A—B", "A, B"),
        ("A — B", "A, B"),
        ("Scotland‑themed", "Scotland-themed"),
        ("2‐3 BHK", "2-3 BHK"),
        ("Prices start at 1.17 Cr — for a 2 BHK.", "Prices start at 1.17 Cr, for a 2 BHK."),
    ],
)
def test_each_kind_of_dash(text, expected):
    assert spoken_punctuation(text) == expected


def test_a_dash_after_a_comma_does_not_double_it():
    assert spoken_punctuation("Yes, — of course") == "Yes, of course"


def test_a_dash_opening_a_sentence_is_dropped():
    assert spoken_punctuation("— Sure, I understand.") == "Sure, I understand."


@pytest.mark.parametrize("text", ["", None, "No dashes here.", "A well-known builder."])
def test_ordinary_text_is_untouched(text):
    assert spoken_punctuation(text) == text


def test_the_pipecat_filter_applies_it():
    assert asyncio.run(DashFilter().filter(LIVE)) == spoken_punctuation(LIVE)


def test_the_voice_engine_is_built_with_it():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "text_filters=[DashFilter()]" in src
