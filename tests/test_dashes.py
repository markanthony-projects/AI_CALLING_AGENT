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
        ("Prices start at 1.17 Cr — for a 2 BHK.", "Prices start at 1.17 Cr, for a 2 BHK."),
    ],
)
def test_each_kind_of_dash(text, expected):
    assert spoken_punctuation(text) == expected


# --- a range is not a comma ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("about 20–30 Lakhs below launch", "about 20 to 30 Lakhs below launch"),
        ("about 20‑30 Lakhs below launch", "about 20 to 30 Lakhs below launch"),
        ("about 20-30 Lakhs below launch", "about 20 to 30 Lakhs below launch"),
        ("It is 1.17–2.64 Crores", "It is 1.17 to 2.64 Crores"),
        ("2‐3 BHK", "2 to 3 BHK"),
        ("20 – 30 Lakhs", "20 to 30 Lakhs"),
    ],
)
def test_a_dash_between_numbers_is_the_word_to(text, expected):
    """The first version of this file made these commas. "20, 30 Lakhs below launch" is two
    figures where the prospect was told one span, and "1.17, 2.64 Crores" is two unrelated
    prices — the money said wrong, in the one part of the pitch that cannot be wrong."""
    assert spoken_punctuation(text) == expected


def test_a_spaced_hyphen_between_words_is_a_comma():
    """From call 6a58a7f4: qwen writes "Varthur - Sarjapur Road" where gpt-oss wrote an
    en-dash, and a plain hyphen went to the engine untouched. Spaces are the whole test —
    a dash standing alone between two words is a clause break to the ear."""
    assert spoken_punctuation("We are launching a new project in Varthur - Sarjapur Road.") == (
        "We are launching a new project in Varthur, Sarjapur Road."
    )
    assert spoken_punctuation("a well-known builder - a good one") == "a well-known builder, a good one"


def test_a_hyphen_between_a_number_and_a_word_stays_a_hyphen():
    """"3-acre golf course" is not a range."""
    assert spoken_punctuation("a 3-acre golf course") == "a 3-acre golf course"
    assert spoken_punctuation("a 3‑acre golf course") == "a 3-acre golf course"


def test_the_range_rule_runs_before_the_clause_rule():
    """An en-dash between numbers must be read as a range, not caught by the clause rule
    first and turned into a comma."""
    assert spoken_punctuation("Varthur – Sarjapur, 20–30 Lakhs off") == "Varthur, Sarjapur, 20 to 30 Lakhs off"


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
