"""Contact details: the phone number and the unit the prospect wants.

A site visit was booked for a Sunday and the prospect was unreachable, because the dialled
number was only ever written to a log line. And two prospects named a unit type out loud
("2 BHK also works fine for me") which was discarded.
"""

import ast
import inspect

import pytest

from app.models.db import Call, Lead
from app.models.schemas import LeadExtraction
from app.worker import _match_unit_type, _normalise

CONFIGS = [
    {"type": "2 BHK", "area": "1200 sqft", "price": "1.2 Cr"},
    {"type": "3 BHK", "area": "1600 sqft", "price": "1.8 Cr"},
    {"type": "Villament", "area": "2500 sqft", "price": "3.5 Cr"},
]


class _Project:
    config_json = CONFIGS


class _Result:
    def scalars(self):
        return self

    def first(self):
        return _Project()


class _Session:
    async def execute(self, *a, **kw):
        return _Result()


class _NoProjectSession:
    async def execute(self, *a, **kw):
        class R:
            def scalars(self):
                return self

            def first(self):
                return None

        return R()


# --- schema carries the fields ---------------------------------------------------


def test_call_and_lead_both_store_a_phone_number():
    assert hasattr(Call, "phone_number")
    assert hasattr(Lead, "phone_number")


def test_lead_records_when_it_was_captured():
    assert hasattr(Lead, "created_at")


def test_extraction_asks_for_the_unit_type():
    assert "preferred_unit_type" in LeadExtraction.model_fields


# --- unit normalisation ----------------------------------------------------------


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("2 BHK", "2 BHK"),
        ("2bhk", "2 BHK"),
        ("2-BHK", "2 BHK"),
        # Still normalised even though the agent no longer says it this way: transcripts
        # from calls placed before the change are re-extracted, and STT can space it too.
        ("2 B H K", "2 BHK"),
        ("  3 bhk  ", "3 BHK"),
        ("villament", "Villament"),
        ("2 BHK apartment", "2 BHK"),
    ],
)
async def test_spoken_unit_maps_onto_the_project_configuration(spoken, expected):
    assert await _match_unit_type(_Session(), "cid", spoken, "sid") == expected


@pytest.mark.parametrize("spoken", [None, "", "5 BHK", "something nice", "whatever you have"])
async def test_unknown_units_are_dropped_not_stored(spoken):
    """Storing a unit the project does not sell is worse than storing nothing.

    "5 BHK" stays here on purpose even though the project has BHKs: a bedroom count is this
    project's own vocabulary, and one it does not offer is the model rounding off rather
    than the prospect naming something else.
    """
    assert await _match_unit_type(_Session(), "cid", spoken, "sid") is None


@pytest.mark.parametrize(
    "spoken,expected",
    [("penthouse", "Penthouse"), ("plots", "Plot"), ("a villa maybe", "Villa")],
)
async def test_a_property_type_this_project_lacks_is_kept_for_the_portfolio(spoken, expected):
    """Asked "what kind of property are you looking for?" a prospect answered "plots". It
    matched none of the project's BHK configurations and was stored as null — discarding
    the one fact a colleague needed to call him back about something else. That question is
    only ever asked once this project has already been ruled out.
    """
    assert await _match_unit_type(_Session(), "cid", spoken, "sid") == expected


async def test_missing_project_does_not_crash_extraction():
    assert await _match_unit_type(_NoProjectSession(), "cid", "2 BHK", "sid") is None


def test_normalise_collapses_punctuation_and_case():
    assert _normalise("2 B-H.K") == _normalise("2bhk") == "2bhk"


# --- wiring ----------------------------------------------------------------------


def test_dial_records_the_number_before_dialling():
    """Compares the calls themselves, not their positions in the text — a comment naming
    trigger_vobiz_call used to be enough to fail this."""
    import ast

    from app.api.routes import campaign

    tree = ast.parse(inspect.getsource(campaign.dial_campaign_vobiz).lstrip())

    def line_of(target: str) -> int:
        # trigger_vobiz_call is handed to add_task as a bare reference, never called here,
        # so match any mention of the name and order by line rather than by walk order.
        lines = [
            n.lineno
            for n in ast.walk(tree)
            if (isinstance(n, ast.Name) and n.id == target)
            or (isinstance(n, ast.Attribute) and n.attr == target)
        ]
        assert lines, f"{target} is not referenced at all"
        return min(lines)

    assert line_of("remember_dialed_number") < line_of("trigger_vobiz_call"), (
        "the number must be recorded before the dial, not after"
    )


def test_call_row_captures_the_number():
    from app.api.routes import webhook

    src = inspect.getsource(webhook._handle_call)
    assert "phone_number=await recall_dialed_number(call_sid)" in src


def test_lead_inherits_the_number_from_its_call():
    from app import worker

    tree = ast.parse(inspect.getsource(worker.process_extraction).lstrip())
    lead = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Lead"
    )
    kwargs = {kw.arg for kw in lead.keywords}
    assert "phone_number" in kwargs, "a lead with no number cannot be called back"
    assert "preferred_unit_type" in kwargs


class _AmbiguousSession:
    """A project selling both '2 BHK' and '2 BHK Premium'."""

    async def execute(self, *a, **kw):
        class P:
            config_json = [{"type": "2 BHK Premium"}, {"type": "2 BHK"}]

        class R:
            def scalars(self):
                return self

            def first(self):
                return P()

        return R()


async def test_exact_match_wins_over_a_longer_substring():
    """Substring matching alone would sell "2 BHK Premium" to someone who said "2 BHK"."""
    assert await _match_unit_type(_AmbiguousSession(), "cid", "2 BHK", "sid") == "2 BHK"


async def test_substring_still_resolves_a_qualified_phrase():
    got = await _match_unit_type(_AmbiguousSession(), "cid", "the 2 BHK Premium one", "sid")
    assert got == "2 BHK Premium"
