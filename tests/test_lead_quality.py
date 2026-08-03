"""What ends up on a lead, and whether the prospect actually said it.

Two live calls put the agent's own pitch on the record as the caller's requirements:

    Agent:    ...starting at 1.17 Crores.
    Prospect: Yeah, that lie in my budget.
    -> budget = 11700000, preferred_location = 'Varthur - Sarjapur Road'

Neither number nor locality was ever spoken by the prospect. The extraction prompt forbids
exactly this, in capitals, with worked examples — and gpt-4o-mini did it anyway, twice. So
the rule is expressed as code here, where it can be tested and cannot quietly stop working.

A third call threw away the one fact worth keeping: asked what he was looking for, the
prospect said "plots", which matched none of this project's BHK configurations and was
stored as null. That question is only ever asked once the pitch has already failed, so its
answer is by definition about a different project.
"""

import ast
import asyncio
import inspect
import time

import pytest

from app.models.db import CallStatus, Purpose
from app.models.schemas import LeadExtraction
from app.utils.answering_machine import is_answering_machine, machine_phrases
from app.utils.attribution import (
    budget_is_grounded,
    money_in_rupees,
    phrase_is_grounded,
    prospect_text,
)

# Verbatim from the call that produced the fabricated lead.
FABRICATED = """Agent: Nice to meet you, Rahul. We are launching a new project in Varthur - Sarjapur Road, near Dommasandra Circle. Are you looking for any property purchase?
Prospect: Yeah, I was very much looking for that.
Agent: That's great, Rahul. Our project has 2, 3, 3.5 and 4.5 BHK configurations, starting at 1.17 Crores. Does this sound interesting to you?
Prospect: Yeah, that lie in my budget."""

# ...and from the two where the prospect really did give a figure.
STATED = """Agent: What budget are you thinking of?
Prospect: Below 60 lakhs.
Agent: Which area are you looking in?
Prospect: I am looking South Bangalore."""

RANGE = """Agent: What budget are you thinking of for your investment?
Prospect: My budget was around 70 to 80 lakhs."""


# --- separating the two speakers ----------------------------------------------------


def test_only_the_prospect_counts():
    said = prospect_text(FABRICATED)
    assert "1.17 Crores" not in said, "the asking price is the agent's line"
    assert "Varthur" not in said
    assert "that lie in my budget" in said


def test_an_unattributed_line_is_not_credited_to_the_prospect():
    """Attributing unknown text to the prospect is the exact failure being prevented."""
    assert prospect_text("stray text with no speaker\nAgent: hello") == ""


# --- money ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Below 60 lakhs.", 6_000_000),
        ("around 1.17 Crores", 11_700_000),
        ("about 75 lakh", 7_500_000),
        ("2 cr maybe", 20_000_000),
    ],
)
def test_sums_of_money_are_read_in_rupees(text, expected):
    assert expected in money_in_rupees(text)


def test_both_ends_of_a_range_are_read():
    """"70 to 80 lakhs" carries its unit only once. Reading the lower bound as unitless
    left the prospect recorded at the top of their range."""
    assert set(money_in_rupees("around 70 to 80 lakhs")) >= {7_000_000, 8_000_000}


# --- the two fabrications -------------------------------------------------------------


def test_the_asking_price_is_not_the_callers_budget():
    """Sales would otherwise ring someone back believing they can spend 1.17 Crores they
    never mentioned."""
    assert not budget_is_grounded(11_700_000.0, FABRICATED)


def test_the_projects_own_locality_is_not_a_stated_preference():
    assert not phrase_is_grounded("Varthur - Sarjapur Road", FABRICATED)


# --- and the things that must survive -------------------------------------------------


def test_a_figure_the_prospect_named_is_kept():
    assert budget_is_grounded(6_000_000.0, STATED)


def test_a_locality_the_prospect_named_is_kept_despite_different_spelling():
    """They said "South Bangalore"; the model writes "South Bengaluru". Same answer."""
    assert phrase_is_grounded("South Bengaluru", STATED)


@pytest.mark.parametrize("budget", [7_000_000.0, 7_500_000.0, 8_000_000.0])
def test_anything_inside_a_stated_range_is_kept(budget):
    """A model reporting the midpoint of "70 to 80 lakhs" has read the prospect correctly."""
    assert budget_is_grounded(budget, RANGE)


@pytest.mark.parametrize("budget", [6_500_000.0, 12_000_000.0])
def test_a_figure_outside_the_stated_range_is_not(budget):
    assert not budget_is_grounded(budget, RANGE)


def test_a_null_value_is_not_a_violation():
    assert budget_is_grounded(None, FABRICATED)
    assert phrase_is_grounded(None, FABRICATED)


def test_a_value_with_nothing_long_enough_to_check_is_left_alone():
    """"2 BHK" is all short tokens. Unverifiable is not the same as wrong, and discarding
    it would be a second bug on top of the first."""
    assert phrase_is_grounded("2 BHK", FABRICATED)


def test_a_generic_geography_word_is_not_evidence():
    """"Road" clears any sane length threshold and proves nothing. Without this, the
    fabricated "Varthur - Sarjapur Road" is grounded by a prospect who happened to say
    "the road was busy" — the same bug, only harder to see."""
    assert not phrase_is_grounded(
        "Varthur - Sarjapur Road", "Prospect: sorry, the road was busy near my area"
    )


def test_a_locality_never_mentioned_is_not_grounded():
    assert not phrase_is_grounded("Whitefield", "Prospect: I want a flat")


def test_a_value_made_only_of_generic_words_is_left_alone():
    """Nothing distinctive to check. Unverifiable is not the same as wrong."""
    assert phrase_is_grounded("Main Road", "Prospect: yes")


def test_a_compass_point_still_counts_as_a_place_name():
    """"South Bangalore" is how people say where they want to live. Treating "south" as
    noise would drop a preference the prospect really stated — and would have dropped the
    one on the call this was built from."""
    assert phrase_is_grounded("South Bengaluru", "Prospect: I am looking South Bangalore.")
    assert not phrase_is_grounded("South Bengaluru", "Prospect: yes, sounds good")


# --- the worker applies it ------------------------------------------------------------


def _worker_source(name: str) -> str:
    from app import worker

    return inspect.getsource(getattr(worker, name))


def _checked_fields() -> set[str]:
    """The fields _drop_ungrounded actually verifies, read off its `checks` tuple.

    Parsed rather than grepped: the docstring names customer_name while explaining why it
    is exempt, so a substring search reports the opposite of the truth.
    """
    tree = ast.parse(_worker_source("_drop_ungrounded").lstrip())
    checks = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "checks" for t in n.targets)
    )
    return {
        e.elts[0].value
        for e in checks.elts
        if isinstance(e, ast.Tuple) and isinstance(e.elts[0], ast.Constant)
    }


def test_the_worker_drops_ungrounded_fields():
    assert _checked_fields() == {"budget", "preferred_location", "preferred_unit_type"}


def test_the_name_is_not_grounded_against_the_transcript():
    """It comes off the dial list. The prospect never has to say it, and a call where they
    only said "Yes" would lose it."""
    assert "customer_name" not in _checked_fields()


def test_grounding_runs_before_the_lead_is_built():
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    assert src.index("_drop_ungrounded") < src.index("lead = Lead(")


def test_grounding_uses_the_transliterated_transcript():
    """The extracted values come back in Latin script. Checked against a Devanagari
    transcript nothing would match and every Hindi answer would be discarded."""
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    call = src[src.index("_drop_ungrounded(") : src.index("if lead_data.transliterated_transcript")]
    assert "transliterated_transcript" in call


# --- what a prospect wants that we do not sell ----------------------------------------


def test_plots_survive_a_project_that_only_sells_flats():
    """The single most useful fact on that record, and it was stored as null."""
    from app.worker import _generic_unit_type

    assert _generic_unit_type("plots") == "Plot"


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("plot", "Plot"),
        ("So I'm looking plots", "Plot"),
        ("a villa maybe", "Villa"),
        ("independent house", "Independent House"),
        ("flat", "Apartment"),
    ],
)
def test_property_types_outside_this_project_are_recognised(spoken, expected):
    from app.worker import _generic_unit_type

    assert _generic_unit_type(spoken) == expected


@pytest.mark.parametrize("spoken", ["something nice", "whatever you have", ""])
def test_free_text_that_names_no_property_type_is_still_dropped(spoken):
    """The fallback must not turn every unmatched string into a unit type."""
    from app.worker import _generic_unit_type

    assert _generic_unit_type(spoken) is None


def test_the_projects_own_configurations_still_win():
    """A prospect asking for a 3 BHK must get the project's exact name for it, not the
    generic bucket — that is the whole reason _match_unit_type exists."""
    from app import worker

    src = inspect.getsource(worker._match_unit_type)
    assert src.index("_normalise(unit) == target") < src.index("_generic_unit_type")


# --- purpose and timeline -------------------------------------------------------------


def test_purpose_is_capturable():
    """Volunteered on two calls — "this was for my investment", "looking for an
    investment" — and discarded both times for want of a column."""
    assert "purpose" in LeadExtraction.model_fields
    assert [p.value for p in Purpose] == ["SELF_USE", "INVESTMENT"]


def test_the_timeline_is_kept_as_a_number_as_well_as_words():
    """'Maybe around in 2 months.' cannot be sorted, filtered, or used to decide who to
    call first."""
    assert "timeline" in LeadExtraction.model_fields
    assert "timeline_months" in LeadExtraction.model_fields


@pytest.mark.parametrize("field", ["purpose", "timeline_months"])
def test_the_new_fields_reach_the_database(field):
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    assert f"{field}=lead_data.{field}" in src, f"{field} is extracted and then dropped"


@pytest.mark.parametrize("field", ["purpose", "timeline_months"])
def test_the_lead_table_has_somewhere_to_put_them(field):
    from app.models.db import Lead

    assert hasattr(Lead, field)


# --- answering machines ---------------------------------------------------------------

VOICEMAIL = (
    "The person you are trying to reach is not available. At the please record your "
    "message. When you have finished recording, you may hang up."
)


def test_the_voicemail_from_the_call_is_recognised():
    assert is_answering_machine(VOICEMAIL)
    assert len(machine_phrases(VOICEMAIL)) >= 2


@pytest.mark.parametrize(
    "said",
    [
        "Yes.",
        "Yeah hi I am Rahul Rajput.",
        "Sorry, I am not available right now, call me tomorrow.",
        "I can't take your call properly, there is a lot of noise",
    ],
)
def test_a_person_is_never_mistaken_for_a_machine(said):
    """Hanging up on a live prospect is far worse than transcribing one voicemail, so a
    single generic phrase is not enough."""
    assert not is_answering_machine(said)


@pytest.mark.parametrize(
    "said",
    ["Please record your message", "The number you have dialled is switched off. Try again later."],
)
def test_unmistakable_machines_are_caught(said):
    assert is_answering_machine(said)


def test_the_check_only_runs_on_the_opening_turn():
    """Later in a real conversation the same words are a person talking about when they
    are free."""
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and n.name == "on_user_turn_stopped"
    )
    guard = next(
        n for n in ast.walk(handler)
        if isinstance(n, ast.If) and "is_answering_machine" in ast.unparse(n.test)
    )
    assert "_turns_heard == 0" in ast.unparse(guard.test), (
        "without this the agent hangs up on anyone who says they are unavailable"
    )


def test_detecting_a_machine_ends_the_call():
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and n.name == "on_user_turn_stopped"
    )
    guard = next(
        n for n in ast.walk(handler)
        if isinstance(n, ast.If) and "is_answering_machine" in ast.unparse(n.test)
    )
    assert "EndFrame" in ast.unparse(guard)


def test_a_machine_is_its_own_call_status():
    """Recorded as COMPLETED it inflates the campaign's answer rate in the direction that
    flatters it; as FAILED it would read as a system fault."""
    assert CallStatus.MACHINE.value == "MACHINE"
    assert CallStatus.MACHINE not in (CallStatus.COMPLETED, CallStatus.FAILED)


def test_the_webhook_records_it():
    from app.api.routes import webhook

    src = inspect.getsource(webhook._handle_call)
    assert "answering_machine" in src and "CallStatus.MACHINE" in src


def test_a_voicemail_is_not_sent_for_extraction():
    """The transcript is somebody's outgoing greeting. Running it through the extractor
    costs an OpenAI call to learn that the person was not home."""
    from app.api.routes import webhook

    src = inspect.getsource(webhook._finalize_call)
    assert src.index("CallStatus.MACHINE") < src.index("enqueue_extraction")


def test_the_result_carries_the_verdict_out_of_the_pipeline():
    from app.services.agent import CallResult

    assert CallResult(transcript="x").answering_machine is False
    assert CallResult(transcript="x", answering_machine=True).answering_machine is True


# --- dialing into an empty token budget -----------------------------------------------


class _FakeRedis:
    def __init__(self, data=None):
        self.data = data or {}

    async def hgetall(self, key):
        return dict(self.data)

    async def hset(self, key, mapping):
        self.data.update({k: str(v) for k, v in mapping.items()})

    async def expire(self, key, ttl):
        return True


def _with_redis(monkeypatch, data):
    from app.core import llm_budget

    monkeypatch.setattr(llm_budget, "get_arq_pool", lambda: _FakeRedis(data))


def test_an_unknown_budget_never_blocks_a_dial(monkeypatch):
    """None is not zero. Refusing on missing telemetry would take the whole campaign down
    the first time Redis blinked."""
    from app.core.llm_budget import tokens_available

    _with_redis(monkeypatch, {})
    assert asyncio.run(tokens_available()) is None


def test_a_reading_is_refilled_forward_to_now(monkeypatch):
    """The provider's bucket refills continuously. Trusting a snapshot would refuse dials
    for a minute after one busy call."""
    from app.core.llm_budget import tokens_available

    _with_redis(monkeypatch, {"remaining": "0", "limit": "12000", "at": str(time.time() - 30)})
    available = asyncio.run(tokens_available())
    assert 5000 < available < 7000, f"30s at 12000/min should be about 6000, got {available}"


def test_the_refill_never_exceeds_the_limit(monkeypatch):
    from app.core.llm_budget import tokens_available

    _with_redis(monkeypatch, {"remaining": "9000", "limit": "12000", "at": str(time.time() - 600)})
    assert asyncio.run(tokens_available()) == 12000


def test_a_malformed_reading_reads_as_unknown(monkeypatch):
    from app.core.llm_budget import tokens_available

    _with_redis(monkeypatch, {"remaining": "lots", "limit": "12000", "at": str(time.time())})
    assert asyncio.run(tokens_available()) is None


def test_the_dial_is_refused_when_the_first_turn_cannot_be_answered(monkeypatch):
    """The carrier leg is billed and a real person's phone rings; then they hear silence
    while the greeting waits on a token bucket."""
    from fastapi import HTTPException

    from app.core import ratelimit

    async def empty():
        return 10.0

    monkeypatch.setattr(ratelimit, "tokens_available", empty)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ratelimit.reserve_llm_headroom())
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_a_healthy_budget_dials(monkeypatch):
    from app.core import ratelimit

    async def plenty():
        return 11_000.0

    monkeypatch.setattr(ratelimit, "tokens_available", plenty)
    asyncio.run(ratelimit.reserve_llm_headroom())  # must not raise


def test_the_gate_can_be_turned_off(monkeypatch):
    """A small plan may prefer a stalled greeting to a refused campaign; that is the
    operator's call, not ours."""
    from app.core import ratelimit

    async def empty():
        return 0.0

    monkeypatch.setattr(ratelimit, "tokens_available", empty)
    monkeypatch.setattr(ratelimit.settings, "LLM_MIN_TOKENS_TO_DIAL", 0)
    asyncio.run(ratelimit.reserve_llm_headroom())  # must not raise


@pytest.mark.parametrize("route", ["app/api/routes/campaign.py", "app/api/routes/dashboard.py"])
def test_both_dial_routes_check_it(route):
    from pathlib import Path

    src = Path(route).read_text(encoding="utf-8")
    assert "reserve_llm_headroom()" in src, f"{route} can dial into an empty token budget"


def test_the_watcher_publishes_what_the_dialer_reads():
    """Two halves of one mechanism; either alone does nothing."""
    from app.services import llm_provider

    assert "record_budget" in inspect.getsource(llm_provider.BudgetWatcher.__call__)


def test_publishing_the_budget_cannot_break_a_live_call():
    """It runs inside an httpx response hook on the call's own client."""
    src = inspect.getsource(
        __import__("app.services.llm_provider", fromlist=["x"]).BudgetWatcher.__call__
    )
    guarded = src[src.index("record_budget") - 200 : src.index("record_budget") + 200]
    assert "try:" in guarded and "except" in guarded
