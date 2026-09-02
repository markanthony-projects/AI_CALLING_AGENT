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

from app.models.db import CallStatus, Lead, LeadStatus, Purpose
from app.models.schemas import LeadExtraction
from app.utils.answering_machine import (
    OPENING_TURNS,
    is_answering_machine,
    machine_in_opening,
    machine_phrases,
)
from app.utils.lead_status import capped_status, qualifying_facts, status_ceiling
from app.utils.attribution import (
    budget_as_stated,
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
    binding = next(line for line in src.splitlines() if "grounding_text =" in line)
    assert "transliterated_transcript" in binding


def test_every_grounding_check_reads_the_same_text():
    """Two of them now — the prospect-owned fields and the name. A second check against the
    raw transcript would quietly disagree with the first on every Hindi call."""
    from app import worker

    src = inspect.getsource(worker.process_extraction)
    assert "_drop_ungrounded(lead_data, grounding_text" in src
    assert "name_spoken_by_prospect(lead_data.customer_name, grounding_text)" in src


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


def _machine_guard():
    """The `if` in the turn handler that decides a call is a machine."""
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and n.name == "on_user_turn_stopped"
    )
    return next(
        n for n in ast.walk(handler)
        if isinstance(n, ast.If) and "machine_in_opening" in ast.unparse(n.test)
    )


def test_the_check_stops_once_the_conversation_has_started():
    """Later in a real conversation the same words are a person talking about when they
    are free, and hanging up on them is far worse than transcribing a voicemail."""
    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and n.name == "on_user_turn_stopped"
    )
    window = next(
        n for n in ast.walk(handler)
        if isinstance(n, ast.If) and "OPENING_TURNS" in ast.unparse(n.test)
    )
    assert "_opening_turns" in ast.unparse(window.test), (
        "without a bound the agent hangs up on anyone who says they are unavailable"
    )


def test_detecting_a_machine_ends_the_call():
    assert "EndFrame" in ast.unparse(_machine_guard())


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

    _with_redis(monkeypatch, {"tokens": "0", "token_limit": "12000", "at": str(time.time() - 30)})
    available = asyncio.run(tokens_available())
    assert 5000 < available < 7000, f"30s at 12000/min should be about 6000, got {available}"


def test_the_refill_never_exceeds_the_limit(monkeypatch):
    from app.core.llm_budget import tokens_available

    _with_redis(monkeypatch, {"tokens": "9000", "token_limit": "12000", "at": str(time.time() - 600)})
    assert asyncio.run(tokens_available()) == 12000


def test_a_malformed_reading_reads_as_unknown(monkeypatch):
    from app.core.llm_budget import tokens_available

    _with_redis(monkeypatch, {"tokens": "lots", "token_limit": "12000", "at": str(time.time())})
    assert asyncio.run(tokens_available()) is None


def test_the_dial_is_refused_when_the_first_turn_cannot_be_answered(monkeypatch):
    """The carrier leg is billed and a real person's phone rings; then they hear silence
    while the greeting waits on a token bucket."""
    from fastapi import HTTPException

    from app.core import ratelimit
    from app.core.llm_budget import Headroom

    async def empty():
        return Headroom(tokens=10.0, requests=None)

    monkeypatch.setattr(ratelimit, "headroom", empty)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ratelimit.reserve_llm_headroom())
    assert exc.value.status_code == 429
    assert "Retry-After" in exc.value.headers


def test_a_healthy_budget_dials(monkeypatch):
    from app.core import ratelimit
    from app.core.llm_budget import Headroom

    async def plenty():
        return Headroom(tokens=11_000.0, requests=4.0)

    monkeypatch.setattr(ratelimit, "headroom", plenty)
    asyncio.run(ratelimit.reserve_llm_headroom())  # must not raise


def test_the_gate_can_be_turned_off(monkeypatch):
    """A small plan may prefer a stalled greeting to a refused campaign; that is the
    operator's call, not ours."""
    from app.core import ratelimit
    from app.core.llm_budget import Headroom

    async def empty():
        return Headroom(tokens=0.0, requests=0.0)

    monkeypatch.setattr(ratelimit, "headroom", empty)
    monkeypatch.setattr(ratelimit.settings, "LLM_MIN_TOKENS_TO_DIAL", 0)
    asyncio.run(ratelimit.reserve_llm_headroom())  # must not raise


def test_the_one_queueing_path_checks_it():
    """Both routes used to carry their own copy of this check, and a copy is a thing that can
    be forgotten. They now share one implementation, so the check has one home."""
    from app.services import dial_queue

    src = inspect.getsource(dial_queue.enqueue)
    assert "reserve_llm_headroom()" in src, "numbers can be queued into an empty token budget"


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


# --- providers run out of different things -------------------------------------------
#
# Measured on 2026-08-03. Groq's free tier binds on tokens: 12,000/minute against a
# 3,400-token request. Cerebras binds on requests first — x-ratelimit-limit-requests-minute
# is 5, against a call that needs six to ten. Watching only tokens would have declared
# Cerebras healthy (29,999 tokens left!) right up until the greeting stalled.


def _headroom(**kw):
    from app.core.llm_budget import Headroom

    return Headroom(tokens=kw.get("tokens"), requests=kw.get("requests"))


def _gate(monkeypatch, room):
    from app.core import ratelimit

    async def _read():
        return room

    monkeypatch.setattr(ratelimit, "headroom", _read)
    return ratelimit


def test_no_requests_left_refuses_the_dial_even_with_tokens_to_spare(monkeypatch):
    """Cerebras exactly: 29,999 tokens and 0 requests. On tokens alone this dials, bills the
    carrier leg, rings a real person and then stalls on the opening line."""
    from fastapi import HTTPException

    ratelimit = _gate(monkeypatch, _headroom(tokens=29_999.0, requests=0.0))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ratelimit.reserve_llm_headroom())
    assert exc.value.status_code == 429
    assert "request" in exc.value.detail.lower()


def test_requests_left_dials(monkeypatch):
    ratelimit = _gate(monkeypatch, _headroom(tokens=29_999.0, requests=3.0))
    asyncio.run(ratelimit.reserve_llm_headroom())  # must not raise


def test_a_provider_that_reports_no_rpm_is_not_blocked(monkeypatch):
    """Groq sends no per-minute request header at all. Absent must mean "unknown", never
    "zero", or Groq would never dial again."""
    ratelimit = _gate(monkeypatch, _headroom(tokens=11_000.0, requests=None))
    asyncio.run(ratelimit.reserve_llm_headroom())  # must not raise


def test_the_token_ceiling_is_still_checked_first(monkeypatch):
    """Both are real; the token message is the more actionable one when both are empty."""
    from fastapi import HTTPException

    ratelimit = _gate(monkeypatch, _headroom(tokens=10.0, requests=0.0))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ratelimit.reserve_llm_headroom())
    assert "token" in exc.value.detail.lower()


def test_the_rpm_refusal_says_how_long_to_wait(monkeypatch):
    from fastapi import HTTPException

    ratelimit = _gate(monkeypatch, _headroom(tokens=29_999.0, requests=0.0))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(ratelimit.reserve_llm_headroom())
    assert int(exc.value.headers["Retry-After"]) >= 1


def test_requests_are_refilled_forward_like_tokens(monkeypatch):
    """5 RPM refills at one request every 12 seconds. A snapshot taken 30s ago that read 0
    is worth about 2.5 requests now, and refusing on it would idle the dialer for a minute
    after every busy call."""
    import time as _t

    from app.core import llm_budget

    class _R:
        async def hgetall(self, key):
            return {
                "tokens": "0", "token_limit": "30000",
                "requests": "0", "request_limit": "5",
                "at": str(_t.time() - 30),
            }

    monkeypatch.setattr(llm_budget, "get_arq_pool", lambda: _R())
    room = asyncio.run(llm_budget.headroom())
    assert 2.0 < room.requests < 3.0, f"expected ~2.5 requests refilled, got {room.requests}"


# --- grounded is not the same as stated -------------------------------------------------
#
# From a live call on 2 Sep 2026. Asked his budget, the prospect said "1 1 to 2 CR"; the
# lead was written down as 1,50,00,000 — the midpoint of the range, a figure nobody uttered.
# budget_is_grounded passes it, and correctly: every rupee of it is inside what he said. But
# it reads to whoever opens the lead as a precise answer, and the error runs the expensive
# way — a rep pitching a 1.5 Cr home to a man who may have meant one Crore.

MAYUR = "Agent: What budget range are you thinking of?\nProspect: 1 1 to 2 CR.\n"


def test_a_midpoint_nobody_said_is_snapped_to_the_figure_they_did_say():
    assert budget_as_stated(15_000_000.0, MAYUR) == 10_000_000.0


@pytest.mark.parametrize("budget", [10_000_000.0, 20_000_000.0])
def test_either_end_of_the_range_is_left_alone(budget):
    """Both ends are figures the prospect actually named. Only the space between them is
    the model's own invention."""
    assert budget_as_stated(budget, MAYUR) == budget


def test_the_ordinary_case_is_untouched():
    """One number, said once. Nothing to snap, and no rounding drift allowed either."""
    assert budget_as_stated(6_000_000.0, STATED) == 6_000_000.0


def test_a_figure_within_rounding_of_a_stated_one_is_not_snapped():
    """"60 lakhs" and 6000000.0 are one figure. Snapping here would move a correct value."""
    assert budget_as_stated(6_000_000.0 * 1.005, STATED) == 6_000_000.0 * 1.005


def test_nothing_stated_is_left_for_the_grounding_check_to_drop():
    """budget_as_stated must not quietly keep a fabricated figure alive by having nothing to
    compare it to — dropping it is budget_is_grounded's job, and it does drop it."""
    assert budget_as_stated(11_700_000.0, FABRICATED) == 11_700_000.0
    assert not budget_is_grounded(11_700_000.0, FABRICATED)


def test_a_null_budget_stays_null():
    assert budget_as_stated(None, MAYUR) is None


def test_the_worker_applies_it_to_what_it_records():
    """The helper being right is not enough: the midpoint reached the database through
    _drop_ungrounded, which only ever nulled fields and had no opinion about this one."""
    import inspect

    from app import worker

    assert "budget_as_stated" in inspect.getsource(worker._drop_ungrounded)


def test_the_extraction_prompt_asks_for_the_low_end_of_a_range():
    """Fixing it after the fact is the safety net, not the plan. The model is told."""
    from app.worker import _build_system_prompt
    from app.utils.timeutils import utc_now

    prompt = _build_system_prompt(utc_now()).lower()
    assert "lowest" in prompt and "range" in prompt


# --- the status the model kept declining to give ----------------------------------------


def test_the_prompt_rules_on_every_lead_status():
    """Two of the three calls on 2 Sep 2026 logged "Extraction returned no status;
    defaulting to WARM". The prompt had rules for is_prospect, budget, locality, unit type,
    appointments and transliteration — and not one word about status, so the field sales
    sorts by was a constant rather than a judgement."""
    from app.worker import _build_system_prompt
    from app.utils.timeutils import utc_now

    prompt = _build_system_prompt(utc_now())
    assert "CRITICAL RULE FOR status" in prompt
    for verdict in ("HOT", "WARM", "COLD"):
        assert f"- {verdict}:" in prompt, f"{verdict} is never defined for the model"


def test_a_call_that_was_cut_off_is_not_warm():
    """WARM is the fallback when the model says nothing, so the prompt has to be explicit
    that a call which ended before the prospect engaged is not one."""
    from app.worker import _build_system_prompt
    from app.utils.timeutils import utc_now

    prompt = _build_system_prompt(utc_now())
    assert "cut off before they engaged is COLD" in prompt


# Three figures, so the lowest and the highest below the model's answer are different
# numbers. With Mayur's transcript they coincide — there is only one figure under the
# midpoint — so it cannot tell snapping down from snapping up, and a mutation that reached
# for the top of the range passed every test above.
SPREAD = "Prospect: my range is 80 lakhs to 2 crore, ideally around 1 crore.\n"


def test_it_snaps_to_the_lowest_stated_figure_not_the_highest():
    """80 lakhs and 1 crore are both under 1.5 Cr and both were said. The floor is the claim
    that survives contact with a rep: understating sends them to cheaper stock, overstating
    sends them to homes the prospect cannot buy."""
    assert budget_as_stated(15_000_000.0, SPREAD) == 8_000_000.0


def test_the_worker_records_the_snapped_figure():
    """Asserting the name appears in _drop_ungrounded is not enough — it appears in the
    comment there too, so removing the call outright still read as present. This runs it."""
    from app.models.schemas import LeadExtraction
    from app.worker import _drop_ungrounded

    lead = LeadExtraction(is_prospect=True, budget=15_000_000.0)
    assert _drop_ungrounded(lead, MAYUR, "sid").budget == 10_000_000.0


def test_the_worker_leaves_a_stated_figure_alone():
    """The snapping must not fire on the ordinary case, where the model answered correctly."""
    from app.models.schemas import LeadExtraction
    from app.worker import _drop_ungrounded

    lead = LeadExtraction(is_prospect=True, budget=6_000_000.0)
    assert _drop_ungrounded(lead, STATED, "sid").budget == 6_000_000.0


def test_the_worker_still_drops_a_fabricated_budget_entirely():
    """Snapping is for figures inside what the prospect said. One that came out of the
    agent's own pitch is not snapped closer — it is removed."""
    from app.models.schemas import LeadExtraction
    from app.worker import _drop_ungrounded

    lead = LeadExtraction(is_prospect=True, budget=11_700_000.0)
    assert _drop_ungrounded(lead, FABRICATED, "sid").budget is None


# --- the voicemail that got a sales pitch -----------------------------------------------
#
# A live call on 2 Sep 2026, 11:11. The machine picked up, talked over the agent's greeting,
# and Deepgram cut its announcement at the pauses:
#
#     AGENT → "Hi, Good afternoon Bharat Dua..." [interrupted]
#     USER  → "At the tone."
#     USER  → "When you have finished recording, you may hang up."
#     AGENT → "We are launching a new project in Varthur..."
#     No speech either way for 60s; abandoning call
#     Call finalised | status=FAILED | duration=76.6s
#
# The second line trips the two-phrase rule on its own. It was never tested, because the
# check ran on the first turn only and the first turn — "At the tone." — is one generic
# phrase. So the agent pitched into the recording, then sat there until the idle timeout:
# 76 seconds billed, a carrier slot held, and an extraction job run against a greeting.

CUT_UP_VOICEMAIL = ["At the tone.", "When you have finished recording, you may hang up."]


def test_the_first_turn_alone_is_still_not_enough():
    """"At the tone." is one generic phrase, and a person could say it. The old check saw
    only this, said no, and closed itself for the rest of the call."""
    assert not machine_in_opening(CUT_UP_VOICEMAIL[:1])


def test_the_announcement_is_caught_once_its_second_turn_arrives():
    assert machine_in_opening(CUT_UP_VOICEMAIL)


def test_an_announcement_with_one_phrase_per_turn_is_still_caught():
    """The reason the turns are read together rather than one at a time: split finely
    enough, no single turn carries two phrases and the rule fires on none of them."""
    split = ["The person you are trying to reach", "is not available", "at the tone"]
    assert not any(is_answering_machine(turn) for turn in split)
    assert machine_in_opening(split)


def test_nothing_is_caught_after_the_opening_window():
    """The safeguard, and the whole reason there is a window at all."""
    assert machine_in_opening(CUT_UP_VOICEMAIL + ["at the beep"] * OPENING_TURNS) is False


@pytest.mark.parametrize(
    "turns",
    [
        ["Hello?", "Yes speaking", "Sorry, I am not available right now"],
        ["Yeah?", "Who is this", "I can't take your call, I am driving"],
        ["Hello", "Haan boliye", "Please try again later, I am busy"],
    ],
)
def test_a_real_prospect_opening_a_call_is_not_a_machine(turns):
    """Joining the turns must not manufacture a second phrase out of a person being
    politely unavailable. This is what the two-phrase rule is protecting."""
    assert not machine_in_opening(turns)


def test_the_window_is_wider_than_one_turn():
    """The whole failure was a window of exactly one. A regression to that would pass every
    other test here, because they all read the joined text."""
    assert OPENING_TURNS >= 2


def test_the_agent_accumulates_the_opening_turns():
    """The helper being right is not enough — the handler has to feed it every opening turn
    rather than only the current one."""
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "_opening_turns.append(transcript)" in src
    assert "machine_in_opening(_opening_turns)" in src


# --- how warm a lead is allowed to be ---------------------------------------------------
#
# A live call on 2 Sep 2026, 11:35. Everything the prospect said:
#
#     Prospect: "Yes. Ok."
#     Prospect: "Tell me tell me."
#
# Six words, and the extraction returned WARM. Every field on the lead was null — the name
# and number came off the dial list, not out of the call — and it went into the dashboard
# beside a lead who had given a budget, a purpose, a timeline, a configuration and a booked
# callback, wearing the same colour.
#
# The prompt already said a call where the prospect stated nothing is COLD. It said so on
# this call too. So the rule is expressed as code as well, the same move _drop_ungrounded
# made for budget and locality after the prompt was ignored twice in capitals.

def _lead(**fields):
    return Lead(**fields)


def test_a_lead_that_carries_nothing_cannot_be_warm():
    """The call this was built from."""
    assert capped_status(LeadStatus.WARM, _lead(customer_name="Abhijit Kumar Singh")) is LeadStatus.COLD


def test_one_stated_requirement_is_enough_to_be_warm():
    assert capped_status(LeadStatus.WARM, _lead(purpose=Purpose.SELF_USE)) is LeadStatus.WARM


@pytest.mark.parametrize(
    "field,value",
    [
        ("budget", 8_000_000),
        ("purpose", "SELF_USE"),
        ("timeline", "next year"),
        ("timeline_months", 12),
        ("preferred_unit_type", "2 BHK"),
        ("preferred_location", "Whitefield"),
    ],
)
def test_every_qualifying_field_counts(field, value):
    """Each of these is something a rep can act on. Leaving one out would quietly cap a real
    lead at COLD."""
    assert qualifying_facts(_lead(**{field: value})) == [field]


def test_the_dial_lists_own_columns_do_not_count():
    """customer_name and phone_number are on the row whether the call happened or not. If
    they counted, no lead could ever be COLD and the cap would do nothing at all."""
    assert qualifying_facts(_lead(customer_name="Abhijit", phone_number="+919449814509")) == []


def test_a_booking_outranks_everything():
    """Someone who agreed to a time has told us more than any answer could, so a lead with
    nothing else on it is still allowed to be HOT."""
    from datetime import datetime

    assert status_ceiling(_lead(site_visit_time=datetime(2026, 9, 6, 11, 0))) is LeadStatus.HOT
    assert status_ceiling(_lead(callback_time=datetime(2026, 9, 2, 18, 0))) is LeadStatus.HOT


def test_the_cap_only_ever_lowers():
    """A COLD verdict on a prospect who gave four facts is the model hearing a refusal, and
    that judgement is left alone. The failure being corrected runs one way — towards
    flattering the pipeline."""
    rich = _lead(budget=8_000_000, preferred_location="Whitefield", timeline_months=3)
    assert capped_status(LeadStatus.COLD, rich) is LeadStatus.COLD


def test_a_missing_status_becomes_the_ceiling_rather_than_a_flat_warm():
    """Two of the first three production calls hit the old flat default, and it was wrong on
    both."""
    assert capped_status(None, _lead()) is LeadStatus.COLD
    assert capped_status(None, _lead(timeline_months=6)) is LeadStatus.WARM


def test_the_four_production_leads_land_where_they_belong():
    """The whole point, read against the calls that produced the rule."""
    from datetime import datetime

    mayur = _lead(budget=10_000_000, purpose=Purpose.SELF_USE, timeline_months=12,
                  preferred_unit_type="2 BHK", callback_time=datetime(2026, 9, 2, 18, 0))
    sachin = _lead(purpose=Purpose.SELF_USE, timeline_months=42, preferred_unit_type="2 BHK",
                   site_visit_time=datetime(2026, 9, 6, 11, 0))

    assert capped_status(LeadStatus.WARM, _lead()) is LeadStatus.COLD            # Abhijit
    assert capped_status(None, _lead()) is LeadStatus.COLD                       # Rutik
    assert capped_status(LeadStatus.WARM, mayur) is LeadStatus.WARM              # unchanged
    assert capped_status(None, sachin) is LeadStatus.HOT                         # unchanged


def test_the_worker_caps_what_it_stores():
    """The helper being right is not enough. Read off the assignment rather than grepping,
    because the comment above it names the module too."""
    tree = ast.parse(_worker_source("process_extraction").lstrip())
    assigned = [
        ast.unparse(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(ast.unparse(t) == "lead.status" for t in n.targets)
    ]
    assert any(a.startswith("capped_status(") for a in assigned), assigned


def test_the_prompt_resolves_its_own_contradiction():
    """is_prospect says MUST BE TRUE on ANY genuine interest, and two lines later says FALSE
    when they never engaged. "Tell me tell me" satisfies both, and the model took the
    capitalised one. The tie is now called explicitly."""
    from app.utils.timeutils import utc_now

    prompt = _build_prompt(utc_now())
    assert "WHEN THE TWO RULES ABOVE DISAGREE" in prompt
    assert "is_prospect is TRUE and status is COLD" in prompt


def test_the_prompt_says_curiosity_is_not_a_requirement():
    from app.utils.timeutils import utc_now

    prompt = _build_prompt(utc_now())
    assert "tell me tell me" in prompt.lower()
    assert "request to hear more" in prompt


def _build_prompt(now):
    from app.worker import _build_system_prompt

    return _build_system_prompt(now)
