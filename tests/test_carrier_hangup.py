"""The carrier telling us a call is over, including the ones that never began.

A call nobody answers never opens a media websocket, so before this callback existed it left
no trace at all. Three consequences, and the middle one is the expensive one:

  * nothing recorded that the dial had happened,
  * the carrier slot stayed held until it aged out twelve minutes later,
  * and the contact sat in DIALING until the reaper swept it twenty minutes later.

On a three-slot account, three unanswered dials stopped the queue for twelve minutes. Most
numbers on a cold list do not answer, so that was the normal case rather than the edge one.
"""

from datetime import timedelta

import pytest

from app.api.routes.webhook import hangup_cause_of, was_answered
from app.models.db import ContactStatus
from app.services import dial_pump
from app.services.dial_pump import (
    DEAD_NUMBER_CAUSES,
    MAX_DIAL_ATTEMPTS,
    carrier_phrase,
    carrier_verdict,
)
from app.utils.timeutils import utc_now


# --- did somebody pick up ----------------------------------------------------------------


def test_an_answer_time_means_it_was_answered():
    assert was_answered({"AnswerTime": "2026-09-01 10:00:05"}) is True


def test_no_answer_time_means_it_was_not():
    assert was_answered({"CallStatus": "completed", "Duration": "0"}) is False


def test_the_field_name_is_read_whatever_its_casing():
    """The callback's casing is the carrier's to change, not ours to depend on."""
    assert was_answered({"answer_time": "2026-09-01 10:00:05"}) is True
    assert was_answered({"ANSWERTIME": "2026-09-01 10:00:05"}) is True


def test_a_duration_is_a_second_witness():
    """For a carrier that reports one field and not the other. Being wrong here retires a
    real conversation or resurrects a finished one, so it is worth two ways of asking."""
    assert was_answered({"Duration": "42"}) is True
    assert was_answered({"billsec": "12"}) is True


def test_an_empty_or_unparseable_body_is_not_an_answer():
    """A body we cannot read looks exactly like a call that never happened, and that is the
    safe reading: the contact gets another attempt rather than being retired unheard."""
    assert was_answered({}) is False
    assert was_answered({"Duration": ""}) is False
    assert was_answered({"Duration": "not-a-number"}) is False
    assert was_answered({"AnswerTime": "None"}) is False


def test_the_cause_is_found_under_any_of_its_names():
    assert hangup_cause_of({"HangupCause": "NO_ANSWER"}) == "NO_ANSWER"
    assert hangup_cause_of({"hangup_cause_name": "USER_BUSY"}) == "USER_BUSY"
    assert hangup_cause_of({}) == ""


# --- what the carrier's verdict means ------------------------------------------------------


def test_an_answered_call_is_not_this_callbacks_business():
    """The session that served it is the only side that knows whether the prospect spoke.
    Overwriting its verdict with 'the call ended' would retire a real conversation."""
    assert carrier_verdict("NORMAL_CLEARING", answered=True, attempts=1) is None


def test_a_ring_out_is_a_no_answer_with_a_retry_left():
    status, when, outcome = carrier_verdict("NO_ANSWER", answered=False, attempts=1)
    assert status is ContactStatus.NO_ANSWER
    assert when is not None
    assert "answered" in outcome


def test_a_ring_out_on_the_last_attempt_is_exhausted():
    status, when, _ = carrier_verdict("NO_ANSWER", answered=False, attempts=MAX_DIAL_ATTEMPTS)
    assert status is ContactStatus.EXHAUSTED
    assert when is None


def test_the_wait_is_the_same_one_every_other_no_answer_gets():
    """Two paths finalise a no-answer — this one and the stale-dial sweep. A carrier verdict
    that used a different gap would make the same outcome behave differently depending on
    which report arrived, which is not something an operator could ever explain."""
    now = utc_now()
    _, from_carrier, _ = carrier_verdict("NO_ANSWER", answered=False, attempts=1, now=now)
    _, from_sweep, _ = dial_pump.stale_dial_verdict(1, now)
    assert from_carrier == from_sweep


@pytest.mark.parametrize("cause", sorted(DEAD_NUMBER_CAUSES))
def test_a_dead_number_is_never_dialled_again(cause):
    """A bought list has some of these in it. Another dial spends money to learn the same
    thing, and the attempt would otherwise be charged to a real prospect's allowance."""
    status, when, outcome = carrier_verdict(cause, answered=False, attempts=1)
    assert status is ContactStatus.INVALID
    assert when is None
    assert "again" in outcome


def test_a_busy_line_is_still_worth_calling_back():
    """Busy means the phone is on and the person is holding it — the opposite of unreachable."""
    status, when, outcome = carrier_verdict("USER_BUSY", answered=False, attempts=1)
    assert status is ContactStatus.NO_ANSWER
    assert when is not None
    assert "busy" in outcome


def test_an_unknown_cause_is_retried_rather_than_retired():
    """Deliberately conservative. A cause this integration has not seen is far more likely to
    be a transient network condition than a dead number, and wrongly retiring a real prospect
    is the expensive mistake."""
    status, when, _ = carrier_verdict("SOME_NEW_CARRIER_CODE", answered=False, attempts=1)
    assert status is ContactStatus.NO_ANSWER
    assert when is not None


def test_a_missing_cause_is_still_a_usable_verdict():
    """The whole decision rests on the answer time; the cause only refines the wording."""
    status, when, outcome = carrier_verdict(None, answered=False, attempts=1)
    assert status is ContactStatus.NO_ANSWER
    assert when is not None
    assert outcome


def test_every_verdict_says_something_an_operator_can_read():
    for cause in ["NO_ANSWER", "USER_BUSY", "CALL_REJECTED", "", "WEIRD_CODE"]:
        phrase = carrier_phrase(cause)
        assert phrase and phrase[0].islower(), cause
        assert "_" not in phrase, cause


# --- the ordering the endpoint depends on --------------------------------------------------


def test_the_slot_is_released_before_anything_that_can_fail():
    """Freeing the slot is what lets the pump dial the next number. Everything after it is a
    database write that can raise, and on a three-slot account losing one to an exception
    costs a third of the account's capacity for twelve minutes."""
    import inspect

    from app.api.routes import webhook

    src = inspect.getsource(webhook.vobiz_hangup)
    assert src.index("call_slots.release") < src.index("record_carrier_outcome")


def test_the_endpoint_answers_even_when_recording_fails():
    """Vobiz retries a failed callback three times. A retry storm would not hold anything
    open — the slot is already free — but it would bury the log."""
    import inspect

    from app.api.routes import webhook

    src = inspect.getsource(webhook.vobiz_hangup)
    assert "except Exception" in src
    assert "return Response(status_code=200)" in src


def test_the_hangup_token_outlives_the_longest_call():
    """A default-TTL token expires during a long call and the callback is refused — which is
    worse than never asking for it, because then the slot is held until it ages out and the
    contact sits in DIALING until the reaper finds it."""
    import inspect

    from app.services import agent, dialer

    src = inspect.getsource(dialer.trigger_vobiz_call)
    assert "hangup_token" in src
    assert "ttl_seconds=int(MAX_CALL_DURATION_SECS) + 1800" in src
    # And the margin is real against the cap it is measured from.
    assert agent.MAX_CALL_DURATION_SECS + 1800 > agent.MAX_CALL_DURATION_SECS


def test_the_dial_asks_for_the_callback_at_all():
    """Without hangup_url none of this runs, and the failure is silent: calls simply stop
    going out as slots fill with rings nobody answered."""
    import inspect

    from app.services import dialer

    src = inspect.getsource(dialer.trigger_vobiz_call)
    assert '"hangup_url": hangup_url' in src
    assert '"hangup_method": "POST"' in src


def test_the_ring_is_capped_and_so_is_the_call():
    """Both bound how long one call can hold a carrier slot. time_limit matters only when our
    own pipeline is the thing that died — and then it is the difference between a stuck leg
    costing a minute and one billing for four hours."""
    import inspect

    from app.core.config import settings
    from app.services import agent, dialer

    src = inspect.getsource(dialer.trigger_vobiz_call)
    assert '"ring_timeout": settings.VOBIZ_RING_SECONDS' in src
    assert '"time_limit"' in src
    # Our own cap has to win in the normal case, or the caller loses the goodbye.
    assert settings.VOBIZ_RING_SECONDS < agent.MAX_CALL_DURATION_SECS


def test_the_ring_cap_is_never_sent_as_a_scheduled_hangup():
    """hangup_on_ring is not a ring limit. The carrier documents it as "schedules the call
    for hangup at a specified time after the call starts ringing" — no condition that the
    call still be ringing when it fires, which is exactly how it behaved.

    On 2 Sep 2026 a call rang at 11:53:24, was answered at 11:53:35, and was cut at 11:54:09
    with both parties mid-sentence: 45 seconds after ring start, to the second, which is what
    we were sending. Three live conversations died before the arithmetic was spotted, because
    measuring from the answer made the times look unrelated — 33.9s, 38.8s, 39.1s.

    Read off the request body rather than the whole function, so the explanation of the bug
    in the comment above it cannot make this pass.
    """
    import ast
    import inspect

    from app.services import dialer

    tree = ast.parse(inspect.getsource(dialer.trigger_vobiz_call).lstrip())
    body = next(
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == "data" for t in n.targets)
    )
    keys = {k.value for k in body.keys if isinstance(k, ast.Constant)}
    assert "ring_timeout" in keys
    assert "hangup_on_ring" not in keys, "this cuts answered calls; ring_timeout is the one"


def test_a_slot_outlives_the_carriers_own_time_limit():
    """The self-healing expiry is the backstop for a callback that never arrives. If it fired
    before the carrier's own limit it would free a slot while the call was still up, and the
    account would carry one more call than it can."""
    from app.core import call_slots
    from app.services.agent import MAX_CALL_DURATION_SECS

    carrier_limit = MAX_CALL_DURATION_SECS + 60
    assert call_slots._stale_after() > carrier_limit


def test_the_reaper_still_covers_a_callback_that_never_arrives():
    """Defence in depth: the callback is the fast path, not the only one. Its threshold has
    to stay above the longest possible call or it would reclaim a live one."""
    import inspect

    from app.services.agent import MAX_CALL_DURATION_SECS

    default = inspect.signature(dial_pump.release_stale_dialing).parameters["older_than"].default
    assert default > timedelta(seconds=MAX_CALL_DURATION_SECS)


# --- delivered more than once --------------------------------------------------------------
#
# Vobiz retries a callback it could not deliver up to three times. The same guard that makes
# those retries harmless also protects an operator who acted on the contact while it rang.


class FakeSession:
    def __init__(self, contact):
        self.contact = contact
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get(self, _model, _pk):
        return self.contact

    async def commit(self):
        self.commits += 1


CONTACT_ID = "11111111-2222-3333-4444-555555555555"


def _contact(status, attempts=1):
    from app.models.db import Contact

    return Contact(status=status, attempts=attempts)


@pytest.fixture
def session(monkeypatch):
    holder = {}

    def factory():
        return holder["session"]

    monkeypatch.setattr(dial_pump, "AsyncSessionLocal", factory)

    def use(contact):
        holder["session"] = FakeSession(contact)
        return holder["session"]

    return use


async def test_a_ringing_contact_takes_the_carriers_verdict(session):
    s = session(_contact(ContactStatus.DIALING))
    assert await dial_pump.record_carrier_outcome(CONTACT_ID, "NO_ANSWER", False) is True
    assert s.contact.status is ContactStatus.NO_ANSWER
    assert s.commits == 1


async def test_a_redelivered_callback_changes_nothing(session):
    """The first delivery moves it out of DIALING, so the second and third find their work
    done. Without this a single ring-out would spend the whole retry allowance."""
    s = session(_contact(ContactStatus.NO_ANSWER))
    assert await dial_pump.record_carrier_outcome(CONTACT_ID, "NO_ANSWER", False) is False
    assert s.commits == 0


async def test_an_operator_decision_made_while_it_rang_survives(session):
    """Somebody marked the number do-not-call during the call it is reporting on. Their
    decision outranks a verdict about a call that is already over."""
    s = session(_contact(ContactStatus.DND))
    assert await dial_pump.record_carrier_outcome(CONTACT_ID, "NO_ANSWER", False) is False
    assert s.contact.status is ContactStatus.DND


async def test_an_answered_call_leaves_the_contact_to_the_session(session):
    s = session(_contact(ContactStatus.DIALING))
    assert await dial_pump.record_carrier_outcome(CONTACT_ID, "NORMAL_CLEARING", True) is False
    assert s.contact.status is ContactStatus.DIALING
    assert s.commits == 0


async def test_a_call_with_no_contact_behind_it_is_not_an_error(session):
    """Browser test calls and anything dialled before the queue existed have none."""
    session(_contact(ContactStatus.DIALING))
    assert await dial_pump.record_carrier_outcome(None, "NO_ANSWER", False) is False


# --- whose call id is it anyway? ---------------------------------------------------------
#
# Our call_sid is a UUID we mint and put in the callback URLs. The carrier has never seen it
# and cannot look it up. Their dashboard showed a completely different id for the same call:
#
#     ours    dd6acaa2-c126-47be-bb85-41a00dd9d3ab
#     theirs  74e163ce-2109-4024-9dac-f9922e4af1db
#
# Which meant that asking their support about one specific call required opening the
# dashboard by hand and matching on phone number and wall-clock time. Both ends of the call
# hand us their identifier and we were discarding both.


class _Reply:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.mark.parametrize(
    "payload",
    [
        {"message": "call fired", "request_uuid": "74e163ce-2109-4024-9dac-f9922e4af1db"},
        {"requestUuid": "74e163ce-2109-4024-9dac-f9922e4af1db"},
        {"RequestUUID": "74e163ce-2109-4024-9dac-f9922e4af1db"},
        {"CallUUID": "74e163ce-2109-4024-9dac-f9922e4af1db"},
        {"request_uuid": ["74e163ce-2109-4024-9dac-f9922e4af1db"]},
    ],
)
def test_the_carriers_id_is_read_however_it_is_spelled(payload):
    """A Plivo-shaped API, so the casing is not ours to rely on — and a list arrives when the
    request named more than one destination."""
    from app.services.dialer import vobiz_request_uuid

    assert vobiz_request_uuid(_Reply(payload)) == "74e163ce-2109-4024-9dac-f9922e4af1db"


@pytest.mark.parametrize(
    "payload",
    [{"message": "call fired"}, {}, "not a dict", ValueError("no body")],
)
def test_a_reply_without_one_costs_nothing(payload):
    """This exists to make a support ticket possible. A dial that worked must never be
    reported as failed because the reply was shaped differently than expected."""
    from app.services.dialer import vobiz_request_uuid

    assert vobiz_request_uuid(_Reply(payload)) is None


def test_the_dial_still_succeeds_when_the_id_is_missing():
    """The guard that matters: reading the id is never allowed to decide the dial."""
    import ast
    import inspect

    from app.services import dialer

    tree = ast.parse(inspect.getsource(dialer.trigger_vobiz_call).lstrip())
    success = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and "status_code in (200, 201)" in ast.unparse(n.test)
    )
    assert "vobiz_request_uuid" not in ast.unparse(success.test)
    assert any(isinstance(n, ast.Return) and n.value.value is True for n in success.body)


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({"CallUUID": "74e163ce"}, "74e163ce"),
        ({"call_uuid": "74e163ce"}, "74e163ce"),
        ({"RequestUUID": "74e163ce"}, "74e163ce"),
        ({"calluuid": "74e163ce"}, "74e163ce"),
        ({"HangupCause": "NORMAL_CLEARING"}, ""),
        ({}, ""),
    ],
)
def test_the_hangup_callback_yields_the_carriers_id(fields, expected):
    """The callback is the one place their id arrives on every call, answered or not."""
    from app.api.routes.webhook import carrier_call_id

    assert carrier_call_id(fields) == expected


def test_the_carriers_id_is_logged_where_the_cause_is():
    """One line carrying both ids and the cause is what a support ticket is written from."""
    import inspect

    from app.api.routes import webhook

    src = inspect.getsource(webhook.vobiz_hangup)
    hangup_line = src[src.index("Carrier hung up"):]
    assert "carrier_call_id(fields)" in hangup_line
