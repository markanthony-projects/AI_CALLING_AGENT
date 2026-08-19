"""Dialling at the rate the carrier can carry, and writing the outcome back.

The endpoint used to place every number in the request at once, and the concurrency cap was
checked when each media websocket opened — after Vobiz had dialled, billed us and rung a real
person, who then had the line closed on them with no Call row written. On a three-slot account
a list of twenty meant seventeen people were called, charged for, and hung up on invisibly.

The pump replaces that. These pin the parts that decide who is dialled, when, and what happens
to a number that does not answer — the retry arithmetic and the SQL predicates, which are
where a mistake is silent: a number that is never eligible again simply stops being called, and
its row looks busy rather than broken.
"""

import ast
import inspect
from datetime import timedelta

import pytest

from app.models.db import (
    RETRIABLE_CONTACT_STATUSES,
    TERMINAL_CONTACT_STATUSES,
    CallStatus,
    ContactStatus,
)
from app.services import dial_pump
from app.services.dial_pump import MAX_DIAL_ATTEMPTS, RETRY_BACKOFF, next_attempt_after
from app.utils.timeutils import BUSINESS_HOURS, utc_now


# --- the retry arithmetic -------------------------------------------------------------


def test_a_first_no_answer_is_retried():
    """Very few people answer a first call from an unknown number. Stopping at one attempt
    throws away most of a reachable list."""
    assert next_attempt_after(1) is not None


def test_retries_run_out():
    assert next_attempt_after(MAX_DIAL_ATTEMPTS) is None


def test_more_attempts_than_the_cap_never_reopens():
    """Guards the arithmetic, not a real state: an off-by-one here dials somebody for ever."""
    assert next_attempt_after(MAX_DIAL_ATTEMPTS + 5) is None


def test_zero_attempts_is_not_a_retry():
    """A contact with no attempts is PENDING, not due for a retry. Returning a time here would
    push a never-dialled number into the future for no reason."""
    assert next_attempt_after(0) is None


def test_the_gap_widens_between_attempts():
    """Calling back in five minutes catches nobody and reads as harassment. The second try
    catches someone who was driving; the third, someone who was travelling."""
    now = utc_now()
    first = next_attempt_after(1, now)
    second = next_attempt_after(2, now)
    assert first < second


def test_the_first_retry_is_hours_away_not_minutes():
    now = utc_now()
    assert next_attempt_after(1, now) - now >= timedelta(hours=1)


def test_the_last_retry_is_within_a_couple_of_days():
    """A lead list has a shelf life; a week later the project has moved on."""
    now = utc_now()
    assert next_attempt_after(MAX_DIAL_ATTEMPTS - 1, now) - now <= timedelta(days=2)


def test_there_is_a_gap_for_every_retry():
    """One backoff short and the last retry silently reuses the previous gap."""
    assert len(RETRY_BACKOFF) >= MAX_DIAL_ATTEMPTS - 1


def test_the_attempt_cap_is_small_enough_not_to_pester():
    assert 2 <= MAX_DIAL_ATTEMPTS <= 4


# --- who is eligible ------------------------------------------------------------------


def _eligible_sql() -> str:
    """The predicate with its values inlined.

    str() on a SQLAlchemy clause renders bind parameters as placeholders, so none of the status
    names appear in it — which made every "this status is not eligible" assertion below pass no
    matter what the predicate actually said. literal_binds is what gives them something to
    check.
    """
    from sqlalchemy.dialects import postgresql

    return str(
        dial_pump.eligible(utc_now()).compile(
            dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True}
        )
    )


def test_the_predicate_renders_its_values():
    """Guards the helper above. Without inlined values the eligibility tests are vacuous."""
    assert ContactStatus.PENDING.value in _eligible_sql()


def test_pending_and_retriable_are_eligible():
    sql = _eligible_sql()
    assert "status" in sql
    for status in (ContactStatus.PENDING, *RETRIABLE_CONTACT_STATUSES):
        assert status.name in sql or status.value in sql


@pytest.mark.parametrize("status", TERMINAL_CONTACT_STATUSES)
def test_terminal_statuses_are_never_dialled(status):
    """DND is the one that matters legally, and COMPLETED is the one that matters to the
    prospect: calling somebody who already spoke to the agent is the worst outcome here."""
    assert status.value not in _eligible_sql()


def test_a_contact_mid_dial_is_not_picked_up_again():
    """DIALING is deliberately absent. The tick runs every five seconds, and treating it as
    eligible would dial the same person twice while the first call was ringing."""
    assert ContactStatus.DIALING.value not in _eligible_sql()


def test_a_null_next_attempt_is_eligible_now():
    """Which is what a fresh import wants — nothing should have to set a time for the first
    attempt."""
    assert "IS NULL" in _eligible_sql().upper()


def test_the_attempt_cap_is_in_the_predicate_too():
    """Belt and braces with EXHAUSTED: a row left NO_ANSWER at three attempts by any path must
    still not be picked up."""
    assert "attempts" in _eligible_sql()


# --- claiming -------------------------------------------------------------------------


def test_rows_are_claimed_with_skip_locked():
    """What makes it safe to run in more than one worker: a row another transaction holds is
    passed over rather than waited for, so two pumps never dial the same number."""
    src = inspect.getsource(dial_pump.claim)
    assert "with_for_update(skip_locked=True)" in src


def test_the_status_flips_inside_the_claiming_transaction():
    """Leaving them PENDING until the dial returned would let the next tick — five seconds
    later — pick them up and dial everybody twice."""
    src = inspect.getsource(dial_pump.claim)
    assert "ContactStatus.DIALING" in src
    assert src.index("with_for_update") < src.index("ContactStatus.DIALING")


def test_claiming_nothing_is_free():
    """The tick runs every five seconds for the life of the worker."""
    import asyncio

    assert asyncio.run(dial_pump.claim(None, "c", 0, utc_now())) == []


def test_the_oldest_contacts_go_first():
    """A list is worked in the order it was uploaded, and a retry does not jump the queue
    ahead of numbers never tried at all."""
    src = inspect.getsource(dial_pump.claim)
    assert "order_by" in src
    assert "nullsfirst" in src


# --- pacing ---------------------------------------------------------------------------


def test_nothing_is_dialled_outside_business_hours():
    """A list uploaded at 9 PM dials in the morning. Enforced here rather than at import so
    the upload itself is never rejected for the time of day."""
    src = inspect.getsource(dial_pump.dial_due_contacts)
    assert "is_within_business_hours" in src
    assert src.index("is_within_business_hours") < src.index("free_slots")


def test_business_hours_are_read_in_ist():
    """The droplet runs on UTC, where 09:30 IST looks like 04:00 — outside every window."""
    assert "to_ist" in inspect.getsource(dial_pump.dial_due_contacts)
    assert BUSINESS_HOURS == (10, 20)


def test_a_full_carrier_dials_nothing_and_marks_nothing():
    """The queue simply does not move, which is correct and needs no special case."""
    tree = ast.parse(inspect.getsource(dial_pump.dial_due_contacts).lstrip())
    guard = next(
        n for n in ast.walk(tree) if isinstance(n, ast.If) and "slots <= 0" in ast.unparse(n.test)
    )
    assert "return 0" in ast.unparse(guard)


def test_the_slot_count_bounds_what_is_claimed():
    """Claiming more than there are slots for would mark contacts DIALING that never get
    dialled, and they would sit there until the stale-dial sweep found them."""
    src = inspect.getsource(dial_pump.dial_due_contacts)
    assert "claim(db, campaign_id, slots, now)" in src


def test_claims_are_committed_before_any_dial_goes_out():
    """A crash between claiming and dialling must leave the rows DIALING, not PENDING — the
    sweep can time those out, whereas losing the claim dials everybody twice."""
    src = inspect.getsource(dial_pump.dial_due_contacts)
    assert src.index("db.commit()") < src.index("_place(db, contact)")


def test_a_paused_campaign_is_not_selected():
    assert "CampaignStatus.ACTIVE" in inspect.getsource(dial_pump.active_campaign_ids)


def test_the_campaign_closest_to_finishing_is_worked_first():
    """Otherwise a newly uploaded list soaks up every slot and the almost-finished campaign
    keeps a long tail for days."""
    src = inspect.getsource(dial_pump.active_campaign_ids)
    assert "remaining.asc()" in src


# --- suppression ----------------------------------------------------------------------


def test_the_suppression_list_is_checked_at_dial_time():
    """Not only at import. A request to stop being called arrives during a run, and the
    numbers already queued behind it have to stop too."""
    src = inspect.getsource(dial_pump.dial_due_contacts)
    assert "suppressed(db" in src
    assert src.index("suppressed(db") < src.index("_place(db, contact)")


def test_a_suppressed_contact_is_not_charged_an_attempt():
    """It was never dialled. Counting it would exhaust a number nobody ever rang, and hide the
    reason behind "no attempts left"."""
    src = inspect.getsource(dial_pump.dial_due_contacts)
    marked = src[src.index("ContactStatus.DND"):]
    assert "attempts" in marked
    assert "last_attempt_at = None" in marked


def test_checking_suppression_is_one_query_for_the_batch():
    """Three slots is three rows today, but the same code runs on a raised cap."""
    assert "in_(numbers)" in inspect.getsource(dial_pump.suppressed)


def test_an_empty_batch_asks_the_database_nothing():
    import asyncio

    assert asyncio.run(dial_pump.suppressed(None, [])) == set()


# --- writing the outcome back ---------------------------------------------------------


def test_a_call_with_no_contact_behind_it_is_ignored():
    """Manual dials, and every call placed before the queue existed, have none."""
    import asyncio

    assert asyncio.run(dial_pump.record_outcome(None, CallStatus.COMPLETED)) is None


def test_a_conversation_is_never_dialled_again():
    """Calling somebody who already spoke to the agent is the worst outcome the queue can
    produce."""
    src = inspect.getsource(dial_pump.record_outcome)
    completed = src[src.index("CallStatus.COMPLETED"):]
    assert "ContactStatus.COMPLETED" in completed
    assert "next_attempt_at = None" in completed


def test_a_pickup_with_nothing_said_counts_as_no_answer():
    """A COMPLETED call with an empty transcript is a pickup that produced no conversation,
    which is the same thing to a dial list as a ring-out."""
    assert "answered_words" in inspect.getsource(dial_pump.record_outcome)
    src = inspect.getsource(dial_pump.record_outcome)
    assert "answered_words > 0" in src


def test_an_operator_decision_outranks_a_call_already_in_flight():
    """Somebody marked DND while the call was up meant it, and the outcome arriving a minute
    later must not overwrite that."""
    src = inspect.getsource(dial_pump.record_outcome)
    assert "ContactStatus.DIALING," in src
    assert "return" in src[src.index("ContactStatus.DIALING,"):]


def test_the_last_attempt_exhausts_rather_than_looping():
    src = inspect.getsource(dial_pump.record_outcome)
    assert "ContactStatus.EXHAUSTED" in src


# --- stuck mid-dial -------------------------------------------------------------------


def test_stuck_dials_are_returned_to_the_queue():
    """A dial whose call never opened a websocket leaves its row DIALING for ever, and DIALING
    is not eligible — so that number is silently never called again while the row looks busy
    rather than broken."""
    src = inspect.getsource(dial_pump.release_stale_dialing)
    assert "ContactStatus.DIALING" in src
    assert "ContactStatus.NO_ANSWER" in src


def test_a_live_call_is_never_reclaimed():
    """Reclaiming one would let a fourth call be dialled on a three-slot account."""
    from app.services.agent import MAX_CALL_DURATION_SECS

    default = inspect.signature(dial_pump.release_stale_dialing).parameters["older_than"].default
    assert default.total_seconds() > MAX_CALL_DURATION_SECS


# --- what the operator is told will happen ---------------------------------------------
#
# Queueing a number that is already on the campaign inserts nothing and changes nothing. The
# endpoint reported only how many rows it wrote, so a paste of one number that had already
# spoken to the agent answered "queued" and placed no call — indistinguishable from a broken
# dialer, and the reason a dial from the dashboard went out to nobody.


def _verdicts(*pairs):
    """Rows as the database returns them: (status, eligible-evaluated-for-that-row)."""
    return list(pairs)


def test_a_fresh_contact_is_reported_as_going_out():
    will_dial, held_back = dial_pump.dial_forecast(_verdicts((ContactStatus.PENDING, True)))
    assert will_dial == 1
    assert held_back == {}


def test_a_contact_that_already_spoke_is_not_reported_as_going_out():
    """The case that sent a dial nowhere. Counting the row would repeat the original lie."""
    will_dial, held_back = dial_pump.dial_forecast(_verdicts((ContactStatus.COMPLETED, False)))
    assert will_dial == 0
    assert held_back == {ContactStatus.COMPLETED.value: 1}


@pytest.mark.parametrize("status", TERMINAL_CONTACT_STATUSES)
def test_no_terminal_status_is_ever_reported_as_going_out(status):
    will_dial, _ = dial_pump.dial_forecast(_verdicts((status, False)))
    assert will_dial == 0


def test_the_reason_names_the_status_so_the_operator_can_act_on_it():
    """"1 will not be dialled" is not actionable; "1 already spoke with the agent" is — it
    points at the row to select and retry."""
    _, held_back = dial_pump.dial_forecast(
        _verdicts((ContactStatus.EXHAUSTED, False), (ContactStatus.DND, False))
    )
    assert held_back == {ContactStatus.EXHAUSTED.value: 1, ContactStatus.DND.value: 1}


def test_held_back_counts_repeats_rather_than_collapsing_them():
    _, held_back = dial_pump.dial_forecast(
        _verdicts((ContactStatus.COMPLETED, False), (ContactStatus.COMPLETED, False))
    )
    assert held_back == {ContactStatus.COMPLETED.value: 2}


def test_a_mixed_paste_reports_both_halves():
    will_dial, held_back = dial_pump.dial_forecast(
        _verdicts(
            (ContactStatus.PENDING, True),
            (ContactStatus.NO_ANSWER, True),
            (ContactStatus.COMPLETED, False),
        )
    )
    assert (will_dial, held_back) == (2, {ContactStatus.COMPLETED.value: 1})


def test_an_unjudgeable_row_is_held_back_rather_than_assumed_dialable():
    """A NULL from the database means the predicate could not decide. Counting it as going
    out would restore exactly the silence this replaces."""
    will_dial, held_back = dial_pump.dial_forecast(_verdicts((ContactStatus.PENDING, None)))
    assert will_dial == 0
    assert held_back == {ContactStatus.PENDING.value: 1}


def test_queueing_nothing_promises_nothing():
    assert dial_pump.dial_forecast([]) == (0, {})
