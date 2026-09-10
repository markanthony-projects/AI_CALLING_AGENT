"""A row for every dial, so "did we ring this number at night" has an answer.

On 9 Sep 2026 the user reported a call from the agent at night. Checking was impossible:

    Call rows           exist only once the media websocket opens — on answer
    Contact.last_attempt_at   overwritten by the next retry
    worker logs         replaced on every --build

Every out-of-hours call in the database turned out to predate the dial pump, but that was
the answer to a different question. Whether a dial rang unanswered that night could not be
known, and a system that cannot answer that about itself is not one to run a campaign on.

dial_attempts is the ledger. _place writes a row when it asks the carrier, whatever the
carrier says; the hangup callback completes it once; nothing updates or deletes it after.
"""

import ast
import asyncio
import inspect
import uuid
from datetime import datetime

import pytest

from app.models.db import Contact, ContactStatus, DialAttempt
from app.services import dial_pump


# --- the row is written when the carrier is asked --------------------------------------------


class _Recorder:
    """The slice of AsyncSession _place touches: it adds, the tick commits."""

    def __init__(self):
        self.added = []

    def add(self, row):
        self.added.append(row)


def _contact(attempts=1):
    return Contact(
        id=uuid.uuid4(),
        campaign_id=uuid.uuid4(),
        phone_number="+919604100447",
        name="Chandan",
        status=ContactStatus.DIALING,
        attempts=attempts,
    )


@pytest.fixture
def carrier(monkeypatch):
    """Stub everything _place talks to except the database. The dial's fate is set per test."""
    state = {"accept": True, "acquire": True, "raise": False}

    async def acquire(_sid):
        return state["acquire"]

    async def release(_sid):
        pass

    async def noop(*_a, **_k):
        pass

    async def trigger(_number, _campaign, _sid):
        if state["raise"]:
            raise RuntimeError("carrier down")
        return state["accept"]

    monkeypatch.setattr(dial_pump.call_slots, "acquire", acquire)
    monkeypatch.setattr(dial_pump.call_slots, "release", release)
    monkeypatch.setattr(dial_pump, "remember_dialed_number", noop)
    monkeypatch.setattr(dial_pump, "remember_customer_name", noop)
    monkeypatch.setattr(dial_pump, "remember_contact", noop)
    monkeypatch.setattr(dial_pump, "trigger_vobiz_call", trigger)
    return state


def _place(db, contact):
    return asyncio.run(dial_pump._place(db, contact))


def test_an_accepted_dial_writes_a_row(carrier):
    db, contact = _Recorder(), _contact(attempts=1)
    assert _place(db, contact) is True
    (row,) = db.added
    assert isinstance(row, DialAttempt)
    assert row.phone_number == "+919604100447"
    assert row.contact_id == contact.id
    assert row.campaign_id == contact.campaign_id
    assert row.attempt_no == 1
    assert row.carrier_accepted is True


def test_a_refused_dial_still_writes_a_row(carrier):
    """We asked. "Asked and refused at 20:03" is a fact about our dialling, and the whole
    point of the ledger is that no dial leaves without a trace."""
    carrier["accept"] = False
    db = _Recorder()
    assert _place(db, _contact()) is False
    (row,) = db.added
    assert row.carrier_accepted is False


def test_a_dial_that_raised_still_writes_a_row(carrier):
    carrier["raise"] = True
    db = _Recorder()
    assert _place(db, _contact()) is False
    (row,) = db.added
    assert row.carrier_accepted is False


def test_the_row_carries_the_sid_the_carrier_will_report_on(carrier):
    """The hangup callback arrives keyed on this sid; without it on the row there is nothing
    to complete."""
    db = _Recorder()
    _place(db, _contact())
    (row,) = db.added
    uuid.UUID(row.call_sid)  # a real sid, not empty


def test_the_retry_is_numbered_as_the_retry(carrier):
    db = _Recorder()
    _place(db, _contact(attempts=2))
    assert db.added[0].attempt_no == 2


def test_a_slot_starved_contact_writes_nothing(carrier):
    """No request went to the carrier, so there was no dial. A row here would say a number
    was rung when it was not — the opposite lie from the one this table fixes."""
    carrier["acquire"] = False
    db = _Recorder()
    assert _place(db, _contact()) is False
    assert db.added == []


def test_the_row_is_added_before_the_refusal_branch():
    """Placed by position, not by test alone: an editor moving the add below `if not ok`
    would drop every refused dial from the ledger and every test above would need the
    carrier to refuse to notice."""
    src = inspect.getsource(dial_pump._place)
    assert src.index("db.add(DialAttempt(") < src.index("if not ok:")


def test_the_row_shares_the_ticks_commit():
    """_place adds to the tick's session and never commits itself. Both the contact's state
    and the ledger row land in the one commit after the loop, or neither does."""
    tree = ast.parse(inspect.getsource(dial_pump._place).lstrip())
    calls = [
        ast.unparse(n.func) for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    ]
    # Checked as calls rather than as a substring: the comment beside the add mentions the
    # tick's commit by name, and a substring search would fail on its own explanation.
    assert not [c for c in calls if c.endswith(".commit")], calls


# --- the carrier's report completes it, once ----------------------------------------------


class _Ledger:
    """AsyncSession for record_dial_outcome: one row, found by sid or not."""

    def __init__(self, row):
        self.row = row
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        pass

    async def execute(self, _stmt):
        row = self.row

        class _Result:
            def scalars(self):
                return self

            def first(self):
                return row

        return _Result()

    async def commit(self):
        self.commits += 1


@pytest.fixture
def ledger(monkeypatch):
    holder = {}
    monkeypatch.setattr(dial_pump, "AsyncSessionLocal", lambda: holder["s"])

    def use(row):
        holder["s"] = _Ledger(row)
        return holder["s"]

    return use


def _row(**over):
    base = dict(
        call_sid="sid-1", phone_number="+919604100447", attempt_no=1,
        carrier_accepted=True, dialed_at=datetime(2026, 9, 9, 16, 39),
    )
    base.update(over)
    return DialAttempt(**base)


def test_a_ring_out_is_written_onto_the_row(ledger):
    s = ledger(_row())
    assert asyncio.run(dial_pump.record_dial_outcome("sid-1", False, "NO_ANSWER")) is True
    assert s.row.answered is False
    assert s.row.hangup_cause == "NO_ANSWER"
    assert s.row.ended_at is not None
    assert s.commits == 1


def test_an_answered_call_is_written_too(ledger):
    """record_carrier_outcome leaves answered calls to the session that served them. The
    ledger does not: "answered" is what the carrier said, and it goes on the row."""
    s = ledger(_row())
    assert asyncio.run(dial_pump.record_dial_outcome("sid-1", True, "NORMAL_CLEARING")) is True
    assert s.row.answered is True


def test_the_carriers_retry_does_not_overwrite_the_first_report(ledger):
    """Vobiz retries a failed callback up to three times. The first report describes the
    call; a later one must not move ended_at or change the cause."""
    s = ledger(_row(answered=False, hangup_cause="NO_ANSWER", ended_at=datetime(2026, 9, 9, 16, 40)))
    assert asyncio.run(dial_pump.record_dial_outcome("sid-1", True, "SOMETHING_ELSE")) is False
    assert s.row.hangup_cause == "NO_ANSWER"
    assert s.row.answered is False
    assert s.commits == 0


def test_a_sid_with_no_row_is_nothing_to_record(ledger):
    """Calls placed by hand, and every call older than the ledger."""
    s = ledger(None)
    assert asyncio.run(dial_pump.record_dial_outcome("unknown", False, "NO_ANSWER")) is False
    assert s.commits == 0


def test_an_empty_sid_is_refused_before_touching_the_database(ledger):
    s = ledger(_row())
    assert asyncio.run(dial_pump.record_dial_outcome("", False, "NO_ANSWER")) is False
    assert s.commits == 0


def test_the_cause_is_normalised_the_way_carrier_verdict_reads_it(ledger):
    s = ledger(_row())
    asyncio.run(dial_pump.record_dial_outcome("sid-1", False, "  no_answer "))
    assert s.row.hangup_cause == "NO_ANSWER"


def test_an_unreported_cause_is_stored_as_null_rather_than_empty(ledger):
    s = ledger(_row())
    asyncio.run(dial_pump.record_dial_outcome("sid-1", False, None))
    assert s.row.hangup_cause is None


def test_the_ledger_is_completed_regardless_of_the_contacts_state():
    """record_carrier_outcome is guarded on the contact still being DIALING, and rightly:
    it decides state. The ledger records history, and must not inherit that guard."""
    tree = ast.parse(inspect.getsource(dial_pump.record_dial_outcome).lstrip())
    fn = tree.body[0]
    # The docstring explains the guard it does not have, naming DIALING while doing so; a
    # substring search would report the opposite of the truth. Only the code is checked.
    body = [n for n in fn.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    code = "\n".join(ast.unparse(n) for n in body)
    assert "ContactStatus" not in code
    assert "DIALING" not in code


# --- the hangup webhook feeds it --------------------------------------------------------------


def test_the_hangup_webhook_completes_the_ledger():
    from app.api.routes import webhook

    src = inspect.getsource(webhook.vobiz_hangup)
    assert "record_dial_outcome(call_sid, answered, cause)" in src


def test_the_ledger_write_cannot_take_the_contact_update_down_with_it():
    """Its own try. A ledger failure — the table missing on an old deployment, say — must
    not stop the contact leaving DIALING, or the 200 the carrier is owed."""
    from app.api.routes import webhook

    tree = ast.parse(inspect.getsource(webhook.vobiz_hangup).lstrip())
    tries = [n for n in ast.walk(tree) if isinstance(n, ast.Try)]
    ledger_try = [t for t in tries if "record_dial_outcome" in ast.unparse(t)]
    assert len(ledger_try) == 1
    assert "record_carrier_outcome" not in ast.unparse(ledger_try[0])


def test_the_slot_is_released_before_the_ledger_is_touched():
    from app.api.routes import webhook

    src = inspect.getsource(webhook.vobiz_hangup)
    assert src.index("call_slots.release") < src.index("record_dial_outcome")


# --- the table itself ---------------------------------------------------------------------------


def test_the_migration_exists_and_chains_from_the_current_head():
    import pathlib

    path = pathlib.Path("alembic/versions/e7c4a2d91f36_dial_attempts.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert 'down_revision: Union[str, Sequence[str], None] = "d5b81f0c3a72"' in text
    assert '"dial_attempts"' in text


def test_the_sid_is_unique_on_the_table():
    """Two rows for one dial would double-count every attempt in any report off this table."""
    assert DialAttempt.__table__.c.call_sid.unique is True


def test_deleting_a_campaign_does_not_delete_its_dialling_history():
    """contacts cascade from campaigns. The ledger must not cascade from contacts, or the
    record of what was dialled goes with the list it was dialled from."""
    fk = next(iter(DialAttempt.__table__.c.contact_id.foreign_keys))
    assert fk.ondelete == "SET NULL"
    assert DialAttempt.__table__.c.campaign_id.foreign_keys == set()


def test_the_reason_is_written_on_the_model():
    doc = DialAttempt.__doc__
    assert "9 Sep 2026" in doc
    assert "never overwritten" in doc
