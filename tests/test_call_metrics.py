"""The numbers a call is judged by reach a table, not only a rotating log.

Asked for the week's p50 on 12 Sep 2026, the answer was grep. Every figure was on a
LATENCY, FIRST WORD or reconnect line, and the lines rotate after a couple of days. This
is the row that survives them, assembled from the same counters the lines print so the
two can never disagree.
"""

import asyncio
import contextlib
from datetime import datetime
from types import SimpleNamespace

import pytest

from app.models.db import CallMetrics, CallStatus
from app.utils.call_metrics import CallMetricsSnapshot

SUMMARY = {"turns": 7, "p50_ms": 812, "p95_ms": 1410, "min_ms": 500, "max_ms": 1900}


def _snapshot(summary=SUMMARY, **overrides):
    kwargs = dict(
        latency_summary=summary,
        first_word_ms=1120,
        held_replies=1,
        tts_reconnects=0,
        tts_revivals=0,
        llm_failures=0,
        end_reason="end_call tool",
        stt="deepgram/nova-2-general",
        llm="cerebras/gpt-oss-120b",
    )
    kwargs.update(overrides)
    return CallMetricsSnapshot.collect(**kwargs)


# --- the row -------------------------------------------------------------------------------


def test_the_latency_summary_becomes_columns():
    row = _snapshot().as_row()
    assert row["turns"] == 7
    assert row["p50_turn_ms"] == 812
    assert row["p95_turn_ms"] == 1410
    assert row["max_turn_ms"] == 1900
    assert row["first_word_ms"] == 1120
    assert row["end_reason"] == "end_call tool"


def test_a_call_with_no_measurable_turn_is_a_real_row_not_a_missing_one():
    """A voicemail, or a hang-up during the greeting. Zero turns with a first-word time is
    exactly what a call nobody spoke on looks like, and it belongs in the table."""
    row = _snapshot(summary=None).as_row()
    assert row["turns"] == 0
    assert row["p50_turn_ms"] is None and row["p95_turn_ms"] is None
    assert row["first_word_ms"] == 1120


def test_every_column_the_table_has_is_filled_and_nothing_else():
    """The row is spread into the model with **kwargs, so a column added to one side and
    not the other is a TypeError on the first finalised call — this catches it in CI."""
    table = {c.name for c in CallMetrics.__table__.columns} - {"id", "call_id", "created_at"}
    assert set(_snapshot().as_row()) == table


# --- finalisation writes it ----------------------------------------------------------------


class _Scalars:
    def __init__(self, call):
        self._call = call

    def first(self):
        return self._call


class _Result:
    def __init__(self, call):
        self._call = call

    def scalars(self):
        return _Scalars(self._call)


class _Session:
    """Just enough of AsyncSession for _finalize_call: the call exists, nothing else does."""

    def __init__(self, call):
        self.call = call
        self.added = []
        self.committed = False

    async def execute(self, *_):
        return _Result(self.call)

    async def scalar(self, *_):
        return None

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.committed = True


@pytest.fixture
def finalise(monkeypatch):
    from app.api.routes import webhook

    call = SimpleNamespace(id="call-row-id", status=None, ended_at=None, duration_seconds=None)
    session = _Session(call)

    @contextlib.asynccontextmanager
    async def _session_factory():
        yield session

    monkeypatch.setattr(webhook, "AsyncSessionLocal", _session_factory)

    async def _no_extraction(call_sid):
        return True

    monkeypatch.setattr(webhook, "enqueue_extraction", _no_extraction)
    return webhook._finalize_call, session


def test_finalisation_writes_the_metrics_row_in_the_same_transaction(finalise):
    finalize_call, session = finalise
    asyncio.run(
        finalize_call("sid", datetime(2026, 9, 14, 9, 0), "Agent: hi", CallStatus.COMPLETED, _snapshot())
    )
    rows = [o for o in session.added if isinstance(o, CallMetrics)]
    assert len(rows) == 1
    assert rows[0].call_id == "call-row-id"
    assert rows[0].p50_turn_ms == 812
    assert session.committed


def test_a_call_that_could_not_count_writes_no_row(finalise):
    finalize_call, session = finalise
    asyncio.run(finalize_call("sid", datetime(2026, 9, 14, 9, 0), "", CallStatus.FAILED, None))
    assert not any(isinstance(o, CallMetrics) for o in session.added)
    assert session.committed


def test_the_agent_hands_the_snapshot_back_on_the_result():
    """Built at the end of run_voice_agent from the observer, the gate and the counters."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "metrics=CallMetricsSnapshot.collect(" in src
    for counter in ("first_word_ms=latency.first_word_ms", "held_replies=turn_gate.dropped", "tts_revivals=tts.revivals"):
        assert counter in src, counter
