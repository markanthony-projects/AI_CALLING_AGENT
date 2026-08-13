"""A call that will not end by itself costs money for as long as it runs.

On call 9b405c1d Sarvam ran out of credits, the pipeline failed to tear itself down, and
nothing in the system noticed:

    09:30:46.792  TTS unavailable after 3 errors; abandoning call
    09:30:50.096  AGENT -> "I understand, Chandan. Please go ahead..."
    (no "Pipeline finished", no "Call finalised" — ever)

The websocket stayed open, so Vobiz kept the phone leg up and kept billing it, one of only
four concurrency slots was gone, and the Call row said IN_PROGRESS until the box was
restarted by hand.

Three things have to hold: the abandon must actually abandon (test_tts_failure.py), a call
must not be able to run for ever whatever the cause, and a row nobody closed must not read
as a live call.
"""

import ast
import inspect
from datetime import timedelta

import pytest

from app.services import agent
from app.services.agent import (
    MAX_LLM_TURN_FAILURES,
    IDLE_TIMEOUT_SECS,
    MAX_CALL_DURATION_SECS,
    MAX_TTS_FAILURES,
    session_error,
)
from app.services.stale_calls import STALE_AFTER, SWEEP_EVERY_SECONDS


def _fn(name):
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in run_voice_agent")


# ─── the duration cap ─────────────────────────────────────────────────────────────────


def test_the_idle_timeout_cannot_do_this_job():
    """It fires only when NEITHER party has spoken. A caller saying "hello?" into a line
    whose voice has died resets it every time, so it can be held off indefinitely by exactly
    the person we are billing for. The cap needs to be independent of who is talking."""
    src = ast.unparse(_fn("enforce_max_duration"))
    assert "asyncio.sleep(MAX_CALL_DURATION_SECS)" in src
    # Nothing about speech, turns or activity may gate it.
    for reset in ("_turn_start_time", "on_user_turn", "_idle", "user_speaking"):
        assert reset not in src


def test_the_cap_is_far_above_a_real_conversation():
    """A backstop against a broken pipeline, not a policy on how long a prospect may talk.
    Cutting off a genuine conversation would be the worse failure of the two."""
    assert MAX_CALL_DURATION_SECS >= 5 * 60
    assert MAX_CALL_DURATION_SECS > IDLE_TIMEOUT_SECS * 5


def test_the_cap_cancels_rather_than_queueing_a_frame():
    """Same reasoning as the TTS path: a queued frame waits behind whatever is wedged, and
    a wedged pipeline is the case this exists for."""
    src = ast.unparse(_fn("enforce_max_duration"))
    assert "abandon_call" in src
    assert "EndFrame" not in src


def test_the_guard_does_not_outlive_the_call():
    """A sleeping task holds the whole call's closure alive. Ten minutes of that per call on
    a box capped at four concurrent calls is a leak worth closing."""
    src = inspect.getsource(agent.run_voice_agent)
    assert "_duration_guard.cancel()" in src
    # In the finally, so a pipeline exception cannot skip it.
    tree = ast.parse(src.lstrip())
    finallys = [
        ast.unparse(n) for node in ast.walk(tree) if isinstance(node, ast.Try)
        for n in [node.finalbody] if n
    ]
    assert any("_duration_guard.cancel()" in f for f in finallys)


def test_a_capped_call_is_recorded_as_failed():
    reason = session_error(None, 0, False, 0, False, True)
    assert reason is not None
    assert "duration" in reason.lower()


def test_a_normal_call_is_not_failed_by_the_cap():
    assert session_error(None, 0, False, 0, False, False) is None


@pytest.mark.parametrize(
    "llm_failures, tts_failures, expect",
    [
        (0, MAX_TTS_FAILURES, "tts"),
        (MAX_LLM_TURN_FAILURES, 0, "llm"),
    ],
)
def test_a_named_cause_outranks_the_cap(llm_failures, tts_failures, expect):
    """The cap only fires when nothing else ended the call, so when a specific cause is also
    set that one is the story and the overrun is its symptom. Reporting "ran too long" for a
    TTS outage would send the next person looking in the wrong place. Every named cause has
    to outrank it, not just the first one that happens to be checked."""
    reason = session_error(None, llm_failures, False, tts_failures, False, True)
    assert expect in reason.lower()
    assert "duration" not in reason.lower()


# ─── the reaper ───────────────────────────────────────────────────────────────────────


def test_the_reaper_waits_longer_than_the_cap():
    """Otherwise it races the process still holding a call the cap is about to end, and
    could overwrite a real COMPLETED with FAILED."""
    assert STALE_AFTER > timedelta(seconds=MAX_CALL_DURATION_SECS)


def test_it_sweeps_on_a_timer_not_only_at_startup():
    """The usual cause is a process that died, which is therefore not around to clean up
    after itself on the way back."""
    assert 0 < SWEEP_EVERY_SECONDS <= 3600


def test_it_discriminates_on_age_and_not_on_process_state():
    """Reaping "everything IN_PROGRESS at startup" is correct only while exactly one process
    serves calls. With two, restarting one would mark the other's live calls failed."""
    from app.services import stale_calls

    src = inspect.getsource(stale_calls)
    assert "Call.started_at < cutoff" in src
    for process_state in ("_STREAMING_CALLS", "ACTIVE_CALLS"):
        assert process_state not in src, "process state cannot decide this; age must"


# ─── what the reaper actually writes ──────────────────────────────────────────────────


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeSession:
    """Enough of AsyncSession to see what the reaper decided, without standing up Postgres."""

    def __init__(self, stale_rows):
        self._stale = stale_rows
        self.updates = []
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def execute(self, statement):
        compiled = str(statement)
        if compiled.lstrip().upper().startswith("SELECT"):
            return _FakeResult(self._stale)
        self.updates.append(statement)
        return _FakeResult([])

    async def commit(self):
        self.committed = True


def _run_reaper(monkeypatch, rows):
    import asyncio

    from app.services import stale_calls

    session = _FakeSession(rows)
    monkeypatch.setattr(stale_calls, "AsyncSessionLocal", lambda: session)
    count = asyncio.run(stale_calls.reap_stale_calls())
    return count, session


class _Row:
    def __init__(self, started_at):
        self.id = "row-id"
        self.call_sid = "9b405c1d"
        self.started_at = started_at


def test_a_quiet_sweep_writes_nothing(monkeypatch):
    """It runs every fifteen minutes for the life of the process. Committing an empty
    transaction each time is pointless load on the managed database."""
    count, session = _run_reaper(monkeypatch, [])
    assert count == 0
    assert not session.committed
    assert not session.updates


def test_an_abandoned_row_is_closed_with_a_duration(monkeypatch):
    from app.utils.timeutils import utc_now

    started = utc_now() - timedelta(hours=3)
    count, session = _run_reaper(monkeypatch, [_Row(started)])

    assert count == 1
    assert session.committed
    assert len(session.updates) == 1
    values = session.updates[0].compile().params
    assert values["status"].value == "FAILED"
    assert values["ended_at"] is not None
    # An upper bound on the real duration, which is the honest direction to be wrong in.
    assert values["duration_seconds"] >= 3 * 3600


def test_the_sweep_cannot_stop_the_app_serving_calls():
    """It runs in the lifespan. An exception escaping the loop would kill the sweeper for the
    life of the process and, uncaught at startup, could take the app with it."""
    from app import main

    src = inspect.getsource(main.lifespan)
    tree = ast.parse(src.lstrip())
    sweep = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef)) and "sweep" in n.name
    )
    assert any(isinstance(n, ast.Try) for n in ast.walk(sweep))


def test_the_sweeper_is_stopped_on_shutdown():
    from app import main

    assert "sweeper.cancel()" in inspect.getsource(main.lifespan)


@pytest.mark.parametrize("field", ["ended_at", "duration_seconds"])
def test_a_reaped_row_is_not_left_half_finished(field):
    """Leaving ended_at null makes duration silently wrong and sorts these rows apart from
    real endings in the dashboard."""
    import app.services.stale_calls as sc

    assert field in inspect.getsource(sc.reap_stale_calls)
