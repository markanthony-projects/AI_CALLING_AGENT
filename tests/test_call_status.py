"""Every call used to be recorded COMPLETED, including the ones that crashed.

These pin the outcome mapping so failed calls are visible in the data rather than
hiding inside the success rate.
"""

from contextlib import asynccontextmanager

import pytest

from app.api.routes import webhook
from app.models.db import Call, CallStatus, Transcript
from app.services.agent import MAX_LLM_TURN_FAILURES, CallResult, session_error


# --- the rule behind FAILED vs COMPLETED ------------------------------------------


def test_clean_session_has_no_error():
    assert session_error(None, 0) is None


def test_one_recovered_llm_failure_is_not_a_failed_call():
    """The agent apologises and carries on; the call is still worth counting as served."""
    assert MAX_LLM_TURN_FAILURES > 1
    assert session_error(None, MAX_LLM_TURN_FAILURES - 1) is None


def test_exhausted_llm_failures_fail_the_session():
    assert session_error(None, MAX_LLM_TURN_FAILURES) is not None


def test_pipeline_error_fails_the_session_and_is_preserved():
    assert session_error("pipeline: transport died", 0) == "pipeline: transport died"


def test_pipeline_error_wins_over_llm_failures():
    assert session_error("pipeline: boom", MAX_LLM_TURN_FAILURES) == "pipeline: boom"


def test_idle_timeout_fails_the_session():
    """A carrier that drops the phone leg without closing the websocket is not a served call."""
    assert session_error(None, 0, idle_timed_out=True) is not None


def test_idle_timeout_is_reported_as_idle():
    assert "idle" in session_error(None, 0, idle_timed_out=True).lower()


def test_idle_timeout_is_shorter_than_pipecat_default():
    """300s of a wedged leg holds one of MAX_CALLS slots for five minutes."""
    from app.services.agent import IDLE_TIMEOUT_SECS

    assert 0 < IDLE_TIMEOUT_SECS <= 120


def test_idle_timeout_is_wired_into_the_pipeline_worker():
    """Source-level guard: the constant is inert unless PipelineWorker is told about it."""
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent))
    workers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "PipelineWorker"
    ]
    assert workers, "run_voice_agent no longer constructs a PipelineWorker"

    kwargs = {kw.arg for kw in workers[0].keywords}
    assert "idle_timeout_secs" in kwargs, (
        "without idle_timeout_secs the pipeline falls back to Pipecat's 300s default"
    )
    assert "cancel_on_idle_timeout" in kwargs


def test_run_voice_agent_wires_session_error_into_its_result():
    """Source-level guard, not a behavioural one.

    run_voice_agent needs a live websocket and Pipecat pipeline to execute, so this one
    seam cannot be exercised in a smoke test. Without it, dropping session_error() from
    the return silently restores the "every call COMPLETED" bug.
    """
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent))
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value]
    assert returns, "run_voice_agent has no return statement"

    wired = any(
        isinstance(r.value, ast.Call)
        and getattr(r.value.func, "id", None) == "CallResult"
        and any(
            kw.arg == "error"
            and isinstance(kw.value, ast.Call)
            and getattr(kw.value.func, "id", None) == "session_error"
            for kw in r.value.keywords
        )
        for r in returns
    )
    assert wired, "run_voice_agent must return CallResult(error=session_error(...))"


class FakeResult:
    def __init__(self, value):
        self._value = value

    def scalars(self):
        return self

    def first(self):
        return self._value


class FakeSession:
    """Minimal AsyncSession good enough for the two write paths under test."""

    def __init__(self, call_record=None, existing_transcript=None):
        self.call_record = call_record
        self.existing_transcript = existing_transcript
        self.added = []
        self.commits = 0

    async def execute(self, *args, **kwargs):
        return FakeResult(self.call_record)

    async def scalar(self, *args, **kwargs):
        return self.existing_transcript

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1


@pytest.fixture
def session_factory(monkeypatch):
    """Route every AsyncSessionLocal() in webhook.py to one inspectable session."""
    holder = {}

    def _install(session):
        holder["session"] = session

        @asynccontextmanager
        async def factory():
            yield session

        monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
        return session

    return _install


async def test_finalize_records_failed_status(session_factory):
    call = Call(call_sid="sid-1", status=CallStatus.IN_PROGRESS)
    session = session_factory(FakeSession(call_record=call))

    await webhook._finalize_call("sid-1", webhook.utc_now(), "", CallStatus.FAILED)

    assert call.status == CallStatus.FAILED
    assert call.ended_at is not None
    assert session.commits == 1


async def test_finalize_records_completed_status(session_factory):
    call = Call(call_sid="sid-2", status=CallStatus.IN_PROGRESS)
    session_factory(FakeSession(call_record=call))

    await webhook._finalize_call("sid-2", webhook.utc_now(), "Agent: hi", CallStatus.COMPLETED)

    assert call.status == CallStatus.COMPLETED


async def test_failed_call_still_persists_and_enqueues_partial_transcript(
    session_factory, monkeypatch
):
    """A crashed session's partial transcript can still contain a qualified lead."""
    call = Call(call_sid="sid-3", status=CallStatus.IN_PROGRESS)
    session = session_factory(FakeSession(call_record=call))

    enqueued = []

    async def fake_enqueue(sid):
        enqueued.append(sid)

    monkeypatch.setattr(webhook, "enqueue_extraction", fake_enqueue)

    await webhook._finalize_call(
        "sid-3", webhook.utc_now(), "Agent: hi\nProspect: budget 75 lakh", CallStatus.FAILED
    )

    assert call.status == CallStatus.FAILED
    assert any(isinstance(o, Transcript) for o in session.added)
    assert enqueued == ["sid-3"]


async def test_missing_call_record_does_not_enqueue(session_factory, monkeypatch):
    session_factory(FakeSession(call_record=None))
    enqueued = []

    async def fake_enqueue(sid):
        enqueued.append(sid)

    monkeypatch.setattr(webhook, "enqueue_extraction", fake_enqueue)

    await webhook._finalize_call("sid-4", webhook.utc_now(), "Agent: hi", CallStatus.COMPLETED)

    assert enqueued == []


# --- outcome mapping through _handle_call -----------------------------------------


class FakeWebSocket:
    def __init__(self):
        self.closed_with = None

    async def accept(self):
        pass

    async def close(self, code=1000):
        self.closed_with = code


@pytest.fixture
def handle_call_env(session_factory, monkeypatch):
    """Drive _handle_call with the voice session and project lookup stubbed out."""

    def _setup(*, agent_result=None, agent_exc=None, project={"name": "P"}):
        call = Call(call_sid="sid", status=CallStatus.IN_PROGRESS)
        session_factory(FakeSession(call_record=call))

        async def fake_project(db, campaign_id):
            return project

        async def fake_agent(*args, **kwargs):
            if agent_exc:
                raise agent_exc
            return agent_result

        async def fake_enqueue(sid):
            return None

        monkeypatch.setattr(webhook, "get_project_by_campaign", fake_project)
        monkeypatch.setattr(webhook, "run_voice_agent", fake_agent)
        monkeypatch.setattr(webhook, "build_campaign_context", lambda p: "ctx")
        monkeypatch.setattr(webhook, "enqueue_extraction", fake_enqueue)
        return call

    return _setup


async def test_clean_session_is_completed(handle_call_env):
    call = handle_call_env(agent_result=CallResult(transcript="Agent: hi"))
    await webhook._handle_call(FakeWebSocket(), "c1", "sid", "vobiz")
    assert call.status == CallStatus.COMPLETED


async def test_llm_failure_is_recorded_failed(handle_call_env):
    """The exact shape of the logged call: Groq rejected the tool call, turn produced nothing."""
    call = handle_call_env(
        agent_result=CallResult(transcript="Agent: hi", error="llm turn failures exhausted")
    )
    await webhook._handle_call(FakeWebSocket(), "c1", "sid", "vobiz")
    assert call.status == CallStatus.FAILED


async def test_pipeline_exception_is_recorded_failed(handle_call_env):
    call = handle_call_env(agent_exc=RuntimeError("transport died"))
    await webhook._handle_call(FakeWebSocket(), "c1", "sid", "vobiz")
    assert call.status == CallStatus.FAILED


async def test_missing_project_is_recorded_failed(handle_call_env):
    call = handle_call_env(project=None)
    ws = FakeWebSocket()
    await webhook._handle_call(ws, "c1", "sid", "vobiz")
    assert call.status == CallStatus.FAILED
    assert ws.closed_with == 1008


async def test_active_call_counter_is_released_on_failure(handle_call_env):
    handle_call_env(agent_exc=RuntimeError("boom"))
    before = webhook.ACTIVE_CALLS
    await webhook._handle_call(FakeWebSocket(), "c1", "sid", "vobiz")
    assert webhook.ACTIVE_CALLS == before
