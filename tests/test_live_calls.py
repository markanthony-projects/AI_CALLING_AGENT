"""The carrier's hangup ends the session it names, instead of leaving it to time out.

Call 8571d93b, 15 Sep 2026: hung up at eight seconds, filed as an idle timeout at seventy.
"""

import asyncio
import inspect

import pytest

from app.services import live_calls


@pytest.fixture(autouse=True)
def clean():
    live_calls._enders.clear()
    yield
    live_calls._enders.clear()


def test_a_registered_session_is_ended_once_with_the_reason():
    reasons = []

    async def ender(reason):
        reasons.append(reason)

    live_calls.register("c1", ender)
    assert live_calls.live() == 1
    assert asyncio.run(live_calls.end("c1", "the carrier reported the hangup")) is True
    assert reasons == ["the carrier reported the hangup"]
    assert asyncio.run(live_calls.end("c1", "again")) is False, "gone once ended"
    assert live_calls.live() == 0


def test_a_session_this_process_does_not_have_is_not_an_error():
    assert asyncio.run(live_calls.end("nobody", "x")) is False


def test_a_session_already_stopping_does_not_raise_into_the_webhook():
    async def ender(reason):
        raise RuntimeError("task finished")

    live_calls.register("c1", ender)
    assert asyncio.run(live_calls.end("c1", "x")) is False


def test_forgetting_is_idempotent():
    live_calls.forget("never")
    live_calls.register("c1", lambda r: None)
    live_calls.forget("c1")
    live_calls.forget("c1")
    assert live_calls.live() == 0


def test_the_agent_registers_before_running_and_forgets_after():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert src.index("live_calls.register(call_sid, end_on_request)") < src.index("await runner.run(task)")
    after = src[src.index("await runner.run(task)") :]
    assert "live_calls.forget(call_sid)" in after[: after.index("# The guard outlives the pipeline")]
    # And the stream closing on its own forgets first, so the hangup cannot end it twice.
    closed = src[src.index("Media stream closed — ending pipeline") :]
    assert closed.index("live_calls.forget(call_sid)") < closed.index("EndFrame(reason=")


def test_the_hangup_route_ends_only_answered_calls():
    from app.api.routes import webhook

    src = inspect.getsource(webhook.vobiz_hangup)
    assert 'if answered and await live_calls.end(call_sid, "the carrier reported the hangup")' in src
    # After the slot is released and the ledger written — ending the session must not be
    # able to delay either.
    assert src.index("call_slots.release(call_sid)") < src.index("live_calls.end(")
    assert src.index("record_dial_outcome(") < src.index("live_calls.end(")


def test_that_reason_finalises_the_call_as_completed():
    """An EndFrame reason reaches the result as end_reason, and any end_reason that is not a
    voicemail is COMPLETED — which is right: the prospect answered and hung up."""
    from app.api.routes import webhook

    src = inspect.getsource(webhook._handle_call)
    assert "end_reason = result.end_reason" in src
    assert "status = CallStatus.COMPLETED" in src
