"""TTS outage handling.

Sarvam ran out of credits mid-call. Only the LLM had an on_error handler, so the pipeline
limped on for 17 seconds emitting no audio at all: the caller said "Hello" twice and hung
up, and the call was recorded as if nothing was wrong.
"""

import ast
import inspect

import pytest

from app.services import agent
from app.services.agent import MAX_LLM_TURN_FAILURES, MAX_TTS_FAILURES, session_error


def _handler(name):
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in run_voice_agent")


def _decorator_targets(name):
    """Which service each on_error handler is attached to."""
    node = _handler(name)
    out = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Call) and getattr(dec.func, "attr", None) == "event_handler":
            out.append((getattr(dec.func.value, "id", None), dec.args[0].value))
    return out


# --- the gap that let this happen ------------------------------------------------


def test_tts_has_its_own_error_handler():
    assert ("tts", "on_error") in _decorator_targets("on_tts_error")


def test_llm_error_handler_still_covers_the_llm():
    assert ("llm", "on_error") in _decorator_targets("on_llm_error")


# --- outcome ---------------------------------------------------------------------


def test_tts_outage_fails_the_call():
    """It was previously recorded COMPLETED — indistinguishable from a real conversation."""
    assert session_error(None, 0, False, MAX_TTS_FAILURES) is not None


def test_tts_outage_says_the_caller_heard_silence():
    reason = session_error(None, 0, False, MAX_TTS_FAILURES)
    assert "tts" in reason.lower() and "silence" in reason.lower()


def test_a_single_tts_blip_does_not_fail_the_call():
    """Pipecat reconnects the socket; one error is not an outage."""
    assert session_error(None, 0, False, 1) is None


def test_tts_outage_outranks_an_llm_failure():
    """With no voice the LLM's state is irrelevant — report the cause that ended the call."""
    reason = session_error(None, MAX_LLM_TURN_FAILURES, False, MAX_TTS_FAILURES)
    assert "tts" in reason.lower()


def test_pipeline_error_still_outranks_tts():
    assert session_error("pipeline: boom", 0, False, MAX_TTS_FAILURES) == "pipeline: boom"


# --- behaviour of the handler ----------------------------------------------------


def test_handler_ends_the_call_without_trying_to_speak():
    """A farewell needs TTS. Queueing TTSSpeakFrame here would be a no-op that delays hangup."""
    src = ast.unparse(_handler("on_tts_error"))
    assert "TTSSpeakFrame" not in src, "cannot speak a goodbye when TTS is the thing that failed"


def test_an_end_frame_cannot_end_a_call_whose_tts_is_wedged():
    """This handler used to queue an EndFrame, and on call 9b405c1d that did nothing at all:

        09:30:46.792  TTS unavailable after 3 errors; abandoning call
        09:30:50.096  AGENT -> "I understand, Chandan. Please go ahead..."

    An assistant turn finalized 3.3s after the abandon, and neither "Pipeline finished" nor
    "Call finalised" ever appeared. EndFrame is a ControlFrame, so it travels in queue order
    and has to pass through the very processor that is wedged. The websocket stayed open,
    Vobiz kept billing the phone leg, and a concurrency slot was held until a manual restart.

    CancelFrame is a SystemFrame and bypasses the queue; task.cancel() is what queues one.
    """
    src = ast.unparse(_handler("on_tts_error"))
    assert "EndFrame" not in src, "EndFrame drains through the dead TTS and never arrives"
    assert "abandon_call" in src


def test_abandon_call_cancels_rather_than_ending():
    src = ast.unparse(_handler("abandon_call"))
    assert "task.cancel" in src
    assert "queue_frames" not in src, "queued frames wait behind whatever is wedged"


def test_abandon_call_survives_its_own_failure():
    """It runs inside an event handler that fires many times a second. An exception escaping
    here would be raised over and over and would take the duration cap down with it."""
    node = _handler("abandon_call")
    assert any(isinstance(n, ast.Try) for n in ast.walk(node))


def test_the_handler_keeps_trying_after_the_first_attempt():
    """It used to fire on == MAX_TTS_FAILURES, exactly once, on the reasoning that repeating
    it was noise. When that one attempt failed to take effect there was nothing behind it,
    and Sarvam went on erroring many times a second with no further action taken."""
    src = ast.unparse(_handler("on_tts_error"))
    assert ">= MAX_TTS_FAILURES" in src
    assert "== MAX_TTS_FAILURES" not in src


def test_handler_ignores_fatal_errors():
    """Pipecat tears the pipeline down itself on fatal; racing it would double-end the call."""
    assert "if error.fatal" in ast.unparse(_handler("on_tts_error"))


def test_failure_count_reaches_session_error():
    """The counter is inert if the call site passes a literal instead of the tallied value."""
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "session_error"
    ]
    assert calls, "run_voice_agent no longer calls session_error"
    names = {a.id for a in calls[0].args if isinstance(a, ast.Name)}
    assert "_tts_failures" in names, "session_error must receive the tallied TTS failures"
    assert "_llm_failures" in names


@pytest.mark.parametrize("threshold", [MAX_TTS_FAILURES])
def test_threshold_is_small_enough_to_matter(threshold):
    """17s of silence is what the caller experienced; the budget must be a few errors."""
    assert 1 < threshold <= 5
