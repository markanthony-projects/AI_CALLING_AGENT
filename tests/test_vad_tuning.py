"""Barge-in sensitivity.

The agent was cut off 4.1s into a 10s sentence while the caller's next utterance did not
begin for another 6 seconds — nobody interrupted it. min_volume was 0.1 against Pipecat's
default of 0.6, so PSTN line noise cleared the bar, and because STT transcribed nothing
the false trigger produced no log line at all.
"""

import ast
import inspect

import pytest
from pipecat.audio.vad.vad_analyzer import (
    VAD_CONFIDENCE,
    VAD_MIN_VOLUME,
    VADParams,
)

from app.core.config import settings
from app.services import agent


def test_min_volume_is_not_wildly_below_pipecat_default():
    """0.1 against a 0.6 default is six times more sensitive than Silero expects."""
    assert settings.VAD_MIN_VOLUME >= VAD_MIN_VOLUME / 2


def test_confidence_is_at_least_pipecat_default():
    assert settings.VAD_CONFIDENCE >= VAD_CONFIDENCE


def test_stop_secs_survives_a_mid_sentence_pause():
    """At 0.2s "Yeah sure. Sunday works for me." split into two turns 328ms apart, running
    two LLM inferences: one asked the time, the other hung up. Measured pauses ran 234-708ms."""
    assert settings.VAD_STOP_SECS >= 0.5


def test_stop_secs_does_not_swallow_the_latency_budget():
    """This is added wait on every turn, on top of a ~730ms voice-to-voice p50."""
    assert settings.VAD_STOP_SECS <= 0.8


def test_vad_settings_remain_a_valid_vadparams():
    """Guards against a typo in .env producing an unusable analyser at call time."""
    params = VADParams(
        min_volume=settings.VAD_MIN_VOLUME,
        confidence=settings.VAD_CONFIDENCE,
        stop_secs=settings.VAD_STOP_SECS,
    )
    assert 0.0 <= params.min_volume <= 1.0
    assert 0.0 <= params.confidence <= 1.0
    assert params.stop_secs > 0


def test_vad_is_configurable_not_hardcoded():
    """These need tuning against real calls; recompiling to try a value is not tuning."""
    source = inspect.getsource(agent.run_voice_agent)
    start = source.index("VADParams(")
    block = source[start : source.index(")", source.index("stop_secs", start))]
    for field in ("min_volume", "confidence", "stop_secs"):
        assert f"settings.VAD_{field.upper()}" in block, f"{field} is hardcoded"


def test_false_barge_ins_are_logged():
    """A noise-triggered turn transcribes to nothing, so without this it is invisible."""
    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    handler = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        and n.name == "on_user_turn_stopped"
    )
    src = ast.unparse(handler)
    assert "logger.warning" in src, "an empty user turn must be reported, not dropped"
    assert "_empty_user_turns" in src, "false barge-ins must be counted for the call"
