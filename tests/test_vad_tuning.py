"""Barge-in sensitivity.

The agent was cut off 4.1s into a 10s sentence while the caller's next utterance did not
begin for another 6 seconds — nobody interrupted it. min_volume was 0.1 against Pipecat's
default of 0.6, so PSTN line noise cleared the bar, and because STT transcribed nothing
the false trigger produced no log line at all.
"""

import ast
import inspect

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


def test_stop_secs_leaves_the_stt_wait_window_open():
    """Pipecat waits max(0, stt_p99 - stop_secs) for transcripts before its turn analyzer
    rules on the turn. Running at 0.6 against Deepgram's 0.35 collapsed that to zero, and
    Pipecat logged a warning saying so on every call."""
    from pipecat.services.deepgram.stt import DEEPGRAM_TTFS_P99

    assert settings.VAD_STOP_SECS < DEEPGRAM_TTFS_P99, (
        f"stop_secs {settings.VAD_STOP_SECS} >= STT p99 {DEEPGRAM_TTFS_P99}; "
        "the transcript wait window collapses to 0 and turn detection runs blind"
    )


def test_stop_secs_matches_what_pipecat_is_calibrated_for():
    """Its built-in latency figures assume the default; departing from it warns at runtime."""
    from pipecat.audio.vad.vad_analyzer import VAD_STOP_SECS

    assert settings.VAD_STOP_SECS == VAD_STOP_SECS


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
