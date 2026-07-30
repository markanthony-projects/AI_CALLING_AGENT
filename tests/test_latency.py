"""Latency instrumentation.

The number the call logs showed was `on_assistant_turn_stopped` — the agent *finishing*
speaking — so it moved by seconds when reply length changed while true response time did
not. These pin the measurement that replaced it: user stops talking -> caller hears audio.
"""

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    MetricsFrame,
    TextFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed

from app.utils.latency import NS_PER_SEC, LatencyObserver, _short, percentile


def pushed(frame, at_seconds: float) -> FramePushed:
    return FramePushed(
        source=None, destination=None, frame=frame,
        direction=None, timestamp=int(at_seconds * NS_PER_SEC),
    )


async def drive(observer, events):
    for frame, at in events:
        await observer.on_push_frame(pushed(frame, at))


# --- the core measurement --------------------------------------------------------


async def test_measures_user_stop_to_bot_start():
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 10.0),
        (BotStartedSpeakingFrame(), 11.2),
    ])
    assert obs.turns == pytest.approx([1.2])


async def test_reply_length_does_not_affect_the_number():
    """The old metric grew with TTS playback; this one must not."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 0.0),
        (BotStartedSpeakingFrame(), 1.0),
        (TextFrame("a forty word reply that takes fifteen seconds to speak"), 1.1),
        (UserStoppedSpeakingFrame(), 30.0),
        (BotStartedSpeakingFrame(), 31.0),
    ])
    assert obs.turns == pytest.approx([1.0, 1.0])


async def test_opening_greeting_is_not_counted():
    """The bot speaks first; there is no user turn to measure it against."""
    obs = LatencyObserver("sid")
    await drive(obs, [(BotStartedSpeakingFrame(), 5.0)])
    assert obs.turns == []
    assert obs.summary() is None


async def test_duplicate_pushes_use_the_first_timestamp():
    """One frame is pushed between every pair of processors, gaining time at each hop.

    Taking the last hop's timestamp would silently under-report latency by the pipeline's
    own traversal time, so the first sighting of a frame is the one that counts.
    """
    obs = LatencyObserver("sid")
    start, speak = UserStoppedSpeakingFrame(), BotStartedSpeakingFrame()
    for hop in range(5):
        await obs.on_push_frame(pushed(start, 10.0 + hop * 0.05))
    for hop in range(5):
        await obs.on_push_frame(pushed(speak, 11.0 + hop * 0.05))
    assert obs.turns == pytest.approx([1.0]), "must measure first-seen to first-seen"


async def test_second_bot_start_without_a_new_user_turn_is_ignored():
    """A resumed turn must not be recorded against a stale start."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 0.0),
        (BotStartedSpeakingFrame(), 1.0),
        (BotStartedSpeakingFrame(), 4.0),
    ])
    assert obs.turns == pytest.approx([1.0])


async def test_barge_in_restarts_the_clock():
    """Non-zero start times on purpose: a 0.0 first timestamp hides an `or`-style bug."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 20.0),
        (UserStoppedSpeakingFrame(), 23.0),   # interrupted, spoke again
        (BotStartedSpeakingFrame(), 23.8),
    ])
    assert obs.turns == pytest.approx([0.8]), "must time from the latest user turn, not the first"


# --- per-service attribution -----------------------------------------------------


async def test_breakdown_attributes_time_to_each_service(caplog):
    obs = LatencyObserver("sid")
    metrics = MetricsFrame(data=[
        TTFBMetricsData(processor="DeepgramSTTService#0", value=0.31),
        TTFBMetricsData(processor="GroqLLMService#0", value=0.42),
        TTFAMetricsData(processor="SarvamTTSService#0", ttfa=0.50, ttfb=0.28, leading_silence=0.22),
    ])
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 0.0),
        (metrics, 0.5),
        (BotStartedSpeakingFrame(), 1.3),
    ])
    line = obs._breakdown()
    assert "deepgram=310ms" in line
    assert "groq=420ms" in line
    assert "sarvam_audio=500ms" in line
    assert "silence=220ms" in line, "TTS padding is latency the caller hears as dead air"


async def test_metrics_are_cleared_between_turns():
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 0.0),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.9)]), 0.2),
        (BotStartedSpeakingFrame(), 1.0),
        (UserStoppedSpeakingFrame(), 5.0),
    ])
    assert obs._breakdown() == "", "stale timings must not be attributed to the next turn"


@pytest.mark.parametrize(
    "processor,expected",
    [
        ("DeepgramSTTService#0", "deepgram"),
        ("GroqLLMService#1", "groq"),
        ("SarvamTTSService#0", "sarvam"),
        ("SomethingElse#2", "somethingelse"),
    ],
)
def test_processor_names_are_shortened(processor, expected):
    assert _short(processor) == expected


# --- summary ---------------------------------------------------------------------


async def test_summary_reports_distribution_not_an_average():
    obs = LatencyObserver("sid")
    for i, delay in enumerate([0.8, 1.0, 1.2, 1.4, 3.0]):
        await drive(obs, [
            (UserStoppedSpeakingFrame(), i * 10.0),
            (BotStartedSpeakingFrame(), i * 10.0 + delay),
        ])
    stats = obs.summary()
    assert stats["turns"] == 5
    assert stats["p50_ms"] == 1200
    assert stats["p95_ms"] == 3000, "the worst turn is what a caller remembers"
    assert (stats["min_ms"], stats["max_ms"]) == (800, 3000)


@pytest.mark.parametrize(
    "values,fraction,expected",
    [
        ([1.0], 0.95, 1.0),
        ([1.0, 2.0], 0.5, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 0.95, 4.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.0),
    ],
)
def test_percentile(values, fraction, expected):
    assert percentile(values, fraction) == expected


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 0.5)


async def test_log_summary_returns_none_without_turns():
    assert LatencyObserver("sid").log_summary() is None


# --- wiring ----------------------------------------------------------------------


def test_metrics_are_enabled_on_the_pipeline():
    """Every TTFB above is None unless PipelineParams turns metrics on."""
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    params = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "PipelineParams"
    ]
    assert params, "run_voice_agent no longer builds PipelineParams"
    enabled = {
        kw.arg for kw in params[0].keywords
        if isinstance(kw.value, ast.Constant) and kw.value.value is True
    }
    assert "enable_metrics" in enabled


def test_summary_is_carried_on_the_call_result():
    """Source-level guard: the observer's numbers are inert if the result drops them."""
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    wired = any(
        isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "CallResult"
        and any(
            kw.arg == "latency"
            and isinstance(kw.value, ast.Call)
            and getattr(kw.value.func, "attr", None) == "log_summary"
            for kw in n.keywords
        )
        for n in ast.walk(tree)
    )
    assert wired, "CallResult must carry latency=latency.log_summary()"


def test_observer_is_attached_to_the_worker():
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    workers = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "PipelineWorker"
    ]
    assert "observers" in {kw.arg for kw in workers[0].keywords}
