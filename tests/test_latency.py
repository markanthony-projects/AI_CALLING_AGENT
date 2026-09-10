"""Latency instrumentation.

The number the call logs showed was `on_assistant_turn_stopped` — the agent *finishing*
speaking — so it moved by seconds when reply length changed while true response time did
not. These pin the measurement that replaced it: user stops talking -> caller hears audio.
"""

import pytest
from loguru import logger
from pipecat.frames.frames import (
    LLMFullResponseStartFrame,
    BotStartedSpeakingFrame,
    MetricsFrame,
    TextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed

from app.utils.latency import NS_PER_SEC, LatencyObserver, _short, percentile


class _Processor:
    """Stands in for the pushing service. The observer reads str(source) to learn which
    TTFB belongs to the LLM."""

    def __init__(self, name):
        self._name = name

    def __str__(self):
        return self._name


def pushed(frame, at_seconds: float, source=None) -> FramePushed:
    return FramePushed(
        source=source, destination=None, frame=frame,
        direction=None, timestamp=int(at_seconds * NS_PER_SEC),
    )


async def drive(observer, events):
    for frame, at in events:
        await observer.on_push_frame(pushed(frame, at))


async def drive_from(observer, events):
    """Like drive, but each event names the processor that pushed the frame."""
    for frame, at, source in events:
        await observer.on_push_frame(pushed(frame, at, source))


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
    line = obs._breakdown(0.0)
    assert "deepgram=310ms" in line
    assert "groq=420ms" in line
    assert "sarvam_audio=500ms" in line
    assert "silence=220ms" in line, "TTS padding is latency the caller hears as dead air"


# --- the time nobody was measuring -----------------------------------------------
#
# Call db5027ae turn 2 reported 3959ms voice-to-voice against groq=619ms and sarvam=200ms.
# Three seconds were missing, and the log gave no line for them — so it read as though every
# service had been fast and the caller had imagined the wait. Each service reports its own
# time-to-first-byte, which starts only once that service has been handed something; the
# stretch before the first of them had no owner.


async def test_the_wait_before_the_llm_is_measured():
    """Deepgram finalizing and the aggregator closing the turn both happen here, and both
    sit before every TTFB the pipeline reports."""
    obs = LatencyObserver("sid")
    llm = _Processor("GroqLLMService#0")
    await drive_from(obs, [
        (UserStoppedSpeakingFrame(), 0.0, None),
        # Request went out at 3.0s; the first token arrived 0.6s later.
        (LLMFullResponseStartFrame(), 3.6, llm),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.6)]), 3.6, None),
    ])
    assert "before_llm=3000ms" in obs._breakdown(3.9)


async def test_a_turn_that_adds_up_says_nothing_extra():
    """Every turn carrying a line for 20ms of frame plumbing would bury the turns where it
    is seconds. The clean turns on that same call ran 20-220ms unattributed."""
    obs = LatencyObserver("sid")
    llm = _Processor("GroqLLMService#0")
    await drive_from(obs, [
        (UserStoppedSpeakingFrame(), 0.0, None),
        (LLMFullResponseStartFrame(), 0.6, llm),
        (MetricsFrame(data=[
            TTFBMetricsData(processor="GroqLLMService#0", value=0.55),
            TTFBMetricsData(processor="SarvamTTSService#0", value=0.2),
        ]), 0.6, None),
    ])
    line = obs._breakdown(0.8)
    assert "before_llm" not in line
    assert "unattributed" not in line


async def test_time_nobody_can_account_for_is_still_reported():
    """The point of the whole thing: if the parts do not add up to the total, say so rather
    than print a breakdown that quietly implies they do."""
    obs = LatencyObserver("sid")
    llm = _Processor("GroqLLMService#0")
    await drive_from(obs, [
        (UserStoppedSpeakingFrame(), 0.0, None),
        (LLMFullResponseStartFrame(), 0.6, llm),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.6)]), 0.6, None),
    ])
    # Two and a half seconds between the reply being generated and the caller hearing it.
    assert "unattributed=2500ms" in obs._breakdown(3.1)


async def test_a_split_turn_is_timed_from_the_first_inference():
    """Two inferences run for one turn. The caller's wait started with the first."""
    obs = LatencyObserver("sid")
    llm = _Processor("GroqLLMService#0")
    await drive_from(obs, [
        (UserStoppedSpeakingFrame(), 0.0, None),
        (LLMFullResponseStartFrame(), 1.5, llm),
        (LLMFullResponseStartFrame(), 2.9, llm),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.5)]), 2.9, None),
    ])
    assert "before_llm=1000ms" in obs._breakdown(3.2)


async def test_every_turn_measures_its_own_wait():
    """The first-token timestamp has to be cleared when a new turn starts, or the "only the
    first inference counts" guard sees the PREVIOUS turn's value still sitting there and
    never records this one. Every turn after the first would then silently report nothing —
    which looks exactly like a pipeline with no problem."""
    obs = LatencyObserver("sid")
    llm = _Processor("GroqLLMService#0")
    await drive_from(obs, [
        (UserStoppedSpeakingFrame(), 0.0, None),
        (LLMFullResponseStartFrame(), 3.6, llm),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.6)]), 3.6, None),
        (BotStartedSpeakingFrame(), 3.9, None),
        # Second turn, a different wait.
        (UserStoppedSpeakingFrame(), 10.0, None),
        (LLMFullResponseStartFrame(), 12.1, llm),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.5)]), 12.1, None),
    ])
    assert "before_llm=1600ms" in obs._breakdown(2.4)


async def test_the_line_that_actually_gets_logged_carries_it():
    """The others exercise the computation directly. This one proves it survives the wiring
    into on_push_frame, where the turn state is torn down as the line is written."""
    written = []
    sink = logger.add(lambda m: written.append(str(m)), level="INFO")
    try:
        obs = LatencyObserver("sid")
        llm = _Processor("GroqLLMService#0")
        await drive_from(obs, [
            (UserStoppedSpeakingFrame(), 0.0, None),
            (LLMFullResponseStartFrame(), 3.6, llm),
            (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.6)]), 3.6, None),
            (BotStartedSpeakingFrame(), 3.9, None),
        ])
    finally:
        logger.remove(sink)
    line = next(m for m in written if "LATENCY turn 1" in m)
    assert "3900ms voice-to-voice" in line
    assert "before_llm=3000ms" in line


async def test_metrics_are_cleared_between_turns():
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 0.0),
        (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.9)]), 0.2),
        (BotStartedSpeakingFrame(), 1.0),
        (UserStoppedSpeakingFrame(), 5.0),
    ])
    # turn_decision is always on the line — it is the point of the line — so the check is
    # that no SERVICE timing survived, not that the breakdown is empty.
    stale = obs._breakdown(0.0)
    assert "groq" not in stale, "stale timings must not be attributed to the next turn"
    assert stale == "  |  turn_decision=0ms", stale


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


# --- the handshake that landed on the prospect ------------------------------------------
#
# Production timings, first turn against second on the same call:
#
#   turn 1: 3382ms voice-to-voice | llm=490ms sarvam=373ms ... unattributed=2519ms
#   turn 2: 1247ms voice-to-voice | llm=498ms sarvam=427ms ... unattributed=322ms
#
# and on another call the same cost showed up inside the LLM's own timing instead: 1525ms
# against a steady state of about 490ms. Deepgram and Sarvam connect when the pipeline starts;
# the LLM is HTTP and connects on first use, which is the first thing the caller waits for.


def test_the_llm_is_warmed_while_the_greeting_plays():
    """Alongside the opening line, which is built locally and takes six to eight seconds to
    speak — several times what a handshake needs, and all of it before the prospect can reply."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    greeting = src[src.index("async def startup_greeting") : src.index("nonlocal _startup_task")]
    assert "llm.warm_up()" in greeting
    assert greeting.index("llm.warm_up()") < greeting.index("spoken(opening_line")


def test_the_warm_up_cannot_take_a_call_down():
    """It runs for a call that is already connected. A warm-up that raises would cost the call
    it was meant to speed up, so every failure is swallowed and logged at debug."""
    import inspect

    from app.services.llm_provider import ResilientLLMService

    src = inspect.getsource(ResilientLLMService.warm_up)
    assert "except Exception" in src
    assert "max_tokens=1" in src, "a warm-up that generates a reply is a cost, not a handshake"


def test_the_timing_line_names_the_provider_that_answered():
    """It read `groq=1525ms` while every request went to Cerebras. The label exists to make the
    fallback visible, and a hard-coded one made the two indistinguishable."""
    from app.services.llm_provider import LLMEndpoint, processor_name
    from app.utils.latency import _short

    cerebras = LLMEndpoint(name="cerebras", model="m", base_url="u", api_key="k")
    groq = LLMEndpoint(name="groq", model="m", base_url="u", api_key="k")
    assert _short(processor_name(cerebras)) == "cerebras"
    assert _short(processor_name(groq)) == "groq"


# --- the six hundred milliseconds that were not on any line -------------------------------------


def captured():
    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="INFO")
    return seen, sink


VAD_STOP_SECS = 0.2


def _vad_stop(at):
    """A VAD stop frame ARRIVING at `at`. Their voice stopped stop_secs earlier — that is
    what stop_secs means, and the frame carries the number so the observer can subtract it.
    So `_vad_stop(1.0)` is a voice that fell silent at 0.8."""
    return (
        VADUserStoppedSpeakingFrame(stop_secs=VAD_STOP_SECS, timestamp=int(at * NS_PER_SEC)),
        at,
    )


def _vad_start(at):
    return (VADUserStartedSpeakingFrame(timestamp=int(at * NS_PER_SEC)), at)


async def test_the_turn_is_timed_from_when_their_voice_stopped():
    """Not from when the turn was declared over. Pipecat broadcasts UserStoppedSpeakingFrame
    out of _on_user_turn_stopped, which is AFTER the VAD's stop_secs and AFTER the blind
    settle — 600ms on this deployment. Timed from there, a call that made the prospect wait
    1.33 seconds reported 729ms, and the metric was called voice-to-voice while it was
    nothing of the sort."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        _vad_stop(1.0),                          # their voice actually stops
        (UserStoppedSpeakingFrame(), 1.6),       # 600ms later the turn is declared over
        (BotStartedSpeakingFrame(), 2.33),       # and 730ms after that they hear something
    ])
    assert obs.turns == pytest.approx([1.53])


async def test_the_decision_is_named_on_the_line():
    obs = LatencyObserver("sid")
    seen, sink = captured()
    try:
        await drive(obs, [
            _vad_stop(1.0),
            (UserStoppedSpeakingFrame(), 1.6),
            (BotStartedSpeakingFrame(), 2.33),
        ])
    finally:
        logger.remove(sink)
    line = next(m for m in seen if "LATENCY turn 1" in m)
    assert "1530ms voice-to-voice" in line
    assert "turn_decision=800ms" in line


async def test_a_pause_for_breath_is_not_the_end_of_their_turn():
    """They stop, start again, and stop again. Only the last stop is when they finished —
    timing from the first would report a turn that includes their own second sentence."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        _vad_stop(1.0),
        _vad_start(1.3),                        # still talking
        _vad_stop(2.0),                         # this is the one
        (UserStoppedSpeakingFrame(), 2.6),
        (BotStartedSpeakingFrame(), 3.0),
    ])
    assert obs.turns == pytest.approx([1.2])


async def test_their_own_speaking_is_never_counted_as_our_delay():
    """The case the VADUserStartedSpeakingFrame handler exists for, and the only one that
    can tell it is there: they resume speaking and the turn is then declared over WITHOUT a
    second stop — the stop strategy has a fallback path that ends a turn on transcripts
    alone. Left holding the stale stop, the observer would bill their own two seconds of
    talking to us and report a turn that never happened.

    Mutation testing found this: with a stop after the restart, the later stop overwrites
    the earlier one anyway and the handler is invisible."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        _vad_stop(1.0),
        _vad_start(1.3),                       # off again, and no stop after it
        (UserStoppedSpeakingFrame(), 3.3),     # turn declared while the VAD says speaking
        (BotStartedSpeakingFrame(), 3.7),
    ])
    assert obs.turns == pytest.approx([0.4]), "their speech was billed as our latency"


async def test_without_vad_frames_it_measures_what_it_always_measured():
    """A transport that sends none must keep working rather than reporting nothing."""
    obs = LatencyObserver("sid")
    await drive(obs, [
        (UserStoppedSpeakingFrame(), 1.0),
        (BotStartedSpeakingFrame(), 1.8),
    ])
    assert obs.turns == pytest.approx([0.8])


async def test_the_voice_stop_does_not_carry_into_the_next_turn():
    obs = LatencyObserver("sid")
    await drive(obs, [
        _vad_stop(1.0),
        (UserStoppedSpeakingFrame(), 1.6),
        (BotStartedSpeakingFrame(), 2.0),
        (UserStoppedSpeakingFrame(), 6.0),       # no VAD frame this time
        (BotStartedSpeakingFrame(), 6.5),
    ])
    assert obs.turns == pytest.approx([1.2, 0.5])


async def test_the_decision_counts_towards_what_is_accounted_for():
    """unattributed is elapsed minus everything named. Leaving the decision out of the sum
    would report it twice — once as turn_decision and again as unattributed."""
    obs = LatencyObserver("sid")
    seen, sink = captured()
    try:
        await drive(obs, [
            _vad_stop(1.0),
            (UserStoppedSpeakingFrame(), 1.6),
            (MetricsFrame(data=[TTFBMetricsData(processor="GroqLLMService#0", value=0.4)]), 1.7),
            (BotStartedSpeakingFrame(), 2.1),
        ])
    finally:
        logger.remove(sink)
    line = next(m for m in seen if "LATENCY turn 1" in m)
    assert "turn_decision=800ms" in line
    assert "unattributed" not in line, line


async def test_the_summary_reports_what_deciding_cost_across_the_call():
    obs = LatencyObserver("sid")
    seen, sink = captured()
    try:
        await drive(obs, [
            _vad_stop(1.0), (UserStoppedSpeakingFrame(), 1.6), (BotStartedSpeakingFrame(), 2.0),
            _vad_stop(5.0), (UserStoppedSpeakingFrame(), 5.6), (BotStartedSpeakingFrame(), 6.0),
        ])
        stats = obs.log_summary()
    finally:
        logger.remove(sink)
    assert stats["decision_p50_ms"] == 800
    assert any("turn_decision_p50=800ms" in m for m in seen)


# --- and the wiring, because an unwired measurement measures nothing ---------------------------


def test_the_agent_hands_the_stream_open_time_to_the_observer():
    """A measurement nobody connects is what SMART_TURN_ENABLED turned out to be: a setting
    that built an object, passed it to a parameter that no longer existed, and was dropped
    in silence. Mutation testing found both ends of this one unwireable-by-any-test until
    these two existed."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "stream_open_at: Optional[float] = None" in src
    assert "LatencyObserver(call_sid, stream_open_at=stream_open_at)" in src


def test_the_webhook_takes_the_time_at_the_moment_the_stream_opens():
    """After accept() and before anything else — the Call row, reading the project, building
    the services. All of that is ours and all of it is on the caller's clock."""
    import inspect

    from app.api.routes import webhook

    src = inspect.getsource(webhook._handle_call)
    assert "stream_open_at = time.monotonic()" in src
    assert src.index("await websocket.accept()") < src.index("stream_open_at = time.monotonic()")
    assert src.index("stream_open_at = time.monotonic()") < src.index("db.add(Call(")
    assert "stream_open_at=stream_open_at," in src, "taken and then not passed on"
