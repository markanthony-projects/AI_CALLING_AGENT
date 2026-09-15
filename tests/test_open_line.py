"""A line that has gone quiet with the prospect still on it is asked into again.

Calls 5a2c8245, 5e247d87 and 81756200, 15 Sep 2026: the agent fell silent, the prospect
said "Hello?" into it, every hello kept the speech service's turn open, and no backstop
fired because every backstop was waiting on that turn to end.
"""

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from app.utils.open_line import (
    CHECK_EVERY_SECS,
    OPEN_LINE_SECS,
    PROSPECT_QUIET_SECS,
    line_has_gone_dead,
)


def _dead(now=100.0, **over):
    base = dict(
        bot_speaking=False,
        bot_stopped_at=now - OPEN_LINE_SECS - 1,
        last_voice_at=now - PROSPECT_QUIET_SECS - 1,
        last_nudge_at=None,
        holding=False,
        ending=False,
    )
    base.update(over)
    return line_has_gone_dead(now, **base)


def test_quiet_both_ways_for_long_enough_is_dead():
    assert _dead() is True


def test_a_prospect_who_never_made_a_sound_still_gets_asked():
    assert _dead(last_voice_at=None) is True


def test_never_while_the_agent_is_speaking():
    assert _dead(bot_speaking=True) is False


def test_never_over_a_prospect_who_is_talking_or_just_was():
    assert _dead(last_voice_at=100.0 - 0.5) is False
    assert _dead(last_voice_at=100.0 - PROSPECT_QUIET_SECS + 0.1) is False


def test_not_before_the_agent_has_ever_spoken():
    """The greeting is on its way; the startup path has its own clock."""
    assert _dead(bot_stopped_at=None) is False


def test_not_until_the_agent_has_been_quiet_long_enough():
    assert _dead(bot_stopped_at=100.0 - OPEN_LINE_SECS + 0.5) is False


def test_a_nudge_restarts_the_clock():
    assert _dead(last_nudge_at=100.0 - 3.0) is False
    assert _dead(last_nudge_at=100.0 - OPEN_LINE_SECS - 0.1) is True


def test_never_on_hold_or_while_ending():
    assert _dead(holding=True) is False
    assert _dead(ending=True) is False


def test_the_thresholds_sit_between_a_slow_turn_and_a_hello():
    assert 2.5 < OPEN_LINE_SECS <= 12
    assert 1.0 <= PROSPECT_QUIET_SECS <= 3.0
    assert CHECK_EVERY_SECS <= 2.0


# --- the observer supplies the facts ----------------------------------------------------


def test_the_observer_records_the_transport_and_the_vad(monkeypatch):
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )

    from app.utils.latency import LatencyObserver

    clock = iter([10.0, 11.0, 12.0, 13.0])
    obs = LatencyObserver("sid", clock=lambda: next(clock))
    assert obs.bot_speaking is False and obs.bot_stopped_at is None and obs.last_voice_at is None

    async def push(frame, at_ms):
        await obs.on_push_frame(SimpleNamespace(frame=frame, timestamp=at_ms * 1_000_000, source="x"))

    async def go():
        await push(VADUserStartedSpeakingFrame(), 1000)
        assert obs.last_voice_at == 10.0
        await push(VADUserStoppedSpeakingFrame(), 1400)
        assert obs.last_voice_at == 11.0
        await push(BotStartedSpeakingFrame(), 2000)
        assert obs.bot_speaking is True
        await push(BotStoppedSpeakingFrame(), 5000)
        assert obs.bot_speaking is False and obs.bot_stopped_at == 12.0

    asyncio.run(go())


# --- wired ------------------------------------------------------------------------------


def test_the_watchdog_runs_for_the_life_of_the_call_and_asks_through_the_bounded_door():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    assert "_open_line_guard = asyncio.create_task(watch_open_line())" in src
    assert src.index("_open_line_guard = asyncio.create_task") < src.index("await runner.run(task)")
    after = src[src.index("finally:") :]
    assert "_open_line_guard.cancel()" in after
    watchdog = src[src.index("async def watch_open_line") : src.index("_open_line_guard = ")]
    assert "line_has_gone_dead(" in watchdog
    assert "await ask_again(" in watchdog, "the same counters and ceiling as every other nudge"
    assert "bot_speaking=latency.bot_speaking" in watchdog
    assert "last_voice_at=latency.last_voice_at" in watchdog


def test_the_greeting_is_something_the_watchdog_can_ask_again():
    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    greeting = src[src.index("async def startup_greeting") : src.index("await task.queue_frames(spoken(opening_line")]
    assert "_last_agent_line = opening_line" in greeting


def test_the_strategy_cap_is_documented_as_inert():
    """So nobody reads it as a backstop again."""
    from app.services import stt_provider

    assert "cannot end a turn on its own" in inspect.getsource(stt_provider._service_turns)
