"""Whether the provider is actually reusing the prompt, read off the call log.

On 10 Sep 2026 the prompt was reordered so that ~4,700 of its ~4,800 tokens are a prefix
shared by every call. That was measured locally, in characters. Whether Cerebras then
serves those tokens from its cache is a different fact, and it was unobservable: pipecat
maps the API's prompt_tokens_details.cached_tokens onto LLMTokenUsage.cache_read_input_tokens
and hands it to the same MetricsFrame the latency observer already reads — which ignored it.

So the LATENCY line each turn now carries prompt= and cached=, and the summary a share. A
zero on every turn is not "no data"; it is the finding that the reorder bought nothing and
the next place to look is the request itself.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTFBMetricsData

from app.utils.latency import LatencyObserver

NS = 1_000_000_000


def _usage(prompt: int, cached: int) -> MetricsFrame:
    return MetricsFrame(
        data=[
            LLMUsageMetricsData(
                processor="OpenAILLMService#0",
                value=LLMTokenUsage(
                    prompt_tokens=prompt,
                    completion_tokens=20,
                    total_tokens=prompt + 20,
                    cache_read_input_tokens=cached,
                ),
            )
        ]
    )


def _push(observer: LatencyObserver, frame, at_secs: float):
    # Only the fields the observer reads off FramePushed. The source name matches the TTFB
    # processor above so the LLM's own timing can be derived as it is on a live call.
    pushed = SimpleNamespace(frame=frame, timestamp=int(at_secs * NS), source="OpenAILLMService#0")
    asyncio.run(observer.on_push_frame(pushed))


def _turn(observer: LatencyObserver, start: float, usage_frames: list) -> None:
    _push(observer, UserStoppedSpeakingFrame(), start)
    _push(observer, MetricsFrame(data=[TTFBMetricsData(processor="OpenAILLMService#0", value=0.4)]), start + 0.5)
    _push(observer, LLMFullResponseStartFrame(), start + 0.5)
    for frame in usage_frames:
        _push(observer, frame, start + 0.6)
    _push(observer, BotStartedSpeakingFrame(), start + 0.8)


# --- per turn -------------------------------------------------------------------------------


def test_the_turn_line_carries_prompt_and_cached_tokens():
    obs = LatencyObserver("sid")
    _push(obs, UserStoppedSpeakingFrame(), 1.0)
    _push(obs, _usage(4800, 4700), 1.6)
    assert "prompt=4800tok cached=4700" in obs._breakdown(0.8)


def test_a_cache_miss_is_shown_as_zero_rather_than_omitted():
    """Omitting the field on a miss makes a run of misses look like the metric is absent."""
    obs = LatencyObserver("sid")
    _push(obs, UserStoppedSpeakingFrame(), 1.0)
    _push(obs, _usage(4800, 0), 1.6)
    assert "cached=0" in obs._breakdown(0.8)


def test_nothing_is_shown_before_any_usage_arrives():
    obs = LatencyObserver("sid")
    _push(obs, UserStoppedSpeakingFrame(), 1.0)
    assert "prompt=" not in obs._breakdown(0.8)


def test_a_split_turn_sums_both_inferences():
    """Two inferences in one turn are two prompts the caller waited on, not one to report
    and one to lose."""
    obs = LatencyObserver("sid")
    _push(obs, UserStoppedSpeakingFrame(), 1.0)
    _push(obs, _usage(4800, 4700), 1.5)
    _push(obs, _usage(4850, 4800), 1.9)
    assert "prompt=9650tok cached=9500" in obs._breakdown(0.8)


def test_the_next_turn_starts_from_nothing():
    obs = LatencyObserver("sid")
    _turn(obs, 1.0, [_usage(4800, 4700)])
    _push(obs, UserStoppedSpeakingFrame(), 5.0)
    assert "prompt=" not in obs._breakdown(0.8)


# --- per call -------------------------------------------------------------------------------


def test_the_summary_reports_the_share_served_from_cache():
    obs = LatencyObserver("sid")
    _turn(obs, 1.0, [_usage(4800, 0)])       # first turn of a cold call: nothing cached yet
    _turn(obs, 5.0, [_usage(4800, 4700)])
    _turn(obs, 9.0, [_usage(4900, 4800)])
    stats = obs.summary()
    assert stats["prompt_tokens"] == 14500
    assert stats["cached_tokens"] == 9500
    assert stats["cached_share"] == pytest.approx(9500 / 14500, abs=0.001)


def test_the_summary_line_prints_it():
    from loguru import logger

    lines = []
    handle = logger.add(lambda m: lines.append(str(m)), level="INFO")
    try:
        obs = LatencyObserver("sid")
        _turn(obs, 1.0, [_usage(4800, 4700)])
        obs.log_summary()
    finally:
        logger.remove(handle)
    assert any("cache=4700/4800tok (98%)" in line for line in lines), lines


def test_a_call_with_no_usage_metrics_summarises_as_before():
    """Older providers and the fallback path may send no usage at all. The summary must
    not grow a zero-division or a misleading 0%."""
    obs = LatencyObserver("sid")
    _turn(obs, 1.0, [])
    stats = obs.summary()
    assert "prompt_tokens" not in stats
    assert "cached_share" not in stats
