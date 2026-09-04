"""Voice-to-voice latency instrumentation.

The call logs reported `on_assistant_turn_stopped`, which fires when the agent finishes
speaking — so the visible gap between turns was dominated by TTS playback, not response
time. Reply length changes moved that number by seconds while latency was unchanged.

This measures the only delay the caller actually experiences: the silence from the moment
they stop talking to the moment they hear the agent, attributed across STT, LLM and TTS.
"""

import math
import statistics
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

NS_PER_SEC = 1_000_000_000

_TRACKED = (
    UserStoppedSpeakingFrame,
    BotStartedSpeakingFrame,
    MetricsFrame,
    LLMFullResponseStartFrame,
    TTSSpeakFrame,
)

# Below this, the remainder is ordinary frame plumbing and saying so on every turn would
# bury the turns where it is not. Set from the clean turns on call db5027ae, whose
# remainders ran 20-220ms.
_WORTH_REPORTING_SECS = 0.3


def _short(processor: str) -> str:
    """'DeepgramSTTService#0' -> 'deepgram'."""
    name = processor.split("#", 1)[0]
    for suffix in ("STTService", "LLMService", "TTSService", "Service"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.lower() or processor


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile. Deterministic and correct for the handful of turns a call has."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class LatencyObserver(BaseObserver):
    """Collects per-turn response latency without sitting in the audio path."""

    def __init__(self, call_sid: str):
        super().__init__()
        self._call_sid = call_sid
        self._turn_start_ns: Optional[int] = None
        self._ttfb: dict[str, float] = {}
        self._ttfa: dict[str, TTFAMetricsData] = {}
        self._turns: list[float] = []
        self._seen: set[int] = set()
        # When the LLM's first token came back, and which processor produced it. Together
        # with that processor's TTFB they give the moment the request was actually sent,
        # which is the one boundary the per-service metrics do not cover.
        self._llm_first_token_ns: Optional[int] = None
        self._llm_processor: Optional[str] = None
        # When the opening line was handed to the voice engine. The stretch between that and
        # the caller actually hearing something was the one part of a call nothing measured:
        # everything before it is in the logs to the millisecond, and everything after it is
        # covered per turn, but the first thing the prospect waits for was invisible.
        self._greeting_queued_ns: Optional[int] = None

    @property
    def turns(self) -> list[float]:
        return list(self._turns)

    async def on_push_frame(self, data: FramePushed):
        frame: Frame = data.frame
        # A frame is pushed between every pair of processors; only count it once.
        if not isinstance(frame, _TRACKED) or frame.id in self._seen:
            return
        self._seen.add(frame.id)

        if isinstance(frame, TTSSpeakFrame):
            # The opening line, which is spoken this way rather than generated. Read off the
            # frame rather than handed in from the agent: this timestamp is the PIPELINE
            # clock, nanoseconds since it started, and a monotonic reading taken anywhere
            # else would be a number from a different epoch subtracted from this one.
            #
            # Only before any turn has run, so the goodbye and the recovery lines — queued
            # the same way, later — cannot claim to be the greeting.
            if self._greeting_queued_ns is None and not self._turns and self._turn_start_ns is None:
                self._greeting_queued_ns = data.timestamp
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_start_ns = data.timestamp
            self._ttfb.clear()
            self._ttfa.clear()
            self._llm_first_token_ns = None
            self._llm_processor = None
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            # The first token of the reply. Only the first one in a turn: a split turn runs
            # two inferences and it is the first that started the caller's wait.
            if self._turn_start_ns is not None and self._llm_first_token_ns is None:
                self._llm_first_token_ns = data.timestamp
                self._llm_processor = str(data.source)
            return

        if isinstance(frame, MetricsFrame):
            for item in frame.data:
                if isinstance(item, TTFAMetricsData):
                    self._ttfa.setdefault(item.processor, item)
                elif isinstance(item, TTFBMetricsData):
                    self._ttfb.setdefault(item.processor, item.value)
            return

        # BotStartedSpeakingFrame. The opening greeting has no preceding user turn, so there
        # is nothing to measure it against as a turn — but it is not nothing. It is measured
        # from the moment it was queued instead, which is the synthesis and the trip out to
        # the carrier: the part of "the opening line started late" that no other line covers.
        if self._turn_start_ns is None:
            if self._greeting_queued_ns is not None:
                waited = (data.timestamp - self._greeting_queued_ns) / NS_PER_SEC
                self._greeting_queued_ns = None
                if waited >= 0:
                    logger.info(
                        f"[{self._call_sid}] GREETING audible after {waited * 1000:.0f}ms "
                        f"(synthesis and the trip to the carrier)"
                    )
            return

        elapsed = (data.timestamp - self._turn_start_ns) / NS_PER_SEC
        breakdown = self._breakdown(elapsed)
        self._turn_start_ns = None
        if elapsed < 0:
            return
        self._turns.append(elapsed)
        logger.info(
            f"[{self._call_sid}] LATENCY turn {len(self._turns)}: "
            f"{elapsed * 1000:.0f}ms voice-to-voice{breakdown}"
        )

    def _before_the_llm(self) -> Optional[float]:
        """Seconds between the prospect falling silent and the request leaving for the LLM.

        Nothing else measures this stretch. Each service reports its own time-to-first-byte,
        so the clock only starts once that service has been handed something — the wait for
        Deepgram to finalize the transcript, and for the aggregator to decide the turn is
        over, sits before all of them and was invisible.

        It has to be derived rather than read: the first token arrives TTFB after the
        request went out, so subtracting the LLM's own TTFB from the arrival time gives the
        moment it was sent.
        """
        if self._turn_start_ns is None or self._llm_first_token_ns is None:
            return None
        ttfb = self._ttfb.get(self._llm_processor)
        if ttfb is None:
            return None
        return max(0.0, (self._llm_first_token_ns - self._turn_start_ns) / NS_PER_SEC - ttfb)

    def _breakdown(self, elapsed: float) -> str:
        parts = [f"{_short(p)}={v * 1000:.0f}ms" for p, v in sorted(self._ttfb.items())]
        for processor, item in sorted(self._ttfa.items()):
            parts.append(
                f"{_short(processor)}_audio={item.ttfa * 1000:.0f}ms"
                f"(silence={item.leading_silence * 1000:.0f}ms)"
            )

        # On call db5027ae turn 2 reported 3959ms voice-to-voice against groq=619ms and
        # sarvam=200ms. Three seconds were missing and there was no line for them, so the
        # log looked like the services were fast and the caller was wrong. These two make
        # the total add up, or say plainly that it does not.
        before_llm = self._before_the_llm()
        if before_llm is not None and before_llm >= _WORTH_REPORTING_SECS:
            parts.append(f"before_llm={before_llm * 1000:.0f}ms")
        accounted = sum(self._ttfb.values()) + (before_llm or 0.0)
        rest = elapsed - accounted
        if rest >= _WORTH_REPORTING_SECS:
            parts.append(f"unattributed={rest * 1000:.0f}ms")
        return "  |  " + "  ".join(parts) if parts else ""

    def summary(self) -> Optional[dict]:
        if not self._turns:
            return None
        return {
            "turns": len(self._turns),
            "p50_ms": round(statistics.median(self._turns) * 1000),
            "p95_ms": round(percentile(self._turns, 0.95) * 1000),
            "min_ms": round(min(self._turns) * 1000),
            "max_ms": round(max(self._turns) * 1000),
        }

    def log_summary(self) -> Optional[dict]:
        stats = self.summary()
        if stats is None:
            logger.info(f"[{self._call_sid}] LATENCY no measurable turns")
            return None
        logger.info(
            f"[{self._call_sid}] LATENCY summary | turns={stats['turns']} "
            f"p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms "
            f"min={stats['min_ms']}ms max={stats['max_ms']}ms"
        )
        return stats
