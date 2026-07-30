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
    MetricsFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

NS_PER_SEC = 1_000_000_000

_TRACKED = (UserStoppedSpeakingFrame, BotStartedSpeakingFrame, MetricsFrame)


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

    @property
    def turns(self) -> list[float]:
        return list(self._turns)

    async def on_push_frame(self, data: FramePushed):
        frame: Frame = data.frame
        # A frame is pushed between every pair of processors; only count it once.
        if not isinstance(frame, _TRACKED) or frame.id in self._seen:
            return
        self._seen.add(frame.id)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_start_ns = data.timestamp
            self._ttfb.clear()
            self._ttfa.clear()
            return

        if isinstance(frame, MetricsFrame):
            for item in frame.data:
                if isinstance(item, TTFAMetricsData):
                    self._ttfa.setdefault(item.processor, item)
                elif isinstance(item, TTFBMetricsData):
                    self._ttfb.setdefault(item.processor, item.value)
            return

        # BotStartedSpeakingFrame. The opening greeting has no preceding user turn, so
        # there is nothing to measure against and it is skipped rather than recorded as 0.
        if self._turn_start_ns is None:
            return

        elapsed = (data.timestamp - self._turn_start_ns) / NS_PER_SEC
        self._turn_start_ns = None
        if elapsed < 0:
            return
        self._turns.append(elapsed)
        logger.info(
            f"[{self._call_sid}] LATENCY turn {len(self._turns)}: "
            f"{elapsed * 1000:.0f}ms voice-to-voice{self._breakdown()}"
        )

    def _breakdown(self) -> str:
        parts = [f"{_short(p)}={v * 1000:.0f}ms" for p, v in sorted(self._ttfb.items())]
        for processor, item in sorted(self._ttfa.items()):
            parts.append(
                f"{_short(processor)}_audio={item.ttfa * 1000:.0f}ms"
                f"(silence={item.leading_silence * 1000:.0f}ms)"
            )
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
