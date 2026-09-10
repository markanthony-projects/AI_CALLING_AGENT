"""Voice-to-voice latency instrumentation.

The call logs reported `on_assistant_turn_stopped`, which fires when the agent finishes
speaking — so the visible gap between turns was dominated by TTS playback, not response
time. Reply length changes moved that number by seconds while latency was unchanged.

This measures the only delay the caller actually experiences: the silence from the moment
they stop talking to the moment they hear the agent, attributed across STT, LLM and TTS.

AND IT USED TO START THE CLOCK TOO LATE. Until 10 Sep 2026 the turn was timed from
UserStoppedSpeakingFrame, which Pipecat broadcasts from `_on_user_turn_stopped` — that is,
once the turn has already been DECLARED over. Two waits happen before that and neither
appeared anywhere:

    0.20s   VAD stop_secs        the VAD making up its mind that the voice stopped
    0.40s   TURN_SETTLE_SECS     the blind wait for them to maybe say more
    ─────
    0.60s   invisible, on every turn of every call

So a reported p50 of 729ms was about 1,330ms of silence for the prospect, and the number
was named "voice-to-voice" while measuring nothing of the sort. config.py said as much in a
comment about TURN_SETTLE_SECS — "it does not appear in the LATENCY log lines" — and the
metric kept its flattering name anyway. The caller told us it felt slower than the logs
said, and the caller was right.

The clock now starts at the last VADUserStoppedSpeakingFrame — the moment their voice
actually stopped — and `turn_decision` reports what the two waits cost, so the thing worth
optimising is on the line rather than under it.
"""

import math
import statistics
import time
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    Frame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TTSSpeakFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TTFAMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed

NS_PER_SEC = 1_000_000_000

_TRACKED = (
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
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

    def __init__(self, call_sid: str, stream_open_at: Optional[float] = None):
        super().__init__()
        self._call_sid = call_sid
        # time.monotonic() at websocket accept, if the caller has it. The pipeline clock
        # starts much later — after two database round trips and the services being built —
        # so it cannot see the part of the wait that happens before it exists.
        self._stream_open_at = stream_open_at
        # The last moment their voice actually stopped. The honest start of a turn: what
        # follows is the VAD settling, the blind wait, and only then the turn being declared.
        self._voice_stopped_ns: Optional[int] = None
        self._turn_start_ns: Optional[int] = None
        self._ttfb: dict[str, float] = {}
        self._ttfa: dict[str, TTFAMetricsData] = {}
        self._turns: list[float] = []
        # What the VAD settle plus the blind wait cost on each turn, kept so the summary can
        # say how much of the call's latency was spent deciding the prospect had finished.
        self._decisions: list[float] = []
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
        # Prompt tokens this turn, and how many the provider served from its prefix cache.
        # The prompt is ~4,800 tokens and is resent every turn; the rules were reordered on
        # 10 Sep so that all but the last ~70 are a prefix shared by every call. Whether the
        # provider actually reuses it is only knowable from here — pipecat maps the API's
        # prompt_tokens_details.cached_tokens onto cache_read_input_tokens, and nothing
        # else in this codebase read it.
        self._prompt_tokens: Optional[int] = None
        self._cached_tokens: Optional[int] = None
        self._reasoning_tokens: Optional[int] = None
        self._prompt_total = 0
        self._cached_total = 0

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

        if isinstance(frame, VADUserStartedSpeakingFrame):
            # Talking again, so the last stop was a pause for breath and not the end of
            # anything. Only the final stop before the turn is declared is the turn's start.
            self._voice_stopped_ns = None
            return

        if isinstance(frame, VADUserStoppedSpeakingFrame):
            # The frame is emitted only once stop_secs of silence has ALREADY passed, so its
            # arrival is not the moment the voice stopped — it is stop_secs afterwards.
            # Subtracting it is the difference between "turn_decision=401ms" and the 601ms
            # the prospect actually sat through, and the frame carries the number itself so
            # this stays right if the setting moves.
            stop_secs = getattr(frame, "stop_secs", None) or 0.0
            self._voice_stopped_ns = data.timestamp - int(stop_secs * NS_PER_SEC)
            return

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_start_ns = data.timestamp
            self._ttfb.clear()
            self._ttfa.clear()
            self._llm_first_token_ns = None
            self._llm_processor = None
            self._prompt_tokens = None
            self._cached_tokens = None
            self._reasoning_tokens = None
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
                elif isinstance(item, LLMUsageMetricsData):
                    # Summed rather than set: a split turn runs two inferences, and both
                    # are prompt tokens the caller waited on.
                    prompt = item.value.prompt_tokens or 0
                    cached = item.value.cache_read_input_tokens or 0
                    self._prompt_tokens = (self._prompt_tokens or 0) + prompt
                    self._cached_tokens = (self._cached_tokens or 0) + cached
                    self._prompt_total += prompt
                    self._cached_total += cached
                    # Tokens the model spent thinking before it said anything. On call
                    # 2eeb48a0 (gpt-oss-120b) turn 1 had 1396ms unattributed after a 473ms
                    # first token; a reasoning model's silence sits exactly there, and
                    # nothing else in the log could name it.
                    self._reasoning_tokens = (self._reasoning_tokens or 0) + (
                        item.value.reasoning_tokens or 0
                    )
            return

        # BotStartedSpeakingFrame. The opening greeting has no preceding user turn, so there
        # is nothing to measure it against as a turn — but it is not nothing.
        #
        # This used to report only "queued -> audible", which was the synthesis and the trip
        # to the carrier and read as 404ms while the caller sat through seconds. What the
        # caller waits through is everything from the media stream opening: two database
        # round trips to write the Call row and read the project, the services being built,
        # the Sarvam websocket handshake — which happens on StartFrame, so the first word
        # cannot be spoken until it completes — and only then the synthesis. All of it is
        # ours, and none of it was on any line.
        if self._turn_start_ns is None:
            if self._greeting_queued_ns is not None:
                synthesis = (data.timestamp - self._greeting_queued_ns) / NS_PER_SEC
                self._greeting_queued_ns = None
                # The pipeline clock counts from pipeline start, so a BotStartedSpeaking
                # timestamp IS "pipeline start -> first audio".
                since_pipeline = data.timestamp / NS_PER_SEC
                if synthesis >= 0:
                    since_open = (
                        f"{(time.monotonic() - self._stream_open_at) * 1000:.0f}ms after the "
                        f"stream opened"
                        if self._stream_open_at is not None
                        else "stream-open time not supplied"
                    )
                    logger.info(
                        f"[{self._call_sid}] FIRST WORD {since_open} | "
                        f"pipeline={since_pipeline * 1000:.0f}ms "
                        f"synthesis={synthesis * 1000:.0f}ms"
                    )
            return

        # From when their voice actually stopped, not from when the turn was declared over.
        # The fallback keeps a transport that sends no VAD frames measuring what it always
        # measured, rather than measuring nothing.
        began_ns = (
            self._voice_stopped_ns
            if self._voice_stopped_ns is not None
            else self._turn_start_ns
        )
        elapsed = (data.timestamp - began_ns) / NS_PER_SEC
        decision = (self._turn_start_ns - began_ns) / NS_PER_SEC
        breakdown = self._breakdown(elapsed, decision)
        self._turn_start_ns = None
        self._voice_stopped_ns = None
        if elapsed < 0:
            return
        self._decisions.append(decision)
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

    def _breakdown(self, elapsed: float, decision: float = 0.0) -> str:
        # First, because it is first in time and because it is the largest fixed cost in the
        # system: VAD stop_secs plus the blind settle, paid on every turn, and invisible
        # until 10 Sep 2026. Shown even at zero — a zero here means the VAD frames did not
        # arrive, which is worth knowing, not worth hiding.
        parts = [f"turn_decision={decision * 1000:.0f}ms"]
        parts += [f"{_short(p)}={v * 1000:.0f}ms" for p, v in sorted(self._ttfb.items())]
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
        accounted = decision + sum(self._ttfb.values()) + (before_llm or 0.0)
        rest = elapsed - accounted
        if rest >= _WORTH_REPORTING_SECS:
            parts.append(f"unattributed={rest * 1000:.0f}ms")
        # Always shown when known, including cached=0: a zero on every turn is the finding.
        if self._prompt_tokens:
            parts.append(f"prompt={self._prompt_tokens}tok cached={self._cached_tokens or 0}")
        # Only when the model reported some: for a non-reasoning model the field is absent
        # and a permanent reasoning=0 would read as a claim the log cannot make.
        if self._reasoning_tokens:
            parts.append(f"reasoning={self._reasoning_tokens}tok")
        return "  |  " + "  ".join(parts) if parts else ""

    def summary(self) -> Optional[dict]:
        if not self._turns:
            return None
        stats = {
            "turns": len(self._turns),
            "p50_ms": round(statistics.median(self._turns) * 1000),
            "p95_ms": round(percentile(self._turns, 0.95) * 1000),
            "min_ms": round(min(self._turns) * 1000),
            "max_ms": round(max(self._turns) * 1000),
        }
        if self._decisions:
            stats["decision_p50_ms"] = round(statistics.median(self._decisions) * 1000)
        if self._prompt_total:
            stats["prompt_tokens"] = self._prompt_total
            stats["cached_tokens"] = self._cached_total
            stats["cached_share"] = round(self._cached_total / self._prompt_total, 3)
        return stats

    def log_summary(self) -> Optional[dict]:
        stats = self.summary()
        if stats is None:
            logger.info(f"[{self._call_sid}] LATENCY no measurable turns")
            return None
        cache = ""
        if "prompt_tokens" in stats:
            cache = (
                f" cache={stats['cached_tokens']}/{stats['prompt_tokens']}tok "
                f"({stats['cached_share'] * 100:.0f}%)"
            )
        decision = (
            f" turn_decision_p50={stats['decision_p50_ms']}ms"
            if "decision_p50_ms" in stats
            else ""
        )
        logger.info(
            f"[{self._call_sid}] LATENCY summary | turns={stats['turns']} "
            f"p50={stats['p50_ms']}ms p95={stats['p95_ms']}ms "
            f"min={stats['min_ms']}ms max={stats['max_ms']}ms{decision}{cache}"
        )
        return stats
