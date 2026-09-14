"""The numbers a call is judged by, gathered once so they can be stored once.

Every one of these was already being logged — the LATENCY summary, the FIRST WORD line, the
held-reply count, the TTS reconnects, the LLM failures — and every one of them died in a
log file that rotates after a couple of days. Asked on 12 Sep what the p50 was for the
week, the honest answer was "grep, if the lines are still there". A product whose USP is
latency has to be able to answer that from a table.

This is the row. It is built from the same counters the log lines use, so the table and
the log can never disagree, and it is written best-effort at finalisation: a call whose
metrics cannot be stored is still a finished call.

Deliberately not a per-turn table. Three lines produce a few hundred rows a day here; the
per-turn breakdown stays in the LATENCY lines, where the attribution is, and this carries
what the dashboard needs to draw a trend and flag a bad day.
"""

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass(frozen=True)
class CallMetricsSnapshot:
    """One call's numbers, as columns."""

    turns: int
    p50_turn_ms: Optional[int]
    p95_turn_ms: Optional[int]
    max_turn_ms: Optional[int]
    # From the media stream opening to the first audio of the greeting.
    first_word_ms: Optional[int]
    # Replies generated to a half-finished sentence and kept off the line (TurnFinalityGate).
    held_replies: int
    # Times the voice websocket was torn down and reopened by a barge-in, beyond the first
    # connection. Four of these in fourteen seconds is what preceded a call going mute.
    tts_reconnects: int
    # Times a socket a failed reconnect had left closed was reopened before speaking.
    tts_revivals: int
    llm_failures: int
    end_reason: Optional[str]
    # Which ears and which brain, so a regression can be tied to a configuration change.
    stt: str
    llm: str

    @classmethod
    def collect(
        cls,
        *,
        latency_summary: Optional[dict],
        first_word_ms: Optional[int],
        held_replies: int,
        tts_reconnects: int,
        tts_revivals: int,
        llm_failures: int,
        end_reason: Optional[str],
        stt: str,
        llm: str,
    ) -> "CallMetricsSnapshot":
        """Build from the pieces run_voice_agent already has at the end of a call.

        The latency summary is None for a call with no measurable turn — a voicemail, a
        hang-up during the greeting — and that is a real row, not a missing one: zero
        turns with a first-word time is exactly what a call nobody spoke on looks like.
        """
        stats = latency_summary or {}
        return cls(
            turns=int(stats.get("turns", 0)),
            p50_turn_ms=stats.get("p50_ms"),
            p95_turn_ms=stats.get("p95_ms"),
            max_turn_ms=stats.get("max_ms"),
            first_word_ms=first_word_ms,
            held_replies=int(held_replies),
            tts_reconnects=int(tts_reconnects),
            tts_revivals=int(tts_revivals),
            llm_failures=int(llm_failures),
            end_reason=end_reason,
            stt=stt,
            llm=llm,
        )

    def as_row(self) -> dict:
        """Column values for the call_metrics table."""
        return asdict(self)
