"""Where the silence before the first word goes.

Every call opens with the prospect holding a live line and hearing nothing. Measured on five
calls it is the same number each time:

    FIRST WORD 2257ms after the stream opened | pipeline=2079ms synthesis=343ms

`synthesis` is the greeting reaching the voice engine and coming back as audio — 343ms, and
not the problem. The other 1736ms is everything before the greeting was even queued, and the
greeting needs nothing to build: it is a local f-string with no model and no network in it.

So something is holding it, and `pipeline=` cannot say what. The greeting is queued from
`on_pipeline_started`, which Pipecat fires only once StartFrame has been through every
processor in the pipeline — and both the speech-to-text and the text-to-speech services open
a websocket when that frame reaches them. A connection the greeting does not need can still
be a connection the greeting waits behind.

That is a theory the arithmetic fits and nothing measures, which is the same shape as the
600ms turn_decision was before app/utils/latency.py existed: a plausible story standing in
for a number. This turns it into a number. Nothing here changes what the call does.
"""

import time
from typing import List, Optional, Tuple

from loguru import logger


class StartupClock:
    """One line naming every wait between the line opening and the first word.

    Milestones are recorded in the order they happen and reported once. A milestone that
    never arrives is simply absent from the line, which is itself worth seeing — a TTS that
    never connected is a call with no voice at all.
    """

    def __init__(self, call_sid: str, stream_open_at: Optional[float] = None):
        self._call_sid = call_sid
        self._stream_open_at = stream_open_at
        self._marks: List[Tuple[str, float]] = []
        self._reported = False

    def mark(self, name: str) -> None:
        """Record that `name` has just happened.

        Silently ignored once the line has been reported: a reconnect later in the call is a
        real event, but it is not part of how long the greeting took and putting it here
        would make the number mean two different things.
        """
        if self._reported or self._stream_open_at is None:
            return
        self._marks.append((name, (time.monotonic() - self._stream_open_at) * 1000))

    def report(self) -> None:
        """Emit the line, once."""
        if self._reported or self._stream_open_at is None or not self._marks:
            return
        self._reported = True
        # Each mark is shown as its own wait as well as its clock reading, because the
        # question is never "when did TTS connect" — it is "what was the greeting waiting
        # for", and that is the gap, not the timestamp.
        parts = []
        previous = 0.0
        for name, at in self._marks:
            parts.append(f"{name}=+{at - previous:.0f}ms")
            previous = at
        logger.info(
            f"[{self._call_sid}] STARTUP {previous:.0f}ms to the greeting | " + "  ".join(parts)
        )
