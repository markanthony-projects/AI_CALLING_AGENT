"""Record what the STT actually said, so an empty turn can be diagnosed rather than guessed.

A live call produced this twice, and there was no way to tell from the logs what had
happened:

    VAD fired with no transcribable speech after 5003ms — likely a false barge-in

Silero heard five seconds of something. Deepgram raised no error and did not reconnect —
both would have logged at WARNING — yet the aggregator finalized the turn with an empty
message. From that line alone the cause is unknowable, and the three candidates need
different fixes:

    interims arrived, no final     -> an endpointing problem; the final never came, or came
                                      after the turn had already closed
    interims arrived and were empty-> Deepgram heard the audio and found no words in it, so
                                      it really was line noise and nothing was lost
    nothing arrived at all         -> audio is not reaching Deepgram; the connection is up
                                      but silent, which is the worst case and the one no
                                      existing log would reveal

This processor is a pure pass-through that keeps a per-turn tally of what came past it.
It is deliberately silent in the normal case: the report is read only when a turn ends with
nothing to show for itself, which is the one moment the answer is worth the log line.
"""

from typing import Optional

from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class SttWitness(FrameProcessor):
    """Counts transcription frames per user turn and reports on request.

    Sits between the STT and the user aggregator so it sees exactly what the aggregator
    sees. It never holds, drops or rewrites a frame — an observability seam in the audio
    path must not be able to break the audio path.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._interims = 0
        self._finals = 0
        self._last_text = ""

    def reset(self) -> None:
        """Start a fresh tally. Called when the prospect's turn begins."""
        self._interims = 0
        self._finals = 0
        self._last_text = ""

    def report(self) -> str:
        """One line describing what the STT produced during the turn just ended."""
        if not self._interims and not self._finals:
            return "STT sent nothing at all — audio may not be reaching it"
        detail = f"{self._interims} interim, {self._finals} final"
        if not self._last_text:
            return f"STT sent {detail}, all empty — the audio carried no words"
        return f"STT sent {detail}, last was {self._last_text[:60]!r}"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        # isinstance order matters: Deepgram's interim frames are not a subclass of the
        # final one in Pipecat 1.5, but checking the narrower type first keeps that true
        # even if that changes.
        if isinstance(frame, InterimTranscriptionFrame):
            self._interims += 1
            self._remember(frame)
        elif isinstance(frame, TranscriptionFrame):
            self._finals += 1
            self._remember(frame)

        await self.push_frame(frame, direction)

    def _remember(self, frame: Frame) -> None:
        text: Optional[str] = (getattr(frame, "text", "") or "").strip()
        if text:
            self._last_text = text
