"""Once the agent has said goodbye, nothing else gets generated.

Live call 3c43b6bc, 4 Sep 2026. The prospect heard two sign-offs, back to back:

    11:21:45  end_call fires, goodbye queued
    11:21:45  USER  → "No, I don't want to visit the site Thank you, you are
                       repeating question."
    11:21:51  AGENT → "I will send you the brochure and price details on
                       WhatsApp. Thank you for your time, Rahul. Have a great day!"
    11:21:58  AGENT → "No problem at all. I apologize for that. I will send you
                       the brochure, floor plans and price details on WhatsApp.
                       Thank you for your time."
    11:21:59  pipeline stopped

The prospect spoke while the goodbye was being played. Their turn finalized, an inference
ran on it, and its answer was spoken after the farewell had already finished.

`_ending` did not stop this and was never meant to: it guards against a SECOND end_call,
which is a different thing. Between queueing the goodbye and the pipeline actually stopping
there is a real gap — the length of the sentence, several seconds — and the rest of the
pipeline is alive for all of it.

The interruption that end_call opens with does not cover it either. It cancels what is in
flight at that instant; an inference that starts a moment later is not in flight yet.

So this closes the source instead. LLMContextFrame is what makes the LLM service run a
completion — base_llm.py acts on exactly that frame and nothing else — so once the call is
ending, that frame stops here and no reply can be generated to speak. The goodbye itself is
queued as a TTSSpeakFrame at the task source and never passes through the LLM, so it is
untouched.
"""

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ClosingGate(FrameProcessor):
    """Stops new inferences once the call is on its way out.

    Placed between the user aggregator and the LLM, which is the only stretch where an
    LLMContextFrame exists: the aggregator emits it and the LLM consumes it.
    """

    def __init__(self, call_sid: str, **kwargs):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._closing = False
        self._dropped = 0

    @property
    def closing(self) -> bool:
        return self._closing

    @property
    def dropped(self) -> int:
        """How many turns were abandoned on the way out, for the log."""
        return self._dropped

    def arm(self) -> None:
        """Called the moment end_call is honoured. There is no way back from here."""
        self._closing = True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if self._closing and isinstance(frame, LLMContextFrame):
            self._dropped += 1
            # Logged every time rather than once. Two dropped turns during a goodbye means
            # the prospect kept talking through it, which says something about the goodbye.
            logger.info(
                f"[{self._call_sid}] Dropped a turn generated after the goodbye "
                f"({self._dropped}); the call is already ending"
            )
            return

        await self.push_frame(frame, direction)
