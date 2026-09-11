"""Never speak an answer to half a question.

From a live call, one prospect turn produced two spoken replies:

    USER  → "Yeah एक साल sounds interesting."   (turn duration 1626ms)
    LATENCY turn 4: 90ms voice-to-voice | sarvam=227ms      <- no LLM timing at all
    AGENT → "That works well, Bhupendra. Since you are planning for a year from now,
             are you looking for a home to live in, or for investment?"
    AGENT → "That sounds good, Bhupendra. Are you looking for a home to stay in, or is
             this for investment?"

The caller was asked the same question twice, back to back, and the 90ms with no LLM
timing is the giveaway: the first reply had already been generated before they finished
speaking, so only its audio was left to measure.

This is how Pipecat works rather than a misconfiguration. Its turn controller fires
inference speculatively, the moment a stop strategy sees enough signal, but declines to
finalize the turn while the user is still audible:

    # Inference-triggered fires only while a turn is active. The turn
    # remains active afterward — only `on_user_turn_stopped` flips state.
    ...
    # Never finalize while the user is audibly speaking.

A prospect who pauses for longer than TURN_SETTLE_SECS and then carries on therefore gets
one inference on the fragment and another on the whole utterance. The aggregator
concatenates the segments, so the second answer is the right one and the first is an answer
to half a sentence that should never have been audible.

Interrupting when the second inference fires does not work: by then the aggregator has
already pushed the new context downstream, so an interruption queued at the task source
arrives behind it and cancels the reply worth keeping. Holding avoids that race entirely.

Nothing waits in the ordinary case. BaseUserTurnStopStrategy.trigger_user_turn_stopped()
fires the inference event and then the finalize event, in that order, so by the time a
model has answered a completed turn the turn is already closed and its text passes straight
through. Only a speculative reply — one generated while the prospect was still talking — is
ever held, and it is held for as long as they keep talking.
"""

from typing import List, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class TurnFinalityGate(FrameProcessor):
    """Holds a reply back until the turn it answers is actually over.

    Placed above ToolSyntaxFilter rather than below it, so a superseded reply never reaches
    the leaked-end_call path either: a model answering half a sentence has been known to
    decide the conversation is finished, and hanging up on that is worse than saying it out
    loud.

    A structured tool call is not carried in these frames and so cannot be held here. That
    is a real gap rather than an oversight — it lives inside the LLM service's function-call
    machinery, which no processor sits in front of.
    """

    def __init__(self, call_sid: str, **kwargs):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._turn_open = False
        # Bumped on every inference. A reply whose generation is no longer current is
        # answering a question the prospect has since finished asking.
        self._generation = 0
        self._reply_generation: Optional[int] = None
        self._held: List[Frame] = []
        self._dropped = 0
        # The generation whose reply must not be spoken at all, whatever it turns out to
        # say. A generation rather than a flag, because the reply being silenced has usually
        # not started arriving yet — so there is no frame to mark, only the inference it
        # will belong to.
        self._muzzled_generation: Optional[int] = None

    @property
    def dropped(self) -> int:
        """How many half-sentence replies were kept off the line this call."""
        return self._dropped

    # --- driven by the aggregator's events --------------------------------------------

    def inference_triggered(self) -> None:
        self._generation += 1

    def discard_reply(self, reason: str) -> None:
        """Do not speak this turn's reply. Not "say something shorter" — say nothing.

        Dropping what is held is the smaller half. In the ordinary case nothing is held at
        all: BaseUserTurnStopStrategy closes the turn before the model has answered, so the
        words are still upstream and arrive to find the gate open. Queueing an interruption
        does not close that window either, because an interruption takes its own lap through
        the pipeline and the reply is already travelling.

        That window is the bug. On call 8d86156e the prospect said "wait" four times and the
        agent replied "Sure, I'll wait" and then said its whole pitch again — the model had
        understood and answered, and answering was the problem. So the muzzle stays on until
        a new response begins, which is the first moment there is anything different to say.
        """
        held = bool(self._held)
        self._drop()
        self._muzzled_generation = self._generation
        logger.info(
            f"[{self._call_sid}] Not speaking this turn's reply — {reason}"
            f"{' (a held one was dropped)' if held else ''}"
        )

    def user_turn_started(self) -> None:
        self._turn_open = True

    async def user_turn_stopped(self) -> None:
        """The prospect has finished. Release the held reply, or drop it if it is stale.

        Race-free by construction: the strategy fires the inference event before this one,
        so a reply that is about to be superseded already reads as superseded here.
        """
        self._turn_open = False
        await self._flush()

    # --- the frame path -----------------------------------------------------------------

    @property
    def _muzzled(self) -> bool:
        """Whether the reply now on the wire is the one that was told not to speak.

        Keyed to the inference rather than latched, and the difference is the whole fix. A
        latch cleared by the next LLMFullResponseStartFrame would be cleared by the very
        reply it was set to stop, because in the ordinary turn the model has not started
        answering at the moment the prospect asks for quiet. The next INFERENCE is what
        earns a voice back, and inference_triggered is what counts them.
        """
        return (
            self._muzzled_generation is not None
            and self._reply_generation == self._muzzled_generation
        )

    @property
    def _superseded(self) -> bool:
        return self._reply_generation is not None and self._reply_generation != self._generation

    async def _flush(self) -> None:
        if not self._held:
            return
        if self._superseded:
            self._drop()
            return
        held, self._held = self._held, []
        for frame in held:
            await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _drop(self) -> None:
        spoken = "".join(getattr(f, "text", "") for f in self._held).strip()
        self._held = []
        if not spoken:
            return
        self._dropped += 1
        logger.info(
            f"[{self._call_sid}] Held back a reply to a half-finished sentence "
            f"(count: {self._dropped}): {spoken[:90]!r}"
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Anything held is irrelevant now, and speaking it after an interruption would
            # answer a question the prospect has already moved on from.
            self._drop()
            self._reply_generation = None
            await self.push_frame(frame, direction)
            return

        if direction != FrameDirection.DOWNSTREAM:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            # A new response supersedes anything still waiting from the previous one.
            self._drop()
            self._reply_generation = self._generation
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (LLMTextFrame, LLMFullResponseEndFrame)):
            if self._muzzled:
                # Swallowed rather than held: nothing here is ever going to be spoken, and
                # the End frame goes with the text so no half-open response is left behind.
                if isinstance(frame, LLMFullResponseEndFrame):
                    self._reply_generation = None
                return
            if self._superseded:
                self._held.append(frame)
                self._drop()
                if isinstance(frame, LLMFullResponseEndFrame):
                    self._reply_generation = None
                return
            if self._turn_open:
                # The prospect has not finished. Hold rather than talk over them; the End
                # frame is held with the text so nothing downstream finalizes an empty turn.
                self._held.append(frame)
                return
            await self._flush()
            if isinstance(frame, LLMFullResponseEndFrame):
                self._reply_generation = None
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)
