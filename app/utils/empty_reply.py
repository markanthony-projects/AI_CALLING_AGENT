"""A reply with nothing in it is noticed, named, and answered — not left as silence.

Call 29b2f355, 14 Sep 2026, turn 3. The prospect said "I was looking something near
Whitefield." and heard nothing for six seconds, until they asked "Did you get it?". No
error, no LATENCY line, no "abandoning the stale turn" — the inference ran, or did not, and
either way the pipeline produced no audio and nothing in it could say so. A reasoning model
that spends its budget thinking and emits no content looks exactly like this from outside,
and so does a request that was never sent.

This sits after OneQuestionPerTurn and before the voice engine, so what it counts is what
would actually have been spoken. A response that ends having forwarded no text and made no
tool call is an empty reply: it is logged as one, counted, and handed to the agent, which
asks its last question again the way it already does when the prospect's turn was empty.
The prospect hears the agent recover instead of a line that went dead on them.
"""

from typing import Awaitable, Callable, Optional

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    FunctionCallInProgressFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class EmptyReplyGuard(FrameProcessor):
    """Counts LLM responses that reach the voice engine with nothing to say."""

    def __init__(
        self,
        call_sid: str,
        on_empty: Optional[Callable[[int], Awaitable[None]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._on_empty = on_empty
        self._in_response = False
        self._spoke = False
        self._called_tool = False
        self._empty = 0

    @property
    def empty(self) -> int:
        """How many replies on this call had nothing in them."""
        return self._empty

    def _reset(self) -> None:
        self._in_response = False
        self._spoke = False
        self._called_tool = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset()
            self._in_response = True
        elif isinstance(frame, InterruptionFrame):
            # Cut short by the prospect is not empty; it is unfinished.
            self._reset()
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._called_tool = True
        elif isinstance(frame, LLMTextFrame) and direction == FrameDirection.DOWNSTREAM:
            if frame.text and frame.text.strip():
                self._spoke = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            empty = self._in_response and not self._spoke and not self._called_tool
            self._reset()
            if empty:
                self._empty += 1
                logger.warning(
                    f"[{self._call_sid}] The model replied with nothing (count: {self._empty}) — "
                    f"no words reached the voice engine and no tool was called. The prospect "
                    f"would hear silence; asking the last question again instead"
                )
                if self._on_empty:
                    await self._on_empty(self._empty)

        await self.push_frame(frame, direction)
