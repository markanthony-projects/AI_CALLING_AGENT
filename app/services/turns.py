"""When the caller has taken the floor, and the agent should stop talking.

Two gates that answer the same question from different evidence, because different speech
services give different evidence. Both expose `relax()`, so the agent asks for a gate and
calls it without knowing which one it got.

The question is not "is somebody making noise" — that one is easy and wrong. On a sales call
the prospect agrees along constantly, says "hello?" when the line goes quiet, and coughs.
Stopping for any of that is what makes an agent feel like a machine. The question is whether
they are *taking the floor*, and the two gates guess at it differently:

    GreetingOnlyMinWords   counts words, so it needs a service that transcribes as it goes.
    SustainedSpeechBargeIn measures how long they have kept talking, and needs no words.

Neither is strictly better. Words are more selective — "haan" and "ek minute ruko" are the
same length in seconds and nothing alike in meaning — but they cost the time to say the
words plus the time to transcribe them, about a second in practice. Duration costs exactly
its threshold and nothing else, and cannot tell those two apart at all.

Which one a call gets is decided in app/services/stt_provider.py, next to the choice of
service, because it follows from that choice rather than standing on its own.
"""

import asyncio
from typing import Optional

from loguru import logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.turns.types import ProcessFrameResult
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

from app.utils.barge_in import takes_the_floor


class GreetingOnlyMinWords(MinWordsUserTurnStartStrategy):
    """A word gate that lifts as soon as the opening line has been delivered.

    A flat gate looked right and was wrong. While the bot is speaking — and Pipecat counts
    that from the first audio frame until the last one has played out, seconds after its
    text is finished — anything shorter than min_words is discarded outright, not deferred.
    So "Yeah sure." answering "Would you like to visit the site?" vanished, and the caller
    sat through 37 seconds of silence saying "Hello" twice before the agent noticed. The
    one-and-two-word replies this dropped are exactly the replies people give.

    The gate only ever existed to stop the "Hello?" on pickup from cutting the greeting off
    at 0.7 seconds. Once that line is out there is nothing left to protect, so it relaxes to
    one word and the prospect can interrupt whenever they like for the rest of the call.

    Whenever they like, with one exception earned on call 5023ff25: "Hello?" spoken while
    the agent is talking is not somebody taking the floor, it is somebody who cannot hear —
    and Sarvam reopens its websocket on every interruption, so four of those left the agent
    mute and the prospect hung up. Those words wait for the sentence to finish instead of
    cutting it off, and are then answered rather than discarded. See app/utils/barge_in.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._deferred_line_check = False

    def relax(self) -> None:
        self._min_words = 1

    async def reset(self):
        await super().reset()
        self._deferred_line_check = False

    async def _handle_transcription(self, frame):
        if self._bot_speaking and not takes_the_floor(getattr(frame, "text", "")):
            # Held, not dropped. The base class would call trigger_reset_aggregation() and
            # the words would be gone; leaving the aggregation alone keeps them for the turn
            # that starts as soon as the agent stops speaking.
            self._deferred_line_check = True
            return ProcessFrameResult.CONTINUE
        self._deferred_line_check = False
        return await super()._handle_transcription(frame)

    async def _handle_bot_stopped_speaking(self, frame):
        await super()._handle_bot_stopped_speaking(frame)
        if self._deferred_line_check:
            # The sentence is finished and they are still owed an answer.
            self._deferred_line_check = False
            await self.trigger_user_turn_started()


class SustainedSpeechBargeIn(BaseUserTurnStartStrategy):
    """Takes the floor for whoever keeps talking, and needs no transcript to do it.

    For a service that says nothing until the turn is over. Deepgram Flux is one, and so is
    Sarvam: neither pushes an InterimTranscriptionFrame, so a word gate has nothing to count
    until the caller has finished — by which point the agent has talked over all of it.

    So this counts seconds instead. While the agent is silent there is no floor to take and
    VAD starting is enough. While the agent is speaking, the caller has to keep speaking for
    `min_speech_secs` before the agent stops, which is the whole mechanism: a cough does not
    last that long, and neither does "hello?", and neither does "haan". A person who
    actually wants the floor keeps going.

    The threshold is doing the job the word count used to do, and it is worth being honest
    about what it cannot do. Three words take about a second to say, so the gate it replaces
    was never fast either — but the word gate knew "ek minute ruko" from "haan haan" and
    this cannot, because they are the same length. That difference is what a hold-and-resume
    buffer would buy back, by making a wrong guess cost a hiccup instead of a sentence.
    Until then the threshold is set where a line-check does not reach it.
    """

    def __init__(self, *, min_speech_secs: float, **kwargs):
        super().__init__(**kwargs)
        self._min_speech_secs = min_speech_secs
        self._bot_speaking = False
        self._waiting: Optional[asyncio.Task] = None

    def relax(self) -> None:
        """Nothing to relax, and that is the point rather than an omission.

        The word gate relaxes because at three words it DISCARDED shorter replies, so it had
        to be loosened the moment the greeting no longer needed protecting. This gate never
        discards anything: speech under the threshold is not thrown away, it is simply not
        an interruption, and the words still arrive as their own turn once the caller stops.
        There is no cost to carrying the same threshold for the whole call.
        """

    async def reset(self):
        await super().reset()
        await self._stand_down()

    async def cleanup(self):
        await super().cleanup()
        await self._stand_down()

    async def process_frame(self, frame: Frame) -> ProcessFrameResult:
        """Watch the agent's speech and the caller's, and start the turn when it is theirs."""
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            # Whatever they were saying is no longer an interruption, and the next VAD start
            # will take the floor immediately. Stopping the clock here rather than letting it
            # run out keeps a turn from starting twice for one utterance.
            await self._stand_down()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if not self._bot_speaking:
                await self.trigger_user_turn_started()
                return ProcessFrameResult.STOP
            await self._stand_by()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._stand_down()
        return ProcessFrameResult.CONTINUE

    async def _stand_by(self):
        """Start the clock. If they are still going when it runs out, the floor is theirs."""
        await self._stand_down()
        self._waiting = self.create_task(self._wait_them_out())

    async def _stand_down(self):
        if self._waiting:
            await self.cancel_task(self._waiting)
            self._waiting = None

    async def _wait_them_out(self):
        await asyncio.sleep(self._min_speech_secs)
        logger.debug(f"{self}: still speaking after {self._min_speech_secs}s — the floor is theirs")
        await self.trigger_user_turn_started()
