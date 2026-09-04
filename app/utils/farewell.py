"""Do not hang up until the goodbye has actually been spoken.

The closing line is the last thing the prospect hears, and on a booked visit it carries the
day and the time. Cutting it off does not merely sound rude — it loses the confirmation the
whole call was for.

Two live-call cutoffs, with different causes:

    17:37:46.471  end_call fired
    17:37:46.896  pipeline finished          <- 425ms, for a three-second sentence

That one was EndFrame stopping the transport in queue order, and it was replaced with
EndWorkerFrame, which makes a round trip to the sink and back before becoming an EndFrame.
The cutoff came back anyway, because the round trip only proves the FRAMES have travelled.
Sarvam is a websocket TTS: run_tts sends the text and returns, and the audio arrives later
on a separate receive task. So the end signal can complete its lap while the voice is still
streaming in behind it.

The second cause is a race in the same queue. end_call queued three frames at once:

    [InterruptionWorkerFrame(), TTSSpeakFrame(line), EndWorkerFrame()]

The interruption is there to discard a stale reply from a split turn. But it only takes
effect after ITS round trip to the sink and back, and the worker's push loop does not wait
for that — it queues the next frame immediately. So the farewell can enter the pipeline
first and the interruption then cancels the goodbye itself.

Nothing here relies on frame ordering. The transport raises BotStoppedSpeakingFrame from
its audio clock task when the turn's audio has actually been played out at realtime pace,
so that is the signal to wait for, with a ceiling so a dead TTS cannot hold the line open.
"""

import asyncio

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# How long to allow for the goodbye before hanging up regardless. Derived from the line
# rather than fixed: a two-word farewell should not hold the leg open for the length of a
# read-back with a day and a time in it.
_WORDS_PER_SECOND = 2.4  # measured on Sarvam bulbul:v3 at pace 1.0
_STARTUP_ALLOWANCE = 2.5  # synthesis latency plus the frame's trip down the pipeline

# Even a maximum-length closing_line (MAX_CLOSING_CHARS = 240) is about forty words. Past
# this the audio is not coming, and every extra second is billed by the carrier.
MAX_FAREWELL_WAIT_SECS = 20.0


def farewell_timeout(line: str) -> float:
    """A ceiling on the wait, sized to the sentence being spoken."""
    words = len((line or "").split())
    return min(MAX_FAREWELL_WAIT_SECS, _STARTUP_ALLOWANCE + words / _WORDS_PER_SECOND)


class FarewellGate(FrameProcessor):
    """Signals when the bot has finished speaking.

    Placed after the output transport, because that is where BotStoppedSpeakingFrame is
    raised — from the audio clock task, once the turn's audio has been written out at
    realtime pace. Upstream of it the frame does not exist yet.

    arm() must be called immediately before the farewell is queued. Clearing the event there
    is what stops a BotStoppedSpeakingFrame from the turn BEFORE the goodbye releasing the
    wait instantly — which would reintroduce the cutoff through the mechanism meant to fix
    it. There is deliberately no second "armed" flag gating process_frame: it would be a
    duplicate of that clear, and a guard that cannot change any outcome is a guard that
    invites the next reader to trust it for something it does not do.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._spoken = asyncio.Event()
        self._speaking = False

    def arm(self) -> None:
        self._spoken.clear()

    @property
    def is_speaking(self) -> bool:
        """True while the transport is playing the bot's audio out.

        Read by end_call, which has to know whether there is anything on the wire before it
        decides between waiting for it and cutting it off.
        """
        return self._speaking

    async def wait_for_quiet(self, timeout: float) -> bool:
        """Let whatever is being spoken right now finish. True if it did, or if nothing was.

        The immediate True matters as much as the wait. end_call fires from inside the LLM
        service, sometimes before a word of the reply has reached the transport and
        sometimes after all of it has; blocking on an event in the second case would add
        the whole timeout to every goodbye.
        """
        if not self._speaking:
            return True
        return await self.wait_until_spoken(timeout)

    async def wait_until_spoken(self, timeout: float) -> bool:
        """True if the bot finished speaking, False if the wait ran out.

        False is not a reason to stay on the line. The caller hangs up either way; the
        return value only says whether the goodbye made it out, which is worth a log line.
        """
        try:
            await asyncio.wait_for(self._spoken.wait(), timeout)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            self._speaking = True
        if isinstance(frame, BotStoppedSpeakingFrame):
            self._speaking = False
            self._spoken.set()
        await self.push_frame(frame, direction)
