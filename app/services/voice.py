"""A voice that comes back after it drops, instead of going quiet for the rest of the call.

Call 5023ff25, 10 Sep 2026. The prospect said "Hello?" four times in fourteen seconds; the
agent went mute for the last thirty and they hung up. app/utils/barge_in.py was written for
the first half of that — the "Hello?" should never have interrupted — and this is the second
half, which was never proved until now.

Sarvam's client is an InterruptibleTTSService, and pipecat's version of that does:

    async def _handle_interruption(self, frame, direction):
        await super()._handle_interruption(frame, direction)
        if self._bot_speaking:
            await self._disconnect()
            await self._connect()

So every barge-in tears the websocket down and opens it again. That is by design — it drops
audio the prospect has spoken over. What is not by design is the failure path:

    except Exception as e:
        await self.push_error(...)
        self._websocket = None

and then, on every later attempt to speak:

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

Nothing reconnects. One failed reconnect and the service is mute for the remaining life of
the call — the pipeline keeps running, the model keeps replying, and none of it reaches the
line. Four reconnects in fourteen seconds is four chances to lose one, which is exactly the
shape of that call.

So the socket is checked at the moment it matters — the next thing the agent tries to say —
and reopened if it has gone. Bounded, because a Sarvam that is genuinely down should not be
hammered once per sentence for the rest of a call, and because a caller hearing nothing is
better served by the call ending than by a pipeline pretending to work.
"""

from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import Frame
from pipecat.services.sarvam.tts import SarvamTTSService

# How many times one call will reopen a socket that has died on it. Past this the failure is
# Sarvam rather than the reconnect race this exists for, and the error already on the wire
# is the more useful signal.
MAX_REVIVALS = 3


class KeepsItsVoice(SarvamTTSService):
    """Reopens a websocket that a failed reconnect left closed, before speaking.

    Checked here rather than on the error event: the error arrives while the barge-in is
    still being handled, and reconnecting into that is the race that lost the socket in the
    first place. The next thing the agent wants to say is both later and the only moment at
    which a missing socket actually costs anything.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._revivals = 0

    @property
    def revivals(self) -> int:
        """How many times this call's voice had to be brought back."""
        return self._revivals

    async def run_tts(self, *args, **kwargs) -> AsyncGenerator[Frame, None]:
        """Speak, having first made sure there is something to speak down.

        Takes *args deliberately. The first version of this declared (self, text) while
        SarvamTTSService.run_tts is (self, text, context_id), so every call raised TypeError,
        three of them tripped MAX_TTS_FAILURES, and the call ended 1.2 seconds in with
        "tts unavailable". This override cares about exactly one thing — is there a socket —
        and has no business restating a signature it does not use. See
        tests/test_voice_recovery.py, which now calls it the way pipecat does.
        """
        if not self._websocket and self._revivals < MAX_REVIVALS:
            self._revivals += 1
            logger.warning(
                f"{self}: the voice socket is closed and nothing reopened it "
                f"(revival {self._revivals}/{MAX_REVIVALS}); reconnecting before speaking"
            )
            await self._connect()
        async for frame in super().run_tts(*args, **kwargs):
            yield frame
