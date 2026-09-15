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

Two more things live here since 15 Sep 2026, both about the same socket:

  * build_tts(settings) — the one place the service is constructed. The answer webhook
    builds it three seconds before the media stream opens (app/services/warm_tts.py) and
    the agent builds it when it did not, and both must produce the same voice.

  * A spare socket. Pipecat's barge-in handling above — close, then reopen — is a 180ms
    handshake the prospect waits through every time they speak over the agent, and each
    one is a chance to lose the socket. With a second socket already open and configured,
    a barge-in becomes a swap: the live socket is discarded, the spare becomes live, and a
    new spare is opened in the background. The reconnect path stays as the fallback for
    the moment there is no spare.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional

from loguru import logger
from pipecat.frames.frames import Frame, InterruptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.services.tts_service import TTSService
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from app.utils.dashes import DashFilter

# How many times one call will reopen a socket that has died on it. Past this the failure is
# Sarvam rather than the reconnect race this exists for, and the error already on the wire
# is the more useful signal.
MAX_REVIVALS = 3
# Sarvam idles a silent socket out; its own client pings every 20s and so does the spare.
SPARE_KEEPALIVE_SECS = 20
SPARE_CONNECT_TIMEOUT_SECS = 5.0


def build_tts(settings, *, spare_socket: Optional[bool] = None) -> "KeepsItsVoice":
    """The voice, built the same way wherever it is built.

    Passed only when somebody has set it: SARVAM_TEMPERATURE unset keeps the key out of the
    connect payload exactly as it has been on every call so far, so a deployment cannot
    change the voice on its own; set, it steadies the prosody Sarvam otherwise re-rolls at
    every full stop. See config.py and tests/test_voice_consistency.py.
    """
    tts_tuning = {}
    if settings.SARVAM_TEMPERATURE is not None:
        tts_tuning["temperature"] = settings.SARVAM_TEMPERATURE

    return KeepsItsVoice(
        api_key=settings.SARVAM_API_KEY,
        spare_socket=settings.TTS_SPARE_SOCKET if spare_socket is None else spare_socket,
        # Dashes the engine misreads become commas and hyphens on the way in. See
        # app/utils/dashes.py for the call that showed it.
        text_filters=[DashFilter()],
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice=settings.SARVAM_VOICE_ID,
            pace=settings.SPEAKING_PACE,
            **tts_tuning,
            max_chunk_length=150,
            # min_buffer_size is deliberately left at Sarvam's default. Setting it to 25
            # was rejected at connect time with "Input parameters has to be a valid
            # dictionary", killing TTS for the whole call. Pipecat forwards the value
            # straight into the config payload with no range check, so any new value here
            # must be validated against Sarvam's API before it reaches a live call.
        ),
    )


class KeepsItsVoice(SarvamTTSService):
    """Reopens a websocket that a failed reconnect left closed, before speaking.

    Checked here rather than on the error event: the error arrives while the barge-in is
    still being handled, and reconnecting into that is the race that lost the socket in the
    first place. The next thing the agent wants to say is both later and the only moment at
    which a missing socket actually costs anything.
    """

    def __init__(self, *, spare_socket: bool = False, **kwargs):
        # Named after the vendor, not after this class. app/utils/latency.py derives its
        # metric label from the instance name, so the first call on this subclass logged
        # "keepsitsvoice=201ms" where every earlier call in this repository logged
        # "sarvam=201ms" — silently breaking every comparison against its own history. The
        # number is Sarvam's latency whatever we wrap it in.
        kwargs.setdefault("name", SarvamTTSService.__name__)
        super().__init__(**kwargs)
        self._revivals = 0
        self._spare_enabled = spare_socket
        self._spare = None
        self._spare_opener: Optional[asyncio.Task] = None
        self._spare_keepalive: Optional[asyncio.Task] = None
        self._swaps = 0

    @property
    def revivals(self) -> int:
        """How many times this call's voice had to be brought back."""
        return self._revivals

    @property
    def swaps(self) -> int:
        """How many barge-ins were handled by swapping sockets instead of reconnecting."""
        return self._swaps

    @property
    def spare_ready(self) -> bool:
        return self._spare is not None and self._spare.state is State.OPEN

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

    # ---------------------------------------------------------------- the spare socket

    def _config_payload(self) -> dict:
        """Exactly what SarvamTTSService._send_config sends, for a socket it did not open.

        Kept in step by tests/test_voice_spare.py, which captures the real one.
        """
        config = {
            "target_language_code": self._settings.language,
            "speaker": self._settings.voice,
            "speech_sample_rate": self._speech_sample_rate,
            "enable_preprocessing": self._settings.enable_preprocessing,
            "min_buffer_size": self._settings.min_buffer_size,
            "max_chunk_length": self._settings.max_chunk_length,
            "output_audio_codec": self._output_audio_codec,
            "output_audio_bitrate": self._output_audio_bitrate,
            "pace": self._settings.pace,
            "model": self._settings.model,
        }
        if self._settings.pitch is not None:
            config["pitch"] = self._settings.pitch
        if self._settings.loudness is not None:
            config["loudness"] = self._settings.loudness
        if self._settings.temperature is not None:
            config["temperature"] = self._settings.temperature
        return config

    async def _open_socket(self):
        """A fresh, configured Sarvam socket — the same handshake _connect_websocket does."""
        socket = await websocket_connect(
            self._websocket_url,
            additional_headers={"api-subscription-key": self._api_key},
            user_agent_header=sdk_headers()["User-Agent"],
        )
        await socket.send(json.dumps({"type": "config", "data": self._config_payload()}))
        return socket

    def _replenish_spare(self):
        """Open the next spare in the background, unless one is open or being opened."""
        if not self._spare_enabled or self.spare_ready:
            return
        if self._spare_opener and not self._spare_opener.done():
            return
        self._spare_opener = asyncio.create_task(self._open_spare())

    async def _open_spare(self):
        try:
            self._spare = await asyncio.wait_for(self._open_socket(), SPARE_CONNECT_TIMEOUT_SECS)
            if not self._spare_keepalive or self._spare_keepalive.done():
                self._spare_keepalive = asyncio.create_task(self._spare_keepalive_handler())
            logger.debug(f"{self}: spare voice socket ready")
        except Exception as e:  # noqa: BLE001 — no spare means the old path, not a failure
            self._spare = None
            logger.warning(
                f"{self}: could not open a spare voice socket ({e}); barge-ins will reconnect"
            )

    async def _spare_keepalive_handler(self):
        while True:
            await asyncio.sleep(SPARE_KEEPALIVE_SECS)
            if self.spare_ready:
                try:
                    await self._spare.send(json.dumps({"type": "ping"}))
                except Exception:  # noqa: BLE001
                    self._spare = None

    async def _close_quietly(self, socket):
        try:
            await socket.close()
        except Exception:  # noqa: BLE001
            pass

    async def _drop_spare(self):
        if self._spare_opener and not self._spare_opener.done():
            self._spare_opener.cancel()
        self._spare_opener = None
        if self._spare_keepalive:
            self._spare_keepalive.cancel()
            self._spare_keepalive = None
        if self._spare is not None:
            spare, self._spare = self._spare, None
            await self._close_quietly(spare)

    async def _connect(self):
        await super()._connect()
        if self._websocket:
            self._replenish_spare()

    async def _disconnect(self):
        await self._drop_spare()
        await super()._disconnect()

    async def _swap_in_spare(self):
        """The spare becomes the live socket; the live one is closed off the hot path."""
        old = self._websocket
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None
        await self.stop_all_metrics()

        self._websocket, self._spare = self._spare, None
        self._receive_task = self.create_task(self._receive_task_handler(self._report_error))
        self._keepalive_task = self.create_task(self._keepalive_task_handler())
        self._swaps += 1

        if old is not None:
            asyncio.create_task(self._close_quietly(old))
        self._replenish_spare()

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        """A barge-in while speaking: swap sockets if a spare is ready, else reconnect.

        InterruptibleTTSService's version is "TTSService's, then disconnect and connect if
        speaking". With a spare open the second half is replaced by the swap, so the
        TTSService half is called directly and Interruptible's is skipped on purpose.
        """
        if self._bot_speaking and self.spare_ready:
            await TTSService._handle_interruption(self, frame, direction)
            await self._swap_in_spare()
            logger.info(f"{self}: barge-in handled by socket swap ({self._swaps} this call)")
            return
        await super()._handle_interruption(frame, direction)
