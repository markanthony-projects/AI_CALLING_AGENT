import asyncio
import base64
import json

from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer


class VobizSerializer(FrameSerializer):
    """
    Production-grade serializer for Vobiz AI WebSockets.
    Vobiz supports native 16kHz Linear PCM (audio/x-l16).

    The stream has two names. It is opened under our call id, and Vobiz's first message —
    "start" — carries the streamId it wants on every playAudio we send back. Until that
    message has been read, outbound audio goes out under our id, which is not theirs. Call
    07709de3, 15 Sep 2026: the greeting was primed and on the wire 246ms after the socket
    opened, faster than any call before it, and the prospect heard nothing for twelve
    seconds and hung up. So the moment the stream is named is exposed (`started`, and
    wait_for_start for the greeting to hold on), and audio sent before it is counted and
    said out loud, so the next silent call can be read rather than guessed at.
    """

    def __init__(self, stream_sid: str):
        super().__init__(params=FrameSerializer.InputParams(resampler_clear_after_secs=None))
        self._call_sid = stream_sid
        self._stream_sid = stream_sid
        self._sample_rate = 16000
        self.started = asyncio.Event()
        self._sent_before_start = 0

    @property
    def sent_before_start(self) -> int:
        """Outbound audio frames serialized before Vobiz had named the stream."""
        return self._sent_before_start

    async def wait_for_start(self, timeout: float) -> bool:
        """True once Vobiz has named the stream; False if it has not within `timeout`."""
        if self.started.is_set():
            return True
        try:
            await asyncio.wait_for(self.started.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            if not self.started.is_set():
                self._sent_before_start += 1
                if self._sent_before_start == 1:
                    logger.warning(
                        f"[{self._call_sid}] Audio sent before Vobiz's start event; it carries "
                        f"our stream id, not theirs, and may not be played"
                    )
            # Vobiz natively accepts 16000Hz PCM audio
            payload = base64.b64encode(frame.audio).decode("utf-8")

            message = {
                "event": "playAudio",
                "streamId": self._stream_sid,
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": 16000,
                    "payload": payload,
                },
            }
            return json.dumps(message)

        if isinstance(frame, CancelFrame):
            # Vobiz clearAudio command to stop playing outbound audio (barge-in)
            message = {"event": "clearAudio", "streamId": self._stream_sid}
            return json.dumps(message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
            event = message.get("event")

            if event == "start":
                # Save the real Vobiz streamId to send audio back to
                self._stream_sid = message.get("streamId", self._stream_sid)
                self.started.set()
                logger.info(f"[{self._call_sid}] Vobiz stream started | streamId={self._stream_sid}")

            elif event == "media":
                # Handle both flat payload and nested media payload just in case
                media_obj = message.get("media")
                if isinstance(media_obj, dict):
                    payload = media_obj.get("payload")
                else:
                    payload = message.get("payload")

                if payload:
                    # Vobiz provides native 16000Hz PCM, just decode and pass along
                    audio_data = base64.b64decode(payload)
                    return InputAudioRawFrame(
                        audio=audio_data, sample_rate=self._sample_rate, num_channels=1
                    )

            elif event == "stop":
                logger.info(f"[{self._call_sid}] Vobiz stop event received")
                # Can return EndFrame, but typically Pipecat handles disconnect via transport

        except Exception as e:
            logger.error(f"Error deserializing Vobiz message: {e}")

        return None
