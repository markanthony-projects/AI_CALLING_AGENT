import base64
import json
from loguru import logger
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    Frame,
    CancelFrame,
)

class VobizSerializer(FrameSerializer):
    """
    Production-grade serializer for Vobiz AI WebSockets.
    Vobiz supports native 16kHz Linear PCM (audio/x-l16).
    """
    
    def __init__(self, stream_sid: str):
        super().__init__(params=FrameSerializer.InputParams(resampler_clear_after_secs=None))
        self._stream_sid = stream_sid
        self._sample_rate = 16000

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            # Vobiz natively accepts 16000Hz PCM audio
            payload = base64.b64encode(frame.audio).decode("utf-8")
            
            message = {
                "event": "playAudio",
                "streamId": self._stream_sid,
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": 16000,
                    "payload": payload
                }
            }
            return json.dumps(message)
            
        if isinstance(frame, CancelFrame):
            # Vobiz clearAudio command to stop playing outbound audio (barge-in)
            message = {
                "event": "clearAudio",
                "streamId": self._stream_sid
            }
            return json.dumps(message)
            
        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
            event = message.get("event")
            
            if event == "start":
                # Save the real Vobiz streamId to send audio back to
                self._stream_sid = message.get("streamId", self._stream_sid)
                logger.info(f"Vobiz stream connected with Stream ID: {self._stream_sid}")

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
                    return InputAudioRawFrame(audio=audio_data, sample_rate=self._sample_rate, num_channels=1)

            elif event == "stop":
                logger.info(f"[{self._stream_sid}] Vobiz stop event received")
                # Can return EndFrame, but typically Pipecat handles disconnect via transport

        except Exception as e:
            logger.error(f"Error deserializing Vobiz message: {e}")
            
        return None
