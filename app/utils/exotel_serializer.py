import base64
import json
from loguru import logger
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

class ProductionExotelSerializer(FrameSerializer):
    """Production-grade serializer for Exotel Media Streams WebSocket protocol.
    
    Ensures absolute compliance with PSTN standard 8kHz G.711 µ-law audio by
    running explicit PCM <-> µ-law encoding on payload delivery, rather than
    naive PCM base64 stringification.
    """

    class InputParams(FrameSerializer.InputParams):
        exotel_sample_rate: int = 8000
        sample_rate: int | None = None

    def __init__(
        self, stream_sid: str, call_sid: str | None = None, params: InputParams | None = None
    ):
        params = params or ProductionExotelSerializer.InputParams()
        super().__init__(params)
        self._params: ProductionExotelSerializer.InputParams = params

        self._stream_sid = stream_sid
        self._call_sid = call_sid

        self._exotel_sample_rate = self._params.exotel_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._input_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )
        self._output_resampler = create_stream_resampler(
            clear_after_secs=self._params.resampler_clear_after_secs
        )

    async def setup(self, frame: StartFrame):
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # 1. Resample to 8kHz if not already 8kHz
            # 2. Encode to mu-law to satisfy Exotel PSTN standard
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._exotel_sample_rate, self._output_resampler
            )
            
            if serialized_data is None or len(serialized_data) == 0:
                return None

            payload = base64.b64encode(serialized_data).decode("ascii")

            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
            
        elif isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            if self.should_ignore_frame(frame):
                return None
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] == "media":
            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # 1. Decode from Exotel's mu-law payload to PCM
            # 2. Resample to target pipeline sample rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._exotel_sample_rate, self._sample_rate, self._input_resampler
            )
            
            if deserialized_data is None or len(deserialized_data) == 0:
                return None

            audio_frame = InputAudioRawFrame(
                audio=deserialized_data,
                num_channels=1, 
                sample_rate=self._sample_rate,
            )
            return audio_frame
            
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")
            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError:
                logger.info(f"Invalid DTMF digit: {digit}")
                return None

        return None
