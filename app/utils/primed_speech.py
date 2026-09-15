"""Speech that was synthesised before the call, played the instant it is asked for.

The greeting is a local f-string known at dial time. Until now it reached the voice engine
only once the media stream had opened and the pipeline had started, and the prospect held a
live line through the synthesis — 222ms on 14 Sep, after a 340ms startup that could not be
avoided. The worker now synthesises each sentence of the opening line while the phone is
still ringing (app/services/greeting_cache.py); this is the other end.

It sits directly in front of the voice engine and watches for a TTSSpeakFrame whose text it
already has audio for. When one arrives it pushes what the engine would have pushed —
TTSStartedFrame, the audio in 20ms frames, TTSStoppedFrame — and drops the text frame, so
the engine never sees it and nothing about the call changes downstream: the transport raises
BotStartedSpeaking off the first frame and BotStoppedSpeaking off the last as it always
has, the farewell gate counts the same utterances, and a barge-in clears the same queue.

Keyed on the exact sentence, so a mismatch is not an error but a fallback: if the time of
day rolled over during the ring and "Good morning" became "Good afternoon", the sentence is
simply not in the cache and the engine synthesises it as before.
"""

from typing import Dict

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

SAMPLE_RATE = 16000
# 20ms of 16kHz mono PCM16 — the frame size the transport paces playback by.
FRAME_BYTES = SAMPLE_RATE * 2 // 50


def chunked(pcm: bytes, size: int = FRAME_BYTES):
    """The audio as playback-sized frames. A short tail is a frame too, not dropped."""
    for start in range(0, len(pcm), size):
        yield pcm[start : start + size]


class PrimedSpeech(FrameProcessor):
    """Plays cached audio for a sentence the pipeline asks to speak."""

    def __init__(self, call_sid: str, primed: Dict[str, bytes] | None = None, **kwargs):
        super().__init__(**kwargs)
        self._call_sid = call_sid
        self._primed = {k.strip(): v for k, v in (primed or {}).items() if v}
        self._played = 0
        if self._primed:
            logger.info(
                f"[{call_sid}] Greeting primed | {len(self._primed)} sentence(s), "
                f"{sum(len(v) for v in self._primed.values()) / (SAMPLE_RATE * 2):.1f}s of audio"
            )

    @property
    def played(self) -> int:
        """How many sentences this call spoke from the cache."""
        return self._played

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            isinstance(frame, TTSSpeakFrame)
            and direction == FrameDirection.DOWNSTREAM
            and frame.text.strip() in self._primed
        ):
            pcm = self._primed.pop(frame.text.strip())
            self._played += 1
            await self.push_frame(TTSStartedFrame(), direction)
            for chunk in chunked(pcm):
                await self.push_frame(
                    TTSAudioRawFrame(audio=chunk, sample_rate=SAMPLE_RATE, num_channels=1),
                    direction,
                )
            await self.push_frame(TTSStoppedFrame(), direction)
            return

        await self.push_frame(frame, direction)
