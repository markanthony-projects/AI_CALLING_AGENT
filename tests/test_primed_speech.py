"""Audio synthesised before the call is what the caller hears, and the engine never sees it.

On 14 Sep the greeting waited 222ms of synthesis after a 340ms startup. The worker now
synthesises it while the phone rings; this processor plays it. What matters here: the
frames it pushes are the frames the voice engine would have pushed, a sentence it does not
have passes through untouched, and it sits directly in front of the engine.
"""

import asyncio

import pipecat.processors.frame_processor as fp
from pipecat.frames.frames import (
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from app.utils.primed_speech import FRAME_BYTES, PrimedSpeech, chunked


class _Captured:
    def __init__(self):
        self.frames = []

    async def push(self, frame, direction):
        self.frames.append(frame)


async def _run(processor, frames, direction=FrameDirection.DOWNSTREAM):
    captured = _Captured()
    processor.push_frame = captured.push

    async def noop(*a, **kw):
        pass

    original = fp.FrameProcessor.process_frame
    fp.FrameProcessor.process_frame = noop
    try:
        for frame in frames:
            await processor.process_frame(frame, direction)
    finally:
        fp.FrameProcessor.process_frame = original
    return captured.frames


def test_a_primed_sentence_is_played_as_the_engine_would_have_played_it():
    pcm = bytes(range(256)) * 15  # 3840 bytes: four full 20ms frames at 24kHz
    processor = PrimedSpeech("c1", {"Hello, Good morning.": pcm})

    out = asyncio.run(_run(processor, [TTSSpeakFrame("Hello, Good morning.")]))

    assert isinstance(out[0], TTSStartedFrame)
    assert isinstance(out[-1], TTSStoppedFrame)
    audio = [f for f in out if isinstance(f, TTSAudioRawFrame)]
    assert len(audio) == 4
    assert all(f.sample_rate == 24000 and f.num_channels == 1 for f in audio)
    assert b"".join(f.audio for f in audio) == pcm
    assert not any(isinstance(f, TTSSpeakFrame) for f in out), "the engine must not see it"
    assert processor.played == 1


def test_a_sentence_it_does_not_have_goes_on_to_the_engine():
    processor = PrimedSpeech("c1", {"Hello, Good morning.": b"\x00" * 640})
    frame = TTSSpeakFrame("Hello, Good afternoon.")

    out = asyncio.run(_run(processor, [frame]))

    assert out == [frame]
    assert processor.played == 0


def test_whitespace_around_the_sentence_does_not_cause_a_miss():
    processor = PrimedSpeech("c1", {"Hello. ": b"\x00" * 640})
    out = asyncio.run(_run(processor, [TTSSpeakFrame(" Hello.")]))
    assert isinstance(out[0], TTSStartedFrame)


def test_each_sentence_is_played_once():
    """The cache is for the greeting. If the same words are said again later in the call
    they are synthesised — the audio was popped, not copied."""
    processor = PrimedSpeech("c1", {"Hello.": b"\x00" * 640})
    first = asyncio.run(_run(processor, [TTSSpeakFrame("Hello.")]))
    second = asyncio.run(_run(processor, [TTSSpeakFrame("Hello.")]))
    assert isinstance(first[0], TTSStartedFrame)
    assert isinstance(second[0], TTSSpeakFrame)


def test_nothing_primed_is_a_plain_pass_through():
    processor = PrimedSpeech("c1", None)
    frame = TTSSpeakFrame("Hello.")
    assert asyncio.run(_run(processor, [frame])) == [frame]


def test_upstream_frames_are_never_intercepted():
    processor = PrimedSpeech("c1", {"Hello.": b"\x00" * 640})
    frame = TTSSpeakFrame("Hello.")
    assert asyncio.run(_run(processor, [frame], FrameDirection.UPSTREAM)) == [frame]


def test_the_frames_are_twenty_milliseconds_and_the_tail_is_kept():
    assert FRAME_BYTES == 960, "20ms of 24kHz mono PCM16"
    pieces = list(chunked(b"x" * 2200))
    assert [len(p) for p in pieces] == [960, 960, 280]


def test_the_frames_carry_the_rate_the_engine_labels_its_own_with():
    """The transport resamples every TTSAudioRawFrame by the rate on the frame. Cached
    audio labelled with any other rate plays at the wrong speed — call 56398497."""
    from app.services.voice import build_tts
    from app.utils.primed_speech import SAMPLE_RATE
    from types import SimpleNamespace

    tts = build_tts(SimpleNamespace(SARVAM_API_KEY="k", SARVAM_VOICE_ID="simran", SPEAKING_PACE=1.0, SARVAM_TEMPERATURE=None, TTS_SPARE_SOCKET=False))
    assert SAMPLE_RATE == tts._init_sample_rate


def test_it_sits_directly_in_front_of_the_voice_engine():
    """Anything between it and the engine would see audio frames it was written to see
    text for; anything upstream of the cutters would see a sentence the cutters have not
    shaped yet."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    start = src.index("Pipeline(")
    order = src[start:]
    assert order.index("empty_reply,") < order.index("primed,") < order.index("tts,")
