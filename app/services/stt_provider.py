"""Which ears the agent listens with, chosen from configuration.

The LLM has had this since the provider moved from Groq to Cerebras: an endpoint described
by settings, resolved in one place, swapped without touching the pipeline. Speech-to-text
never got it. The model name and language were written into run_voice_agent as literals,
so trying a different one meant editing the agent and redeploying.

That was tolerable while there was nothing to try. It is not tolerable now, because the
next thing on the list is a three-way comparison — the current Deepgram model against
Sarvam and against Deepgram's newer ones — and running that by editing and redeploying
between each is exactly how an experiment ends up measuring the deploy rather than the
model.

Nothing about the default changes. STT_PROVIDER, STT_MODEL and STT_LANGUAGE carry the same
values that were hard-coded, so an untouched deployment builds the identical service it
built before. See tests/test_stt_provider.py, which pins that.

Deepgram and Sarvam do not take the same settings, and this does not pretend otherwise —
each provider gets its own construction. What is shared is the decision, so there is one
place to look when a call transcribes badly and somebody asks what it was listening with.
"""

from dataclasses import dataclass

from loguru import logger
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.stt_service import STTService

# The audio the transport hands over. Not configurable: the serializer, the VAD and the
# turn analyzer all assume it, and a provider that cannot take it needs its own resampling
# rather than a quiet change here.
SAMPLE_RATE = 16000

DEEPGRAM = "deepgram"
SARVAM = "sarvam"
PROVIDERS = (DEEPGRAM, SARVAM)


@dataclass(frozen=True)
class SttEndpoint:
    """One speech-to-text provider the agent can listen with."""

    provider: str
    model: str
    language: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


def stt_endpoint(settings) -> SttEndpoint:
    """What the configuration says to listen with."""
    provider = (settings.STT_PROVIDER or "").strip().lower()
    if provider not in PROVIDERS:
        # Refused rather than defaulted. A typo that silently falls back to Deepgram would
        # make a comparison run report the wrong winner, which is worse than not running.
        raise ValueError(
            f"STT_PROVIDER must be one of {', '.join(PROVIDERS)}; got {settings.STT_PROVIDER!r}"
        )
    return SttEndpoint(
        provider=provider,
        model=settings.STT_MODEL.strip(),
        language=settings.STT_LANGUAGE.strip(),
    )


def _deepgram(endpoint: SttEndpoint, settings) -> STTService:
    return DeepgramSTTService(
        api_key=settings.DEEPGRAM_API_KEY,
        sample_rate=SAMPLE_RATE,
        encoding="linear16",
        channels=1,
        settings=DeepgramSTTService.Settings(
            model=endpoint.model,
            language=endpoint.language,
            interim_results=True,
            smart_format=True,
            # Deepgram's own silence window before it finalises a transcript. Left where it
            # was: it sits under TURN_SETTLE_SECS, and moving both at once would make the
            # turn timing impossible to attribute.
            endpointing=settings.STT_ENDPOINTING_MS,
        ),
    )


def _sarvam(endpoint: SttEndpoint, settings) -> STTService:
    return SarvamSTTService(
        api_key=settings.SARVAM_API_KEY,
        sample_rate=SAMPLE_RATE,
        settings=SarvamSTTService.Settings(
            model=endpoint.model,
            language=endpoint.language,
        ),
    )


_BUILDERS = {DEEPGRAM: _deepgram, SARVAM: _sarvam}


def build_stt_service(call_sid: str, settings) -> STTService:
    """Assemble the call's speech-to-text service from configuration alone."""
    endpoint = stt_endpoint(settings)
    logger.info(f"[{call_sid}] Listening with {endpoint}")
    return _BUILDERS[endpoint.provider](endpoint, settings)
