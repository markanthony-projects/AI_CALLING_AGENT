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
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop import (
    ExternalUserTurnStopStrategy,
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_stop.base_user_turn_stop_strategy import BaseUserTurnStopStrategy
from pipecat.turns.user_start.base_user_turn_start_strategy import BaseUserTurnStartStrategy

from app.services.turns import (
    GreetingOnlyMinWords,
    ServiceDecidesButNotForever,
    SustainedSpeechBargeIn,
)

# The audio the transport hands over. Not configurable: the serializer, the VAD and the
# turn analyzer all assume it, and a provider that cannot take it needs its own resampling
# rather than a quiet change here.
SAMPLE_RATE = 16000

DEEPGRAM = "deepgram"
SARVAM = "sarvam"
FLUX = "flux"
PROVIDERS = (DEEPGRAM, SARVAM, FLUX)


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


# Deepgram takes "hi"; Sarvam wants "hi-IN" and warns on anything it does not recognise,
# then sends the unrecognised code anyway. One STT_LANGUAGE setting feeds both providers, so
# the dialect belongs here rather than in the environment file — otherwise switching provider
# silently changes what language the call is transcribed in.
_SARVAM_LANGUAGE = {
    "hi": "hi-IN",
    "en": "en-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "mr": "mr-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "pa": "pa-IN",
}


def languages(endpoint: SttEndpoint) -> list:
    """STT_LANGUAGE as a list. One setting, because one call has one language policy.

    Most services take a single code and read the first. Flux's multilingual model takes
    several, which is what a Hinglish call needs: "hi,en-IN".
    """
    return [part.strip() for part in (endpoint.language or "").split(",") if part.strip()]


def _timer_turns(settings) -> BaseUserTurnStopStrategy:
    """The turn is over when nobody has spoken for a while. A guess, and a measured 600ms
    of it on every turn — VAD stop_secs plus this window — but the only thing available
    when the service does no more than transcribe."""
    return SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=settings.TURN_SETTLE_SECS)


def _service_turns(settings) -> BaseUserTurnStopStrategy:
    """The turn is over when the speech service says so. It pushes the frames itself, so
    there is nothing here to wait on and no window to pay.

    One thing to read the first live call for. This strategy only fires on
    UserStoppedSpeakingFrame if a transcript has already reached it; otherwise it falls
    through to a 0.5s aggregation timeout, which would cost more than the 600ms window it
    was brought in to remove. Flux pushes them in the right order — TranscriptionFrame, then
    UserStoppedSpeakingFrame, both in _handle_end_of_turn — but the transcript is pushed
    downstream and the stop frame is broadcast, and those are not the same path. If the
    first Flux call shows turns landing ~500ms late, this is the reason, and the fix is the
    strategy's `timeout`, not the settle window.
    """
    return ServiceDecidesButNotForever(max_open_secs=settings.STT_MAX_TURN_SECS)


def _word_gate(settings) -> BaseUserTurnStartStrategy:
    """Count the words before stopping the agent. Needs a service that transcribes as it
    goes, which in practice means Deepgram's interim results."""
    return GreetingOnlyMinWords(min_words=settings.INTERRUPT_MIN_WORDS)


def _duration_gate(settings) -> BaseUserTurnStartStrategy:
    """Time the speech instead of reading it, for a service with nothing to read yet.

    `enable_user_speaking_frames=False` because Flux broadcasts UserStartedSpeakingFrame
    itself from StartOfTurn; letting the aggregator broadcast a second one puts two starts
    on the wire for one utterance. Sarvam does not, and gets the default.
    """
    return SustainedSpeechBargeIn(min_speech_secs=settings.BARGE_IN_MIN_SPEECH_SECS)


def _duration_gate_quietly(settings) -> BaseUserTurnStartStrategy:
    return SustainedSpeechBargeIn(
        min_speech_secs=settings.BARGE_IN_MIN_SPEECH_SECS,
        enable_user_speaking_frames=False,
    )


def sarvam_language(code: str) -> str:
    """Sarvam's spelling of a language code, or the code unchanged if it already is one."""
    lowered = (code or "").strip().lower()
    return _SARVAM_LANGUAGE.get(lowered, code.strip())


def _sarvam(endpoint: SttEndpoint, settings) -> STTService:
    return SarvamSTTService(
        api_key=settings.SARVAM_API_KEY,
        sample_rate=SAMPLE_RATE,
        settings=SarvamSTTService.Settings(
            model=endpoint.model,
            language=sarvam_language(endpoint.language),
        ),
    )


class ConnectsWhileTheGreetingPlays(DeepgramFluxSTTService):
    """Opens its websocket in the background, so the greeting does not queue behind it.

    Measured on five calls, every one the same shape:

        STARTUP 1244ms to the greeting | services built=+151ms  stt=+907ms
                                         tts=+172ms  pipeline=+12ms  greeting queued=+2ms

    The greeting is a local f-string with no network in it, and it waited 1.2 seconds. The
    speech-to-text handshake is 800ms of that, and the greeting never uses speech-to-text.

    It waits because StartFrame walks the pipeline one processor at a time, and
    DeepgramFluxSTTBase.start does `await super().start(frame); await self._connect()`. This
    service sits before the voice engine, so the voice engine cannot even begin its own
    handshake — 172ms — until this one has finished. on_pipeline_started, which is what
    queues the greeting, fires after all of them.

    So the first connect is started and not awaited. Nothing is lost by that: audio cannot
    arrive before the prospect speaks, which is after the greeting, and both run_stt and
    _transport_send_audio already return quietly when the socket is not open yet. There is
    about two seconds of slack against a handshake that has never taken one.

    Only the FIRST connect. A reconnect mid-call is awaited exactly as before — by then the
    caller is handling a live failure and "connected" has to mean connected.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._opened_once = False
        self._opening = None

    async def _connect(self):
        # _task_manager, not the task_manager property: the property raises when the
        # service has not been set up, and a raise here would be a call with no
        # transcription at all rather than a slow greeting.
        if self._opened_once or getattr(self, "_task_manager", None) is None:
            await super()._connect()
            return
        self._opened_once = True
        logger.debug(f"{self}: opening in the background so the greeting need not wait")
        self._opening = self.create_task(self._open())

    async def _open(self):
        await super()._connect()

    async def cleanup(self):
        if self._opening:
            await self.cancel_task(self._opening)
            self._opening = None
        await super().cleanup()


def _flux(endpoint: SttEndpoint, settings) -> STTService:
    """Deepgram Flux: transcription and end-of-turn from the same socket.

    `should_interrupt=False` on purpose, and it is the whole reason this can be adopted one
    piece at a time. Flux will happily interrupt the agent the moment it hears speech, which
    would take barge-in away from GreetingOnlyMinWords — and that is where "Hello?" was
    taught not to cut the agent off mid-sentence, earned on call 5023ff25. So Flux is bought
    for the end of a turn and nothing else; who is allowed to interrupt stays where it is.
    """
    return ConnectsWhileTheGreetingPlays(
        api_key=settings.DEEPGRAM_API_KEY,
        sample_rate=SAMPLE_RATE,
        should_interrupt=False,
        settings=DeepgramFluxSTTService.Settings(
            model=endpoint.model,
            language_hints=_flux_hints(endpoint),
            # Unset means unset: the service builds its connection URL with `is not None`,
            # so None and the SDK's own NOT_GIVEN sentinel both leave the parameter off it
            # and Deepgram applies its defaults (0.7 and 5000ms). Checked, because the
            # obvious alternative — omitting the keys when unset — is a branch that buys
            # nothing. tests/test_stt_provider.py reads the query string it connects with.
            eot_threshold=settings.STT_EOT_THRESHOLD,
            eot_timeout_ms=settings.STT_EOT_TIMEOUT_MS,
        ),
    )


def _flux_hints(endpoint: SttEndpoint) -> list:
    """STT_LANGUAGE read as the list Flux's multilingual model wants.

    "hi,en-IN" is how a Hinglish call is described to it — the language people actually
    speak on these calls is neither of them on its own. Unknown codes are dropped rather
    than sent: Flux ignores hints it does not know, and a typo that silently changes nothing
    is worse than one that shows up as a missing hint here.
    """
    hints = []
    for code in languages(endpoint):
        try:
            hints.append(Language(code))
        except ValueError:
            logger.warning(f"Ignoring STT_LANGUAGE entry {code!r}: not a language Flux knows")
    return hints


# Each provider knows how to build itself and what its arrival means for turn-taking. Adding
# one is a function and a line here; nothing else in the codebase learns its name.
_BUILDERS = {DEEPGRAM: _deepgram, SARVAM: _sarvam, FLUX: _flux}

# Who decides a user turn is over. The timer is the default because most services only
# transcribe and somebody has to guess; Flux is the exception that knows, and says so.
_TURN_OWNERS = {DEEPGRAM: _timer_turns, SARVAM: _timer_turns, FLUX: _service_turns}

# What has to happen before the agent stops talking. It follows from the same fact as the
# turn owner and a different one: whether this service produces any words BEFORE the turn
# ends. Deepgram does, via interim results. Flux and Sarvam both push nothing until the
# utterance is over, so a word gate would let the agent talk over all of it.
_BARGE_IN = {DEEPGRAM: _word_gate, SARVAM: _duration_gate, FLUX: _duration_gate_quietly}


@dataclass(frozen=True)
class Listening:
    """What the agent listens with, and what that choice implies for turn-taking.

    These two travel together because they are one decision. A service that only
    transcribes leaves the pipeline to guess when a turn ended, with a timer — 600ms per
    turn on this deployment, measured, and 47% of everything the caller waits through. A
    service that decides end-of-turn itself removes that guess, and removing it is the whole
    reason for choosing such a service.

    Returned as one object so no caller has to ask which provider is configured in order to
    wire the pipeline correctly. That question has exactly one right answer per provider,
    and it is answered here.
    """

    endpoint: SttEndpoint
    service: STTService
    start_strategy: BaseUserTurnStartStrategy
    stop_strategy: BaseUserTurnStopStrategy

    def __str__(self) -> str:
        return (
            f"{self.endpoint} (barge-in: {type(self.start_strategy).__name__}, "
            f"turns: {type(self.stop_strategy).__name__})"
        )


def build_listening(call_sid: str, settings) -> Listening:
    """Assemble the call's ears, and the end-of-turn decision that comes with them."""
    endpoint = stt_endpoint(settings)
    listening = Listening(
        endpoint=endpoint,
        service=_BUILDERS[endpoint.provider](endpoint, settings),
        start_strategy=_BARGE_IN[endpoint.provider](settings),
        stop_strategy=_TURN_OWNERS[endpoint.provider](settings),
    )
    logger.info(f"[{call_sid}] Listening with {listening}")
    return listening
