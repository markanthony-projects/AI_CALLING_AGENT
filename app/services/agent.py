import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.frames.frames import EndFrame, InterruptionWorkerFrame, TextFrame, TTSSpeakFrame
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.sarvam.tts import SarvamTTSService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMAssistantAggregator,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies


class GreetingOnlyMinWords(MinWordsUserTurnStartStrategy):
    """A word gate that lifts as soon as the opening line has been delivered.

    A flat gate looked right and was wrong. While the bot is speaking — and Pipecat counts
    that from the first audio frame until the last one has played out, seconds after its
    text is finished — anything shorter than min_words is discarded outright, not deferred.
    So "Yeah sure." answering "Would you like to visit the site?" vanished, and the caller
    sat through 37 seconds of silence saying "Hello" twice before the agent noticed. The
    one-and-two-word replies this dropped are exactly the replies people give.

    The gate only ever existed to stop the "Hello?" on pickup from cutting the greeting off
    at 0.7 seconds. Once that line is out there is nothing left to protect, so it relaxes to
    one word and the prospect can interrupt whenever they like for the rest of the call.
    """

    def relax(self) -> None:
        self._min_words = 1
from app.core.config import settings
from app.services.llm_provider import (
    MAX_THROTTLE_WAIT_SECS,
    build_llm_service,
    is_transient_throttle,
    primary_endpoint,
)
from app.utils.answering_machine import is_answering_machine, machine_phrases
from app.utils.latency import LatencyObserver
from app.utils.vobiz_serializer import VobizSerializer
from app.utils.spoken_text import ToolSyntaxFilter
from app.prompts.agent_prompts import get_system_prompt
import sys
from loguru import logger

# Suppress verbose Pipecat DEBUG logs, keep only INFO and above
logger.remove()
logger.add(sys.stderr, level="INFO")

# Which model serves calls is configuration now, not a constant here: Groq gave six
# weeks' notice on llama-3.3-70b-versatile, and the next such notice should cost an
# env change rather than a release. See LLM_MODEL in app/core/config.py.
GROQ_MODEL = settings.LLM_MODEL

# Groq rejects a malformed tool call server-side ("Failed to call a function"), which ends
# the turn with no speech at all. Left unhandled the caller just hears silence until they
# hang up, so every failed turn must still produce audio.
# Pipecat's default is 300s. A carrier can drop the PSTN leg without closing the websocket,
# and the pipeline then sits there holding one of the concurrency slots. Sixty seconds with
# neither party speaking is dead air on a phone call, not a pause for thought.
IDLE_TIMEOUT_SECS = 60.0

MAX_LLM_TURN_FAILURES = 2

# TTS failure is not recoverable the way an LLM failure is: with no voice there is no
# agent, and the caller pays for every second of the silence. Pipecat already retries the
# socket, so repeated errors mean the service itself is refusing us (expired key, no
# credits, outage). Give up quickly rather than hold a line that can never speak.
MAX_TTS_FAILURES = 3
LLM_RECOVERY_LINE = "Sorry, I missed that. Could you say it once more?"
LLM_SIGNOFF_LINE = "Apologies, I'm having some trouble on this line. I'll call you right back. Thank you!"

# Spoken when the provider throttled us for a second or two. Deliberately not
# LLM_RECOVERY_LINE: the caller said nothing wrong, and asking them to repeat a sentence we
# heard perfectly well blames them for our rate limit.
LLM_BUSY_LINE = "One moment please."

# Spoken before we hang up when the model gives us nothing usable to say.
FAREWELL_LINE = "Thank you so much for your time. Have a wonderful day!"

AGENT_NAME = "Ananya"


def build_opening_line(project_name: str, customer_name: Optional[str] = None) -> str:
    """The first thing the caller hears.

    The name comes off the dial list, so the call opens by confirming we reached the right
    person rather than asking a stranger who they are. Without a name there is nothing to
    confirm, so it asks instead — never a guessed name.
    """
    intro = f"Hi, I am {AGENT_NAME} calling you on behalf of {project_name}."
    name = (customer_name or "").strip()
    if name:
        return f"{intro} Am I speaking with {name}?"
    return f"{intro} May I know your good name?"

# A sign-off is two or three sentences. Anything longer is the model monologuing into a
# hangup, and the caller waits through all of it before the line clears.
MAX_CLOSING_CHARS = 240
_DEVANAGARI = range(0x0900, 0x0980)

# Told to read the booking back and having agreed no time, the model produced "that's Sunday
# at a time to be decided" and hung up. A sign-off that hedges the slot is not a confirmation
# and saying it out loud is worse than a plain goodbye.
_UNCONFIRMED = (
    "to be decided",
    "to be confirmed",
    "to be finalised",
    "to be finalized",
    "yet to be",
    "tbd",
)


def closing_line(spoken: Optional[str]) -> str:
    """Vet the sign-off the model wants spoken as it hangs up.

    The model supplies this so a booked visit can be read back on the way out; a fixed line
    cannot name the day and time, and a prospect who is never told the booking is confirmed
    does not turn up. Everything it sends is still checked, because this is the last thing
    the caller hears and there is no turn left in which to recover from a bad one.
    """
    if not spoken or not spoken.strip():
        return FAREWELL_LINE
    line = " ".join(spoken.split())
    # Sarvam breaks up mid-word on mixed scripts, so Devanagari here would garble the goodbye.
    if len(line) > MAX_CLOSING_CHARS or any(ord(ch) in _DEVANAGARI for ch in line):
        return FAREWELL_LINE
    if any(hedge in line.lower() for hedge in _UNCONFIRMED):
        return FAREWELL_LINE
    return line


# Groq rejects a malformed tool call with this text. end_call is the only tool we expose,
# so a rejected call means the model was trying to hang up — closing properly beats asking
# a prospect who just said goodbye to repeat themselves.
FUNCTION_CALL_FAILURE = "failed to call a function"

# Provider strings meaning "out of budget", not "that request went wrong". Retrying inside
# the call cannot help — Groq's daily-token 429 said "try again in 28m37s" — so the generic
# recovery path made a caller repeat a sentence we had heard perfectly well, then hung up on
# them anyway. Matched case-insensitively against the provider's own message.
LLM_QUOTA_MARKERS = (
    "rate limit reached",
    "rate_limit_exceeded",
    "insufficient_quota",
    "exceeded your current quota",
    "tokens per day",
)


def is_quota_error(message: Optional[str], delay: Optional[float] = None) -> bool:
    """True when the provider is refusing on budget rather than on this particular request.

    A throttle longer than a caller will wait counts too. Cerebras answers an exhausted
    per-minute allowance with "Requests per minute limit exceeded" and Retry-After: 56 —
    no phrase in LLM_QUOTA_MARKERS appears anywhere in it, so on the words alone this would
    fall through to the generic path and ask the caller to repeat a sentence into an account
    that cannot answer for another minute. From their end, a 56-second wait and a dead
    account are the same event.
    """
    if delay is not None and delay > MAX_THROTTLE_WAIT_SECS:
        return True
    text = (message or "").lower()
    return any(marker in text for marker in LLM_QUOTA_MARKERS)


@dataclass(frozen=True)
class CallResult:
    """Outcome of one voice session: the transcript, plus why it ended badly if it did.

    The caller needs the distinction to record CallStatus. A session that produced a
    transcript can still have failed, and a failed session's partial transcript is
    still worth extracting a lead from.
    """

    transcript: str
    error: Optional[str] = None
    latency: Optional[dict] = None
    # An answering machine picked up. Neither a completed call nor a failed one, and the
    # transcript is somebody's outgoing greeting rather than a conversation, so there is
    # nothing in it to extract a lead from.
    answering_machine: bool = False


def session_error(
    pipeline_error: Optional[str],
    llm_failures: int,
    idle_timed_out: bool = False,
    tts_failures: int = 0,
    llm_quota_exhausted: bool = False,
) -> Optional[str]:
    """Decide whether a finished session counts as failed, and why.

    Kept out of the pipeline so the rule behind CallStatus.FAILED can be tested without
    standing up a transport.
    """
    if pipeline_error:
        return pipeline_error
    if llm_quota_exhausted:
        # Distinct from a turn failure: nothing is wrong with the code or the call, the
        # provider account is out of budget. Reads differently in the logs on purpose.
        return "llm quota exhausted: top up the provider account"
    if tts_failures >= MAX_TTS_FAILURES:
        return "tts unavailable: caller heard only silence"
    if llm_failures >= MAX_LLM_TURN_FAILURES:
        return "llm turn failures exhausted"
    if idle_timed_out:
        return "idle timeout: neither party spoke"
    return None


async def run_voice_agent(
    websocket,
    campaign_context: str,
    call_sid: str,
    client_type: str = "vobiz",
    project_name: str = "your project",
    customer_name: Optional[str] = None,
):
    logger.info(
        f"[{call_sid}] Voice agent starting | client={client_type} | project='{project_name}' "
        f"| lead={customer_name or 'unnamed'} | llm={primary_endpoint(settings)}"
    )

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            min_volume=settings.VAD_MIN_VOLUME,
            confidence=settings.VAD_CONFIDENCE,
            stop_secs=settings.VAD_STOP_SECS,
        )
    )

    serializer = VobizSerializer(stream_sid=call_sid)

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            serializer=serializer,
        ),
    )

    # Not GroqLLMService directly: the SDK's default two silent retries honour Retry-After
    # inside the same await, so a 429 reached the logs as `groq=14605ms` with no error and
    # the caller sat through all fourteen seconds of it. See app/services/llm_provider.py.
    llm = build_llm_service(call_sid, settings)
    stt = DeepgramSTTService(
        api_key=settings.DEEPGRAM_API_KEY,
        sample_rate=16000,
        encoding="linear16",
        channels=1,
        settings=DeepgramSTTService.Settings(
            model="nova-2-general",
            language="hi", # 'hi' model natively supports Hinglish and English mixed
            interim_results=True,
            smart_format=True,
            endpointing=300,
        ),
    )
    
    # Low-latency streaming WebSocket Sarvam TTS with pace 1.0
    tts = SarvamTTSService(
        api_key=settings.SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice=settings.SARVAM_VOICE_ID,
            pace=1.0,
            max_chunk_length=150,
            # min_buffer_size is deliberately left at Sarvam's default. Setting it to 25
            # was rejected at connect time with "Input parameters has to be a valid
            # dictionary", killing TTS for the whole call. Pipecat forwards the value
            # straight into the config payload with no range check, so any new value here
            # must be validated against Sarvam's API before it reaches a live call.
        ),
    )

    system_prompt = get_system_prompt(campaign_context, customer_name)
    messages = [{"role": "system", "content": system_prompt}]
    
    task_ref = []
    
    # 1. Dummy function to generate the JSON schema via Pipecat's inspect logic
    async def end_call(params: dict, closing_line: str):
        """
        Speaks your closing line and hangs up. This is irreversible.

        Call this ONLY when the prospect has said goodbye, refused to continue, or the booking is confirmed with a specific day AND time.

        DO NOT call it when they agree to something: "Yes", "sure" and "okay" are commitments to act on, not farewells. Agreed to a visit but no specific day AND time yet? Ask for it instead.

        Args:
            closing_line: Exactly what to speak as you hang up, in Latin script. This is your only goodbye. If a visit or callback was booked, read it back with the day and an exact clock time. If you cannot name the hour, do not call this function at all.
        """
        pass

    # 2. Actual handler that intercepts the tool execution
    async def end_call_handler(params=None, *args, **kwargs):
        spoken = getattr(params, "arguments", None) or {}
        line = closing_line(spoken.get("closing_line") if isinstance(spoken, dict) else None)
        logger.info(f"[{call_sid}] AGENT initiated call end via tool → \"{line}\"")
        if task_ref:
            # A split user turn can run two inferences at once: one asked "What time on
            # Sunday?" while the other hung up, and the caller heard the stale question
            # followed by the goodbye. The interruption discards whatever is still queued.
            # It completes its round-trip before the farewell leaves the push queue, so the
            # farewell itself survives. EndFrame sits behind it, or the line is cut off.
            await task_ref[0].queue_frames(
                [InterruptionWorkerFrame(), TTSSpeakFrame(line), EndFrame()]
            )
            
    llm.register_function("end_call", end_call_handler)

    # 3. Same intent, wrong channel: the model wrote the tool call into its spoken text.
    # ToolSyntaxFilter has already stopped the caller hearing it, so all that is left is to
    # hang up the way the structured call would have.
    async def on_leaked_end_call(line: Optional[str], already_spoke: bool):
        if not task_ref:
            return
        if already_spoke:
            # The words before the markup were the goodbye. Speaking the leaked closing
            # line too would be two farewells in a row.
            logger.info(f"[{call_sid}] Ending call after leaked end_call syntax (goodbye already spoken)")
            await task_ref[0].queue_frames([EndFrame()])
            return
        spoken = closing_line(line)
        logger.info(f"[{call_sid}] Ending call after leaked end_call syntax → \"{spoken}\"")
        await task_ref[0].queue_frames([TTSSpeakFrame(spoken), EndFrame()])

    tool_syntax_filter = ToolSyntaxFilter(call_sid, on_leaked_end_call=on_leaked_end_call)
    
    # Pass the dummy function so Pipecat parses the docstring into a ToolSchema
    context = LLMContext(messages=messages, tools=[end_call])
    
    # Replaces the default start strategies rather than joining them. The defaults are
    # [VADUserTurnStartStrategy, TranscriptionUserTurnStartStrategy] and any one of them
    # firing starts the turn — so leaving VAD in place would keep barging in on the first
    # syllable and this would change nothing. Stop strategies are untouched.
    greeting_gate = GreetingOnlyMinWords(min_words=settings.INTERRUPT_MIN_WORDS)
    # The stop strategy is named rather than left to default for the same reason. Pipecat's
    # default is a Smart Turn ONNX model that predicts semantic completeness from the audio;
    # on PSTN it ruled "Maybe around in 2" a finished turn, so "months." arrived as a second
    # turn and each half cost its own full LLM request. A settle window is blunter but it
    # cannot mispredict, it costs no CPU per turn, and both observed splits were under 0.75s.
    stop_strategy = SpeechTimeoutUserTurnStopStrategy(
        user_speech_timeout=settings.TURN_SETTLE_SECS
    )
    user_agg = LLMUserAggregator(
        context=context,
        params=LLMUserAggregatorParams(
            vad_analyzer=vad_analyzer,
            user_turn_strategies=UserTurnStrategies(
                start=[greeting_gate], stop=[stop_strategy]
            ),
        ),
    )
    assistant_agg = LLMAssistantAggregator(context=context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        # Between the LLM and the TTS on purpose: it is the last point at which leaked tool
        # syntax can be removed before it is spoken, and being upstream of the assistant
        # aggregator keeps the leak out of the context too.
        tool_syntax_filter,
        tts,
        transport.output(),
        assistant_agg,
    ])

    # enable_metrics makes each service report its TTFB; the observer correlates those
    # with the turn boundaries to produce the caller's actual wait.
    latency = LatencyObserver(call_sid)
    task = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # allow_interruptions is not a field on PipelineParams in Pipecat 1.5 and was
            # being dropped silently — interruptions are governed by the turn strategies on
            # the user aggregator below.
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency],
        idle_timeout_secs=IDLE_TIMEOUT_SECS,
        cancel_on_idle_timeout=True,
    )
    task_ref.append(task)
    
    _turn_start_time: float = 0.0
    _user_has_spoken: bool = False
    _startup_task = None
    _llm_failures: int = 0
    _llm_quota_exhausted: bool = False
    _tts_failures: int = 0
    _empty_user_turns: int = 0
    _idle_timed_out: bool = False
    # True between the moment a turn is sent for inference and the moment its answer starts
    # coming back. If the prospect speaks again inside that window, whatever is being
    # generated is an answer to half of what they said.
    _llm_in_flight: bool = False
    # Counted so the machine check only ever runs on the opening turn.
    _turns_heard: int = 0
    _answering_machine: bool = False

    # ─── Pipeline Started ──────────────────────────────────────────────────────
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        async def startup_greeting():
            await asyncio.sleep(0.2)
            if not _user_has_spoken:
                opening_line = build_opening_line(project_name, customer_name)
                context.add_message({"role": "assistant", "content": opening_line})
                logger.info(f"[{call_sid}] AGENT → \"{opening_line}\"")
                await task.queue_frames([TTSSpeakFrame(opening_line)])
            else:
                # They spoke first, so the greeting was cancelled and there is nothing left
                # to protect. Lifting it here matters: otherwise the gate stays armed for a
                # call that never had an opening line to guard.
                greeting_gate.relax()

        nonlocal _startup_task
        _startup_task = asyncio.create_task(startup_greeting())

    # ─── Client Disconnected ───────────────────────────────────────────────────
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[{call_sid}] Client disconnected — ending pipeline")
        await task.queue_frames([EndFrame()])

    # ─── Idle ──────────────────────────────────────────────────────────────────
    # Fires when the carrier dropped the phone leg but left the websocket open, which
    # otherwise pins a concurrency slot until Pipecat's 300s default expires.
    @task.event_handler("on_idle_timeout")
    async def on_idle_timeout(worker):
        nonlocal _idle_timed_out
        _idle_timed_out = True
        logger.warning(f"[{call_sid}] No speech either way for {IDLE_TIMEOUT_SECS:.0f}s; abandoning call")

    # ─── User Starts Speaking ──────────────────────────────────────────────────
    @user_agg.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        # Deliberately does not set _user_has_spoken: this fires on VAD, and five seconds of
        # connection noise once cancelled the opening line, leaving the caller in silence
        # until they said "Hello?" themselves. Only a real transcript counts as speech.
        nonlocal _turn_start_time, _llm_in_flight
        _turn_start_time = time.time()

        # The prospect has started talking again while we are still generating a reply to
        # what they said before. That reply answers a sentence they were not finished
        # saying, and paying for it twice is the smaller half of the problem: a caller once
        # heard a stale "What time on Sunday?" followed immediately by the goodbye, because
        # both inferences finished. Abandon it — the next one sees the whole utterance.
        #
        # Only fires before the answer starts arriving. Once it does, a new user turn is an
        # ordinary barge-in and Pipecat raises its own interruption for it.
        if _llm_in_flight:
            _llm_in_flight = False
            logger.info(f"[{call_sid}] Prospect resumed mid-inference; abandoning the stale turn")
            await task.queue_frames([InterruptionWorkerFrame()])

        if client_type == "exotel":
            try:
                await websocket.send_json({"event": "clear_client_buffer"})
            except Exception:
                pass

    # ─── User Stops Speaking ───────────────────────────────────────────────────
    @user_agg.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        nonlocal _empty_user_turns, _user_has_spoken, _turns_heard, _answering_machine
        transcript = (message.content or "").strip() if message and hasattr(message, "content") else ""
        total_turn_time = f"{(time.time() - _turn_start_time) * 1000:.0f}ms" if _turn_start_time else "?"
        if transcript:
            _user_has_spoken = True
            logger.info(f"[{call_sid}] USER  → \"{transcript}\" (Total Turn Duration: {total_turn_time})")

            # Only ever the opening turn. A recorded greeting is the first thing a machine
            # says; the same words later in a real conversation are a person talking about
            # their availability, and hanging up on them would be much worse than
            # transcribing one voicemail.
            if not _answering_machine and _turns_heard == 0 and is_answering_machine(transcript):
                _answering_machine = True
                logger.info(
                    f"[{call_sid}] Answering machine detected; hanging up without leaving a "
                    f"message. Matched: {machine_phrases(transcript)}"
                )
                await task.queue_frames([EndFrame()])
            _turns_heard += 1
            return
        # VAD heard speech but the STT produced nothing, so this turn was line noise that
        # cut the agent off mid-sentence. Without this line a false barge-in leaves no
        # trace at all, which is what made the interruptions look inexplicable.
        _empty_user_turns += 1
        logger.warning(
            f"[{call_sid}] VAD fired with no transcribable speech after {total_turn_time} "
            f"— likely a false barge-in (count: {_empty_user_turns})"
        )

    # The two edges of "a reply is being generated". Together they bound the window in
    # which a new user turn makes the in-flight inference worthless.
    @user_agg.event_handler("on_user_turn_inference_triggered")
    async def on_user_turn_inference_triggered(aggregator, *_):
        nonlocal _llm_in_flight
        _llm_in_flight = True

    @assistant_agg.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator, *_):
        nonlocal _llm_in_flight
        _llm_in_flight = False

    # ─── Agent Generation Logging ──────────────────────────────────────────────
    # GroqLLMService is HTTP-based: it has no on_client_connected. on_completion_timeout
    # is the signal that actually matters — a stalled turn is dead air on a live call.
    @llm.event_handler("on_completion_timeout")
    async def on_completion_timeout(service):
        logger.warning(f"[{call_sid}] LLM   → completion timed out; caller is hearing silence")

    # A non-fatal LLM error aborts the turn without emitting any speech, so recovery has to
    # be driven from here or the line goes dead. Escalate to a sign-off rather than looping
    # an apology at a caller whose every turn is failing.
    @llm.event_handler("on_error")
    async def on_llm_error(service, error):
        nonlocal _llm_failures
        if error.fatal:
            return

        # A rejected tool call is the model trying to hang up, not a lost turn.
        if FUNCTION_CALL_FAILURE in (error.error or "").lower():
            logger.info(f"[{call_sid}] Tool call rejected upstream; closing the call as intended")
            await task.queue_frames([TTSSpeakFrame(FAREWELL_LINE), EndFrame()])
            return

        # How long the provider asked us to wait, taken from its Retry-After header where it
        # sent one. This is the only signal that separates a hiccup from a dead account, and
        # providers disagree on where to put it: Groq writes it into English prose in the
        # body, Cerebras sends only the header and a message with no number in it at all.
        # Reading the header here means neither has to be special-cased below.
        throttled_for = service.last_throttle_delay

        # A per-minute throttle, not an exhausted account. These reach us only now that the
        # SDK no longer swallows them, and is_quota_error() cannot tell the two apart —
        # Groq words them identically apart from the number of seconds. Falling through to
        # the quota branch would hang up on a caller over a three-second wait, which is a
        # worse bug than the silence this change removes.
        if is_transient_throttle(error.error, throttled_for):
            _llm_failures += 1
            logger.warning(
                f"[{call_sid}] LLM throttled ({_llm_failures}/{MAX_LLM_TURN_FAILURES}); "
                f"provider asked for {throttled_for or 0.0:.1f}s. Configure LLM_FALLBACK_API_KEY "
                f"to ride these out instead of losing the turn."
            )
            if _llm_failures >= MAX_LLM_TURN_FAILURES:
                await task.queue_frames([TTSSpeakFrame(LLM_SIGNOFF_LINE), EndFrame()])
            else:
                await task.queue_frames([TTSSpeakFrame(LLM_BUSY_LINE)])
            return

        # Out of budget, or throttled for longer than the caller will stay on the line —
        # which amounts to the same thing from their end. LLM_RECOVERY_LINE asks them to
        # repeat themselves, which blames them for our billing and buys nothing, because
        # the next turn fails identically. Close cleanly instead.
        if is_quota_error(error.error, throttled_for):
            nonlocal _llm_quota_exhausted
            _llm_quota_exhausted = True
            logger.error(
                f"[{call_sid}] LLM quota exhausted; signing off without asking the caller "
                f"to repeat. Provider said: {error.error}"
            )
            await task.queue_frames([TTSSpeakFrame(LLM_SIGNOFF_LINE), EndFrame()])
            return

        _llm_failures += 1
        logger.warning(
            f"[{call_sid}] LLM turn failed ({_llm_failures}/{MAX_LLM_TURN_FAILURES}): {error.error}"
        )
        if _llm_failures >= MAX_LLM_TURN_FAILURES:
            logger.error(f"[{call_sid}] LLM unrecoverable; signing off to avoid dead air")
            await task.queue_frames([TTSSpeakFrame(LLM_SIGNOFF_LINE), EndFrame()])
        else:
            await task.queue_frames([TTSSpeakFrame(LLM_RECOVERY_LINE)])

    # ─── TTS Failure ───────────────────────────────────────────────────────────
    # Sarvam reports "No credits available" per synthesis attempt, so this fires many
    # times per second. Log the cause once, then end the call — there is no way to say
    # anything to the caller, including a goodbye.
    @tts.event_handler("on_error")
    async def on_tts_error(service, error):
        nonlocal _tts_failures
        if error.fatal:
            return
        _tts_failures += 1
        if _tts_failures == 1:
            logger.error(f"[{call_sid}] TTS failing — caller is hearing silence: {error.error}")
        if _tts_failures == MAX_TTS_FAILURES:
            logger.error(f"[{call_sid}] TTS unavailable after {MAX_TTS_FAILURES} errors; abandoning call")
            await task.queue_frames([EndFrame()])

    # Log the assistant's finalized turn. Note the event is on_assistant_turn_stopped —
    # on_assistant_message_added is not an event this aggregator registers, which is why
    # every AGENT line was missing from the logs.
    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        content = (getattr(message, "content", "") or "").strip()
        if content:
            suffix = " [interrupted]" if getattr(message, "interrupted", False) else ""
            logger.info(f"[{call_sid}] AGENT → \"{content}\"{suffix}")
        # The greeting is the agent's first turn, so by the time any assistant turn has
        # finished the line it guards is already spoken. From here a single word starts the
        # prospect's turn, and a short answer can never be swallowed again.
        greeting_gate.relax()

    runner = PipelineRunner()
    error: Optional[str] = None
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"[{call_sid}] Pipeline exception: {e}")
        error = f"pipeline: {e}"

    # Deliberately not in a finally block: returning from finally discards any in-flight
    # exception, which is how a crashed session used to be recorded as a clean one.
    logger.info(f"[{call_sid}] Pipeline finished. Extracting transcript...")
    transcript_str = ""
    try:
        prev_content = ""
        for msg in context.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Deduplicate consecutive identical messages (fixes Pipecat context double-append on startup greeting)
            if role in ("user", "assistant") and content and content != prev_content:
                speaker = "Prospect" if role == "user" else "Agent"
                transcript_str += f"{speaker}: {content}\n"
                prev_content = content
    except Exception as e:
        logger.error(f"[{call_sid}] Failed to compile transcript: {e}")

    return CallResult(
        transcript=transcript_str.strip(),
        error=session_error(
            error, _llm_failures, _idle_timed_out, _tts_failures, _llm_quota_exhausted
        ),
        latency=latency.log_summary(),
        answering_machine=_answering_machine,
    )
