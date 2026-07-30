import asyncio
import time
from dataclasses import dataclass
from typing import Optional
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.frames.frames import EndFrame, InterruptionWorkerFrame, TextFrame, TTSSpeakFrame
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.services.groq.llm import GroqLLMService
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
from app.core.config import settings
from app.utils.latency import LatencyObserver
from app.utils.vobiz_serializer import VobizSerializer
from app.prompts.agent_prompts import get_system_prompt
import sys
from loguru import logger

# Suppress verbose Pipecat DEBUG logs, keep only INFO and above
logger.remove()
logger.add(sys.stderr, level="INFO")

# llama3-70b-8192 was decommissioned by Groq on 2026-07-15.
# llama-3.3-70b-versatile is the official recommended replacement.
GROQ_MODEL = "llama-3.3-70b-versatile"

# Groq rejects a malformed tool call server-side ("Failed to call a function"), which ends
# the turn with no speech at all. Left unhandled the caller just hears silence until they
# hang up, so every failed turn must still produce audio.
# Pipecat's default is 300s. A carrier can drop the PSTN leg without closing the websocket,
# and the pipeline then sits there holding one of MAX_CALLS slots. Sixty seconds with
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

# Spoken before we hang up when the model gives us nothing usable to say.
FAREWELL_LINE = "Thank you so much for your time. Have a wonderful day!"

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


def session_error(
    pipeline_error: Optional[str],
    llm_failures: int,
    idle_timed_out: bool = False,
    tts_failures: int = 0,
) -> Optional[str]:
    """Decide whether a finished session counts as failed, and why.

    Kept out of the pipeline so the rule behind CallStatus.FAILED can be tested without
    standing up a transport.
    """
    if pipeline_error:
        return pipeline_error
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
):
    logger.info(f"[{call_sid}] Voice agent starting | client={client_type} | project='{project_name}' | model={GROQ_MODEL}")

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

    llm = GroqLLMService(api_key=settings.GROQ_API_KEY, settings=GroqLLMService.Settings(model=GROQ_MODEL))
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

    system_prompt = get_system_prompt(campaign_context)
    messages = [{"role": "system", "content": system_prompt}]
    
    task_ref = []
    
    # 1. Dummy function to generate the JSON schema via Pipecat's inspect logic
    async def end_call(params: dict, closing_line: str):
        """
        Speaks your closing line and then hangs up. This is irreversible and the prospect cannot be reached again on this call.

        Call this ONLY when the prospect has said goodbye, refused to continue, or the booking is fully confirmed with a specific day and time.

        DO NOT call this when the prospect agrees to something. "Yes", "sure", "okay" and "sounds good" are commitments to act on, not farewells. If they have agreed to a site visit or callback but you do not yet have a specific day AND time, ask for it instead of calling this function.

        Args:
            closing_line: The exact words to speak as you hang up, in English/Latin script. This is your only goodbye, so do not also say one in a normal reply. If a site visit or callback was booked, read it back here with the day and an exact clock time, e.g. "Perfect Rahul, that's Sunday at 3 PM at Lakeview Residency. I'll send you the details. Thank you!". If you cannot name the hour, the booking is NOT finished — do not call this function at all, ask what time instead. Never write a placeholder like "at a time to be decided". If nothing was booked, a warm thank-you is enough.
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
    
    # Pass the dummy function so Pipecat parses the docstring into a ToolSchema
    context = LLMContext(messages=messages, tools=[end_call])
    
    user_agg = LLMUserAggregator(context=context, params=LLMUserAggregatorParams(vad_analyzer=vad_analyzer))
    assistant_agg = LLMAssistantAggregator(context=context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
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
            allow_interruptions=True,
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
    _tts_failures: int = 0
    _empty_user_turns: int = 0
    _idle_timed_out: bool = False

    # ─── Pipeline Started ──────────────────────────────────────────────────────
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        async def startup_greeting():
            await asyncio.sleep(0.2)
            if not _user_has_spoken:
                opening_line = f"Hi there! This is Priya calling from {project_name}. How are you doing today?"
                context.add_message({"role": "assistant", "content": opening_line})
                logger.info(f"[{call_sid}] AGENT → \"{opening_line}\"")
                await task.queue_frames([TTSSpeakFrame(opening_line)])
                
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
        nonlocal _turn_start_time
        _turn_start_time = time.time()
        if client_type == "exotel":
            try:
                await websocket.send_json({"event": "clear_client_buffer"})
            except Exception:
                pass

    # ─── User Stops Speaking ───────────────────────────────────────────────────
    @user_agg.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        nonlocal _empty_user_turns, _user_has_spoken
        transcript = (message.content or "").strip() if message and hasattr(message, "content") else ""
        total_turn_time = f"{(time.time() - _turn_start_time) * 1000:.0f}ms" if _turn_start_time else "?"
        if transcript:
            _user_has_spoken = True
            logger.info(f"[{call_sid}] USER  → \"{transcript}\" (Total Turn Duration: {total_turn_time})")
            return
        # VAD heard speech but the STT produced nothing, so this turn was line noise that
        # cut the agent off mid-sentence. Without this line a false barge-in leaves no
        # trace at all, which is what made the interruptions look inexplicable.
        _empty_user_turns += 1
        logger.warning(
            f"[{call_sid}] VAD fired with no transcribable speech after {total_turn_time} "
            f"— likely a false barge-in (count: {_empty_user_turns})"
        )

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
        error=session_error(error, _llm_failures, _idle_timed_out, _tts_failures),
        latency=latency.log_summary(),
    )
