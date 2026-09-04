import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.frames.frames import (
    EndFrame,
    EndWorkerFrame,
    InterruptionWorkerFrame,
    TTSSpeakFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
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
from app.services.stt_provider import build_stt_service
from app.services.llm_provider import (
    MAX_THROTTLE_WAIT_SECS,
    build_llm_service,
    is_transient_throttle,
    primary_endpoint,
)
from app.utils.answering_machine import OPENING_TURNS, machine_in_opening, machine_phrases
from app.utils.asked import REPEAT_LIMIT, AskedSoFar
from app.utils.closing_gate import ClosingGate
from app.utils.latency import LatencyObserver
from app.utils.pace import adjusted_pace, pace_request
from app.utils.person_name import spoken_name
from app.utils.vobiz_serializer import VobizSerializer
from app.utils.farewell import FarewellGate, farewell_timeout
from app.utils.reprompt import MAX_DEAD_AIR_NUDGES, dead_air_nudge
from app.utils.socket_witness import SocketWitness
from app.utils.spoken_text import ToolSyntaxFilter
from app.utils.timeutils import time_of_day_greeting
from app.utils.turn_analyzer import build_turn_analyzer
from app.utils.stt_witness import SttWitness
from app.utils.turn_gate import TurnFinalityGate
from app.prompts.agent_prompts import AGENT_NAME, get_system_prompt
import sys
from loguru import logger

# Suppress verbose Pipecat DEBUG logs, keep only INFO and above
logger.remove()
logger.add(sys.stderr, level="INFO")

# Groq rejects a malformed tool call server-side ("Failed to call a function"), which ends
# the turn with no speech at all. Left unhandled the caller just hears silence until they
# hang up, so every failed turn must still produce audio.
# Pipecat's default is 300s. A carrier can drop the PSTN leg without closing the websocket,
# and the pipeline then sits there holding one of the concurrency slots. Sixty seconds with
# neither party speaking is dead air on a phone call, not a pause for thought.
IDLE_TIMEOUT_SECS = 60.0

# The idle timeout only fires when NEITHER party has spoken, so it cannot end a call whose
# voice has died while the caller keeps asking "hello?" — every one of those resets it. On a
# live call Sarvam ran out of credits, the pipeline wedged, and nothing ended it at all: the
# websocket stayed open, so Vobiz kept the phone leg up and kept billing, and one of four
# concurrency slots was held until the box was restarted by hand.
#
# Ten minutes is well past a qualifying call, which runs two to four. Deliberately not close
# to the real distribution: this is a backstop against a broken pipeline, not a policy on how
# long a prospect may talk, and cutting off a genuine conversation would be the worse failure.
MAX_CALL_DURATION_SECS = 600.0

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

# How fast the agent speaks when nobody has asked otherwise. Named rather than inline
# because it is now two things: the value the call opens with, and the ceiling a
# prospect can walk back up to after asking for slower. See app/utils/pace.py.
SPEAKING_PACE = 1.0


def caller_identity(project_name: str, developer_name: Optional[str] = None) -> str:
    """Who the agent says it is calling from.

    The developer, when the project has one recorded. On a live call the agent said "I am
    Priya calling you from Abhee Codename New Dimension" — which is the project, not an
    employer. A person calls from the company and names the project when they describe it.

    Falls back to the project name, which is what every call has said until now, so a
    project nobody has filled this in for sounds exactly as it did before.
    """
    return (developer_name or "").strip() or project_name


def build_opening_line(
    project_name: str,
    customer_name: Optional[str] = None,
    now: Optional[datetime] = None,
    developer_name: Optional[str] = None,
) -> str:
    """The first thing the caller hears.

    Greets by time of day and says who is calling, and nothing else. The name off the dial
    list is used to address them, so a prospect who does hear their own name knows the call
    is meant for them; without one the greeting simply omits it and the agent asks in its
    first reply — never a guessed name.

    It used to end "Can I speak to you for a minute?", and those eight words did two things
    wrong. They carried no information, on a line where the prospect had already sat through
    twenty words before saying anything but hello — and they invited a "no" to a question
    that was not the one worth asking. The question that follows in the opening gate, "Are
    you looking for any property purchase?", asks for the same permission and sorts the call
    at the same time.

    Written as short sentences rather than one comma-spliced line. Pipecat synthesises one
    sentence per request, so a full stop is a real gap the caller hears while a comma is not:
    measured on bulbul:v3, the same words with and without commas take the same time to say.
    As one comma-spliced line this arrives in a single flat rush.
    """
    part = time_of_day_greeting(now)
    # The lead list holds "Abhijit Kumar Singh", "RAHUL" and "mahantesha"; none of those is
    # how a person is greeted. Idempotent, so applying it here as well as before the prompt
    # costs nothing and means this line is safe whoever calls it. See app/utils/person_name.
    name = spoken_name(customer_name)
    address = f" {name}" if name else ""
    return (
        f"Hi, Good {part}{address}. I am {AGENT_NAME} calling you from "
        f"{caller_identity(project_name, developer_name)}."
    )

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
    # What stopped the pipeline, in words, as reported by on_pipeline_finished. None means it
    # stopped without saying — which is itself worth seeing, because that is exactly how an
    # ending nobody chose looks from the outside.
    end_reason: Optional[str] = None


def ending_reason(frame) -> str:
    """Why the pipeline stopped, in words, from the frame that stopped it.

    Falls back to the frame's class rather than to None or to "completed": an ending nobody
    labelled is the case worth seeing, and naming it EndFrame at least says which mechanism
    fired. Blank reasons are treated as absent, because an empty string reads in a log as
    though something was recorded when nothing was.
    """
    reason = getattr(frame, "reason", None)
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return type(frame).__name__


def session_error(
    pipeline_error: Optional[str],
    llm_failures: int,
    idle_timed_out: bool = False,
    tts_failures: int = 0,
    llm_quota_exhausted: bool = False,
    ran_too_long: bool = False,
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
    # Below the specific causes on purpose. The cap only fires when nothing else ended the
    # call, so if a named cause is also set that one is the story and this is its symptom.
    if ran_too_long:
        return "exceeded max call duration: the pipeline did not end on its own"
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
    developer_name: Optional[str] = None,
):
    # Converted once, here, so the greeting and the prompt address the prospect the same
    # way. Told the full name, the model uses the full name for the rest of the call — and
    # then the opening line and every turn after it disagree about who it is talking to.
    customer_name = spoken_name(customer_name) or None

    logger.info(
        f"[{call_sid}] Voice agent starting | client={client_type} | project='{project_name}' "
        f"| calling as='{caller_identity(project_name, developer_name)}' "
        f"| lead={customer_name or 'unnamed'} | llm={primary_endpoint(settings)}"
    )

    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            min_volume=settings.VAD_MIN_VOLUME,
            confidence=settings.VAD_CONFIDENCE,
            stop_secs=settings.VAD_STOP_SECS,
        )
    )

    turn_analyzer = build_turn_analyzer(call_sid, settings)

    serializer = VobizSerializer(stream_sid=call_sid)

    # Wrapped, not replaced: every call the transport makes falls through to the real socket.
    # It is here only so the close code survives the close — Pipecat's message iterator sees
    # the ASGI disconnect frame and raises a bare StopAsyncIteration, discarding the one field
    # that says whether the peer closed on purpose or the connection died. Two live calls have
    # now ended this way with nothing in the logs to tell those apart. See socket_witness.py.
    socket = SocketWitness(websocket)

    transport = FastAPIWebsocketTransport(
        websocket=socket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            serializer=serializer,
            # None unless SMART_TURN_ENABLED, and None is what the transport had before —
            # so an untouched deployment keeps the silence timer it has always used.
            turn_analyzer=turn_analyzer,
        ),
    )

    # Not GroqLLMService directly: the SDK's default two silent retries honour Retry-After
    # inside the same await, so a 429 reached the logs as `groq=14605ms` with no error and
    # the caller sat through all fourteen seconds of it. See app/services/llm_provider.py.
    llm = build_llm_service(call_sid, settings)
    stt = build_stt_service(call_sid, settings)
    
    # Passed only when somebody has set it. Unset, the key stays out of the connect payload
    # exactly as it has on every call so far, so a deployment cannot change the voice on its
    # own; set, it steadies the prosody Sarvam otherwise re-rolls at every full stop. See
    # config.py and tests/test_voice_consistency.py.
    tts_tuning = {}
    if settings.SARVAM_TEMPERATURE is not None:
        tts_tuning["temperature"] = settings.SARVAM_TEMPERATURE
        logger.info(f"[{call_sid}] Voice steadiness set | temperature={settings.SARVAM_TEMPERATURE}")

    # Low-latency streaming WebSocket Sarvam TTS with pace 1.0
    tts = SarvamTTSService(
        api_key=settings.SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice=settings.SARVAM_VOICE_ID,
            pace=SPEAKING_PACE,
            **tts_tuning,
            max_chunk_length=150,
            # min_buffer_size is deliberately left at Sarvam's default. Setting it to 25
            # was rejected at connect time with "Input parameters has to be a valid
            # dictionary", killing TTS for the whole call. Pipecat forwards the value
            # straight into the config payload with no range check, so any new value here
            # must be validated against Sarvam's API before it reaches a live call.
        ),
    )

    # Sarvam's websocket is torn down and reopened on every interruption — Pipecat's own
    # InterruptibleTTSService does it, to drop audio the prospect has just spoken over. On a
    # live call four barge-ins in fourteen seconds meant four reconnects, and one of them
    # timed out during the opening handshake and left the agent mute until the caller hung
    # up. None of that churn appeared anywhere, so the reconnects had to be inferred from a
    # docstring. These two lines make it countable.
    _tts_reconnects: int = 0

    @tts.event_handler("on_connected")
    async def on_tts_connected(service):
        nonlocal _tts_reconnects
        _tts_reconnects += 1
        # The first is the call opening its voice, which is not news. Every one after it is
        # a reconnect, and a run of them is the shape that preceded the failure.
        if _tts_reconnects > 1:
            logger.info(f"[{call_sid}] TTS reconnected ({_tts_reconnects - 1})")

    @tts.event_handler("on_connection_error")
    async def on_tts_connection_error(service, message):
        logger.error(
            f"[{call_sid}] TTS could not reach Sarvam after {_tts_reconnects} connection(s); "
            f"the caller hears silence until it comes back: {message}"
        )

    system_prompt = get_system_prompt(campaign_context, customer_name)
    messages = [{"role": "system", "content": system_prompt}]
    
    task_ref = []
    # Raises the signal that the goodbye has actually been played out. Constructed here
    # rather than beside the other processors because the end_call handler below closes
    # over it. See app/utils/farewell.py.
    farewell = FarewellGate()
    # One hangup per call. Both end paths can fire for the same turn — the model can emit a
    # structured tool call and leaked syntax together — and two farewells racing would
    # interrupt each other, which is the failure this whole path exists to stop.
    _ending = False

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

    async def say_goodbye_then_hang_up(line: str) -> None:
        """Speak the closing line, wait for it to actually be heard, then end the call.

        Written as three awaited steps rather than one queue_frames call because the frames
        do not order the way the old one-liner assumed, and the caller heard the difference.

        The interruption clears a stale reply from a split turn — one inference once asked
        "What time on Sunday?" while another hung up, and the prospect heard the question
        followed straight by the goodbye. But an InterruptionWorkerFrame only takes effect
        after its own round trip to the sink and back, and the worker's push loop does not
        wait for that before queueing the next frame. Queued together, the farewell could
        enter the pipeline first and then be cancelled by the very interruption meant to
        protect it. flush_pipeline waits for the lap to finish, so the order is not a guess.

        Stale is the word that had to be earned. On 4 Sep 2026 one inference produced a
        reply AND the tool call, and this interruption cut the reply off mid-sentence:

            AGENT initiated call end via tool -> "Thank you for sharing these details..."
            AGENT -> "That works well. Since you are looking in North Bangalore, I will
                      have our property expert suggest some better options for you."
                      [interrupted]

        That reply was not stale. It was this turn's own lead-in, and the prospect heard
        half of it and then a goodbye. So the interruption now applies only when the words
        on the wire came from some earlier inference; when they came from the one hanging
        up, they are allowed to finish first, with a ceiling sized to the sentence.

        Then the wait. EndWorkerFrame is not enough on its own: its round trip proves the
        frames have travelled, and Sarvam's audio does not travel with them — run_tts sends
        the text and returns, and the voice arrives afterwards on a receive task.
        BotStoppedSpeakingFrame comes off the transport's audio clock once the turn has
        actually been played out, so that is what is waited on, with a ceiling sized to the
        sentence so a dead TTS cannot hold the carrier leg open.
        """
        lead_in = tool_syntax_filter.lead_in
        if lead_in and farewell.is_speaking:
            logger.info(
                f"[{call_sid}] Letting this turn's own words finish before the goodbye: "
                f"{lead_in.strip()[:80]!r}"
            )
            if not await farewell.wait_for_quiet(farewell_timeout(lead_in)):
                logger.warning(
                    f"[{call_sid}] The lead-in never finished playing; saying the goodbye "
                    f"over it rather than holding the line open"
                )
        else:
            await task_ref[0].queue_frames([InterruptionWorkerFrame()])
            await task_ref[0].flush_pipeline(timeout=2.0)

        # From here nothing may cancel the closing line. See ClosingGate.protect_goodbye:
        # placed after the branch above so end_call's own interruption, which clears a stale
        # reply from a split turn, still lands.
        closing_gate.protect_goodbye()

        farewell.arm()
        await task_ref[0].queue_frames([TTSSpeakFrame(line)])
        if not await farewell.wait_until_spoken(farewell_timeout(line)):
            logger.warning(
                f"[{call_sid}] Goodbye never finished playing; hanging up anyway so the "
                f"carrier leg stops billing"
            )
        await task_ref[0].queue_frames([EndWorkerFrame(reason="end_call tool")])

    # 2. Actual handler that intercepts the tool execution
    async def end_call_handler(params=None, *args, **kwargs):
        nonlocal _ending
        spoken = getattr(params, "arguments", None) or {}
        line = closing_line(spoken.get("closing_line") if isinstance(spoken, dict) else None)
        logger.info(f"[{call_sid}] AGENT initiated call end via tool → \"{line}\"")
        if not task_ref or _ending:
            return
        _ending = True
        closing_gate.arm()
        # Detached on purpose. This handler runs inside the LLM service's function-call
        # machinery, which is waiting to push the tool result; holding it for the length of
        # a spoken sentence would block the very pipeline that has to carry the audio.
        asyncio.create_task(say_goodbye_then_hang_up(line))

    llm.register_function("end_call", end_call_handler)

    # 3. Same intent, wrong channel: the model wrote the tool call into its spoken text.
    # ToolSyntaxFilter has already stopped the caller hearing it, so all that is left is to
    # hang up the way the structured call would have.
    async def on_leaked_end_call(line: Optional[str], already_spoke: bool):
        nonlocal _ending
        if not task_ref or _ending:
            return
        _ending = True
        closing_gate.arm()
        if already_spoke:
            # The words before the markup were the goodbye, and they are on the wire now.
            # Speaking the leaked closing line too would be two farewells in a row — but the
            # first one still has to finish, so this waits on the same signal.
            logger.info(f"[{call_sid}] Ending call after leaked end_call syntax (goodbye already spoken)")
            farewell.arm()
            await farewell.wait_until_spoken(farewell_timeout(line or ""))
            await task_ref[0].queue_frames([EndWorkerFrame(reason="leaked end_call, goodbye already spoken")])
            return
        spoken = closing_line(line)
        logger.info(f"[{call_sid}] Ending call after leaked end_call syntax → \"{spoken}\"")
        await say_goodbye_then_hang_up(spoken)

    # Above the tool-syntax filter on purpose: a reply to half a sentence must not reach
    # the leaked-end_call path either. See app/utils/turn_gate.py.
    turn_gate = TurnFinalityGate(call_sid)
    # Armed by end_call. From then on no turn can be generated, so nothing can be
    # spoken after the goodbye. See app/utils/closing_gate.py.
    closing_gate = ClosingGate(call_sid)
    stt_witness = SttWitness()
    tool_syntax_filter = ToolSyntaxFilter(
        call_sid, on_leaked_end_call=on_leaked_end_call, campaign_context=campaign_context
    )
    
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
        # Pure pass-through. Sees exactly what the aggregator sees, so a turn that finalizes
        # empty can say whether the STT sent nothing, sent silence, or sent interims whose
        # final never arrived. See app/utils/stt_witness.py.
        stt_witness,
        user_agg,
        closing_gate,
        llm,
        # Drops a reply generated while the prospect was still talking, before anything
        # downstream can speak it or act on it.
        turn_gate,
        # Between the LLM and the TTS on purpose: it is the last point at which leaked tool
        # syntax can be removed before it is spoken, and being upstream of the assistant
        # aggregator keeps the leak out of the context too.
        tool_syntax_filter,
        tts,
        transport.output(),
        # After the output transport, never before it: BotStoppedSpeakingFrame is raised by
        # the transport's audio clock once the turn has actually been played out at realtime
        # pace, so upstream of here the signal does not exist yet.
        farewell,
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
    
    # What the agent has already asked about, and what to do instead of asking it again.
    # On a live call it asked for a site visit nineteen times and lost a three-crore lead to
    # the repetition. See app/utils/asked.py.
    asked = AskedSoFar()
    _asked_shown: str = ""

    def refresh_asked_brief() -> None:
        """Put the block in front of the model, or take it away again.

        Appended to the system message rather than added as its own. A system turn sitting
        between the conversation's own would have better recency, but Gemma's chat template
        is not something this repository controls and a context the provider rejects is a
        dead call — while a block the model reads a little less closely is a worse call at
        worst. Mutated in place: LLMContext holds this very list, so there is nothing to
        re-set.
        """
        nonlocal _asked_shown
        brief = asked.brief()
        if brief == _asked_shown:
            return
        _asked_shown = brief
        messages[0]["content"] = "\n\n".join([system_prompt, brief]) if brief else system_prompt

    # Moves only when the prospect asks about the speed, and never past SPEAKING_PACE.
    _pace: float = SPEAKING_PACE
    _turn_start_time: float = 0.0
    _user_has_spoken: bool = False
    _startup_task = None
    _llm_failures: int = 0
    _llm_quota_exhausted: bool = False
    _tts_failures: int = 0
    _empty_user_turns: int = 0
    _idle_timed_out: bool = False
    _ran_too_long: bool = False
    # What stopped the pipeline, filled in by on_pipeline_finished below. Reported on the
    # CallResult so a call whose ending nobody chose is visible in the data, not just the log.
    _end_reason: Optional[str] = None
    # True between the moment a turn is sent for inference and the moment its answer starts
    # coming back. If the prospect speaks again inside that window, whatever is being
    # generated is an answer to half of what they said.
    _llm_in_flight: bool = False
    # True from the moment a reply starts being spoken until its audio has finished. An
    # empty user turn inside this window is a false barge-in on speech already in progress;
    # one outside it is a prospect answering into silence.
    _agent_speaking: bool = False
    # The agent's last finalized reply, kept so its question can be asked again without an
    # LLM round trip when the answer never reaches us.
    _last_agent_line: str = ""
    _dead_air_nudges: int = 0
    _last_nudged: str = ""
    # Counted so the machine check only ever runs on the opening turn.
    _turns_heard: int = 0
    _answering_machine: bool = False
    # What the prospect said in their opening turns, kept so a voicemail announcement
    # split across several of them can be read as the one sentence it is.
    _opening_turns: list[str] = []

    # ─── Pipeline Started ──────────────────────────────────────────────────────
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        async def startup_greeting():
            # No delay before the opening line. A caller who picks up already waits about
            # four seconds — 611ms for the answer webhook to become a websocket, 2.4s for
            # Vobiz's start event, then the pipeline — and every one of those is silence on
            # a sales call. The 200ms that used to sit here was guarding against greeting
            # someone who spoke first, which _user_has_spoken checks on the line below and
            # GreetingOnlyMinWords protects for the rest of the opening line anyway.
            if not _user_has_spoken:
                opening_line = build_opening_line(
                    project_name, customer_name, developer_name=developer_name
                )
                context.add_message({"role": "assistant", "content": opening_line})
                logger.info(f"[{call_sid}] AGENT → \"{opening_line}\"")
                # Alongside the greeting, not before it: the opening line is built locally and
                # needs no model, so this is six to eight seconds of speech during which the
                # connection to the provider can be opened for free. Without it the handshake
                # lands on the first thing the prospect actually waits for — measured at 3382ms
                # for a first turn against 1247ms for the second on the same call.
                asyncio.create_task(llm.warm_up())
                await task.queue_frames([TTSSpeakFrame(opening_line)])
            else:
                # They spoke first, so the greeting was cancelled and there is nothing left
                # to protect. Lifting it here matters: otherwise the gate stays armed for a
                # call that never had an opening line to guard.
                greeting_gate.relax()

        nonlocal _startup_task
        _startup_task = asyncio.create_task(startup_greeting())

    async def abandon_call(reason: str) -> None:
        """Tear the pipeline down now, without asking it to drain first.

        EndFrame cannot do this job and a live call proved it. When Sarvam ran out of
        credits the handler queued one, logged "abandoning call", and the pipeline carried
        straight on — an assistant turn finalized 3.3 seconds later and neither "Pipeline
        finished" nor "Call finalised" ever appeared. The websocket stayed open, so Vobiz
        kept the phone leg up and kept billing for it, the Call row stayed IN_PROGRESS
        forever, and one of only four concurrency slots was gone until the box was restarted.

        The reason is the frame class. EndFrame is a ControlFrame, so it travels in queue
        order and has to pass through the TTS to get anywhere — and the TTS is precisely
        what is wedged, looping through reconnects to a service refusing us. CancelFrame is
        a SystemFrame; it bypasses the ordered queue. task.cancel() queues one, which is the
        same path Pipecat itself takes for a fatal error and the same path cancel_on_idle
        _timeout already uses successfully on this pipeline.

        Safe to call repeatedly: PipelineWorker.cancel() is guarded on both _finished and
        _cancelled, which matters because the TTS error that triggers it arrives many times
        a second.
        """
        logger.error(f"[{call_sid}] Abandoning call ({reason}); cancelling the pipeline")
        try:
            await task.cancel(reason=reason)
        except Exception as e:
            # Never let teardown raise out of an event handler. If cancel itself fails the
            # duration cap below is the last line of defence, and it must still get to run.
            logger.error(f"[{call_sid}] Pipeline cancel failed: {e}")

    # ─── Client Disconnected ───────────────────────────────────────────────────
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        # "the media stream closed", not "the caller hung up". All this event reports is
        # that the websocket went away, and that happens for two quite different reasons:
        # the prospect pressed end, or the stream itself dropped while the PSTN leg was
        # still up. Claiming the first cost a real investigation on 2 Sep 2026 — a call was
        # read off this line as the prospect walking away mid-pitch, when the carrier's own
        # record said the hangup source was the carrier. The distinction lives in the
        # carrier's hangup cause, which arrives separately; this line must not pre-empt it.
        logger.info(f"[{call_sid}] Media stream closed — ending pipeline")
        # The whole point of SocketWitness, on its own line so it can be counted: `SOCKET`
        # appears nowhere else in these logs.
        logger.warning(f"[{call_sid}] SOCKET closed | {socket.report()}")
        await task.queue_frames([EndFrame(reason="the media stream closed")])

    # ─── Hard Duration Cap ─────────────────────────────────────────────────────
    async def enforce_max_duration():
        """Last line of defence against a call that will not end by itself.

        The idle timeout does not cover this. It fires only when NEITHER party has spoken,
        so a caller who keeps saying "hello?" into a line whose voice has died resets it
        forever, and every second of that is billed by Vobiz and holds a concurrency slot.
        A wedged pipeline is the same shape from out here: nothing arrives to end it.

        Sized well above a real conversation rather than near it. A sales call that has run
        this long is not a call any more.
        """
        try:
            await asyncio.sleep(MAX_CALL_DURATION_SECS)
        except asyncio.CancelledError:
            return
        nonlocal _ran_too_long
        _ran_too_long = True
        logger.error(
            f"[{call_sid}] Call has run {MAX_CALL_DURATION_SECS:.0f}s — past any real "
            f"conversation. Cancelling so the carrier leg stops billing."
        )
        await abandon_call("exceeded the maximum call duration")

    _duration_guard = asyncio.create_task(enforce_max_duration())

    # ─── Why the call ended ────────────────────────────────────────────────────
    # Every deliberate ending in this module logs its own reason before queueing a frame.
    # A live call still finished with none of them in the log: the agent asked a question,
    # the audio played out, and 43ms later the pipeline was done with nothing to say why.
    #
    # That leaves the ending unattributable, which is the one thing a call log must never be
    # — "the prospect hung up" and "we hung up on the prospect" look identical from here. So
    # the pipeline reports its own terminator, and the reason travels on the frame itself so
    # the paths below can name themselves rather than relying on a log line written earlier.
    @task.event_handler("on_pipeline_finished")
    async def on_pipeline_finished(worker, frame):
        nonlocal _end_reason
        _end_reason = ending_reason(frame)
        logger.info(f"[{call_sid}] Pipeline stopped by {type(frame).__name__}: {_end_reason}")

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
        stt_witness.reset()

        # The prospect has started talking again while we are still generating a reply to
        # what they said before. That reply answers a sentence they were not finished
        # saying, and paying for it twice is the smaller half of the problem: a caller once
        # heard a stale "What time on Sunday?" followed immediately by the goodbye, because
        # both inferences finished. Abandon it — the next one sees the whole utterance.
        #
        # Only fires before the answer starts arriving. Once it does, a new user turn is an
        # ordinary barge-in and Pipecat raises its own interruption for it.
        turn_gate.user_turn_started()
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
        nonlocal _dead_air_nudges, _last_nudged
        # Releases a held reply, or drops it when a newer inference has superseded it. The
        # strategy fires the inference event before this one, so the check is race-free.
        await turn_gate.user_turn_stopped()
        transcript = (message.content or "").strip() if message and hasattr(message, "content") else ""
        total_turn_time = f"{(time.time() - _turn_start_time) * 1000:.0f}ms" if _turn_start_time else "?"
        if transcript:
            _user_has_spoken = True
            logger.info(f"[{call_sid}] USER  → \"{transcript}\" (Total Turn Duration: {total_turn_time})")

            # "Can you say it again with... little bit slow hai?" — asked twice on a live
            # call, and answered twice at exactly the same speed, because the pace was a
            # constant. Being heard and ignored is worse than not being understood.
            nonlocal _pace
            wanted = adjusted_pace(_pace, pace_request(transcript), SPEAKING_PACE)
            if wanted != _pace:
                _pace = wanted
                logger.info(f"[{call_sid}] Prospect asked about the speed; pace now {_pace}")
                await task.queue_frames(
                    [TTSUpdateSettingsFrame(delta=SarvamTTSService.Settings(pace=_pace))]
                )

            # Only the opening turns. A recorded greeting is the first thing a machine says;
            # the same words later in a real conversation are a person talking about their
            # availability, and hanging up on them would be much worse than transcribing one
            # voicemail.
            #
            # Turns plural, and read together: this was the first turn alone, which is not
            # how a voicemail announcement arrives. See OPENING_TURNS.
            if not _answering_machine and len(_opening_turns) < OPENING_TURNS:
                _opening_turns.append(transcript)
                if machine_in_opening(_opening_turns):
                    _answering_machine = True
                    logger.info(
                        f"[{call_sid}] Answering machine detected over {len(_opening_turns)} "
                        f"opening turn(s); hanging up without leaving a message. "
                        f"Matched: {machine_phrases(' '.join(_opening_turns))}"
                    )
                    await task.queue_frames([EndFrame(reason="answering machine")])
            _turns_heard += 1
            return
        # VAD heard speech but the STT produced nothing. Without this line a false barge-in
        # leaves no trace at all, which is what made the interruptions look inexplicable.
        # The witness distinguishes the harmless case from the one that costs a real answer.
        _empty_user_turns += 1
        logger.warning(
            f"[{call_sid}] VAD fired with no transcribable speech after {total_turn_time} "
            f"(count: {_empty_user_turns}) — {stt_witness.report()}"
        )

        # No transcript means no inference, which means the agent says nothing at all. If it
        # had asked a question, the prospect has now answered into a line that went silent
        # on them, and on a live call the only thing that restarted the conversation was
        # them saying "Hello?" eleven seconds later. Ask again ourselves.
        #
        # Not while the agent is still talking: an empty turn during its own speech is the
        # false barge-in this counter was built for, the question is already being asked,
        # and repeating it over the top would be worse than the noise that triggered it.
        if _agent_speaking or _llm_in_flight:
            return
        if _dead_air_nudges >= MAX_DEAD_AIR_NUDGES:
            return
        nudge = dead_air_nudge(_last_agent_line)
        # None when the last turn asked nothing — a sign-off must never be said twice.
        # Equal to the previous nudge when this is the same question going unanswered a
        # second time, which is a line that cannot carry the call, not a prospect to badger.
        if not nudge or nudge == _last_nudged:
            return
        _dead_air_nudges += 1
        _last_nudged = nudge
        logger.info(f"[{call_sid}] Nothing heard back; asking again → \"{nudge}\"")
        await task.queue_frames([TTSSpeakFrame(nudge)])

    # The two edges of "a reply is being generated". Together they bound the window in
    # which a new user turn makes the in-flight inference worthless.
    @user_agg.event_handler("on_user_turn_inference_triggered")
    async def on_user_turn_inference_triggered(aggregator, *_):
        nonlocal _llm_in_flight
        _llm_in_flight = True
        # Pipecat fires this more than once in a turn when the prospect pauses and carries
        # on. Each one makes the previous reply an answer to half a sentence.
        turn_gate.inference_triggered()

    @assistant_agg.event_handler("on_assistant_turn_started")
    async def on_assistant_turn_started(aggregator, *_):
        nonlocal _llm_in_flight, _agent_speaking
        _llm_in_flight = False
        _agent_speaking = True

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
            await task.queue_frames([TTSSpeakFrame(FAREWELL_LINE), EndWorkerFrame(reason="provider rejected the tool call")])
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
                await task.queue_frames([TTSSpeakFrame(LLM_SIGNOFF_LINE), EndWorkerFrame(reason="llm throttled past recovery")])
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
            await task.queue_frames([TTSSpeakFrame(LLM_SIGNOFF_LINE), EndWorkerFrame(reason="llm quota exhausted")])
            return

        _llm_failures += 1
        logger.warning(
            f"[{call_sid}] LLM turn failed ({_llm_failures}/{MAX_LLM_TURN_FAILURES}): {error.error}"
        )
        if _llm_failures >= MAX_LLM_TURN_FAILURES:
            logger.error(f"[{call_sid}] LLM unrecoverable; signing off to avoid dead air")
            await task.queue_frames([TTSSpeakFrame(LLM_SIGNOFF_LINE), EndWorkerFrame(reason="llm turn failures exhausted")])
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
            # Pipecat already queues a CancelFrame for a fatal ErrorFrame, so counting this
            # one would only race its own shutdown.
            return
        _tts_failures += 1
        # Every one, not only the first. On a live call on 4 Sep the voice engine failed
        # twice and went silent for twelve seconds; only the first failure carried this line,
        # so reading the log afterwards the second one existed solely as a Pipecat traceback
        # with no call id on it. A failure nobody can attribute to a call is not a diagnosis.
        logger.error(
            f"[{call_sid}] TTS failing ({_tts_failures}) — caller is hearing silence: "
            f"{error.error}"
        )
        # >=, not ==. This used to fire once and never again, so when the one attempt failed
        # to take effect there was nothing behind it.
        if _tts_failures >= MAX_TTS_FAILURES:
            logger.error(f"[{call_sid}] TTS unavailable after {_tts_failures} errors; abandoning call")
            await abandon_call("tts unavailable")

    # Log the assistant's finalized turn. Note the event is on_assistant_turn_stopped —
    # on_assistant_message_added is not an event this aggregator registers, which is why
    # every AGENT line was missing from the logs.
    @assistant_agg.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message):
        nonlocal _agent_speaking, _last_agent_line
        _agent_speaking = False
        content = (getattr(message, "content", "") or "").strip()
        if content:
            suffix = " [interrupted]" if getattr(message, "interrupted", False) else ""
            logger.info(f"[{call_sid}] AGENT → \"{content}\"{suffix}")
            # An interrupted reply is still what the prospect last heard us ask, so it is
            # still the right thing to repeat if their answer then goes missing.
            _last_agent_line = content

            topic = asked.record(content)
            if topic:
                count = asked.counts[topic.key]
                if count == REPEAT_LIMIT:
                    logger.info(
                        f"[{call_sid}] Asked about {topic.label} {count} times; telling the "
                        f"model to stop and what to do instead"
                    )
                elif count > REPEAT_LIMIT:
                    # It was already told. This is the model going round the loop anyway,
                    # which is the thing worth counting across calls.
                    logger.warning(
                        f"[{call_sid}] Asked about {topic.label} {count} times despite being "
                        f"told not to"
                    )
                refresh_asked_brief()
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
    finally:
        # The guard outlives the pipeline otherwise, and a sleeping task holding this
        # closure keeps the whole call's state alive for ten minutes after the caller has
        # gone. On a box capped at four concurrent calls that is a leak worth closing.
        _duration_guard.cancel()

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
        end_reason=_end_reason,
        error=session_error(
            error,
            _llm_failures,
            _idle_timed_out,
            _tts_failures,
            _llm_quota_exhausted,
            _ran_too_long,
        ),
        latency=latency.log_summary(),
        answering_machine=_answering_machine,
    )
