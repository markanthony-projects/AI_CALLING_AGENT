import asyncio
import time
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.frames.frames import EndFrame, TextFrame, TTSSpeakFrame
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


async def run_voice_agent(
    websocket,
    campaign_context: str,
    call_sid: str,
    client_type: str = "vobiz",
    project_name: str = "your project",
):
    logger.info(f"[{call_sid}] Voice agent starting | client={client_type} | project='{project_name}' | model={GROQ_MODEL}")

    # 1. Strict VAD Endpointing (0.2s) to prevent Pipecat from collapsing STT wait timeout and forcing aggregator fallback delays
    vad_analyzer = SileroVADAnalyzer(params=VADParams(min_volume=0.1, confidence=0.5, stop_secs=0.2))

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
        ),
    )

    system_prompt = get_system_prompt(campaign_context)
    messages = [{"role": "system", "content": system_prompt}]
    
    task_ref = []
    
    # 1. Dummy function to generate the JSON schema via Pipecat's inspect logic
    async def end_call(params: dict):
        """
        Ends the call. Call this function ONLY when the prospect explicitly and unambiguously says goodbye or ends the conversation.
        """
        pass
        
    # 2. Actual handler that intercepts the tool execution
    async def end_call_handler(*args, **kwargs):
        logger.info(f"[{call_sid}] AGENT initiated call end via tool.")
        if task_ref:
            await task_ref[0].queue_frames([EndFrame()])
            
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

    task = PipelineWorker(pipeline, params=PipelineParams(allow_interruptions=True))
    task_ref.append(task)
    
    _turn_start_time: float = 0.0
    _user_has_spoken: bool = False
    _startup_task = None

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

    # ─── User Starts Speaking ──────────────────────────────────────────────────
    @user_agg.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
        nonlocal _turn_start_time, _user_has_spoken
        _user_has_spoken = True
        _turn_start_time = time.time()
        if client_type == "exotel":
            try:
                await websocket.send_json({"event": "clear_client_buffer"})
            except Exception:
                pass

    # ─── User Stops Speaking ───────────────────────────────────────────────────
    @user_agg.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message):
        transcript = (message.content or "").strip() if message and hasattr(message, "content") else ""
        if transcript:
            total_turn_time = f"{(time.time() - _turn_start_time) * 1000:.0f}ms" if _turn_start_time else "?"
            logger.info(f"[{call_sid}] USER  → \"{transcript}\" (Total Turn Duration: {total_turn_time})")

    # ─── Agent Generation Logging ──────────────────────────────────────────────
    @llm.event_handler("on_client_connected")
    async def on_llm_connected(service, client):
        logger.info(f"[{call_sid}] LLM   → connected and generating...")

    # Log the assistant's finalized message text before it finishes speaking
    @assistant_agg.event_handler("on_assistant_message_added")
    async def on_assistant_message(aggregator, message):
        if message and hasattr(message, "content"):
            logger.info(f"[{call_sid}] AGENT → \"{message.content}\"")

    runner = PipelineRunner()
    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"[{call_sid}] Pipeline exception: {e}")
    finally:
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
            
        return transcript_str.strip()
