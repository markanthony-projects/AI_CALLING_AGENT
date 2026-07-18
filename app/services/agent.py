import asyncio
import time
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.frames.frames import EndFrame, TextFrame, TTSSpeakFrame
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
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
from app.utils.exotel_serializer import ProductionExotelSerializer
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
    client_type: str = "exotel",
    project_name: str = "your project",
):
    logger.info(f"[{call_sid}] Voice agent starting | client={client_type} | project='{project_name}' | model={GROQ_MODEL}")

    # 1. Faster VAD Endpointing (0.5s instead of default 0.8s) to reduce latency gap
    vad_analyzer = SileroVADAnalyzer(params=VADParams(min_volume=0.1, confidence=0.5, stop_secs=0.5))

    # Single unified serializer for both browser and Exotel (G.711 µ-law @ 8 kHz)
    serializer = ProductionExotelSerializer(
        stream_sid=call_sid,
        params=ProductionExotelSerializer.InputParams(auto_hang_up=False, exotel_sample_rate=8000),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            serializer=serializer,
        ),
    )

    llm = GroqLLMService(api_key=settings.GROQ_API_KEY, settings=GroqLLMService.Settings(model=GROQ_MODEL))
    stt = SarvamSTTService(api_key=settings.SARVAM_API_KEY, settings=SarvamSTTService.Settings(model="saaras:v3"))
    
    # 2. Revert TTS Pace to 1.0
    tts = SarvamTTSService(
        api_key=settings.SARVAM_API_KEY,
        settings=SarvamTTSService.Settings(model="bulbul:v3", voice=settings.SARVAM_VOICE_ID, pace=1.0),
    )

    system_prompt = get_system_prompt(campaign_context)
    messages = [{"role": "system", "content": system_prompt}]
    
    task_ref = []
    
    # 1. Dummy function to generate the JSON schema via Pipecat's inspect logic
    # Pipecat requires the first parameter to be named 'params' for direct functions.
    async def end_call(params: dict):
        """
        Ends the call. Call this function ONLY when the conversation is completely finished.
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

    # ─── Pipeline Started ──────────────────────────────────────────────────────
    @task.event_handler("on_pipeline_started")
    async def on_pipeline_started(worker, frame):
        opening_line = f"Hello, I am Priya calling from {project_name}. Do you have 2 minutes?"
        context.add_message({"role": "assistant", "content": opening_line})
        logger.info(f"[{call_sid}] AGENT → \"{opening_line}\"")
        await task.queue_frames([TTSSpeakFrame(opening_line)])

    # ─── Client Disconnected ───────────────────────────────────────────────────
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"[{call_sid}] Client disconnected — ending pipeline")
        await task.queue_frames([EndFrame()])

    # ─── User Starts Speaking ──────────────────────────────────────────────────
    @user_agg.event_handler("on_user_turn_started")
    async def on_user_turn_started(aggregator, strategy):
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
        transcript = (message.content or "").strip() if message and hasattr(message, "content") else ""
        if transcript:
            stt_latency = f"{(time.time() - _turn_start_time) * 1000:.0f}ms" if _turn_start_time else "?"
            logger.info(f"[{call_sid}] USER  → \"{transcript}\" (Turn Duration + STT latency: {stt_latency})")

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
            for msg in context.messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    speaker = "Prospect" if role == "user" else "Agent"
                    transcript_str += f"{speaker}: {content}\n"
        except Exception as e:
            logger.error(f"[{call_sid}] Failed to compile transcript: {e}")
            
        return transcript_str.strip()
