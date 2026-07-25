import asyncio
from loguru import logger
from app.worker import process_extraction

async def dispatch_extraction_task(transcript: str, call_sid: str):
    try:
        # Schedule the background extraction natively in Uvicorn's event loop
        asyncio.create_task(process_extraction(transcript, call_sid))
        logger.info(f"Dispatched background extraction task natively for {call_sid}")
    except Exception as e:
        logger.error(f"Failed to dispatch extraction task for {call_sid}: {e}")
