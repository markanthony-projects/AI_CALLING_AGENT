from loguru import logger

from app.core.queue import get_arq_pool


async def enqueue_extraction(call_sid: str) -> bool:
    """Hand the call off to the arq worker; the transcript is already durable in Postgres."""
    try:
        job = await get_arq_pool().enqueue_job("process_extraction", call_sid, _job_id=f"extract:{call_sid}")
    except Exception as e:
        logger.error(f"[{call_sid}] Failed to enqueue extraction job: {e}")
        return False

    if job is None:
        logger.info(f"[{call_sid}] Extraction job already queued or recently completed; skipping duplicate")
        return False

    logger.info(f"[{call_sid}] Enqueued extraction job {job.job_id}")
    return True
