from arq import create_pool
from arq.connections import RedisSettings
from loguru import logger

_redis_pool = None

async def get_redis_pool():
    global _redis_pool
    if not _redis_pool:
        _redis_pool = await create_pool(RedisSettings(host="localhost", port=6379))
    return _redis_pool

async def dispatch_extraction_task(transcript: str, call_sid: str):
    pool = await get_redis_pool()
    await pool.enqueue_job('process_extraction', transcript, call_sid)
    logger.info(f"Dispatched extraction task to Redis for {call_sid}")
