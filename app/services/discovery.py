import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.db import Project, Campaign, CampaignStatus
from app.models.schemas import ProjectFilterParams
import redis.asyncio as redis
from app.core.config import settings
from loguru import logger

_redis_client = None
_l1_project_cache = {}  # In-memory L1 cache: campaign_id -> (timestamp, project_dict)
L1_CACHE_TTL = 300  # 5 minutes in-memory TTL

# Redis runs as a container on the same host, so a healthy round trip is sub-millisecond and
# anything slower than a second or two is not slow, it is down. Left at redis-py's default of
# no timeout, an unreachable Redis waits on the OS TCP timeout instead — measured at 23
# seconds. That is now on the path that opens a live call: the carrier slot check, the dialled
# number, the lead's name. Twenty-three seconds of that is a prospect listening to silence
# while a container restarts, and it happens on every call until Redis is back.
_REDIS_TIMEOUT_SECONDS = 2.0


async def get_redis_client():
    global _redis_client
    if not _redis_client:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            socket_connect_timeout=_REDIS_TIMEOUT_SECONDS,
            socket_timeout=_REDIS_TIMEOUT_SECONDS,
            # The client is long-lived and mostly idle between calls. Without this it hands
            # out a connection the server closed hours ago and the first command fails.
            health_check_interval=30,
            retry_on_timeout=True,
        )
    return _redis_client

async def discover_projects(db: AsyncSession, filters: ProjectFilterParams):
    query = select(Project)
    
    if filters.city:
        query = query.where(Project.city.ilike(f"%{filters.city}%"))
    if filters.locality:
        query = query.where(Project.locality.ilike(f"%{filters.locality}%"))
    if filters.min_budget is not None:
        query = query.where(Project.min_price >= filters.min_budget)
    if filters.max_budget is not None:
        query = query.where(Project.max_price <= filters.max_budget)
        
    result = await db.execute(query)
    return result.scalars().all()

async def get_project_by_campaign(db: AsyncSession, campaign_id: str):
    import time
    now = time.time()
    
    # 1. Check In-Memory L1 Cache (0ms latency)
    if campaign_id in _l1_project_cache:
        ts, cached_dict = _l1_project_cache[campaign_id]
        if now - ts < L1_CACHE_TTL:
            logger.info(f"L1 In-Memory Cache HIT for campaign {campaign_id}")
            return cached_dict

    # 2. Check Redis L2 Cache
    cache = await get_redis_client()
    cache_key = f"project_context:{campaign_id}"
    
    try:
        cached_data = await cache.get(cache_key)
        if cached_data:
            logger.info(f"Redis L2 Cache HIT for campaign {campaign_id}")
            project_dict = json.loads(cached_data)
            _l1_project_cache[campaign_id] = (now, project_dict)
            return project_dict
    except Exception as e:
        logger.warning(f"Redis L2 cache check failed: {e}")
        
    logger.info(f"Cache MISS for campaign {campaign_id}, querying DB")
    
    # Fallback for out-of-the-box browser testing
    if campaign_id == "demo":
        stmt = select(Campaign).where(Campaign.status == CampaignStatus.ACTIVE).limit(1)
        result = await db.execute(stmt)
        demo_campaign = result.scalar_one_or_none()
        if not demo_campaign:
            return None
        campaign_id = demo_campaign.id

    query = select(Project).join(Campaign, Campaign.project_id == Project.id).where(Campaign.id == campaign_id)
    result = await db.execute(query)
    project = result.scalars().first()
    
    if not project:
        return None
        
    project_dict = {
        "name": project.name,
        "locality": project.locality,
        "min_price": float(project.min_price) if project.min_price else None,
        "max_price": float(project.max_price) if project.max_price else None,
        "amenities": project.amenities or [],
        "nearby_facilities": project.nearby_facilities or {},
        "possession_status": project.possession_status,
        "usps": project.usps or [],
        "rera_id": project.rera_id,
        "config_json": project.config_json or []
    }
    
    _l1_project_cache[campaign_id] = (now, project_dict)
    try:
        await cache.setex(cache_key, 86400, json.dumps(project_dict))
    except Exception as e:
        logger.warning(f"Redis L2 cache store failed: {e}")
        
    return project_dict

async def invalidate_campaign_context(campaign_id: str):
    """Drop the cached project context for one campaign.

    Named for what it takes. It was called invalidate_project_cache, and two callers duly
    handed it a project id — which matches no key, so editing a project cleared nothing and
    live calls kept quoting the old prices for up to a day. Both the L1 dict and the Redis
    key are campaign-scoped, because that is what the caller of get_project_context has.
    """
    _l1_project_cache.pop(campaign_id, None)
    try:
        cache = await get_redis_client()
        await cache.delete(f"project_context:{campaign_id}")
    except Exception as e:
        logger.warning(f"Redis cache invalidate failed: {e}")
    logger.info(f"Invalidated L1 & L2 cache for campaign {campaign_id}")


async def invalidate_project_everywhere(db, project_id) -> int:
    """Drop the cached context of every campaign selling this project.

    A project edit changes what the agent says on calls for every campaign attached to it,
    and the cache has no key for the project itself. Resolving the campaigns is the only way
    an edit reaches the caller; without it the dashboard saves happily and the next prospect
    hears the old price for the next twenty-four hours.

    Returns how many were cleared, so a caller can log it and a test can see it happened.
    """
    ids = (
        await db.execute(select(Campaign.id).where(Campaign.project_id == project_id))
    ).scalars().all()
    for campaign_id in ids:
        await invalidate_campaign_context(str(campaign_id))
    return len(ids)
