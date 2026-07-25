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

async def get_redis_client():
    global _redis_client
    if not _redis_client:
        _redis_client = redis.from_url(settings.REDIS_URL)
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

async def invalidate_project_cache(campaign_id: str):
    _l1_project_cache.pop(campaign_id, None)
    try:
        cache = await get_redis_client()
        await cache.delete(f"project_context:{campaign_id}")
    except Exception as e:
        logger.warning(f"Redis cache invalidate failed: {e}")
    logger.info(f"Invalidated L1 & L2 cache for campaign {campaign_id}")
