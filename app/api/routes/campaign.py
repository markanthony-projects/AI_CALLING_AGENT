from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List
import uuid

from app.api.dependencies import get_db
from app.models.db import Campaign, Project
from app.services.dialer import trigger_exotel_call

router = APIRouter(prefix="/api/v1/campaigns", tags=["campaigns"])

class CampaignCreate(BaseModel):
    project_id: str
    name: str

class DialRequest(BaseModel):
    phone_numbers: List[str]

@router.post("/")
async def create_campaign(req: CampaignCreate, db: AsyncSession = Depends(get_db)):
    project = await db.get(Project, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
        
    campaign = Campaign(project_id=project.id, name=req.name)
    db.add(campaign)
    await db.commit()
    await db.refresh(campaign)
    return {"id": campaign.id, "name": campaign.name, "status": campaign.status}

@router.post("/{campaign_id}/dial/exotel")
async def dial_campaign_exotel(campaign_id: str, req: DialRequest, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")
        
    for number in req.phone_numbers:
        call_sid = str(uuid.uuid4())
        background_tasks.add_task(trigger_exotel_call, number, str(campaign_id), call_sid)
        
    campaign.total_leads_dialed = (campaign.total_leads_dialed or 0) + len(req.phone_numbers)
    await db.commit()
    
    return {"status": "dialing_started", "total_numbers": len(req.phone_numbers)}

@router.post("/{campaign_id}/dial/browser")
async def get_browser_call_link(campaign_id: str):
    """
    Browser based call testing endpoint. Returns a link to start the call in the web UI.
    """
    call_sid = str(uuid.uuid4())
    link = f"/static/index.html?campaign_id={campaign_id}&call_sid={call_sid}"
    return {"browser_test_link": link, "call_sid": call_sid}
