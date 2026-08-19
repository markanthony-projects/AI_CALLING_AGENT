import uuid
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import settings
from app.core.ratelimit import reserve_dial_quota, reserve_llm_headroom
from app.core.security import issue_call_token, require_api_key
from app.models.db import Campaign, Project
from app.services.call_context import remember_customer_name, remember_dialed_number
from app.services.dialer import trigger_vobiz_call
from app.utils.phone import to_e164

router = APIRouter(prefix="/api/v1/campaigns", tags=["campaigns"], dependencies=[Depends(require_api_key)])

MAX_DIAL_BATCH = 500

PhoneNumber = Annotated[str, AfterValidator(to_e164)]


class CampaignCreate(BaseModel):
    project_id: uuid.UUID
    name: str = Field(min_length=1, max_length=200)


class DialTarget(BaseModel):
    """One lead to call: the number, and the name to greet them by if the list has it."""

    number: PhoneNumber
    name: Optional[str] = Field(default=None, max_length=100)


def _as_dial_target(value: Union[str, dict]) -> Union[dict, str]:
    """Accept a bare number as well as {"name": ..., "number": ...}.

    The CRM export now carries names, but existing integrations post a flat list of
    strings and must keep dialing unchanged.
    """
    if isinstance(value, str):
        return {"number": value}
    if isinstance(value, dict) and "number" not in value:
        for alias in ("phone_number", "phone", "mobile"):
            if alias in value:
                return {**value, "number": value[alias]}
    return value


DialEntry = Annotated[DialTarget, BeforeValidator(_as_dial_target)]


class DialRequest(BaseModel):
    phone_numbers: List[DialEntry] = Field(min_length=1, max_length=MAX_DIAL_BATCH)


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


@router.post("/{campaign_id}/dial/vobiz")
async def dial_campaign_vobiz(
    campaign_id: uuid.UUID,
    req: DialRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    # Charged before anything is dialled: once trigger_vobiz_call runs the money is spent.
    await reserve_dial_quota(len(req.phone_numbers))
    # Money is already committed by the line above; this asks the separate question
    # of whether the LLM can answer the greeting once the phone is picked up.
    await reserve_llm_headroom()

    call_sids = []
    for target in req.phone_numbers:
        call_sid = str(uuid.uuid4())
        call_sids.append(call_sid)
        # Recorded before dialling: the Call row is not written until the media stream
        # opens, and by then this request is long gone. The name rides along the same way,
        # so the agent can open with "Am I speaking with ...?".
        await remember_dialed_number(call_sid, target.number)
        await remember_customer_name(call_sid, target.name)
        background_tasks.add_task(trigger_vobiz_call, target.number, str(campaign_id), call_sid)

    campaign.total_leads_dialed = (campaign.total_leads_dialed or 0) + len(call_sids)
    await db.commit()

    return {"status": "dialing_started", "total_numbers": len(call_sids), "call_sids": call_sids}


@router.post("/{campaign_id}/dial/browser")
async def get_browser_call_link(campaign_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    call_sid = str(uuid.uuid4())
    token = issue_call_token(str(campaign_id), call_sid)
    link = f"/static/index.html?campaign_id={campaign_id}&call_sid={call_sid}&token={token}"
    return {"browser_test_link": link, "call_sid": call_sid, "expires_in": settings.CALL_TOKEN_TTL_SECONDS}
