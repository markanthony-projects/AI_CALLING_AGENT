import re
import uuid
from typing import Annotated, List

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import AfterValidator, BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import settings
from app.core.ratelimit import reserve_dial_quota
from app.core.security import issue_call_token, require_api_key
from app.models.db import Campaign, Project
from app.services.call_context import remember_dialed_number
from app.services.dialer import trigger_vobiz_call

router = APIRouter(prefix="/api/v1/campaigns", tags=["campaigns"], dependencies=[Depends(require_api_key)])

MAX_DIAL_BATCH = 500

_E164 = re.compile(r"^\+[1-9]\d{7,14}$")
_SEPARATORS = re.compile(r"[\s\-().]")


def to_e164(raw: str) -> str:
    """Accept the formats a lead list actually contains and return strict E.164.

    Spreadsheets and CRM exports carry numbers as '98765 43210', '098765-43210' or
    '91 9876543210'; dialing must not fail on punctuation the operator never chose.
    """
    cc = settings.DEFAULT_COUNTRY_CODE
    cleaned = _SEPARATORS.sub("", raw.strip())

    if cleaned.startswith("+"):
        candidate = cleaned
    else:
        digits = cleaned.lstrip("0")  # national trunk prefix, or 00 international prefix
        # A bare national number is 10 digits; only a longer one can already carry
        # the country code. Guards against 10-digit mobiles that start with 91.
        if len(digits) > 10 and digits.startswith(cc):
            candidate = f"+{digits}"
        else:
            candidate = f"+{cc}{digits}"

    if not _E164.match(candidate):
        raise ValueError(
            f"'{raw}' is not a dialable number. Use E.164 (+919876543210) "
            f"or a local number that a +{cc} prefix completes."
        )
    return candidate


PhoneNumber = Annotated[str, AfterValidator(to_e164)]


class CampaignCreate(BaseModel):
    project_id: uuid.UUID
    name: str = Field(min_length=1, max_length=200)


class DialRequest(BaseModel):
    phone_numbers: List[PhoneNumber] = Field(min_length=1, max_length=MAX_DIAL_BATCH)


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

    call_sids = []
    for number in req.phone_numbers:
        call_sid = str(uuid.uuid4())
        call_sids.append(call_sid)
        # Recorded before dialling: the Call row is not written until the media stream
        # opens, and by then this request is long gone.
        await remember_dialed_number(call_sid, number)
        background_tasks.add_task(trigger_vobiz_call, number, str(campaign_id), call_sid)

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
