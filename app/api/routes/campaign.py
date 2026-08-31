import uuid
from typing import Annotated, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AfterValidator, BaseModel, BeforeValidator, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import settings
from app.services import dial_queue
from app.core.security import issue_call_token, require_api_key
from app.models.db import Campaign, Project
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
    db: AsyncSession = Depends(get_db),
):
    """Add numbers to the campaign's dial queue, for callers holding the API key.

    This used to dial. It created one background task per number — up to five hundred — and
    asked Vobiz to call every one at once. The concurrency cap was still enforced, but at the
    media websocket, which opens only after the carrier has dialled, billed us and rung a
    real person; everyone past the third slot was called, charged for, and hung up on with no
    Call row to say it had happened. The dashboard route was rebuilt to fix exactly that, and
    this door was left open behind it — with the deployment guide pointing operators at it.

    It now enqueues through the same service the dashboard uses, so the carrier limit is
    reserved before any number is dialled. The response no longer carries call_sids: no call
    has been placed yet, and returning identifiers for calls that do not exist is what let
    the old shape read as success.
    """
    report = await dial_queue.enqueue(db, campaign_id, req.phone_numbers, requested_by="api key")
    return report.as_response()


@router.post("/{campaign_id}/dial/browser")
async def get_browser_call_link(campaign_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    campaign = await db.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(status_code=404, detail="Campaign not found")

    call_sid = str(uuid.uuid4())
    token = issue_call_token(str(campaign_id), call_sid)
    link = f"/static/index.html?campaign_id={campaign_id}&call_sid={call_sid}&token={token}"
    return {"browser_test_link": link, "call_sid": call_sid, "expires_in": settings.CALL_TOKEN_TTL_SECONDS}
