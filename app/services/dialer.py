from typing import Optional

import httpx
from app.core.config import settings
from app.core.security import issue_call_token
from loguru import logger

def vobiz_request_uuid(response) -> Optional[str]:
    """The carrier's own identifier for a call it has just accepted.

    Best effort on purpose: this exists to make a support ticket possible, and a dial that
    worked must never fail because the reply was shaped differently than expected.
    """
    try:
        body = response.json()
    except Exception:
        return None
    if not isinstance(body, dict):
        return None
    for key in ("request_uuid", "requestUuid", "RequestUUID", "call_uuid", "CallUUID"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        # Some carriers return one entry per destination number.
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0].strip()
    return None


async def trigger_vobiz_call(customer_number: str, campaign_id: str, call_sid: str) -> bool:
    """Trigger outbound call via Vobiz AI REST API"""
    # Imported here rather than at module scope: app.services.agent pulls in the whole
    # pipecat runtime, and the worker imports this module only to place dials.
    from app.services.agent import MAX_CALL_DURATION_SECS

    if not settings.VOBIZ_AUTH_ID or not settings.VOBIZ_AUTH_TOKEN or not settings.VOBIZ_PHONE_NUMBER:
        logger.error("Vobiz credentials missing in .env")
        return False
        
    # Vobiz outbound call endpoint requires Auth ID in the path
    url = f"https://api.vobiz.ai/api/v1/Account/{settings.VOBIZ_AUTH_ID}/Call/"
    
    headers = {
        "X-Auth-ID": settings.VOBIZ_AUTH_ID,
        "X-Auth-Token": settings.VOBIZ_AUTH_TOKEN,
        "Content-Type": "application/json"
    }
    base_url = settings.WEBHOOK_BASE_URL.rstrip('/')
    
    # URL that Vobiz will hit when the customer answers the phone
    token = issue_call_token(campaign_id, call_sid)
    answer_url = f"{base_url}/vobiz/answer/{campaign_id}/{call_sid}?token={token}"

    # The hangup callback is the only thing that reports a call nobody answered. It arrives
    # after the call is over, so its token has to outlive the call — a default-TTL token
    # would expire during a long one and the callback would be refused, which is worse than
    # not asking for it: the slot would then be held until it aged out and the contact would
    # sit in DIALING until the reaper found it.
    hangup_token = issue_call_token(
        campaign_id, call_sid, ttl_seconds=int(MAX_CALL_DURATION_SECS) + 1800
    )
    hangup_url = f"{base_url}/vobiz/hangup/{campaign_id}/{call_sid}?token={hangup_token}"

    data = {
        "to": customer_number,
        "from": settings.VOBIZ_PHONE_NUMBER,
        "answer_url": answer_url,
        # Without this a ring-out is invisible: the media websocket only opens on answer, so
        # nothing tells us the call happened, the carrier slot stays held until it ages out,
        # and the contact sits in DIALING until the reaper sweeps it twenty minutes later.
        "hangup_url": hangup_url,
        "hangup_method": "POST",
        # Stop ringing long before the carrier would. See VOBIZ_RING_SECONDS.
        #
        # ring_timeout, NOT hangup_on_ring. They sound interchangeable and are not:
        #
        #   ring_timeout     how long the destination may ring before the call is abandoned
        #   hangup_on_ring   "schedules the call for hangup at a specified time after the
        #                    call starts ringing" — a scheduled hangup, with no condition
        #                    that the call still be ringing when it fires
        #
        # We had the second one, which is why live conversations died. On 2 Sep 2026 a call
        # rang at 11:53:24, was answered at 11:53:35, and was cut at 11:54:09 with both
        # parties mid-sentence — 45 seconds after ring start, to the second, which is the
        # value below. The carrier's own cause code says "Scheduled Hangup", which is this
        # parameter's own name. Three calls died this way before the arithmetic was spotted,
        # because the obvious place to measure from is the answer, and from there the times
        # looked unrelated: 33.9s, 38.8s, 39.1s. From ring start they were all exactly 45.
        "ring_timeout": settings.VOBIZ_RING_SECONDS,
        # Vobiz defaults this to four hours. Our own pipeline hangs up at
        # MAX_CALL_DURATION_SECS, so this only matters when that pipeline is the thing that
        # died — and then it is the difference between a stuck leg costing seconds and one
        # billing until somebody notices. The margin lets our cap win in the normal case, so
        # the caller hears a goodbye rather than the line vanishing.
        "time_limit": int(MAX_CALL_DURATION_SECS) + 60,
    }
    
    logger.info(f"Triggering Vobiz outbound call to {customer_number}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data)
            if response.status_code in (200, 201):
                # The carrier's own id for this call, logged beside ours. Vobiz does not know
                # our call_sid — it is a UUID we mint and put in the callback URLs — so every
                # support ticket about a specific call meant opening their dashboard by hand
                # to find their identifier. The reply carries it and we were discarding it.
                logger.info(
                    f"Successfully triggered Vobiz call for {customer_number}"
                    f" | vobiz_request_uuid={vobiz_request_uuid(response) or 'unreported'}"
                )
                return True
            else:
                logger.error(f"Vobiz call failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error calling Vobiz: {e}")
            return False
