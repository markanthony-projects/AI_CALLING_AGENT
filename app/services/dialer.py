import httpx
from app.core.config import settings
from app.core.security import issue_call_token
from loguru import logger

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
        "hangup_on_ring": settings.VOBIZ_RING_SECONDS,
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
                logger.info(f"Successfully triggered Vobiz call for {customer_number}")
                return True
            else:
                logger.error(f"Vobiz call failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error calling Vobiz: {e}")
            return False
