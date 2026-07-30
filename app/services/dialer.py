import httpx
from app.core.config import settings
from app.core.security import issue_call_token
from loguru import logger

async def trigger_vobiz_call(customer_number: str, campaign_id: str, call_sid: str) -> bool:
    """Trigger outbound call via Vobiz AI REST API"""
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
    
    data = {
        "to": customer_number,
        "from": settings.VOBIZ_PHONE_NUMBER,
        "answer_url": answer_url
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
