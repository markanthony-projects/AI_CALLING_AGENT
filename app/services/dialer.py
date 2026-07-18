import httpx
from app.core.config import settings
from loguru import logger

async def trigger_exotel_call(customer_number: str, campaign_id: str, call_sid: str) -> bool:
    """Trigger outbound call via Exotel REST API"""
    if not settings.EXOTEL_ACCOUNT_SID or not settings.EXOTEL_API_KEY:
        logger.error("Exotel credentials missing in .env")
        return False
        
    url = f"https://api.exotel.com/v1/Accounts/{settings.EXOTEL_ACCOUNT_SID}/Calls.json"
    auth = (settings.EXOTEL_API_KEY, settings.EXOTEL_API_TOKEN)
    
    # URL that Exotel will hit when the customer answers the phone
    answer_url = f"{settings.WEBHOOK_BASE_URL}/exotel/answer/{campaign_id}/{call_sid}"
    
    data = {
        "From": customer_number,
        "To": settings.EXOTEL_CALLER_ID,
        "CallerId": settings.EXOTEL_CALLER_ID,
        "Url": answer_url,
        "CallType": "trans",
        "Record": "true"
    }
    
    logger.info(f"Triggering Exotel outbound call to {customer_number}")
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, auth=auth, data=data)
            if response.status_code == 200:
                logger.info(f"Successfully triggered Exotel call for {customer_number}")
                return True
            else:
                logger.error(f"Exotel call failed: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error calling Exotel: {e}")
            return False
