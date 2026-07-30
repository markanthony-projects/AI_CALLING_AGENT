import secrets
import time
from typing import Optional

import jwt
from fastapi import HTTPException, Query, Security, WebSocketException, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

_ALGORITHM = "HS256"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def issue_call_token(campaign_id: str, call_sid: str, ttl_seconds: Optional[int] = None) -> str:
    """Mint a short-lived token binding a telephony callback to one campaign and call."""
    now = int(time.time())
    ttl = settings.CALL_TOKEN_TTL_SECONDS if ttl_seconds is None else ttl_seconds
    return jwt.encode(
        {"cid": campaign_id, "sub": call_sid, "iat": now, "exp": now + ttl},
        settings.CALL_TOKEN_SECRET,
        algorithm=_ALGORITHM,
    )


def verify_call_token(token: str, campaign_id: str, call_sid: str) -> bool:
    if not token:
        return False
    try:
        claims = jwt.decode(token, settings.CALL_TOKEN_SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return False
    return claims.get("cid") == campaign_id and claims.get("sub") == call_sid


async def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    if not settings.AUTH_ENABLED:
        return
    if not api_key or not secrets.compare_digest(api_key, settings.API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


async def require_call_token(campaign_id: str, call_sid: str, token: str = Query(default="")) -> None:
    if not settings.AUTH_ENABLED:
        return
    if not verify_call_token(token, campaign_id, call_sid):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid or expired call token")


async def require_call_token_ws(campaign_id: str, call_sid: str, token: str = Query(default="")) -> None:
    if not settings.AUTH_ENABLED:
        return
    if not verify_call_token(token, campaign_id, call_sid):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid or expired call token")
