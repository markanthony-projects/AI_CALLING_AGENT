import secrets
import time
from typing import Optional

import jwt
from fastapi import Cookie, Depends, HTTPException, Query, Request, Security, WebSocketException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from fastapi.security import APIKeyHeader

from app.core.config import settings

_ALGORITHM = "HS256"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Named so it cannot collide with the call token. In production the __Host- prefix is used,
# which the browser enforces to mean Secure, Path=/ and no Domain — so a sibling subdomain
# cannot write a session for this one. That prefix is illegal over plain HTTP, so local dev
# falls back to the bare name.
SESSION_COOKIE_NAME = "cai_session"
HOST_PREFIXED_COOKIE_NAME = f"__Host-{SESSION_COOKIE_NAME}"


def session_cookie_name() -> str:
    return HOST_PREFIXED_COOKIE_NAME if settings.DASHBOARD_COOKIE_SECURE else SESSION_COOKIE_NAME


def session_cookie_samesite() -> str:
    """Read from configuration, not inferred from whether CORS is on.

    Deriving it from the CORS list conflated two different questions. A dashboard on
    dashboard.homebble.in calling ai-calls.homebble.in is cross-ORIGIN but same-SITE, so Lax
    still works and the browser keeps its own CSRF protection. Only a genuinely cross-site
    dashboard needs None, and that makes the session a third-party cookie — which Safari and
    Firefox block by default.

    Whichever it is, every unsafe method also verifies the Origin header below.
    """
    return settings.DASHBOARD_COOKIE_SAMESITE


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


# --- Dashboard sessions -------------------------------------------------------------
#
# Signed with DASHBOARD_SESSION_SECRET rather than CALL_TOKEN_SECRET so the two cannot be
# swapped: a call token is handed to a telephony vendor and appears in URLs, and must never
# be presentable as a dashboard login.


class SessionClaims:
    __slots__ = ("user_id", "email", "role")

    def __init__(self, user_id: str, email: str, role: str) -> None:
        self.user_id = user_id
        self.email = email
        self.role = role

    @property
    def is_admin(self) -> bool:
        return self.role == "ADMIN"


def issue_session_token(user_id: str, email: str, role: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            "sub": user_id,
            "email": email,
            "role": role,
            "iat": now,
            "exp": now + settings.DASHBOARD_SESSION_TTL_SECONDS,
        },
        settings.DASHBOARD_SESSION_SECRET,
        algorithm=_ALGORITHM,
    )


def decode_session_token(token: str) -> Optional[SessionClaims]:
    if not token or not settings.dashboard_enabled:
        return None
    try:
        claims = jwt.decode(token, settings.DASHBOARD_SESSION_SECRET, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        return None
    user_id, email = claims.get("sub"), claims.get("email")
    if not user_id or not email:
        return None
    return SessionClaims(user_id, email, claims.get("role", "VIEWER"))


_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def _assert_same_origin(request: "Request") -> None:
    """Reject a cross-site write.

    Cookie auth means the browser attaches the session to a request the user never made.
    SameSite=Lax covers the same-origin deployment, but a cross-origin dashboard needs
    SameSite=None, and then only this check stands between a malicious page and a POST that
    dials. Origin is set by the browser on every unsafe method and cannot be forged by
    script, so an unexpected value is a request no dashboard made.
    """
    if request.method in _SAFE_METHODS:
        return

    origin = request.headers.get("origin")
    if origin is None:
        # Same-origin XHR from some browsers, and every non-browser client (curl, tests).
        # Neither can be a cross-site forgery: CSRF requires a browser, and browsers send
        # Origin on cross-site writes.
        return

    allowed = set(settings.dashboard_cors_origins)
    forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host")
    if forwarded_host:
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        allowed.add(f"{proto}://{forwarded_host}")

    if origin.rstrip("/") not in allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-origin request rejected",
        )


async def require_session(
    request: Request,
    host_session: Optional[str] = Cookie(default=None, alias=HOST_PREFIXED_COOKIE_NAME),
    session: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    db: AsyncSession = Depends(get_db),
) -> SessionClaims:
    """Authenticate a dashboard request from its cookie.

    Unlike require_api_key this ignores AUTH_ENABLED. That flag exists so a developer can
    place test calls without a key; it is not a reason to serve every lead's phone number
    and transcript to an unauthenticated request.

    Both cookie names are accepted so that flipping DASHBOARD_COOKIE_SECURE does not sign
    every open session out mid-shift.
    """
    if not settings.dashboard_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard is not configured on this deployment",
        )

    _assert_same_origin(request)

    claims = decode_session_token(host_session or session or "")
    if claims is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )

    # The token stays valid for its whole TTL, so without this a user deactivated an hour ago
    # keeps reading every lead's phone number and every transcript for the rest of the day.
    # One indexed primary-key lookup on a table with a handful of rows is the price of
    # revocation actually revoking. Role is re-read too, so a demotion takes effect at once
    # rather than leaving a former admin able to dial until their session expires.
    from app.models.db import DashboardUser

    user = await db.get(DashboardUser, claims.user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is no longer valid"
        )
    return SessionClaims(str(user.id), user.email, user.role.value)


async def require_admin(claims: SessionClaims = Depends(require_session)) -> SessionClaims:
    """Guards everything that spends money or changes state."""
    if not claims.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This action requires an admin account",
        )
    return claims
