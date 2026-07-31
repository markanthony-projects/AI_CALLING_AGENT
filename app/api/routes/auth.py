"""Dashboard login.

Sessions live in an httpOnly cookie rather than a token the page holds in JavaScript. The
dashboard reads every lead's phone number and every call transcript, so an XSS on that page
should not also hand the attacker a portable credential they can replay from elsewhere.
"""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from loguru import logger
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.core.config import settings
from app.core.passwords import hash_password, needs_rehash, verify_password
from app.core.security import (
    HOST_PREFIXED_COOKIE_NAME,
    SESSION_COOKIE_NAME,
    SessionClaims,
    issue_session_token,
    require_session,
    session_cookie_name,
    session_cookie_samesite,
)
from app.models.db import DashboardUser
from app.utils.timeutils import utc_now

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# Brute-force ceiling per email+IP. Generous enough that a person fumbling their password
# is never locked out, tight enough that an online guessing run is pointless.
_MAX_ATTEMPTS = 10
_ATTEMPT_WINDOW_SECONDS = 900

# Verifying against this costs the same as verifying a real password, so a wrong email and
# a wrong password take the same time. Without it, response latency enumerates valid users.
_DUMMY_HASH = hash_password("this-hash-exists-only-to-equalise-timing")


# Deliberately not pydantic's EmailStr: that pulls in email-validator to enforce RFC
# compliance on a field whose only job is to match a row this box provisioned. A shape
# check is what the form needs, and it keeps the dependency list where it is.
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)

    @field_validator("email")
    @classmethod
    def looks_like_an_email(cls, v: str) -> str:
        v = v.strip().lower()
        if not _EMAIL.match(v):
            raise ValueError("Enter a valid email address")
        return v


class UserOut(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    role: str


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=session_cookie_name(),
        value=token,
        max_age=settings.DASHBOARD_SESSION_TTL_SECONDS,
        httponly=True,
        secure=settings.DASHBOARD_COOKIE_SECURE,
        samesite=session_cookie_samesite(),
        path="/",
    )


async def _throttle(key: str) -> None:
    """Count failed logins in Redis. Fails open — Redis being down must not lock everyone out."""
    try:
        from app.core.queue import get_arq_pool

        redis = get_arq_pool()
        attempts = await redis.incr(f"login:fail:{key}")
        if attempts == 1:
            await redis.expire(f"login:fail:{key}", _ATTEMPT_WINDOW_SECONDS)
        if attempts > _MAX_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed sign-in attempts. Try again later.",
                headers={"Retry-After": str(_ATTEMPT_WINDOW_SECONDS)},
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - availability of login outranks the throttle
        logger.warning(f"Login throttle unavailable: {exc}")


async def _clear_throttle(key: str) -> None:
    try:
        from app.core.queue import get_arq_pool

        await get_arq_pool().delete(f"login:fail:{key}")
    except Exception:  # noqa: BLE001
        pass


@router.post("/login", response_model=UserOut)
async def login(
    req: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
):
    if not settings.dashboard_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dashboard is not configured on this deployment",
        )

    email = req.email.strip().lower()
    client_ip = request.client.host if request.client else "unknown"
    throttle_key = f"{email}|{client_ip}"
    await _throttle(throttle_key)

    user = (
        await db.execute(select(DashboardUser).where(DashboardUser.email == email))
    ).scalars().first()

    # Always run a verification, even with no user, so the timing does not differ.
    stored = user.password_hash if user else _DUMMY_HASH
    password_ok = verify_password(req.password, stored)

    if not user or not password_ok or not user.is_active:
        logger.warning(f"Failed dashboard login for {email!r} from {client_ip}")
        # One message for every failure mode: naming which part was wrong tells an attacker
        # which emails are registered.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password"
        )

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(req.password)
    user.last_login_at = utc_now()
    await db.commit()

    await _clear_throttle(throttle_key)
    _set_session_cookie(response, issue_session_token(str(user.id), user.email, user.role.value))
    logger.info(f"Dashboard login: {user.email} ({user.role.value}) from {client_ip}")

    return UserOut(
        id=str(user.id), email=user.email, full_name=user.full_name, role=user.role.value
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout():
    # Deleted unconditionally: a caller with an already-expired or absent cookie still
    # expects logout to succeed rather than 401 them into a dead end. Both names are cleared
    # so a session survives neither a DASHBOARD_COOKIE_SECURE flip nor a stale duplicate.
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    for name in (HOST_PREFIXED_COOKIE_NAME, SESSION_COOKIE_NAME):
        response.delete_cookie(
            key=name,
            path="/",
            httponly=True,
            secure=settings.DASHBOARD_COOKIE_SECURE,
            samesite=session_cookie_samesite(),
        )
    return response


@router.get("/me", response_model=UserOut)
async def me(claims: SessionClaims = Depends(require_session), db: AsyncSession = Depends(get_db)):
    """Resolve the session against the database, not just the token.

    The token is valid for its whole TTL, so a user deactivated an hour ago would still be
    carrying a signed session. Re-reading the row is what actually revokes access.
    """
    user = await db.get(DashboardUser, claims.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session is no longer valid")
    return UserOut(
        id=str(user.id), email=user.email, full_name=user.full_name, role=user.role.value
    )
