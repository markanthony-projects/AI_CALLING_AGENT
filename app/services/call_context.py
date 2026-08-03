"""Carries the dialled number from the dial request to the call that answers it.

The number is known when we ask Vobiz to dial, but the Call row is not written until the
media stream opens, in a different request. Redis bridges the two. It is deliberately not
the system of record: the number is copied onto the Call row as soon as the stream opens,
and losing the key only costs us the number on that one call.
"""

from typing import Optional

from loguru import logger

from app.services.discovery import get_redis_client

# Comfortably longer than the gap between dialling and answering, short enough that
# abandoned dials do not accumulate.
_TTL_SECONDS = 3600


def _key(call_sid: str) -> str:
    return f"call_phone:{call_sid}"


def _name_key(call_sid: str) -> str:
    return f"call_name:{call_sid}"


async def remember_dialed_number(call_sid: str, phone_number: str) -> None:
    try:
        client = await get_redis_client()
        await client.setex(_key(call_sid), _TTL_SECONDS, phone_number)
    except Exception as e:
        logger.warning(f"[{call_sid}] Could not record dialled number: {e}")


async def recall_dialed_number(call_sid: str) -> Optional[str]:
    try:
        client = await get_redis_client()
        value = await client.get(_key(call_sid))
    except Exception as e:
        logger.warning(f"[{call_sid}] Could not recall dialled number: {e}")
        return None
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


async def remember_customer_name(call_sid: str, customer_name: Optional[str]) -> None:
    """Carry the lead-list name across to the call, so the agent can greet by name.

    Optional by design: a dial list without names still dials, the agent just asks for the
    name on the call instead of confirming one.
    """
    name = (customer_name or "").strip()
    if not name:
        return
    try:
        client = await get_redis_client()
        await client.setex(_name_key(call_sid), _TTL_SECONDS, name)
    except Exception as e:
        logger.warning(f"[{call_sid}] Could not record customer name: {e}")


async def recall_customer_name(call_sid: str) -> Optional[str]:
    try:
        client = await get_redis_client()
        value = await client.get(_name_key(call_sid))
    except Exception as e:
        logger.warning(f"[{call_sid}] Could not recall customer name: {e}")
        return None
    if value is None:
        return None
    return value.decode() if isinstance(value, bytes) else str(value)


async def forget_dialed_number(call_sid: str) -> None:
    try:
        client = await get_redis_client()
        await client.delete(_key(call_sid), _name_key(call_sid))
    except Exception:
        pass
