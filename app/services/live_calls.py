"""The sessions this process is serving, so a carrier event can reach the one it is about.

Call 8571d93b, 15 Sep 2026. The prospect hung up eight seconds in. The carrier's hangup
callback said so at once — answered, NORMAL_CLEARING — and the media websocket stayed
open, so the pipeline sat on a dead line for sixty seconds until the idle timeout gave up
and filed the call as FAILED. The hangup route released the slot and updated the ledger
and had no way to tell the session it was over.

This is that way. A session registers how to end itself when it starts and forgets it when
it stops; the hangup route ends whichever session the callback names. In-process only, which
is the deployment: one API container serves every stream. The slot and the ledger are still
handled where they were — this only shortens the wait.
"""

from typing import Awaitable, Callable, Dict

from loguru import logger

Ender = Callable[[str], Awaitable[None]]

_enders: Dict[str, Ender] = {}


def register(call_sid: str, ender: Ender) -> None:
    _enders[call_sid] = ender


def forget(call_sid: str) -> None:
    _enders.pop(call_sid, None)


async def end(call_sid: str, reason: str) -> bool:
    """End the live session for this call, if this process has one. True if it did."""
    ender = _enders.pop(call_sid, None)
    if ender is None:
        return False
    try:
        await ender(reason)
    except Exception as e:  # noqa: BLE001 — a session already stopping is not an error here
        logger.debug(f"[{call_sid}] Could not end the session ({e}); it is probably ending")
        return False
    return True


def live() -> int:
    return len(_enders)
