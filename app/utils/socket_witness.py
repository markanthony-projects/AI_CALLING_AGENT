"""Keeps the reason a call's media stream died, because nothing else does.

Twice — 2 Sep and 3 Sep 2026 — a live call ended when the websocket to Vobiz went away
mid-conversation. The second one happened with somebody sitting next to the phone, so it is
settled that the prospect did not hang up: our stream dropped and Vobiz then hung up on a
connected caller. Both calls left the same three lines and nothing else:

    Media stream closed — ending pipeline
    Carrier hung up | answered=True | cause=ORIGINATOR_CANCEL
    Call cut off mid-conversation | the agent never closed it

nginx recorded a clean 101 with no error, no timeout, and outbound audio flowing at 32.6
KB/s — full rate — right up to the last moment. So the fault is not upstream of nginx and
it is not us stalling. What is missing is the one number that separates the remaining
possibilities, and it was being thrown away one layer below us: the ASGI disconnect message
carries a close code, and Pipecat's message iterator turns it into a bare StopAsyncIteration.

    1000 / 1001  the peer closed on purpose      -> a Vobiz-side decision, take it to them
    1006         no close frame; TCP just died   -> the network path, not either application
    1011         the peer hit an internal error  -> a Vobiz-side fault

The gap between the last inbound frame and the close separates them further: audio arriving
20ms before a 1006 is a connection cut out from under a healthy stream, while silence for
two seconds first says Vobiz stopped sending before it stopped talking.

This wraps the socket rather than patching Pipecat: everything delegates to the real object,
and only `receive` is observed on the way past.
"""

import time
from typing import Optional

# https://www.rfc-editor.org/rfc/rfc6455#section-7.4.1, plus the two Starlette synthesises.
_CLOSE_CODES = {
    1000: "normal — the peer closed deliberately",
    1001: "going away — the peer's endpoint is shutting down or the call leg ended",
    1002: "protocol error — the peer rejected something we sent",
    1003: "unsupported data",
    1005: "no status — closed with an empty close frame",
    1006: "abnormal — no close frame arrived; the connection itself died",
    1007: "invalid payload",
    1008: "policy violation",
    1009: "message too big",
    1011: "internal error on the peer",
    1012: "service restart on the peer",
    1013: "try again later — the peer is overloaded",
    1015: "TLS handshake failure",
}


def close_meaning(code: Optional[int]) -> str:
    """What a websocket close code says about who ended the call, in words."""
    if code is None:
        return "no disconnect frame seen — this side closed first"
    return _CLOSE_CODES.get(code, "unrecognised close code")


class SocketWitness:
    """A pass-through wrapper around the call's WebSocket that remembers how it ended.

    Attribute lookups that are not ours fall through to the real socket, so the transport
    keeps using `client_state`, `send_bytes`, `send_text` and `close` exactly as before.
    """

    def __init__(self, websocket, clock=time.monotonic):
        # The clock is a parameter so a test can drive it. Patching time.monotonic globally
        # is not an option here: asyncio's own event loop calls it, and a scripted sequence
        # gets eaten by the loop before the code under test ever sees it.
        self._clock = clock
        self._ws = websocket
        self.close_code: Optional[int] = None
        self.close_reason: Optional[str] = None
        self.inbound_messages = 0
        self.inbound_bytes = 0
        self._opened_at = clock()
        self._last_inbound_at: Optional[float] = None
        self._closed_at: Optional[float] = None

    def __getattr__(self, name):
        # Only reached for names this object does not define, so every real socket method
        # and property still resolves to the socket itself.
        return getattr(self._ws, name)

    async def receive(self):
        message = await self._ws.receive()
        now = self._clock()
        kind = message.get("type") if isinstance(message, dict) else None

        if kind == "websocket.disconnect":
            self.close_code = message.get("code")
            # An empty reason is the norm and reads better as absent than as ''.
            self.close_reason = (message.get("reason") or "").strip() or None
            self._closed_at = now
        elif kind == "websocket.receive":
            self.inbound_messages += 1
            payload = message.get("bytes")
            # Vobiz sends text frames carrying base64 audio; length in characters is close
            # enough for "was audio still arriving", which is all this is asked.
            self.inbound_bytes += len(payload) if payload is not None else len(message.get("text") or "")
            self._last_inbound_at = now

        return message

    @property
    def silence_before_close(self) -> Optional[float]:
        """Seconds between the last inbound frame and the disconnect, if both happened."""
        if self._closed_at is None or self._last_inbound_at is None:
            return None
        return self._closed_at - self._last_inbound_at

    def report(self) -> str:
        """One line naming the close code, what it means, and the traffic behind it."""
        held = (self._closed_at or self._clock()) - self._opened_at
        code = "none" if self.close_code is None else str(self.close_code)
        parts = [f"code={code} ({close_meaning(self.close_code)})"]
        if self.close_reason:
            parts.append(f"reason={self.close_reason!r}")

        gap = self.silence_before_close
        if gap is None:
            parts.append("no inbound frame seen before the close")
        else:
            parts.append(f"last inbound {gap * 1000:.0f}ms before close")

        rate = (self.inbound_bytes / held / 1024) if held > 0 else 0
        parts.append(
            f"{self.inbound_messages} frames, {self.inbound_bytes / 1024:.0f}KB "
            f"over {held:.1f}s ({rate:.1f}KB/s in)"
        )
        return " | ".join(parts)
