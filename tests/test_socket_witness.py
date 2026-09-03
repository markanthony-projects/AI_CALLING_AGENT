"""Why a call's media stream died, kept where somebody can read it.

Twice — 2 Sep and 3 Sep 2026 — a live call ended when the websocket to Vobiz went away
mid-conversation. The second happened with somebody standing next to the phone, so it is
settled that the prospect did not hang up. Both left the same three lines:

    Media stream closed — ending pipeline
    Carrier hung up | answered=True | cause=ORIGINATOR_CANCEL
    Call cut off mid-conversation | the agent never closed it

None of which says whether the peer closed on purpose or the connection died — and those
two lead to completely different places. The close code answers it, and Pipecat's message
iterator was throwing it away one layer below us.
"""

import asyncio

import pytest

from app.utils.socket_witness import SocketWitness, close_meaning


class FakeSocket:
    """The parts of a Starlette WebSocket this wrapper is asked to stand in front of."""

    def __init__(self, messages):
        self.messages = list(messages)
        self.client_state = "CONNECTED"
        self.sent = []

    async def receive(self):
        return self.messages.pop(0)

    async def send_bytes(self, data):
        self.sent.append(data)

    async def close(self):
        self.client_state = "CLOSED"


def drain(witness, count):
    async def _run():
        for _ in range(count):
            await witness.receive()

    asyncio.run(_run())


def audio(n=640):
    return {"type": "websocket.receive", "text": "x" * n}


# --- the number the whole thing exists for ----------------------------------------------


def test_the_close_code_survives_the_close():
    witness = SocketWitness(FakeSocket([audio(), {"type": "websocket.disconnect", "code": 1006}]))
    drain(witness, 2)
    assert witness.close_code == 1006


def test_the_close_reason_is_kept_when_the_peer_sends_one():
    witness = SocketWitness(
        FakeSocket([{"type": "websocket.disconnect", "code": 1011, "reason": "stream error"}])
    )
    drain(witness, 1)
    assert witness.close_reason == "stream error"


def test_an_empty_reason_reads_as_absent():
    """'' in a log line looks like something was recorded when nothing was."""
    witness = SocketWitness(FakeSocket([{"type": "websocket.disconnect", "code": 1000, "reason": ""}]))
    drain(witness, 1)
    assert witness.close_reason is None


@pytest.mark.parametrize(
    "code,word",
    [
        (1000, "deliberately"),
        (1001, "going away"),
        (1006, "the connection itself died"),
        (1011, "internal error"),
        (1013, "overloaded"),
    ],
)
def test_every_code_that_decides_the_investigation_is_named(code, word):
    """Nobody reading this at 2am should have to look up an RFC. 1000 and 1001 send us to
    Vobiz; 1006 sends us to the network path between them and the droplet."""
    assert word in close_meaning(code)


def test_an_unrecognised_code_says_so_rather_than_guessing():
    assert "unrecognised" in close_meaning(4999)


def test_no_disconnect_frame_means_this_side_closed_first():
    """The ordinary end_call ending. It has to read differently from a drop, or the two are
    indistinguishable in a log grep — which is the state this replaces."""
    witness = SocketWitness(FakeSocket([audio()]))
    drain(witness, 1)
    assert witness.close_code is None
    assert "this side closed first" in witness.report()


# --- was audio still arriving when it died? ----------------------------------------------


def test_the_gap_between_the_last_frame_and_the_close_is_measured():
    """A 1006 twenty milliseconds after a frame is a connection cut out from under a healthy
    stream. The same code after two seconds of silence says Vobiz stopped sending first."""
    ticks = iter([100.0, 101.0, 103.4])  # opened, frame, disconnect

    witness = SocketWitness(
        FakeSocket([audio(), {"type": "websocket.disconnect", "code": 1006}]),
        clock=lambda: next(ticks),
    )
    drain(witness, 2)
    assert witness.silence_before_close == pytest.approx(2.4)
    assert "2400ms before close" in witness.report()


def test_no_gap_is_reported_when_nothing_ever_arrived():
    witness = SocketWitness(FakeSocket([{"type": "websocket.disconnect", "code": 1006}]))
    drain(witness, 1)
    assert witness.silence_before_close is None
    assert "no inbound frame" in witness.report()


def test_inbound_traffic_is_counted():
    witness = SocketWitness(FakeSocket([audio(1024), audio(1024), {"type": "websocket.disconnect", "code": 1000}]))
    drain(witness, 3)
    assert witness.inbound_messages == 2
    assert witness.inbound_bytes == 2048


def test_binary_frames_are_counted_too():
    """Vobiz sends text today. A serializer change must not silently zero this."""
    witness = SocketWitness(FakeSocket([{"type": "websocket.receive", "bytes": b"\x00" * 320}]))
    drain(witness, 1)
    assert witness.inbound_bytes == 320


def test_the_disconnect_frame_is_not_counted_as_traffic():
    witness = SocketWitness(FakeSocket([{"type": "websocket.disconnect", "code": 1006}]))
    drain(witness, 1)
    assert witness.inbound_messages == 0


# --- it must not change the socket it wraps ----------------------------------------------


def test_the_message_is_passed_through_unchanged():
    """The transport reads this dictionary itself. Observing it must not consume it."""
    sent = {"type": "websocket.receive", "text": "hello"}
    witness = SocketWitness(FakeSocket([sent]))

    async def _run():
        return await witness.receive()

    assert asyncio.run(_run()) is sent


def test_everything_else_falls_through_to_the_real_socket():
    """Pipecat calls send_bytes, close and client_state on this object. If any of them
    stopped resolving, the wrapper would take down every call rather than diagnose one."""
    real = FakeSocket([])
    witness = SocketWitness(real)

    async def _run():
        await witness.send_bytes(b"audio")
        await witness.close()

    asyncio.run(_run())
    assert real.sent == [b"audio"]
    assert witness.client_state == "CLOSED"


# --- and it has to actually be wired in ---------------------------------------------------


def test_the_transport_is_given_the_witness_and_not_the_raw_socket():
    """The helper being right is worth nothing if the transport still holds the bare socket."""
    import ast
    import inspect

    from app.services import agent

    tree = ast.parse(inspect.getsource(agent.run_voice_agent).lstrip())
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "FastAPIWebsocketTransport"
    )
    passed = next(k.value for k in call.keywords if k.arg == "websocket")
    assert isinstance(passed, ast.Name), ast.unparse(passed)

    wrapped = [
        ast.unparse(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "id", None) == passed.id for t in n.targets)
    ]
    assert wrapped == ["SocketWitness(websocket)"], wrapped


def test_the_close_is_reported_on_its_own_countable_line():
    """`SOCKET` appears nowhere else in these logs, so one grep counts every drop."""
    import inspect

    from app.services import agent

    src = inspect.getsource(agent.run_voice_agent)
    handler = src[src.index("async def on_client_disconnected"):]
    handler = handler[: handler.index("# ─── Hard Duration Cap")]
    assert "SOCKET closed" in handler
    assert "socket.report()" in handler
