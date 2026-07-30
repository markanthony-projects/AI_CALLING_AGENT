"""Regression tests for the answer webhook's call-control XML.

A live call was terminated ~2s in because <Hangup/> followed a <Stream> with
keepCallAlive="false". Per https://vobiz.ai/docs/xml/stream, false runs subsequent
elements *concurrently* with the stream; only "true" makes <Stream> execute exclusively
so that <Hangup/> waits for the stream to disconnect.
"""

import re
import uuid

import pytest

from app.api.routes import webhook
from app.core.security import issue_call_token

CID = str(uuid.uuid4())
SID = str(uuid.uuid4())


@pytest.fixture
def answer(client):
    def _get(call_sid=SID):
        return client.post(
            f"/vobiz/answer/{CID}/{call_sid}?token={issue_call_token(CID, call_sid)}"
        )

    return _get


@pytest.fixture(autouse=True)
def clean_streaming_set():
    webhook._STREAMING_CALLS.clear()
    yield
    webhook._STREAMING_CALLS.clear()


@pytest.fixture(autouse=True)
def no_prior_call(monkeypatch):
    """The route opens its own AsyncSessionLocal for the replay check; keep it off Postgres."""
    from contextlib import asynccontextmanager

    class Session:
        async def scalar(self, *a, **kw):
            return None

    @asynccontextmanager
    async def factory():
        yield Session()

    monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)


def test_stream_keeps_call_alive(answer, auth_enabled):
    """The one-word bug: false hangs up the caller while the agent is still talking."""
    body = answer().text
    assert 'keepCallAlive="true"' in body, (
        "Stream must execute exclusively, or the <Hangup/> after it fires concurrently "
        "and kills the live call"
    )
    assert 'keepCallAlive="false"' not in body


def test_hangup_follows_stream(answer, auth_enabled):
    """<Hangup/> after the stream is what ends the PSTN leg and stops billing."""
    body = answer().text
    assert "<Hangup/>" in body
    assert body.index("<Stream") < body.index("<Hangup/>")


def test_stream_is_bidirectional_16k(answer, auth_enabled):
    body = answer().text
    assert 'bidirectional="true"' in body
    assert "audio/x-l16;rate=16000" in body


def test_stream_url_carries_a_fresh_token(answer, auth_enabled):
    from app.core.security import verify_call_token

    body = answer().text
    m = re.search(r"/ws/vobiz/([^/]+)/([^?]+)\?token=([\w\-.]+)<", body)
    assert m, "stream URL must carry a call token"
    assert verify_call_token(m.group(3), CID, SID)


def test_mid_stream_refetch_does_not_hang_up(answer, auth_enabled):
    """The actual failure: the carrier re-asked for instructions 1.9s into a live call."""
    webhook._STREAMING_CALLS.add(SID)
    body = answer().text
    assert "<Hangup/>" not in body, "hanging up a live stream terminates the caller"
    assert "<Stream" not in body, "a second stream would open a duplicate leg"


def test_replay_after_call_ended_hangs_up(answer, auth_enabled, monkeypatch):
    """Once the stream is gone, a re-fetch is a genuine replay and must end the leg."""
    from contextlib import asynccontextmanager

    class Session:
        async def scalar(self, *a, **kw):
            return uuid.uuid4()  # a Call row exists

    @asynccontextmanager
    async def factory():
        yield Session()

    monkeypatch.setattr(webhook, "AsyncSessionLocal", factory)
    assert SID not in webhook._STREAMING_CALLS

    body = answer().text
    assert "<Hangup/>" in body
    assert "<Stream" not in body


def test_streaming_set_is_cleared_after_a_call():
    """A leaked entry would make every later retry of that sid un-hangupable."""
    import inspect

    source = inspect.getsource(webhook._handle_call)
    assert "_STREAMING_CALLS.add(call_sid)" in source
    assert "_STREAMING_CALLS.discard(call_sid)" in source
    finally_block = source[source.index("finally:") :]
    assert "_STREAMING_CALLS.discard(call_sid)" in finally_block
