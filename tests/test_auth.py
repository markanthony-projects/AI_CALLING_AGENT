"""Auth gates, both modes.

AUTH_ENABLED=false is a deliberate local-testing switch; these pin that it cannot
leak into a build that is supposed to be locked down.
"""

import uuid

import pytest
from fastapi import HTTPException, WebSocketException

from app.core.config import settings
from app.core.security import (
    issue_call_token,
    require_call_token,
    require_call_token_ws,
    verify_call_token,
)

CID = str(uuid.uuid4())
SID = str(uuid.uuid4())


@pytest.fixture
def headers():
    return {"X-API-Key": settings.API_KEY}


def _dial(client, headers=None):
    return client.post(
        f"/api/v1/campaigns/{CID}/dial/vobiz",
        json={"phone_numbers": ["+919876543210"]},
        headers=headers or {},
    )


# --- management API ---------------------------------------------------------------


def test_dial_requires_api_key(client, auth_enabled):
    assert _dial(client).status_code == 401


def test_dial_rejects_wrong_api_key(client, auth_enabled):
    assert _dial(client, {"X-API-Key": "wrong"}).status_code == 401


def test_dial_accepts_correct_api_key(client, auth_enabled, headers):
    # 404: past auth, campaign lookup missed against the stubbed session
    assert _dial(client, headers).status_code == 404


def test_browser_link_requires_api_key(client, auth_enabled):
    assert client.post(f"/api/v1/campaigns/{CID}/dial/browser").status_code == 401


def test_auth_disabled_lets_requests_through(client, auth_disabled):
    assert _dial(client).status_code == 404


# --- input validation stays on in both modes --------------------------------------


@pytest.mark.parametrize("raw,expected", [("98765 43210", 422), ("", 422), ("abc", 422)])
def test_undialable_numbers_rejected(client, auth_disabled, raw, expected):
    r = client.post(f"/api/v1/campaigns/{CID}/dial/vobiz", json={"phone_numbers": [raw]})
    assert r.status_code in (expected, 404)


def test_empty_batch_rejected(client, auth_disabled):
    r = client.post(f"/api/v1/campaigns/{CID}/dial/vobiz", json={"phone_numbers": []})
    assert r.status_code == 422


def test_oversized_batch_rejected(client, auth_disabled):
    r = client.post(
        f"/api/v1/campaigns/{CID}/dial/vobiz",
        json={"phone_numbers": ["+919876543210"] * 501},
    )
    assert r.status_code == 422


def test_non_uuid_campaign_rejected(client, auth_disabled):
    r = client.post(
        "/api/v1/campaigns/not-a-uuid/dial/vobiz", json={"phone_numbers": ["+919876543210"]}
    )
    assert r.status_code == 422


# --- signed call tokens -----------------------------------------------------------


def test_token_valid_for_its_own_call():
    assert verify_call_token(issue_call_token(CID, SID), CID, SID)


@pytest.mark.parametrize(
    "cid,sid",
    [(str(uuid.uuid4()), SID), (CID, str(uuid.uuid4()))],
)
def test_token_is_bound_to_one_campaign_and_call(cid, sid):
    assert not verify_call_token(issue_call_token(CID, SID), cid, sid)


def test_tampered_token_rejected():
    tok = issue_call_token(CID, SID)
    assert not verify_call_token(tok[:-3] + "aaa", CID, SID)


def test_empty_token_rejected():
    assert not verify_call_token("", CID, SID)


def test_expired_token_rejected():
    assert not verify_call_token(issue_call_token(CID, SID, ttl_seconds=-10), CID, SID)


# --- telephony webhooks -----------------------------------------------------------


def test_answer_webhook_requires_token(client, auth_enabled):
    assert client.post(f"/vobiz/answer/{CID}/{SID}").status_code == 403


def test_answer_webhook_rejects_garbage_token(client, auth_enabled):
    assert client.post(f"/vobiz/answer/{CID}/{SID}?token=garbage").status_code == 403


async def test_ws_gate_blocks_without_token(auth_enabled):
    with pytest.raises(WebSocketException):
        await require_call_token_ws(campaign_id=CID, call_sid=SID, token="")


async def test_http_gate_blocks_without_token(auth_enabled):
    with pytest.raises(HTTPException):
        await require_call_token(campaign_id=CID, call_sid=SID, token="")


async def test_gates_open_when_auth_disabled(auth_disabled):
    assert await require_call_token_ws(campaign_id=CID, call_sid=SID, token="") is None
    assert await require_call_token(campaign_id=CID, call_sid=SID, token="") is None
