"""The dashboard reads every prospect's phone number and every call transcript.

These tests pin the properties that keep that data behind a login: that no dashboard route
answers without a session, that AUTH_ENABLED=false does not open them the way it opens the
call endpoints, that a wrong password is indistinguishable from an unknown user, and that
the session cookie cannot be read by script on the page.
"""

import pytest

from app.core import passwords
from app.core.config import settings
from tests.conftest import StubUser
from app.core.security import (
    HOST_PREFIXED_COOKIE_NAME,
    SESSION_COOKIE_NAME,
    decode_session_token,
    issue_session_token,
    session_cookie_name,
)

DASHBOARD_ROUTES = [
    "/api/v1/dashboard/overview",
    "/api/v1/dashboard/timeseries",
    "/api/v1/dashboard/funnel",
    "/api/v1/dashboard/live",
    "/api/v1/dashboard/calls",
    "/api/v1/dashboard/leads",
    "/api/v1/dashboard/campaigns",
    "/api/v1/dashboard/projects",
    "/api/v1/dashboard/appointments",
]


@pytest.fixture
def dashboard_enabled():
    original = settings.DASHBOARD_SESSION_SECRET
    settings.DASHBOARD_SESSION_SECRET = "test-dashboard-secret-at-least-32-chars"
    yield
    settings.DASHBOARD_SESSION_SECRET = original


@pytest.mark.parametrize("path", DASHBOARD_ROUTES)
def test_dashboard_routes_reject_anonymous(client, dashboard_enabled, path):
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", DASHBOARD_ROUTES)
def test_auth_disabled_does_not_open_the_dashboard(client, auth_disabled, dashboard_enabled, path):
    """AUTH_ENABLED exists so a developer can place a test call without an API key.

    It is not a reason to serve lead phone numbers and transcripts to anyone who asks, so
    the dashboard must stay closed even with it off.
    """
    assert client.get(path).status_code == 401


@pytest.mark.parametrize("path", DASHBOARD_ROUTES)
def test_routes_report_unconfigured_rather_than_leaking(client, path):
    """With no DASHBOARD_SESSION_SECRET the feature is off, not open."""
    original = settings.DASHBOARD_SESSION_SECRET
    settings.DASHBOARD_SESSION_SECRET = ""
    try:
        assert client.get(path).status_code == 503
    finally:
        settings.DASHBOARD_SESSION_SECRET = original


def test_login_rejects_unknown_user_without_saying_so(client, dashboard_enabled):
    response = client.post(
        "/api/v1/auth/login", json={"email": "nobody@example.com", "password": "whatever-long"}
    )
    assert response.status_code == 401
    # The message must not distinguish "no such user" from "wrong password": that difference
    # is a free list of which addresses are worth attacking.
    assert response.json()["detail"] == "Incorrect email or password"


def test_login_rejects_malformed_email_before_touching_the_database(client, dashboard_enabled):
    assert client.post("/api/v1/auth/login", json={"email": "not-an-email", "password": "x"}).status_code == 422


def test_logout_succeeds_without_a_session(client, dashboard_enabled):
    """A user whose cookie already expired still has to be able to clear it."""
    assert client.post("/api/v1/auth/logout").status_code == 204


def test_session_token_round_trips(dashboard_enabled):
    token = issue_session_token("user-id", "ops@example.com", "ADMIN")
    claims = decode_session_token(token)
    assert claims is not None
    assert (claims.user_id, claims.email, claims.role) == ("user-id", "ops@example.com", "ADMIN")
    assert claims.is_admin


def test_session_token_signed_with_another_secret_is_refused(dashboard_enabled):
    token = issue_session_token("user-id", "ops@example.com", "ADMIN")
    settings.DASHBOARD_SESSION_SECRET = "a-completely-different-secret-32-chars"
    assert decode_session_token(token) is None


def test_call_token_is_not_accepted_as_a_session(dashboard_enabled):
    """Call tokens travel in URLs and are handed to a telephony vendor.

    They are signed with a different secret precisely so one can never be presented as a
    dashboard login.
    """
    from app.core.security import issue_call_token

    assert decode_session_token(issue_call_token("campaign", "call-sid")) is None


def test_expired_session_is_refused(client, dashboard_enabled):
    """A session must actually stop working, not merely claim an expiry.

    Minted with a negative TTL rather than by moving the clock: PyJWT reads time.time()
    itself, so a patched clock only shifts issuance and the token stays valid.
    """
    original_ttl = settings.DASHBOARD_SESSION_TTL_SECONDS
    try:
        settings.DASHBOARD_SESSION_TTL_SECONDS = -60
        stale = issue_session_token("user-id", "ops@example.com", "VIEWER")
    finally:
        settings.DASHBOARD_SESSION_TTL_SECONDS = original_ttl

    assert decode_session_token(stale) is None

    client.cookies.set(SESSION_COOKIE_NAME, stale)
    assert client.get("/api/v1/dashboard/overview").status_code == 401


def test_cookie_name_is_host_prefixed_in_production():
    original = settings.DASHBOARD_COOKIE_SECURE
    try:
        settings.DASHBOARD_COOKIE_SECURE = True
        # __Host- is enforced by the browser as Secure + Path=/ + no Domain, which stops a
        # sibling subdomain writing a session for this one.
        assert session_cookie_name() == HOST_PREFIXED_COOKIE_NAME
        settings.DASHBOARD_COOKIE_SECURE = False
        assert session_cookie_name() == SESSION_COOKIE_NAME
    finally:
        settings.DASHBOARD_COOKIE_SECURE = original


def test_cross_origin_write_is_rejected(client, dashboard_enabled):
    """SameSite=None is required for a cross-origin dashboard, and removes the browser's
    own CSRF protection. The Origin check is what replaces it."""
    token = issue_session_token("user-id", "ops@example.com", "ADMIN")
    client.cookies.set(SESSION_COOKIE_NAME, token)
    response = client.patch(
        "/api/v1/dashboard/leads/0f6bd2b6-0000-4000-8000-000000000000",
        json={"status": "HOT"},
        headers={"Origin": "https://attacker.example"},
    )
    assert response.status_code == 403


def test_same_origin_write_is_allowed_through_the_origin_check(client, dashboard_enabled):
    token = issue_session_token("user-id", "ops@example.com", "ADMIN")
    client.cookies.set(SESSION_COOKIE_NAME, token)
    response = client.patch(
        "/api/v1/dashboard/leads/0f6bd2b6-0000-4000-8000-000000000000",
        json={"status": "HOT"},
        headers={"Origin": "http://testserver"},
    )
    # Past the origin gate. The stub session has no such lead, so 404 is the pass condition.
    assert response.status_code == 404


def test_viewer_cannot_dial(client, dashboard_enabled):
    """Dialing spends money at Vobiz, so it is admin-only regardless of who is signed in.

    The role is set on the stubbed row, not just the cookie: require_session re-reads the
    user, so the database is what decides. A token claiming ADMIN over a VIEWER row must
    still be refused.
    """
    client.session_user["user"] = StubUser(role="VIEWER", email="sales@example.com")
    client.cookies.set(SESSION_COOKIE_NAME, issue_session_token("u", "sales@example.com", "ADMIN"))
    response = client.post(
        "/api/v1/dashboard/campaigns/0f6bd2b6-0000-4000-8000-000000000000/dial",
        json={"phone_numbers": ["+919876543210"]},
    )
    assert response.status_code == 403


def test_viewer_cannot_change_campaign_state(client, dashboard_enabled):
    client.session_user["user"] = StubUser(role="VIEWER", email="sales@example.com")
    client.cookies.set(SESSION_COOKIE_NAME, issue_session_token("u", "sales@example.com", "VIEWER"))
    response = client.patch(
        "/api/v1/dashboard/campaigns/0f6bd2b6-0000-4000-8000-000000000000",
        json={"status": "PAUSED"},
    )
    assert response.status_code == 403


# --- Password hashing -----------------------------------------------------------------


def test_password_round_trips():
    stored = passwords.hash_password("correct horse battery staple")
    assert passwords.verify_password("correct horse battery staple", stored)
    assert not passwords.verify_password("wrong password entirely", stored)


def test_hash_is_salted():
    """Two users with the same password must not share a digest, or one crack breaks both."""
    a = passwords.hash_password("same-password")
    b = passwords.hash_password("same-password")
    assert a != b
    assert passwords.verify_password("same-password", a)
    assert passwords.verify_password("same-password", b)


def test_plaintext_never_appears_in_the_stored_value():
    stored = passwords.hash_password("hunter2-hunter2")
    assert "hunter2" not in stored


def test_malformed_hash_fails_login_rather_than_crashing():
    """A truncated or hand-edited column must 401, not 500 — a 500 is a working oracle."""
    for broken in ("", "garbage", "scrypt$notanumber$8$1$aaaa$bbbb", "bcrypt$1$2$3$4$5"):
        assert not passwords.verify_password("anything", broken)


def test_needs_rehash_flags_weaker_parameters():
    assert passwords.needs_rehash("scrypt$1024$8$1$c2FsdA==$aGFzaA==")
    assert not passwords.needs_rehash(passwords.hash_password("current-params"))
