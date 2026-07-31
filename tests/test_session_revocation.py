"""Revocation, and the cookie policy a separately-hosted dashboard needs.

The dashboard is a separate repo deployed to Vercel, so the session cookie crosses origins.
Two things follow, and both were wrong before:

1. The guard used to trust the token alone. A user deactivated an hour ago kept reading
   every lead's phone number and every transcript until their 12h session expired — only
   /me noticed. Now every dashboard request re-reads the row.

2. SameSite was derived from whether CORS was configured, which confused cross-ORIGIN with
   cross-SITE. dashboard.homebble.in -> ai-calls.homebble.in is cross-origin but same-site,
   where Lax works; only a genuinely cross-site host needs None, and None makes the session
   a third-party cookie that Safari and Firefox discard.
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import SESSION_COOKIE_NAME, issue_session_token, session_cookie_samesite
from tests.conftest import StubUser

BASE = dict(
    API_KEY="k" * 32,
    CALL_TOKEN_SECRET="s" * 32,
    DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
    OPENAI_API_KEY="x",
    SARVAM_API_KEY="x",
    # Pinned, or these inherit the developer's .env and the SameSite cases change meaning.
    DASHBOARD_COOKIE_SECURE=True,
)

LEAD = "/api/v1/dashboard/leads/0f6bd2b6-0000-4000-8000-000000000000"


# --- revocation ---------------------------------------------------------------------


def test_a_deactivated_user_is_refused(client, dashboard_enabled):
    """The token is still cryptographically valid; the row is what revokes."""
    client.session_user["user"] = StubUser(is_active=False)
    client.cookies.set(SESSION_COOKIE_NAME, issue_session_token("user-id", "ops@example.com", "ADMIN"))
    assert client.get(LEAD).status_code == 401


def test_a_deleted_user_is_refused(client, dashboard_enabled):
    client.session_user["user"] = None
    client.cookies.set(SESSION_COOKIE_NAME, issue_session_token("user-id", "ops@example.com", "ADMIN"))
    assert client.get(LEAD).status_code == 401


def test_an_active_user_still_gets_through(client, dashboard_enabled):
    """The stub has no such lead, so 404 means the guard let the request past."""
    client.session_user["user"] = StubUser(role="ADMIN")
    client.cookies.set(SESSION_COOKIE_NAME, issue_session_token("user-id", "ops@example.com", "ADMIN"))
    assert client.get(LEAD).status_code == 404


def test_a_demotion_takes_effect_without_waiting_for_expiry():
    """A token minted while the user was an admin must not keep admin powers after the
    role is changed — the guard rebuilds the claims from the row."""
    import inspect

    from app.core import security

    src = inspect.getsource(security.require_session)
    assert "db.get(DashboardUser" in src, "the guard never re-reads the user"
    assert "SessionClaims(str(user.id), user.email, user.role.value)" in src, (
        "claims are returned from the token rather than rebuilt from the row"
    )


# --- cookie policy ------------------------------------------------------------------


def test_samesite_comes_from_configuration(monkeypatch):
    from app.core import security

    monkeypatch.setattr(security.settings, "DASHBOARD_COOKIE_SAMESITE", "none")
    assert session_cookie_samesite() == "none"
    monkeypatch.setattr(security.settings, "DASHBOARD_COOKIE_SAMESITE", "lax")
    assert session_cookie_samesite() == "lax"


def test_samesite_is_not_inferred_from_cors(monkeypatch):
    """Deriving it from CORS forced None on a same-site subdomain deployment, making the
    session a third-party cookie for no reason."""
    from app.core import security

    monkeypatch.setattr(security.settings, "DASHBOARD_COOKIE_SAMESITE", "lax")
    monkeypatch.setattr(
        security.settings, "DASHBOARD_CORS_ORIGINS", "https://ops.vercel.app"
    )
    assert session_cookie_samesite() == "lax"


def test_lax_is_the_default():
    assert Settings(**BASE).DASHBOARD_COOKIE_SAMESITE == "lax"


@pytest.mark.parametrize("value", ["Lax", "NONE", " strict "])
def test_samesite_is_normalised(value):
    assert Settings(**BASE, DASHBOARD_COOKIE_SAMESITE=value).DASHBOARD_COOKIE_SAMESITE == value.strip().lower()


def test_a_nonsense_samesite_is_rejected():
    with pytest.raises(ValidationError):
        Settings(**BASE, DASHBOARD_COOKIE_SAMESITE="sometimes")


def test_samesite_none_without_secure_fails_at_startup():
    """Browsers discard a SameSite=None cookie that is not Secure, so login would appear to
    succeed and every request after it would be anonymous. Better to refuse to boot."""
    with pytest.raises(ValidationError) as exc:
        Settings(**{**BASE, "DASHBOARD_COOKIE_SECURE": False}, DASHBOARD_COOKIE_SAMESITE="none")
    assert "Secure" in str(exc.value)


def test_lax_without_secure_is_fine_for_local_dev():
    s = Settings(**{**BASE, "DASHBOARD_COOKIE_SECURE": False}, DASHBOARD_COOKIE_SAMESITE="lax")
    assert s.DASHBOARD_COOKIE_SAMESITE == "lax"
