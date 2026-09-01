"""Test harness.

These are smoke tests: they must run with no Postgres, no Redis and no API keys, so
that CI catches a broken deploy without needing the world stood up. Anything that
would touch a real service is stubbed at the dependency boundary.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Set before app.core.config is imported anywhere: the real .env must not leak into tests.
#
# APP_ENV_FILE is what makes that true rather than merely intended. Environment variables beat
# the file, so every value below was already honoured — but anything missing from this dict
# fell through to whatever .env the developer happened to have. CEREBRAS_API_KEY was missing,
# so the suite passed everywhere except CI, and DOCS_ENABLED leaked in and kept two tests red.
#
# Everything the settings require has to be here now, because there is no longer a file behind
# it to quietly fill the gaps.
os.environ.update(
    {
        "APP_ENV_FILE": str(ROOT / "tests" / "no-such-env-file"),
        "AUTH_ENABLED": "true",
        "API_KEY": "test-api-key-that-is-at-least-32-chars",
        "CALL_TOKEN_SECRET": "test-call-token-secret-at-least-32-chars",
        "CALL_TOKEN_TTL_SECONDS": "900",
        "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/testdb",
        "REDIS_URL": "redis://localhost:6379/15",
        "OPENAI_API_KEY": "test",
        "SARVAM_API_KEY": "test",
        "GROQ_API_KEY": "test",
        "CEREBRAS_API_KEY": "test",
        "DEEPGRAM_API_KEY": "test",
        "VOBIZ_AUTH_ID": "test-auth-id",
        "VOBIZ_AUTH_TOKEN": "test-auth-token",
        "VOBIZ_PHONE_NUMBER": "+911234567890",
        "WEBHOOK_BASE_URL": "https://test.example.com",
    }
)

from types import SimpleNamespace

import pytest

from app.api.dependencies import get_db
from app.core.config import settings
from app.main import app


class StubResult:
    """An empty result set — every lookup misses, so routes take their not-found path."""

    def scalars(self):
        return self

    def first(self):
        return None

    def all(self):
        return []

    def one_or_none(self):
        return None


class StubUser:
    """The signed-in row require_session re-reads on every request.

    Role comes from here rather than the cookie because the guard re-reads it, so a
    demotion takes effect immediately instead of at token expiry.
    """

    def __init__(self, role="ADMIN", is_active=True, user_id="user-id", email="ops@example.com"):
        self.id = user_id
        self.email = email
        self.full_name = None
        self.role = SimpleNamespace(value=role)
        self.is_active = is_active


class StubSession:
    """Stands in for AsyncSession where a route only needs a lookup to miss."""

    def __init__(self, get_result=None, user=None):
        self._get_result = get_result
        self._user = user

    async def get(self, model=None, *args, **kwargs):
        # Only the session guard's own lookup resolves; every other get() still misses, so
        # routes keep taking their not-found path.
        if self._user is not None and getattr(model, "__name__", "") == "DashboardUser":
            return self._user
        return self._get_result

    async def scalar(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        return StubResult()

    def add(self, *args, **kwargs):
        pass

    async def commit(self):
        pass

    async def refresh(self, *args, **kwargs):
        pass


@pytest.fixture
def client():
    from starlette.testclient import TestClient

    # Mutable so a test can change who is signed in before it makes its request.
    session_user = {"user": StubUser()}

    async def stub_db():
        yield StubSession(user=session_user["user"])

    app.dependency_overrides[get_db] = stub_db
    # No context manager: lifespan would demand a live Redis.
    test_client = TestClient(app)
    test_client.session_user = session_user
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_enabled():
    original = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = True
    yield
    settings.AUTH_ENABLED = original


@pytest.fixture
def auth_disabled():
    original = settings.AUTH_ENABLED
    settings.AUTH_ENABLED = False
    yield
    settings.AUTH_ENABLED = original


@pytest.fixture
def dashboard_enabled():
    """Turn the dashboard on for one test. An empty secret disables it entirely."""
    from app.core.config import settings

    original = settings.DASHBOARD_SESSION_SECRET
    settings.DASHBOARD_SESSION_SECRET = "test-dashboard-secret-at-least-32-chars"
    yield
    settings.DASHBOARD_SESSION_SECRET = original
