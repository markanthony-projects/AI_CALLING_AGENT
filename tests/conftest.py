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
os.environ.update(
    {
        "AUTH_ENABLED": "true",
        "API_KEY": "test-api-key-that-is-at-least-32-chars",
        "CALL_TOKEN_SECRET": "test-call-token-secret-at-least-32-chars",
        "CALL_TOKEN_TTL_SECONDS": "900",
        "DATABASE_URL": "postgresql+asyncpg://user:pass@localhost:5432/testdb",
        "REDIS_URL": "redis://localhost:6379/15",
        "OPENAI_API_KEY": "test",
        "SARVAM_API_KEY": "test",
        "GROQ_API_KEY": "test",
        "DEEPGRAM_API_KEY": "test",
        "VOBIZ_AUTH_ID": "test-auth-id",
        "VOBIZ_AUTH_TOKEN": "test-auth-token",
        "VOBIZ_PHONE_NUMBER": "+911234567890",
        "WEBHOOK_BASE_URL": "https://test.example.com",
    }
)

import pytest

from app.api.dependencies import get_db
from app.core.config import settings
from app.main import app


class StubSession:
    """Stands in for AsyncSession where a route only needs a lookup to miss."""

    def __init__(self, get_result=None):
        self._get_result = get_result

    async def get(self, *args, **kwargs):
        return self._get_result

    async def scalar(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        pass

    async def commit(self):
        pass

    async def refresh(self, *args, **kwargs):
        pass


@pytest.fixture
def client():
    from starlette.testclient import TestClient

    async def stub_db():
        yield StubSession()

    app.dependency_overrides[get_db] = stub_db
    # No context manager: lifespan would demand a live Redis.
    yield TestClient(app)
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
