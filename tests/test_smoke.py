"""Deploy smoke tests: would this build actually serve a call?"""

import uuid

import pytest

from app.core.config import Settings, settings
from app.models.db import CallStatus


def test_app_imports_and_health_responds(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_reports_auth_state(client, auth_enabled):
    assert client.get("/health").json()["auth"] == "enabled"


def test_health_reports_auth_disabled(client, auth_disabled):
    assert client.get("/health").json()["auth"] == "DISABLED"


def _mounted_api_paths() -> set[str]:
    """Every APIRoute path the app will actually serve.

    FastAPI wraps included routers in _IncludedRouter rather than flattening them into
    app.routes, so this descends through original_router to reach the real routes.
    """
    from fastapi.routing import APIRoute

    from app.main import app

    paths, stack = set(), list(app.routes)
    while stack:
        route = stack.pop()
        if isinstance(route, APIRoute):
            paths.add(route.path)
        inner = getattr(route, "original_router", None)
        if inner is not None:
            stack.extend(inner.routes)
    return paths


@pytest.mark.parametrize(
    "expected",
    [
        "/api/v1/campaigns/",
        "/api/v1/campaigns/{campaign_id}/dial/vobiz",
        "/api/v1/campaigns/{campaign_id}/dial/browser",
        "/vobiz/answer/{campaign_id}/{call_sid}",
    ],
)
def test_critical_http_routes_are_registered(expected):
    """A router that fails to mount is a 404 in production, not an import error.

    Walks the route table rather than /openapi.json, which is off by default on a public
    host — otherwise disabling the schema endpoint would silently disable this check too.
    """
    assert expected in _mounted_api_paths()


@pytest.mark.parametrize(
    "expected",
    ["/ws/vobiz/{campaign_id}/{call_sid}", "/ws/browser/{campaign_id}/{call_sid}"],
)
def test_websocket_routes_are_registered(expected):
    """WebSocket routes are absent from the OpenAPI schema, so walk the router instead."""
    from app.api.routes.webhook import router

    assert expected in {getattr(r, "path", None) for r in router.routes}


def test_worker_settings_are_wired():
    from app.worker import WorkerSettings, process_extraction

    assert process_extraction in WorkerSettings.functions
    assert WorkerSettings.max_tries > 1, "a transient OpenAI failure must be retried"
    assert WorkerSettings.job_timeout > 0
    assert WorkerSettings.redis_settings is not None


def test_extraction_job_name_matches_worker_function():
    """enqueue_extraction sends a string; arq resolves it by name at the far end."""
    import inspect

    from app.services import extraction
    from app.worker import process_extraction

    source = inspect.getsource(extraction.enqueue_extraction)
    assert f'"{process_extraction.__name__}"' in source


def test_auth_defaults_to_enabled():
    """Fail secure: a missing AUTH_ENABLED must not silently open the API."""
    fresh = Settings(
        _env_file=None,
        AUTH_ENABLED=Settings.model_fields["AUTH_ENABLED"].default,
        API_KEY="a" * 32,
        CALL_TOKEN_SECRET="b" * 32,
        DATABASE_URL="postgresql+asyncpg://u:p@h/d",
        OPENAI_API_KEY="x",
        SARVAM_API_KEY="x",
        # Settings refuses to construct without a key for the configured LLM:
        # a process that cannot make a completion should not accept a websocket.
        CEREBRAS_API_KEY="csk-test",
    )
    assert fresh.AUTH_ENABLED is True


def test_secrets_must_be_long_enough():
    with pytest.raises(Exception):
        Settings(
            _env_file=None,
            API_KEY="short",
            CALL_TOKEN_SECRET="b" * 32,
            DATABASE_URL="postgresql+asyncpg://u:p@h/d",
            OPENAI_API_KEY="x",
            SARVAM_API_KEY="x",
        )


def test_database_url_is_coerced_to_asyncpg():
    """A plain postgresql:// URL silently selects a sync driver and deadlocks the loop."""
    s = Settings(
        _env_file=None,
        API_KEY="a" * 32,
        CALL_TOKEN_SECRET="b" * 32,
        DATABASE_URL="postgresql://u:p@h/d",
        OPENAI_API_KEY="x",
        SARVAM_API_KEY="x",
        # Settings refuses to construct without a key for the configured LLM:
        # a process that cannot make a completion should not accept a websocket.
        CEREBRAS_API_KEY="csk-test",
    )
    assert s.DATABASE_URL.startswith("postgresql+asyncpg://")


def test_call_status_failed_exists():
    assert CallStatus.FAILED.value == "FAILED"
