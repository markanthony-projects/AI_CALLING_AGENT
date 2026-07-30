"""/docs and /openapi.json sit outside the API-key dependency.

On a public host they publish every route and request shape to anyone who asks. Nothing
becomes callable, but there is no reason to hand out the map, so they are off by default and
have to be asked for.
"""

import ast
import inspect

from app import main
from app.core.config import settings


def test_schema_endpoints_are_off_by_default():
    assert settings.DOCS_ENABLED is False, "a public host would publish the full route list"


def test_the_app_actually_disables_them():
    assert main.app.openapi_url is None
    assert main.app.docs_url is None
    assert main.app.redoc_url is None


def test_all_three_are_gated_on_the_setting():
    """Leaving redoc or openapi.json wired up would defeat closing /docs."""
    src = inspect.getsource(main)
    tree = ast.parse(src)
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "FastAPI"
    )
    gated = {
        kw.arg for kw in call.keywords
        if kw.arg in ("docs_url", "redoc_url", "openapi_url")
        and "DOCS_ENABLED" in ast.unparse(kw.value)
    }
    assert gated == {"docs_url", "redoc_url", "openapi_url"}, f"only gated: {gated or 'none'}"


def test_health_stays_public():
    """It is the deploy check for whether auth came up, so it must not need auth."""
    from fastapi.routing import APIRoute

    health = next(r for r in main.app.routes if isinstance(r, APIRoute) and r.path == "/health")
    assert not health.dependencies
