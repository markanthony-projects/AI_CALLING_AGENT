"""How many media streams the host will carry at once.

Each concurrent call runs Silero VAD and audio resampling on the CPU, so the cap has to
match the droplet it is deployed on. It was hardcoded to 8, which silently assumed a host
big enough for 8 — on a smaller one every call in progress degrades instead of the extra
one being rejected.
"""

import ast
import inspect

from app.api.routes import webhook
from app.core.config import Settings, settings


def test_the_cap_is_configurable():
    """Resizing the droplet must not need a code change."""
    src = inspect.getsource(webhook._handle_call)
    assert "settings.MAX_CONCURRENT_CALLS" in src, "the cap is hardcoded again"


def test_the_cap_is_actually_enforced():
    """The check has to gate the accept, not merely log."""
    tree = ast.parse(inspect.getsource(webhook._handle_call).lstrip())
    guard = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.If) and "MAX_CONCURRENT_CALLS" in ast.unparse(n.test)
    )
    body = ast.unparse(guard)
    assert "close" in body, "an over-cap call is logged but still accepted"
    assert "return" in body, "the handler falls through after rejecting"


def test_rejection_happens_before_accept():
    """Accepting first and closing after still spins up a pipeline for a call we refuse."""
    src = inspect.getsource(webhook._handle_call)
    assert src.index("MAX_CONCURRENT_CALLS") < src.index("websocket.accept()")


def test_default_suits_a_small_droplet():
    """Roughly two calls per vCPU; the deployment guide specifies 2 vCPU."""
    assert settings.MAX_CONCURRENT_CALLS == 4


def test_cap_must_be_positive():
    """Zero would reject every call while looking like a valid configuration."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Settings(
            MAX_CONCURRENT_CALLS=0,
            API_KEY="k" * 32,
            CALL_TOKEN_SECRET="s" * 32,
            DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
            OPENAI_API_KEY="x",
            SARVAM_API_KEY="x",
        )
