"""How hard the primary model is allowed to think before it speaks.

Call 2eeb48a0, 10 Sep 2026, gpt-oss-120b with nothing set: turn 1 had a 473ms first token
and then 1396ms of silence before the first sentence; turn 8, 1434ms. A reasoning model
thinks in that gap, on the caller's clock, for a reply that is one sentence and a question.
Cerebras takes reasoning_effort=low|medium|high for it. gemma takes nothing, and a
parameter a model does not know is a 400 — so blank sends nothing.
"""

import asyncio
import os

import pytest

from app.core.config import Settings
from app.services.llm_provider import LLMEndpoint, ResilientLLMService, primary_endpoint

BASE = dict(
    API_KEY="k" * 32,
    CALL_TOKEN_SECRET="s" * 32,
    DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
    OPENAI_API_KEY="x",
    SARVAM_API_KEY="x",
    CEREBRAS_API_KEY="csk-test",
)


@pytest.fixture(autouse=True)
def _restore_environment():
    """_settings() strips the LLM variables out of os.environ so the assertions are about
    the arguments. Stripped permanently, every later test that builds its own Settings loses
    CEREBRAS_API_KEY and fails the "the LLM has a key" validator — which is exactly what
    happened the first time this file imported test_llm_throttle's helper without its
    matching fixture: thirteen unrelated tests failed in a full run and passed alone."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _settings(**over):
    for key in ("LLM_REASONING_EFFORT", "LLM_PROVIDER_NAME", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        os.environ.pop(key, None)
    return Settings(**{**BASE, **over}, _env_file=None)


def _endpoint(**over):
    base = dict(name="cerebras", api_key="k", base_url="https://api.cerebras.ai/v1", model="gpt-oss-120b")
    return LLMEndpoint(**{**base, **over})


def test_blank_sends_nothing():
    """Blank is for a model that does not reason. Distinct from "none", which is a value
    qwen understands and gpt-oss refuses — see the default, which is "none" because the
    default model is qwen."""
    assert primary_endpoint(_settings(LLM_REASONING_EFFORT="")).extra_params == {}


def test_the_default_turns_qwens_thinking_off():
    assert primary_endpoint(_settings()).extra_params == {"reasoning_effort": "none"}


def test_low_is_sent_as_reasoning_effort():
    ep = primary_endpoint(_settings(LLM_REASONING_EFFORT="low"))
    assert ep.reasoning_effort == "low"
    assert ep.extra_params == {"reasoning_effort": "low"}


def test_it_is_normalised():
    assert primary_endpoint(_settings(LLM_REASONING_EFFORT="  Low ")).extra_params == {"reasoning_effort": "low"}


def test_the_service_carries_it_into_the_request_params():
    svc = ResilientLLMService(call_sid="t", endpoint=_endpoint(reasoning_effort="low"))
    assert svc._settings.extra == {"reasoning_effort": "low"}


def test_a_model_that_does_not_reason_gets_no_extra_params():
    svc = ResilientLLMService(call_sid="t", endpoint=_endpoint(model="gemma-4-31b"))
    assert svc._settings.extra == {}


class _Completions:
    def __init__(self):
        self.sent = {}

    async def create(self, **params):
        self.sent.update(params)


def _fallback_params(primary, fallback):
    from pipecat.processors.aggregators.llm_context import LLMContext

    svc = ResilientLLMService(call_sid="t", endpoint=primary, fallback=fallback)
    stub = _Completions()
    svc._fallback_client.chat.completions = stub
    asyncio.run(svc._complete_on_fallback(LLMContext(messages=[{"role": "user", "content": "hi"}])))
    return stub.sent


def test_the_fallback_does_not_inherit_it():
    """gpt-4o-mini rejects reasoning_effort. A rescued turn must not die of the primary's
    parameter."""
    sent = _fallback_params(
        _endpoint(reasoning_effort="low"),
        LLMEndpoint(name="openai", api_key="k2", base_url="https://api.openai.com/v1", model="gpt-4o-mini"),
    )
    assert "reasoning_effort" not in sent
    assert sent["model"] == "gpt-4o-mini"


def test_a_fallback_with_its_own_effort_sends_its_own():
    sent = _fallback_params(_endpoint(reasoning_effort="low"), _endpoint(reasoning_effort="medium"))
    assert sent["reasoning_effort"] == "medium"
