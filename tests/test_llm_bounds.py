"""A reply is bounded before it is generated, on whichever provider answers.

Call cfb7a957, 12 Sep 2026: one reply, fourteen seconds, six turns of a conversation that
had not happened — both sides of it — and a booking nobody agreed to. OneQuestionPerTurn
keeps that off the line; these bounds keep it from being generated or billed at all.

Two of them. max_completion_tokens caps the spend, reasoning included. The stop sequences
are the labels of the other side of the conversation, so a model that starts narrating the
prospect is cut off by the provider before a token of it streams. Both have to reach the
fallback request too: a runaway rescued by the fallback is still a runaway.
"""

import asyncio

from app.core.config import settings
from app.services.llm_provider import (
    STOP_SEQUENCES,
    LLMEndpoint,
    ResilientLLMService,
    build_llm_service,
)

PRIMARY = LLMEndpoint(
    name="cerebras", api_key="k", base_url="https://api.cerebras.ai/v1", model="gpt-oss-120b",
    reasoning_effort="low",
)
FALLBACK = LLMEndpoint(name="openai", api_key="k", base_url="https://api.openai.com/v1", model="gpt-4o-mini")


def test_the_cap_comes_from_settings():
    svc = build_llm_service("t", settings)
    assert svc._settings.max_completion_tokens == settings.LLM_MAX_COMPLETION_TOKENS
    assert settings.LLM_MAX_COMPLETION_TOKENS == 400


def test_the_stop_sequences_name_the_other_side_of_the_conversation():
    """Labels, not words: none of these can occur inside a spoken sales sentence, so a
    reply is never cut for saying something ordinary."""
    assert "\nUser:" in STOP_SEQUENCES and "\nProspect:" in STOP_SEQUENCES
    assert all(s.startswith("\n") and s.endswith(":") for s in STOP_SEQUENCES)
    assert len(STOP_SEQUENCES) <= 4  # the most any OpenAI-wire provider here accepts


class _Completions:
    def __init__(self):
        self.sent = {}

    async def create(self, **params):
        self.sent.update(params)


def test_the_bounds_reach_the_fallback_request_too():
    from types import SimpleNamespace

    from pipecat.processors.aggregators.llm_context import LLMContext

    svc = ResilientLLMService(
        call_sid="t", endpoint=PRIMARY, fallback=FALLBACK, max_completion_tokens=400
    )
    completions = _Completions()
    svc._fallback_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    asyncio.run(svc._complete_on_fallback(LLMContext(messages=[{"role": "user", "content": "hi"}])))

    assert completions.sent["model"] == "gpt-4o-mini"
    assert completions.sent["max_completion_tokens"] == 400
    assert completions.sent["stop"] == list(STOP_SEQUENCES)
    # The primary's reasoning_effort is the primary's; gpt-4o-mini answers it with a 400.
    assert "reasoning_effort" not in completions.sent
