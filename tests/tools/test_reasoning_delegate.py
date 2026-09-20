"""delegate_reasoning spec + GeminiReasoningProvider behavior with injectable
clients (no network, no real API key)."""

import asyncio
from types import SimpleNamespace

import pytest

from tools.reasoning_delegate import (
    DEFAULT_REASONING_MODEL,
    GeminiReasoningProvider,
    ReasoningProvider,
    make_reasoning_delegate_spec,
)
from tools.registry import RiskTier


class _FakeProvider(ReasoningProvider):
    def __init__(self, text: str = "reasoned answer") -> None:
        super().__init__()
        self.text = text
        self.calls: list[tuple[str, str]] = []

    async def think(self, task: str, context: str = "") -> str:
        self.calls.append((task, context))
        return self.text


async def test_successful_mocked_provider_returns_expected_shape():
    provider = _FakeProvider("the answer is 42")
    spec = make_reasoning_delegate_spec(provider=provider)
    result = await spec.handler(task="what is 6 * 7?", context="joking")
    assert result == {"status": "ok", "result": "the answer is 42"}
    assert provider.calls == [("what is 6 * 7?", "joking")]


async def test_handler_only_requires_task():
    provider = _FakeProvider()
    spec = make_reasoning_delegate_spec(provider=provider)
    result = await spec.handler(task="just a task")
    assert result["status"] == "ok"
    assert provider.calls == [("just a task", "")]


def test_spec_metadata():
    spec = make_reasoning_delegate_spec(provider=_FakeProvider())
    assert spec.name == "delegate_reasoning"
    assert spec.risk_tier is RiskTier.TRIVIAL
    assert spec.parameters["required"] == ["task"]


def test_constructing_provider_without_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY not set"):
        GeminiReasoningProvider()


async def test_think_concatenates_task_and_context():
    class _Models:
        async def generate_content(self, model, contents):
            return SimpleNamespace(text=f"got: {contents}")

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    provider = GeminiReasoningProvider(model="model-x", client=_Client())
    assert await provider.think("question", "context line") == "got: context line\n\nquestion"


async def test_think_raises_when_no_text_returned():
    class _Models:
        async def generate_content(self, model, contents):
            return SimpleNamespace(text=None)

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()

    provider = GeminiReasoningProvider(model="model-x", client=_Client())
    with pytest.raises(RuntimeError, match="no text"):
        await provider.think("question")


def test_default_model_is_configurable_constant():
    assert DEFAULT_REASONING_MODEL