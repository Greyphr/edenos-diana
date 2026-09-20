"""Reasoning delegation: offload deep-thinking tasks to a non-Live model.

The realtime voice model is meant to be quick and conversational. For
questions that genuinely need multi-step reasoning, Eden delegates the
full question to a slower, deeper text model via ``delegate_reasoning``.
The delegate call is a normal background tool task, so reasoning time
never blocks the live audio session.
"""

import logging
import os
from abc import ABC, abstractmethod

from google import genai

from tools.registry import RiskTier, ToolSpec

logger = logging.getLogger(__name__)

# Fallback deep-reasoning model when config/voice.yaml omits "reasoning_model".
# Overridable per-config, mirroring how the live model name is kept out of
# code (see DEFAULT_MODEL in the Live provider).
DEFAULT_REASONING_MODEL = "gemini-3.5-pro"


class ReasoningProvider(ABC):
    @abstractmethod
    async def think(self, task: str, context: str = "") -> str:
        """Return a considered answer to ``task``, optionally given the
        relevant conversation ``context``.

        This is a single-shot call: one task in, one reasoned answer out.
        It runs as an ordinary background tool task, so a slow answer never
        blocks the live voice session.
        """


class GeminiReasoningProvider(ReasoningProvider):
    """Non-Live text model for deep thinking, reusing the same
    GEMINI_API_KEY as the voice session (no new credential needed)."""

    def __init__(self, model: str | None = None, client=None):
        """``model`` defaults to the configured reasoning model; ``client``
        is injectable for tests, production builds from the env."""
        self.model = model or DEFAULT_REASONING_MODEL
        if client is not None:
            self._client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
                )
            self._client = genai.Client(
                api_key=api_key, http_options={"api_version": "v1alpha"}
            )

    async def think(self, task: str, context: str = "") -> str:
        contents = task
        if context and context.strip():
            contents = f"{context.strip()}\n\n{task}"
        logger.info("Reasoning delegate: %r via %s", task[:120], self.model)
        response = await self._client.aio.models.generate_content(
            model=self.model, contents=contents
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError(
                f"Reasoning delegate returned no text for model {self.model!r}"
            )
        return text


def make_reasoning_delegate_spec(
    provider: ReasoningProvider | None = None,
) -> ToolSpec:
    """Tool that hands a deep-thought question to the reasoning provider.

    TRIVIAL tier: read-only thinking, owner-only, confirmation-free - it has
    no real-world effect, so it is treated like any other READ/TRIVIAL tool
    rather than an action needing confirmation.
    """
    provider = provider or GeminiReasoningProvider()

    async def _delegate_handler(task: str, context: str = "") -> dict:
        result = await provider.think(task, context)
        return {"status": "ok", "result": result}

    return ToolSpec(
        name="delegate_reasoning",
        description=(
            "Delegate a question that needs real multi-step reasoning or "
            "careful analysis to a dedicated deep-thinking model, rather than "
            "working through it inline. Pass the full question in 'task', and "
            "any relevant conversation context in 'context'. Returns the "
            "considered answer as text. Use this for complex logic, planning, "
            "or analysis - not for quick conversational replies."
        ),
        parameters={
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The full question or problem to reason about.",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Relevant conversation context to ground the reasoning "
                        "in. Optional."
                    ),
                },
            },
            "required": ["task"],
        },
        risk_tier=RiskTier.TRIVIAL,
        handler=_delegate_handler,
    )