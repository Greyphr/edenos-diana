from abc import ABC, abstractmethod
from collections.abc import Callable


class VoiceProvider(ABC):
    @abstractmethod
    async def start_session(
        self,
        system_instruction: str,
        voice_config: dict,
        tool_declarations: list[dict] | None = None,
    ) -> None:
        """Open the voice session with the given instruction and voice config.

        ``tool_declarations`` is an optional list of Gemini function
        declarations (``[{"name": ..., "description": ...,
        "parameters": {...JSON schema...}}]``). Each declared name must have
        been registered via ``register_tool``.
        """

    @abstractmethod
    def register_tool(self, name: str, handler) -> None:
        """Register an async ``handler(**args) -> dict`` for a function name.

        When a ``tool_declarations`` entry matching ``name`` is invoked by
        the model, the handler is awaited off the receiving hot path and its
        dict result is sent back to the session.
        """

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Stream captured audio to the session."""

    @abstractmethod
    def on_audio_response(self, callback: Callable[[bytes], None]) -> None:
        """Register the callback invoked for each model audio chunk."""

    @abstractmethod
    def on_interrupted(self, callback: Callable[[], None]) -> None:
        """Register the callback invoked when the model reply is interrupted."""

    @abstractmethod
    async def stop_session(self) -> None:
        """Close the session and release resources."""


def get_voice_provider(config: dict) -> VoiceProvider:
    provider_name = config.get("provider")
    if provider_name == "gemini_live":
        from conversation.providers.gemini_live_provider import (
            GeminiLiveProvider,
        )

        return GeminiLiveProvider()
    raise ValueError(f"Unknown voice provider: {provider_name!r}")