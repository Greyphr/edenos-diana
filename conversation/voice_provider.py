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
    def on_input_transcript(self, callback: Callable[[str], None]) -> None:
        """Register the callback invoked for each finalized input transcription.

        The callback receives the finalized text of what the owner actually
        said. Interim (partial-utterance) fragments are never surfaced: the
        receiver only sees a complete, settled transcript. This is driven by
        the server-side ``input_audio_transcription`` config; the callback
        runs synchronously, so the provider must dispatch it off its receive
        hot path (e.g. via ``asyncio.create_task``).
        """

    @abstractmethod
    async def send_status_note(self, text: str) -> None:
        """Inject an owner-visible status line into the live session.

        The note is surfaced as a brief client turn (a ``[System: ...]``
        part), so the owner hears/spies it the same way they hear any other
        conversational turn. Used to acknowledge things that happen in the
        background of the voice session — e.g. reporting the outcome of a
        spoken confirmation that was resolved through the transcript path.
        Providers should ignore the note (and log a warning) when no session
        is currently open, rather than raising.
        """

    @abstractmethod
    def on_disconnected(self, callback: Callable[[], None]) -> None:
        """Register the callback invoked when the session is lost mid-conversation.

        Called as soon as a connection error is detected, before any reconnect
        attempt is made. The caller should stop feeding audio into the dead
        session (e.g. drop back to an idle state).
        """

    @abstractmethod
    def on_reconnected(self, callback: Callable[[bool], None]) -> None:
        """Register the callback invoked after a reconnected session is live
        again. The callback receives whether the reconnection resumed the
        previous session (True) or started a fresh one (False)."""

    @abstractmethod
    async def wait_for_session(self) -> None:
        """Block until the receive loop ends.

        Returns when the session is closed for shutdown, or raises if the
        provider gave up reconnecting after its retry window (the caller's
        outer supervisor should restart the process).
        """

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