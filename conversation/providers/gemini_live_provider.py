# Model: gemini-3.1-flash-live-preview
# (current low-latency Live/native-audio model as of Sept 2026, per Google docs)
import asyncio
import logging
import os
from collections.abc import Callable

from google import genai
from google.genai import types

from conversation.voice_provider import VoiceProvider

logger = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-live-preview"

# Initial-connection smoothing: a transient blip at startup shouldn't fail the
# whole boot, so retry a few times with a short fixed delay before giving up.
INITIAL_CONNECT_ATTEMPTS = 3
INITIAL_CONNECT_DELAY_SECONDS = 2.0

# Mid-session reconnect backoff, same shape as daemon.py's supervisor.
RECONNECT_START_BACKOFF = 2.0
RECONNECT_MAX_BACKOFF = 30.0
# Total elapsed retry window; when exceeded, give up and raise so the daemon's
# outer restart takes over.
RECONNECT_MAX_TOTAL_SECONDS = 120.0


class GeminiLiveProvider(VoiceProvider):
    def __init__(self, client=None):
        """``client`` is injectable for tests; production builds from the env."""
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
        self._session = None
        self._session_ctx = None
        self._audio_callback: Callable[[bytes], None] | None = None
        self._interrupted_callback: Callable[[], None] | None = None
        self._disconnected_callback: Callable[[], None] | None = None
        self._reconnected_callback: Callable[[], None] | None = None
        self._tool_handlers: dict[str, Callable] = {}
        self._receive_task: asyncio.Task | None = None
        # start_session params cached so a reconnect can replay the same setup
        # without the caller needing to call start_session again.
        self._session_args: dict | None = None

    def register_tool(self, name: str, handler) -> None:
        self._tool_handlers[name] = handler

    async def start_session(
        self,
        system_instruction: str,
        voice_config: dict,
        tool_declarations: list[dict] | None = None,
    ) -> None:
        self._session_args = {
            "system_instruction": system_instruction,
            "voice_config": voice_config,
            "tool_declarations": tool_declarations,
        }
        last_error: Exception | None = None
        for attempt in range(1, INITIAL_CONNECT_ATTEMPTS + 1):
            try:
                if self._session is not None:
                    await self._close_session()
                await self._open_session()
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                if attempt < INITIAL_CONNECT_ATTEMPTS:
                    logger.warning(
                        "Initial Live connection failed (attempt %d/%d): %r",
                        attempt,
                        INITIAL_CONNECT_ATTEMPTS,
                        e,
                    )
                    await asyncio.sleep(INITIAL_CONNECT_DELAY_SECONDS)
        else:
            raise ConnectionError(
                f"Could not connect to the Live session after "
                f"{INITIAL_CONNECT_ATTEMPTS} attempts: {last_error}"
            ) from last_error

        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _open_session(self) -> None:
        if self._session_args is None:
            raise RuntimeError("start_session must be called before opening a session")
        args = self._session_args
        tools = None
        if args["tool_declarations"]:
            tools = [{"function_declarations": list(args["tool_declarations"])}]
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=args["system_instruction"],
            tools=tools,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=args["voice_config"]["voice_name"]
                    )
                ),
                language_code=args["voice_config"]["language_code"],
            ),
        )
        self._session_ctx = self._client.aio.live.connect(model=MODEL, config=config)
        self._session = await self._session_ctx.__aenter__()

    async def _close_session(self) -> None:
        ctx = self._session_ctx
        self._session_ctx = None
        self._session = None
        if ctx is not None:
            await ctx.__aexit__(None, None, None)

    async def _receive_loop(self):
        while True:
            try:
                async for response in self._session.receive():
                    await self._handle_response(response)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._handle_disconnect(e)

    async def _handle_response(self, response) -> None:
        if response.server_content and response.server_content.interrupted:
            if self._interrupted_callback:
                self._interrupted_callback()
            return

        if response.server_content and response.server_content.model_turn:
            for part in response.server_content.model_turn.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    if self._audio_callback:
                        self._audio_callback(part.inline_data.data)
            return

        if response.tool_call:
            await self._handle_tool_call(response.tool_call)

    async def _handle_disconnect(self, error: Exception) -> None:
        """Reconnect after a connection-level failure, or give up and re-raise."""
        if self._disconnected_callback:
            self._disconnected_callback()
        logger.error("Realtime session disconnected: %r", error, exc_info=True)

        delay = RECONNECT_START_BACKOFF
        elapsed = 0.0
        attempt = 0
        while True:
            if elapsed >= RECONNECT_MAX_TOTAL_SECONDS:
                logger.error(
                    "Could not reconnect within %.0fs; raising so the daemon "
                    "restarts Eden",
                    elapsed,
                )
                raise error
            attempt += 1
            logger.warning(
                "Reconnecting to the Live session (attempt %d) in %.0fs...",
                attempt,
                delay,
            )
            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, RECONNECT_MAX_BACKOFF)
            try:
                await self._close_session()
                await self._open_session()
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %r", attempt, e)

        logger.info("Live session reconnected")
        if self._reconnected_callback:
            self._reconnected_callback()

    async def _handle_tool_call(self, tool_call) -> None:
        if not tool_call.function_calls:
            return
        # Run handlers off the receive hot path; each is created as a task so
        # one slow tool can't stall audio/interruption handling.
        tasks = [asyncio.create_task(self._run_tool(fc)) for fc in tool_call.function_calls]
        responses = await asyncio.gather(*tasks)
        if self._session is not None:
            await self._session.send_tool_response(function_responses=responses)

    async def _run_tool(self, fc) -> types.FunctionResponse:
        handler = self._tool_handlers.get(fc.name)
        if handler is None:
            return types.FunctionResponse(
                id=fc.id,
                name=fc.name,
                response={"error": f"No handler registered for {fc.name!r}"},
            )
        try:
            result = await handler(**(fc.args or {}))
            if not isinstance(result, dict):
                result = {"result": result}
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Tool handler %r failed", fc.name)
            result = {"error": str(e)}
        return types.FunctionResponse(id=fc.id, name=fc.name, response=result)

    async def send_audio(self, chunk: bytes) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type="audio/pcm;rate=16000",
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Transient: the session may just have dropped (the receive loop is
            # reconnecting). Don't take the mic task down with it.
            logger.warning("Dropping audio chunk, session unavailable: %r", e)

    def on_audio_response(self, callback: Callable[[bytes], None]) -> None:
        self._audio_callback = callback

    def on_interrupted(self, callback: Callable[[], None]) -> None:
        self._interrupted_callback = callback

    def on_disconnected(self, callback: Callable[[], None]) -> None:
        self._disconnected_callback = callback

    def on_reconnected(self, callback: Callable[[], None]) -> None:
        self._reconnected_callback = callback

    async def wait_for_session(self) -> None:
        if self._receive_task is None:
            return
        await self._receive_task

    async def stop_session(self) -> None:
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        await self._close_session()