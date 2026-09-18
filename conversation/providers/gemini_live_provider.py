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


class GeminiLiveProvider(VoiceProvider):
    def __init__(self):
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
        self._tool_handlers: dict[str, Callable] = {}
        self._receive_task: asyncio.Task | None = None

    def register_tool(self, name: str, handler) -> None:
        self._tool_handlers[name] = handler

    async def start_session(
        self,
        system_instruction: str,
        voice_config: dict,
        tool_declarations: list[dict] | None = None,
    ) -> None:
        tools = None
        if tool_declarations:
            tools = [{"function_declarations": list(tool_declarations)}]
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_instruction,
            tools=tools,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_config["voice_name"]
                    )
                ),
                language_code=voice_config["language_code"],
            ),
        )
        self._session_ctx = self._client.aio.live.connect(model=MODEL, config=config)
        self._session = await self._session_ctx.__aenter__()
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        try:
            while True:
                async for response in self._session.receive():
                    if (
                        response.server_content
                        and response.server_content.interrupted
                    ):
                        if self._interrupted_callback:
                            self._interrupted_callback()
                        continue

                    if response.server_content and response.server_content.model_turn:
                        for part in response.server_content.model_turn.parts:
                            if (
                                part.inline_data
                                and isinstance(part.inline_data.data, bytes)
                            ):
                                if self._audio_callback:
                                    self._audio_callback(part.inline_data.data)
                        continue

                    if response.tool_call:
                        await self._handle_tool_call(response.tool_call)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Realtime session error: {e}")

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
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=chunk,
                mime_type="audio/pcm;rate=16000",
            )
        )

    def on_audio_response(self, callback: Callable[[bytes], None]) -> None:
        self._audio_callback = callback

    def on_interrupted(self, callback: Callable[[], None]) -> None:
        self._interrupted_callback = callback

    async def stop_session(self) -> None:
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session_ctx.__aexit__(None, None, None)