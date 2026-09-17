# Model: gemini-3.1-flash-live-preview
# (current low-latency Live/native-audio model as of Sept 2026, per Google docs)
import asyncio
import os
from collections.abc import Callable

from google import genai
from google.genai import types

from conversation.voice_provider import VoiceProvider

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
        self._receive_task: asyncio.Task | None = None

    async def start_session(self, system_instruction: str, voice_config: dict) -> None:
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=system_instruction,
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

                    if (
                        response.server_content
                        and response.server_content.model_turn
                    ):
                        for part in response.server_content.model_turn.parts:
                            if (
                                part.inline_data
                                and isinstance(part.inline_data.data, bytes)
                            ):
                                if self._audio_callback:
                                    self._audio_callback(part.inline_data.data)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Realtime session error: {e}")

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