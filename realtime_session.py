# Model: gemini-3.1-flash-live-preview
# (current low-latency Live/native-audio model as of Sept 2026, per Google docs)
import asyncio
from collections.abc import Callable

from google import genai
from google.genai import types


MODEL = "gemini-3.1-flash-live-preview"

SYSTEM_INSTRUCTION = (
    "You are Eden, Christopher's personal AI assistant. "
    "You are conversational and concise. "
    "Respond naturally and keep replies brief unless asked for detail."
)


class RealtimeSession:
    def __init__(self, api_key: str):
        self._client = genai.Client(
            api_key=api_key, http_options={"api_version": "v1alpha"}
        )
        self._session = None
        self._session_ctx = None
        self._audio_callback: Callable[[bytes], None] | None = None
        self._receive_task: asyncio.Task | None = None

    async def connect(self, on_audio: Callable[[bytes], None]):
        self._audio_callback = on_audio
        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=SYSTEM_INSTRUCTION,
        )
        self._session_ctx = self._client.aio.live.connect(
            model=MODEL, config=config
        )
        self._session = await self._session_ctx.__aenter__()
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self):
        try:
            while True:
                async for response in self._session.receive():
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

    async def send_audio(self, chunk: bytes):
        if self._session is None:
            return
        await self._session.send_realtime_input(
            audio=types.Blob(
                data=chunk,
                mime_type="audio/pcm;rate=16000",
            )
        )

    async def close(self):
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session_ctx.__aexit__(None, None, None)