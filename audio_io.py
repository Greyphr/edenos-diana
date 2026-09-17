import asyncio
import sounddevice as sd


SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024


class AudioIO:
    def __init__(self):
        self._mic_stream: sd.InputStream | None = None
        self._speaker_stream: sd.OutputStream | None = None
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def start_mic(self, on_chunk) -> asyncio.Task:
        self._mic_stream = sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        self._mic_stream.start()

        async def _capture_loop():
            while True:
                data = await asyncio.to_thread(
                    self._mic_stream.read, CHUNK_SIZE
                )
                await on_chunk(data.tobytes())

        return asyncio.create_task(_capture_loop())

    async def start_speaker(self) -> asyncio.Task:
        self._speaker_stream = sd.OutputStream(
            samplerate=RECEIVE_SAMPLE_RATE,
            channels=1,
            dtype="int16",
        )
        self._speaker_stream.start()

        async def _playback_loop():
            while True:
                chunk = await self._playback_queue.get()
                await asyncio.to_thread(self._speaker_stream.write, chunk)

        return asyncio.create_task(_playback_loop())

    def enqueue_audio(self, data: bytes):
        self._playback_queue.put_nowait(data)

    def close(self):
        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
        if self._speaker_stream:
            self._speaker_stream.stop()
            self._speaker_stream.close()