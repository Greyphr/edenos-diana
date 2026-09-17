import asyncio
import sounddevice as sd
import numpy as np


SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

CHIME_FREQUENCY_HZ = 880
CHIME_DURATION_SECONDS = 0.2
CHIME_AMPLITUDE = 0.3


class AudioIO:
    def __init__(self):
        self._mic_stream: sd.InputStream | None = None
        self._speaker_stream: sd.OutputStream | None = None
        self._playback_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._engaged = False

    def set_engaged(self, engaged: bool):
        self._engaged = engaged

    async def start_mic(
        self, wake_detector, on_wake, on_forward_chunk
    ) -> asyncio.Task:
        self._mic_stream = sd.InputStream(
            samplerate=SEND_SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=CHUNK_SIZE,
        )
        self._mic_stream.start()

        async def _capture_loop():
            while True:
                data, overflowed = await asyncio.to_thread(
                    self._mic_stream.read, CHUNK_SIZE
                )
                chunk = data.tobytes()

                if wake_detector.process(chunk):
                    on_wake()

                if self._engaged:
                    await on_forward_chunk(chunk)

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
                samples = np.frombuffer(chunk, dtype=np.int16)
                await asyncio.to_thread(self._speaker_stream.write, samples)

        return asyncio.create_task(_playback_loop())

    def enqueue_audio(self, data: bytes):
        self._playback_queue.put_nowait(data)

    def clear_queue(self):
        while not self._playback_queue.empty():
            try:
                self._playback_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def play_chime(self):
        self.clear_queue()
        n = int(RECEIVE_SAMPLE_RATE * CHIME_DURATION_SECONDS)
        t = np.arange(n) / RECEIVE_SAMPLE_RATE
        tone = (
            CHIME_AMPLITUDE
            * np.sin(2 * np.pi * CHIME_FREQUENCY_HZ * t)
            * np.iinfo(np.int16).max
        ).astype(np.int16)
        self._playback_queue.put_nowait(tone.tobytes())

    def close(self):
        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
        if self._speaker_stream:
            self._speaker_stream.stop()
            self._speaker_stream.close()