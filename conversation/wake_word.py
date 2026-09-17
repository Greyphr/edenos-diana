import logging
import os

import numpy as np
import pvporcupine

from conversation.audio_io import SEND_SAMPLE_RATE

logger = logging.getLogger(__name__)


class WakeWordDetector:
    def __init__(self, access_key: str, model_path: str | None = None):
        self._buffer = bytearray()
        if model_path and os.path.isfile(model_path):
            logger.warning("Using custom wake word model: %s", model_path)
            self._porcupine = pvporcupine.create(
                access_key=access_key, keyword_paths=[model_path]
            )
        else:
            if model_path:
                logger.warning(
                    "WAKE_WORD_MODEL_PATH is set (%s) but the file does not "
                    "exist; falling back to the built-in 'computer' keyword.",
                    model_path,
                )
            else:
                logger.warning(
                    "WAKE_WORD_MODEL_PATH is not set; using the built-in "
                    "'computer' keyword as a placeholder until a custom "
                    "Eden .ppn file is trained."
                )
            self._porcupine = pvporcupine.create(
                access_key=access_key, keywords=["computer"]
            )

        if self._porcupine.sample_rate != SEND_SAMPLE_RATE:
            raise RuntimeError(
                f"Porcupine requires {self._porcupine.sample_rate} Hz input, "
                f"but the mic stream is {SEND_SAMPLE_RATE} Hz. "
                "Resampling is not supported."
            )

        self._frame_length = self._porcupine.frame_length
        self._frame_bytes = self._frame_length * 2

    def process(self, audio_chunk: bytes) -> bool:
        self._buffer.extend(audio_chunk)
        detected = False
        while len(self._buffer) >= self._frame_bytes:
            frame = np.frombuffer(self._buffer[: self._frame_bytes], dtype=np.int16)
            del self._buffer[: self._frame_bytes]
            if self._porcupine.process(frame) >= 0:
                detected = True
        return detected

    def close(self):
        self._porcupine.delete()