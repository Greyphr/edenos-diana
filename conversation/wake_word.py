import logging

import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models

from conversation.audio_io import SEND_SAMPLE_RATE

logger = logging.getLogger(__name__)

OPENWAKEWORD_SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # openwakeword processes audio in 80 ms (1280 sample) frames
WAKEWORD_MODEL = "hey_jarvis"
DETECTION_THRESHOLD = 0.5


class WakeWordDetector:
    def __init__(self):
        if OPENWAKEWORD_SAMPLE_RATE != SEND_SAMPLE_RATE:
            raise RuntimeError(
                f"openwakeword requires {OPENWAKEWORD_SAMPLE_RATE} Hz input, "
                f"but the mic stream is {SEND_SAMPLE_RATE} Hz. "
                "Resampling is not supported."
            )

        download_models([WAKEWORD_MODEL])

        logger.warning(
            "Using the stock '%s' model as a placeholder until a custom "
            "Eden wake-word model is trained.",
            WAKEWORD_MODEL,
        )

        self._model = Model(
            wakeword_models=[WAKEWORD_MODEL],
            inference_framework="onnx",
        )
        self._frame_bytes = FRAME_SAMPLES * 2
        self._buffer = bytearray()

    def process(self, audio_chunk: bytes) -> bool:
        self._buffer.extend(audio_chunk)
        detected = False
        while len(self._buffer) >= self._frame_bytes:
            frame = np.frombuffer(self._buffer[: self._frame_bytes], dtype=np.int16)
            del self._buffer[: self._frame_bytes]
            scores = self._model.predict(frame)
            # predict() returns the per-model score dict (timing=False, which
            # this detector never enables). Guard the shape anyway so a future
            # SDK change can't crash wake-word processing.
            if isinstance(scores, dict) and any(
                float(score) >= DETECTION_THRESHOLD for score in scores.values()
            ):
                detected = True
        return detected

    def close(self):
        pass