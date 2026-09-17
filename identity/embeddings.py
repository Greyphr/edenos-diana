import logging
import threading
import warnings

import numpy as np

# webrtcvad (a resemblyzer dep) still imports pkg_resources; keep its
# deprecation warning from cluttering the console at startup.
warnings.filterwarnings(
    "ignore", message="pkg_resources is deprecated", category=UserWarning
)
from resemblyzer import VoiceEncoder, preprocess_wav

logger = logging.getLogger(__name__)

VOICE_ENCODER_SAMPLE_RATE = 16000


def wav_to_float32(audio: bytes) -> np.ndarray:
    """Convert int16 PCM bytes to the float32 waveform resemblyzer expects."""
    return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0


class EmbeddingExtractor:
    """Extracts speaker embeddings using resemblyzer's VoiceEncoder.

    The encoder model is loaded once at construction (which is slow) and
    reused for every utterance.
    """

    def __init__(self) -> None:
        logger.info(
            "Loading voice encoder model (first load takes a few seconds)..."
        )
        self._encoder = VoiceEncoder()
        # embed_utterance mutates the encoder's internal LSTM state, so
        # concurrent scoring calls must not interleave.
        self._lock = threading.Lock()
        logger.info("Voice encoder ready.")

    def extract(self, audio: bytes) -> np.ndarray:
        wav = wav_to_float32(audio)
        wav = preprocess_wav(wav)
        if wav.shape[0] < VOICE_ENCODER_SAMPLE_RATE // 10:
            raise ValueError("utterance too short for embedding extraction")
        with self._lock:
            embedding = self._encoder.embed_utterance(wav)
        return np.asarray(embedding, dtype=np.float32)