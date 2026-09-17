import asyncio
import logging
import threading

import numpy as np

from identity.embeddings import EmbeddingExtractor
from identity.vad import UtteranceVAD
from identity.voiceprint_store import VoiceprintStore

logger = logging.getLogger(__name__)

# Cosine-similarity thresholds against the enrolled profile.
CONFIDENCE_RECOGNIZED = 0.90  # >= this: recognized
CONFIDENCE_UNCERTAIN = 0.70   # 0.70-0.89: uncertain, reported as not recognized


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SpeakerRecognizer:
    """Segments engaged audio into utterances and scores them against
    enrolled voiceprints.

    - ``feed(chunk)`` is the hot audio-forwarding path: it only appends to
      the VAD and never runs embedding extraction inline.
    - ``on_result(callback)`` registers a callback of the form
      ``callback(name: str | None, confidence: float, recognized: bool)``
      fired whenever a completed utterance has been scored.
    """

    def __init__(
        self, extractor: EmbeddingExtractor | None = None,
        store: VoiceprintStore | None = None,
    ) -> None:
        self._extractor = extractor or EmbeddingExtractor()
        self._store = store or VoiceprintStore()
        self._vad = UtteranceVAD()
        self._vad.on_utterance = self._on_utterance
        self._callbacks = []

    def on_result(self, callback) -> None:
        self._callbacks.append(callback)

    def feed(self, chunk: bytes) -> None:
        """Cheap, non-blocking: append to the VAD buffer. Never scores inline."""
        try:
            self._vad.process(chunk)
        except Exception:
            logger.exception("VAD processing failed")

    def _on_utterance(self, audio: bytes) -> None:
        # Scored off the hot path: schedule a background task/thread.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            thread = threading.Thread(
                target=self._score_background, args=(audio,), daemon=True
            )
            thread.start()
        else:
            asyncio.create_task(self._score_async(audio))

    async def _score_async(self, audio: bytes) -> None:
        try:
            name, confidence, recognized = await asyncio.to_thread(
                self._score_sync, audio
            )
        except Exception:
            logger.exception("Speaker recognition failed for an utterance")
            name, confidence, recognized = None, 0.0, False
        self._emit(name, confidence, recognized)

    def _score_background(self, audio: bytes) -> None:
        try:
            name, confidence, recognized = self._score_sync(audio)
        except Exception:
            logger.exception("Speaker recognition failed for an utterance")
            name, confidence, recognized = None, 0.0, False
        self._emit(name, confidence, recognized)

    def _score_sync(self, audio: bytes):
        profiles = self._store.load_all()
        if not profiles:
            # Default-deny: no profile enrolled yet, always unrecognized.
            return None, 0.0, False
        embedding = self._extractor.extract(audio)
        best_name, best_similarity = None, 0.0
        for profile in profiles:
            profile_embedding = np.asarray(
                profile.get("embedding", ()), dtype=np.float32
            )
            similarity = cosine_similarity(embedding, profile_embedding)
            if similarity > best_similarity:
                best_similarity, best_name = similarity, profile.get("name")
        if best_similarity >= CONFIDENCE_RECOGNIZED:
            return best_name, best_similarity, True
        return None, best_similarity, False

    def _emit(self, name, confidence, recognized) -> None:
        for callback in list(self._callbacks):
            try:
                callback(name, confidence, recognized)
            except Exception:
                logger.exception("on_result callback raised")