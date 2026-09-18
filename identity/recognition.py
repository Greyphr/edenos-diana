import asyncio
import logging
import threading

import numpy as np

from identity.embeddings import EmbeddingExtractor
from identity.vad import SAMPLE_RATE, UtteranceVAD
from identity.voiceprint_store import VoiceprintStore

logger = logging.getLogger(__name__)

# Cosine-similarity thresholds against the enrolled profile.
CONFIDENCE_RECOGNIZED = 0.80  # >= this: recognized
CONFIDENCE_UNCERTAIN = 0.70   # 0.70-0.89: uncertain, reported as not recognized
# Utterances shorter than this rarely produce a reliable embedding, so they
# are skipped rather than scored and reported as low-confidence.
SCORING_MIN_MS = 1000


def utterance_duration_ms(audio: bytes) -> int:
    """Duration of an int16/16 kHz utterance, in milliseconds."""
    samples = len(audio) // 2
    return round(samples * 1000 / SAMPLE_RATE)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SpeakerRecognizer:
    """Segments engaged audio into utterances and scores them against
    enrolled voiceprints.

    - ``feed(chunk)`` is the hot audio-forwarding path: it only appends to
      the VAD and never runs embedding extraction inline.
    - ``on_result(callback)`` registers a callback of the form
      ``callback(name: str | None, confidence: float, recognized: bool,
      duration_s: float | None)`` fired whenever a scored utterance ends.
      Utterances shorter than ``SCORING_MIN_MS`` are not scored and fire
      nothing.
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
            result = await asyncio.to_thread(self._score_sync, audio)
        except Exception:
            logger.exception("Speaker recognition failed for an utterance")
            result = (None, 0.0, False, None)
        if result is None:
            return  # too short to score reliably; no result emitted
        self._emit(*result)

    def _score_background(self, audio: bytes) -> None:
        try:
            result = self._score_sync(audio)
        except Exception:
            logger.exception("Speaker recognition failed for an utterance")
            result = (None, 0.0, False, None)
        if result is None:
            return  # too short to score reliably; no result emitted
        self._emit(*result)

    def _score_sync(self, audio: bytes):
        dur_ms = utterance_duration_ms(audio)
        if dur_ms < SCORING_MIN_MS:
            logger.debug(
                "Skipping utterance too short to score: %d ms (< %d ms)",
                dur_ms, SCORING_MIN_MS,
            )
            return None
        profiles = self._store.load_all()
        if not profiles:
            # Default-deny: no profile enrolled yet, always unrecognized.
            return None, 0.0, False, dur_ms
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
            return best_name, best_similarity, True, dur_ms
        return None, best_similarity, False, dur_ms

    def _emit(self, name, confidence, recognized, duration_ms=None) -> None:
        duration_s = duration_ms / 1000.0 if duration_ms is not None else None
        for callback in list(self._callbacks):
            try:
                callback(name, confidence, recognized, duration_s)
            except Exception:
                logger.exception("on_result callback raised")