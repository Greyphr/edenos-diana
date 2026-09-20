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
CONFIDENCE_UNCERTAIN = 0.70   # 0.70-0.79: uncertain band - neither a
                              # confident match nor a confident non-match
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
    - ``collect_enrollment(target, on_sample, on_complete)`` switches the
      recognizer into enrollment mode: instead of matching completed
      utterances against stored profiles, it extracts an embedding from each
      one and fires ``on_sample(count, target)``. After ``target`` samples
      are gathered, ``on_complete(embeddings)`` fires, the recognizer
      silently drops back to normal scoring, and the same VAD + extraction
      pipeline as ongoing recognition is reused (no second mic session).
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
        # In-flight async scoring tasks; kept so callers can wait on the
        # current utterance's result instead of judging on nothing yet, and
        # so the tasks aren't dropped (asyncio.create_task result held here).
        self._pending_tasks: set[asyncio.Task] = set()
        # Enrollment-collection state (None when in normal scoring mode).
        self._enroll_target: int | None = None
        self._enroll_collector_callback = None
        self._enroll_complete_callback = None
        self._enroll_embeddings: list[np.ndarray] = []

    def collect_enrollment(
        self, target: int, on_sample, on_complete, *, start_fresh: bool = True
    ) -> None:
        """Start collecting ``target`` phrase embeddings for enrollment.

        ``on_sample(count, target)`` fires per completed utterance (on the
        scoring thread/loop), ``on_complete(embeddings)`` fires once ``target``
        samples are gathered, after which scoring resumes normally.
        """
        if start_fresh:
            self._enroll_embeddings = []
        self._enroll_target = target
        self._enroll_collector_callback = on_sample
        self._enroll_complete_callback = on_complete

    def _enroll_mode(self) -> bool:
        return self._enroll_target is not None

    def on_result(self, callback) -> None:
        self._callbacks.append(callback)

    def reset(self) -> None:
        """Drop any half-scored utterance from the previous session.

        Called on the way back to idle alongside the trust-window reset, so
        a partially-buffered utterance can't survive into the next engaged
        session and get stitched onto new audio.
        """
        self._vad.reset()

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
            task = asyncio.create_task(self._score_async(audio))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

    async def wait_for_pending(self, timeout: float = 1.5) -> None:
        """Wait briefly for any in-flight scoring task to finish.

        Returns immediately when nothing is genuinely pending — either
        identity has already resolved or there's been no speech at all, so
        this costs nothing in the common cases.
        """
        pending = list(self._pending_tasks)
        if not pending:
            return
        await asyncio.wait(pending, timeout=timeout)

    async def _score_async(self, audio: bytes) -> None:
        try:
            result = await asyncio.to_thread(self._score_sync, audio)
        except Exception:
            # A recognizer exception is not a measurement: building a synthetic
            # (None, 0.0, False) result here and emitting it would feed the
            # trust window as though it were a real non-match. Just log and
            # emit nothing - the same no-result path the too-short case takes.
            logger.exception("Speaker recognition failed for an utterance")
            return
        if result is None:
            return  # too short to score reliably; no result emitted
        self._emit(*result)

    def _score_background(self, audio: bytes) -> None:
        try:
            result = self._score_sync(audio)
        except Exception:
            # See _score_async: never emit a synthetic unrecognized result
            # from an exception - only genuine measurements reach _emit.
            logger.exception("Speaker recognition failed for an utterance")
            return
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
        if self._enroll_mode():
            return self._enroll_sync(audio, dur_ms)
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

    def _enroll_sync(self, audio: bytes, dur_ms: int):
        """Enrollment mode: extract one embedding per utterance and collect.

        Runs on the same thread-shape as scoring (off the audio hot path),
        reusing the exact VAD + EmbeddingExtractor pipeline as recognition.
        On the target count the recognizer drops back to normal scoring
        silently before firing the completion callback.
        """
        embedding = self._extractor.extract(audio)
        self._enroll_embeddings.append(embedding)
        count = len(self._enroll_embeddings)
        collector = self._enroll_collector_callback
        if collector is not None:
            try:
                collector(check_count=count, target=self._enroll_target)
            except Exception:
                logger.exception("enrollment sample callback raised")
        if count >= self._enroll_target:
            self._enroll_target = None
            completed = self._enroll_embeddings
            self._enroll_embeddings = []
            complete = self._enroll_complete_callback
            self._enroll_complete_callback = None
            self._enroll_collector_callback = None
            if complete is not None:
                try:
                    complete(completed)
                except Exception:
                    logger.exception("enrollment completion callback raised")
        return None

    def _emit(self, name, confidence, recognized, duration_ms=None) -> None:
        duration_s = duration_ms / 1000.0 if duration_ms is not None else None
        for callback in list(self._callbacks):
            try:
                callback(name, confidence, recognized, duration_s)
            except Exception:
                logger.exception("on_result callback raised")