"""SpeakerRecognizer scoring paths.

The headline regression: an exception raised by the embedding extractor while
scoring must NEVER reach on_result/_emit - this has regressed twice before, so
it is asserted directly for both the async and background scoring paths.
"""

import numpy as np
import pytest

from identity.recognition import SpeakerRecognizer, SCORING_MIN_MS
from identity.voiceprint_store import VoiceprintStore


class _RaisingExtractor:
    def extract(self, audio):
        raise RuntimeError("simulated extractor failure")


class _FixedExtractor:
    def __init__(self, embedding) -> None:
        self.embedding = np.asarray(embedding, dtype=np.float32)
        self.calls = 0

    def extract(self, audio):
        self.calls += 1
        return self.embedding


def _recognizer(extractor, tmp_path):
    store = VoiceprintStore(profiles_dir=str(tmp_path / "profiles"),
                            owner_file=str(tmp_path / "owner.json"))
    return SpeakerRecognizer(extractor=extractor, store=store)


def _long_utts() -> bytes:
    """int16 PCM long enough to cross SCORING_MIN_MS (1 s + a bit)."""
    samples = (SCORING_MIN_MS + 50) * 16
    return b"\x00\x00" * samples


# --- The never-emit-on-exception regression --------------------------------

async def test_extractor_exception_never_reaches_on_result_async(tmp_path):
    recognizer = _recognizer(_RaisingExtractor(), tmp_path)
    results = []
    recognizer.on_result(lambda *args: results.append(args))
    # A profile must exist, otherwise scoring bails out before the extractor.
    recognizer._store.save_profile("alex", np.array([1.0, 0.0], dtype=np.float32))

    await recognizer._score_async(_long_utts())

    assert results == []


def test_extractor_exception_never_reaches_emit_background(tmp_path):
    recognizer = _recognizer(_RaisingExtractor(), tmp_path)
    emitted = []
    recognizer.on_result(lambda *args: emitted.append(args))
    recognizer._store.save_profile("alex", np.array([1.0, 0.0], dtype=np.float32))

    recognizer._score_background(_long_utts())

    assert emitted == []


async def test_extractor_exception_never_synthesizes_unrecognized_result(tmp_path):
    recognizer = _recognizer(_RaisingExtractor(), tmp_path)
    results = []
    recognizer.on_result(lambda *args: results.append(args))
    recognizer._store.save_profile("alex", np.array([1.0, 0.0], dtype=np.float32))

    await recognizer._score_async(_long_utts())

    # No (None, low_confidence, False) placeholder either - silence is the
    # only honest outcome of a failed measurement.
    assert results == []


# --- Too short to score ------------------------------------------------------

async def test_too_short_utterance_is_skipped_without_extractor(tmp_path):
    extractor = _FixedExtractor([1.0, 0.0])
    recognizer = _recognizer(extractor, tmp_path)
    results = []
    recognizer.on_result(lambda *args: results.append(args))
    short = b"\x00\x00" * (SCORING_MIN_MS - 100) * 16

    await recognizer._score_async(short)

    assert extractor.calls == 0
    assert results == []


# --- Happy path --------------------------------------------------------------

async def test_confident_match_emits_recognized_result(tmp_path):
    extractor = _FixedExtractor([1.0, 0.0])
    recognizer = _recognizer(extractor, tmp_path)
    recognizer._store.save_profile("alex", np.array([1.0, 0.0], dtype=np.float32))
    results = []
    recognizer.on_result(lambda *args: results.append(args))

    await recognizer._score_async(_long_utts())

    assert len(results) == 1
    name, confidence, recognized, duration_s = results[0]
    assert name == "alex"
    assert confidence >= 0.8
    assert recognized is True
    assert duration_s is not None and duration_s > 1.0


async def test_no_profiles_is_default_deny(tmp_path):
    extractor = _FixedExtractor([1.0, 0.0])
    recognizer = _recognizer(extractor, tmp_path)
    results = []
    recognizer.on_result(lambda *args: results.append(args))

    await recognizer._score_async(_long_utts())

    assert results == [(None, 0.0, False, pytest.approx(1.05, abs=0.06))]


# --- Enrollment collection ----------------------------------------------------

async def test_collect_enrollment_gathers_samples_then_returns_to_scoring(tmp_path):
    extractor = _FixedExtractor([1.0, 0.0])
    recognizer = _recognizer(extractor, tmp_path)
    samples = []
    completions = []
    recognizer.collect_enrollment(
        2,
        on_sample=lambda **kw: samples.append(kw),
        on_complete=lambda emb: completions.append(emb),
    )

    await recognizer._score_async(_long_utts())
    assert samples == [{"check_count": 1, "target": 2}]
    assert completions == []

    await recognizer._score_async(_long_utts())
    assert samples == [
        {"check_count": 1, "target": 2},
        {"check_count": 2, "target": 2},
    ]
    assert len(completions) == 1
    assert len(completions[0]) == 2
    assert recognizer._enroll_mode() is False