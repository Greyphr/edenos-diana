"""UtteranceVAD: utterance completion, blip rejection, hangover stitching.

Uses synthetic int16/16 kHz PCM (440 Hz sine) so nothing touches real audio
or the microphone.

Frame accounting: each 10 ms frame is 160 samples (320 bytes). Trailing quiet
finishes an utterance after exactly ``hangover_ms / 10`` decrementing frames
plus the one transition frame that entered TRAILING, i.e. the emitted audio
ends ``hangover_ms + FRAME_MS`` of trailing quiet after the last speech frame.
"""

import numpy as np

from identity.vad import DEFAULT_HANGOVER_MS, SAMPLE_RATE, UtteranceVAD

FRAME_MS = 10
_SPEECH_AMPLITUDE = 22000
_QUIET_AMPLITUDE = 20

# Quiet that guarantees an utterance completes: hangover plus two frames of
# slack (the finish happens after hangover + one frame).
_TRAIL_MS = DEFAULT_HANGOVER_MS + 2 * FRAME_MS


def _wave(ms: int, amplitude: int) -> bytes:
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.arange(n) / SAMPLE_RATE
    samples = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    return samples.tobytes()


def _chunks(ms: int, amplitude: int):
    full = _wave(ms, amplitude)
    step = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 320 bytes per 10 ms frame
    for i in range(0, len(full), step):
        yield full[i : i + step]


class _Recorder:
    def __init__(self) -> None:
        self.utterances: list[bytes] = []

    def __call__(self, audio: bytes) -> None:
        self.utterances.append(audio)


def _emit_bytes(speech_ms: int) -> int:
    """Bytes captured for one utterance ending after hangover + one frame."""
    return (speech_ms + DEFAULT_HANGOVER_MS + FRAME_MS) * SAMPLE_RATE // 1000 * 2


def test_sustained_speech_produces_one_completed_utterance():
    recorder = _Recorder()
    vad = UtteranceVAD()
    vad.on_utterance = recorder
    for chunk in _chunks(500, _SPEECH_AMPLITUDE):
        vad.process(chunk)
    for chunk in _chunks(_TRAIL_MS, _QUIET_AMPLITUDE):
        vad.process(chunk)
    assert len(recorder.utterances) == 1
    assert len(recorder.utterances[0]) == _emit_bytes(500)


def test_blip_under_minimum_duration_is_discarded():
    recorder = _Recorder()
    vad = UtteranceVAD()
    vad.on_utterance = recorder
    for chunk in _chunks(100, _SPEECH_AMPLITUDE):  # 100 ms < 300 ms minimum
        vad.process(chunk)
    for chunk in _chunks(_TRAIL_MS, _QUIET_AMPLITUDE):
        vad.process(chunk)
    assert recorder.utterances == []

    # A real utterance afterward still works (the blip didn't wedge the vad).
    for chunk in _chunks(400, _SPEECH_AMPLITUDE):
        vad.process(chunk)
    for chunk in _chunks(_TRAIL_MS, _QUIET_AMPLITUDE):
        vad.process(chunk)
    assert len(recorder.utterances) == 1
    assert len(recorder.utterances[0]) == _emit_bytes(400)


def test_hangover_prevents_premature_cutoff():
    recorder = _Recorder()
    vad = UtteranceVAD()
    vad.on_utterance = recorder
    # Speech -> quiet-shaped gap (100 ms, inside the 350 ms hangover) -> more
    # speech: must stay ONE utterance and stitch the gap in, not cut at it.
    for chunk in _chunks(300, _SPEECH_AMPLITUDE):
        vad.process(chunk)
    for chunk in _chunks(100, _QUIET_AMPLITUDE):
        vad.process(chunk)
    assert recorder.utterances == []  # not cut off by the brief quiet gap
    for chunk in _chunks(300, _SPEECH_AMPLITUDE):
        vad.process(chunk)
    for chunk in _chunks(_TRAIL_MS, _QUIET_AMPLITUDE):
        vad.process(chunk)

    assert len(recorder.utterances) == 1
    # 300 speech + 100 gap + 300 speech, then hangover + one frame of quiet.
    total_ms = 300 + 100 + 300 + DEFAULT_HANGOVER_MS + FRAME_MS
    assert len(recorder.utterances[0]) == total_ms * SAMPLE_RATE // 1000 * 2


def test_min_duration_parameter_is_honored():
    recorder = _Recorder()
    vad = UtteranceVAD(min_utterance_ms=600)
    vad.on_utterance = recorder
    for chunk in _chunks(500, _SPEECH_AMPLITUDE):
        vad.process(chunk)
    for chunk in _chunks(_TRAIL_MS, _QUIET_AMPLITUDE):
        vad.process(chunk)
    assert recorder.utterances == []