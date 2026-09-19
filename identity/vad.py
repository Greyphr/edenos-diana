import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

DEFAULT_MIN_UTTERANCE_MS = 300
DEFAULT_HANGOVER_MS = 350
DEFAULT_MAX_UTTERANCE_MS = 10000

# Energy is the RMS of int16 samples. A rising edge only begins speech when
# it clears BOTH the noise-floor margin and an absolute floor, so faint
# ambience can never trigger an utterance.
SPEECH_MARGIN = 1.6
SPEECH_ABSOLUTE_FLOOR = 150.0
HANGOVER_QUIET_RATIO = 1.3

NOISE_FLOOR_DOWN = 0.05  # fast follow when ambience drops
NOISE_FLOOR_UP = 0.005   # slow drift when ambience rises


def _rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))


class UtteranceVAD:
    """Energy-based utterance boundary detector.

    Segments a stream of 16 kHz int16 PCM chunks into discrete utterances.
    Each completed utterance is emitted through the ``on_utterance`` callback
    as raw int16 PCM bytes. This is separate from Gemini's own turn detection:
    it exists purely to hand clean, buffered audio to the embedding extractor.
    """

    IDLE = 0
    SPEAKING = 1
    TRAILING = 2

    def __init__(
        self,
        min_utterance_ms: int = DEFAULT_MIN_UTTERANCE_MS,
        hangover_ms: int = DEFAULT_HANGOVER_MS,
        max_utterance_ms: int = DEFAULT_MAX_UTTERANCE_MS,
    ) -> None:
        self.min_utterance_ms = min_utterance_ms
        self.hangover_ms = hangover_ms
        self.max_utterance_ms = max_utterance_ms
        self.on_utterance: Callable[[bytes], None] | None = None
        self._floor: float | None = None
        self.reset()

    def reset(self) -> None:
        """Abandon any utterance in progress.

        Clears the mid-utterance buffer, speech/hangover tracking, and state
        without discarding the adaptively-learned noise floor — relearning
        ambience takes seconds of quiet audio, and an idle gap between
        engaged sessions shouldn't throw it away. Used on the way back to
        idle so a half-finished utterance buffer from one session can never
        be stitched onto the first audio of the next.
        """
        self._state = self.IDLE
        self._buffer = bytearray()
        self._speech_ms = 0
        self._hangover_left = 0

    def process(self, chunk: bytes) -> None:
        frame_ms = max(1, round(len(chunk) // 2 * 1000 / SAMPLE_RATE))
        energy = _rms(chunk)
        floor = self._floor if self._floor is not None else 0.0
        onset_threshold = max(floor * SPEECH_MARGIN, SPEECH_ABSOLUTE_FLOOR)
        quiet_threshold = max(
            floor * HANGOVER_QUIET_RATIO, SPEECH_ABSOLUTE_FLOOR * 0.6
        )

        if self._state == self.IDLE:
            if energy >= onset_threshold:
                self._state = self.SPEAKING
                self._speech_ms = frame_ms
                self._buffer = bytearray(chunk)
                logger.debug(
                    "speech started (energy=%.0f above floor=%.0f)", energy, floor
                )
            else:
                self._adapt_floor(energy)
            return

        self._buffer.extend(chunk)

        if self._state == self.SPEAKING:
            self._speech_ms += frame_ms
            if energy <= quiet_threshold:
                self._state = self.TRAILING
                self._hangover_left = max(1, round(self.hangover_ms / frame_ms))
            elif self._speech_ms >= self.max_utterance_ms:
                self._finish_utterance()
            return

        # TRAILING: keep accumulating while we wait out the hangover window.
        if energy > quiet_threshold:
            self._state = self.SPEAKING
            self._speech_ms += frame_ms
            self._hangover_left = 0
        else:
            self._adapt_floor(energy)
            self._hangover_left -= 1
            if self._hangover_left <= 0:
                self._finish_utterance()

    def _adapt_floor(self, energy: float) -> None:
        if self._floor is None:
            self._floor = max(energy, SPEECH_ABSOLUTE_FLOOR * 0.2)
        elif energy < self._floor:
            self._floor += NOISE_FLOOR_DOWN * (energy - self._floor)
        else:
            self._floor += NOISE_FLOOR_UP * (energy - self._floor)

    def _finish_utterance(self) -> None:
        speech_ms = self._speech_ms
        audio = bytes(self._buffer)
        self.reset()
        if speech_ms < self.min_utterance_ms:
            logger.debug("discarded %.0f ms as a noise blip", speech_ms)
            return
        if callable(self.on_utterance):
            try:
                self.on_utterance(audio)
            except Exception:
                logger.exception("on_utterance callback raised")