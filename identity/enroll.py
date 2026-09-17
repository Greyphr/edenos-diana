"""Standalone speaker enrollment.

Run:  python -m identity.enroll [--name christopher]

Opens its own short mic session and prompts you to say a few phrases. Each
captured phrase is segmented with the same VAD logic used during recognition,
embedded, and the average embedding is saved as your voiceprint profile.
"""

import argparse
import logging
import sys
import threading
import time

import sounddevice as sd

from identity.embeddings import EmbeddingExtractor
from identity.vad import SAMPLE_RATE, UtteranceVAD
from identity.voiceprint_store import VoiceprintStore

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 1024
CAPTURE_TIMEOUT_SECONDS = 15.0
MIN_PHRASE_SECONDS = 0.5

PHRASES = [
    "This is my voice, and Eden will know it.",
    "Good morning Eden, it's good to talk to you.",
    "I speak for myself, in my own home.",
    "Play something soothing for the evening, Eden.",
    "My name is my name, and my voice says the rest.",
]


def capture_one_phrase(vad, stream, capturer) -> bytes | None:
    """Listens until one utterance completes (or times out) and returns it."""
    vad.reset()
    capturer.reset()
    deadline = time.monotonic() + CAPTURE_TIMEOUT_SECONDS
    while not capturer.done.is_set() and time.monotonic() < deadline:
        data, _overflowed = stream.read(CHUNK_SAMPLES)
        vad.process(data.tobytes())
    return capturer.audio


class _Capturer:
    def __init__(self, vad):
        self.vad = vad
        self.audio = None
        self.done = threading.Event()
        self.vad.on_utterance = self._on_utterance

    def reset(self):
        self.audio = None
        self.done.clear()

    def _on_utterance(self, audio):
        if not self.done.is_set():
            self.audio = audio
            self.done.set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enroll a speaker voiceprint for Eden."
    )
    parser.add_argument(
        "--name", help="Name to save the voiceprint under (e.g. christopher)."
    )
    parser.add_argument(
        "--phrases", nargs="+", help="Phrases to say during enrollment."
    )
    args = parser.parse_args()

    name = (args.name or "").strip()
    if not name:
        name = input(
            "What name should this voiceprint be saved under? [christopher] "
        ).strip()
        if not name:
            name = "christopher"
    phrases = args.phrases or PHRASES

    print("=" * 60)
    print("SPEAKER ENROLLMENT")
    print("=" * 60)
    print(f"Voiceprint will be saved under the name: {name}")
    print(f"You will be asked to say {len(phrases)} phrases.")
    print("Speak naturally, at your normal volume and distance from the mic.")
    print()

    extractor = EmbeddingExtractor()
    store = VoiceprintStore()

    vad = UtteranceVAD()
    capturer = _Capturer(vad)
    embeddings = []

    print(f"Opening microphone ({SAMPLE_RATE} Hz)...\n")
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            for i, phrase in enumerate(phrases, start=1):
                print(f"[{i}/{len(phrases)}] When you see 'Listening', say:")
                print(f"      \"{phrase}\"")
                input("Press Enter when you are ready, then speak...")
                print("Listening...", flush=True)
                audio = None
                while audio is None:
                    audio = capture_one_phrase(vad, stream, capturer)
                    if audio is None:
                        print("  No speech detected - please try again.")
                        continue
                    seconds = len(audio) / 2 / SAMPLE_RATE
                    if seconds < MIN_PHRASE_SECONDS:
                        print(
                            f"  That was only {seconds:.1f}s - "
                            "please say the whole phrase."
                        )
                        audio = None
                embedding = extractor.extract(audio)
                embeddings.append(embedding)
                print(
                    f"  Captured {seconds:.1f}s of audio - "
                    f"phrase {i} of {len(phrases)} recorded."
                )
    except KeyboardInterrupt:
        print("\nEnrollment cancelled.")
        sys.exit(1)

    combined = store.average_embeddings(embeddings)
    store.save_profile(name, combined)
    print()
    print(f"Voiceprint for \"{name}\" saved to:")
    print(f"  {store.profiles_dir}")
    print("Done. Run `python main.py` and when you talk during an engaged")
    print("session the console will log whether your voice is recognized.")


if __name__ == "__main__":
    main()