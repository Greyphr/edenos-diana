"""Standalone speaker enrollment.

Run:  python -m identity.enroll [--name <your-name>]

Opens its own short mic session and prompts you to say a few phrases. Each
captured phrase is segmented with the same VAD logic used during recognition,
embedded, and the average embedding is saved as your voiceprint profile.
The first-ever enrollment claims ownership (see identity/enrollment_core.py);
re-runs never change it.
"""

import argparse
import logging
import sys
import threading
import time

import numpy as np
import sounddevice as sd

from identity.embeddings import EmbeddingExtractor
from identity.enrollment_core import enrollment_is_coherent, save_enrollment
from identity.vad import SAMPLE_RATE, UtteranceVAD
from identity.voiceprint_store import VoiceprintStore

logger = logging.getLogger(__name__)

CHUNK_SAMPLES = 1024
CAPTURE_TIMEOUT_SECONDS = 15.0
MIN_PHRASE_SECONDS = 0.5

# Minimum pairwise cosine similarity between captured phrases before an
# enrollment is trusted. The threshold itself lives in
# identity/enrollment_core.py (AGREEMENT_MIN_SIMILARITY); this mirror keeps
# the CLI warning's wording on the same number.
AGREEMENT_MIN_SIMILARITY = 0.6

# Matches the enrollment target used by the voice-driven bootstrap.
ENROLLMENT_PHRASE_COUNT = 5

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


def minimum_pairwise_similarity(
    embeddings: list[np.ndarray],
) -> tuple[float, int, int] | None:
    """Least-similar pair of collected embeddings: (similarity, i, j).

    Returns None when fewer than two embeddings were collected, so a single-
    phrase enrollment has nothing to disagree with. Used only to surface the
    offending pair in the CLI warning; enrollment_core.enrollment_is_coherent
    is the author of the actual threshold check.
    """
    if len(embeddings) < 2:
        return None
    worst = None
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            if worst is None or sim < worst[0]:
                worst = (sim, i, j)
    return worst


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
        "--name", help="Name to save the voiceprint under (e.g. alex)."
    )
    parser.add_argument(
        "--phrases", nargs="+", help="Phrases to say during enrollment."
    )
    args = parser.parse_args()

    name = (args.name or "").strip()
    if not name:
        while True:
            name = input("What's your name? ").strip()
            if name:
                break
            print(
                "  Please enter a name - this is how Eden will refer to you "
                "after enrollment."
            )
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

    # Before averaging: if the two phrases that agree least are still far apart,
    # the capture is suspect (a different speaker, or wildly shifting mic
    # placement) and the average will be a muddled embedding that recognizes
    # nobody. Default to NOT saving - redo the capture instead. A "yes" here
    # is an explicit override, passed through as ``force`` so the shared
    # enrollment core doesn't raise a second, redundant complaint.
    force_save = False
    worst = minimum_pairwise_similarity(embeddings)
    if worst is not None and not enrollment_is_coherent(embeddings):
        sim, i, j = worst
        print()
        print(
            f"  Warning: phrases {i + 1} and {j + 1} scored only {sim:.2f} "
            f"similarity to each other (below {AGREEMENT_MIN_SIMILARITY:.1f})."
        )
        print("  Those two phrases may not have been spoken by the same person.")
        answer = input(
            "Some phrases sounded inconsistent - save anyway? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Enrollment cancelled - nothing was saved.")
            sys.exit(1)
        force_save = True

    # Never silently replace an existing voiceprint: warn and require an
    # explicit yes, archiving the old profile first (same recovery path
    # reenroll_voice uses) so it isn't destroyed by the overwrite.
    if name in store.list_profiles():
        print()
        print(f"  Warning: a voiceprint for \"{name}\" already exists.")
        answer = input(
            f"A voiceprint for '{name}' already exists - overwrite? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Enrollment cancelled - existing voiceprint kept.")
            sys.exit(1)
        archived = store.archive_profile(name)
        if archived:
            print(f"  Archived the previous voiceprint to:")
            print(f"    {archived}")

    result = save_enrollment(name, embeddings, store=store, force=force_save)
    print()
    print(f"Voiceprint for \"{name}\" saved to:")
    print(f"  {result['profiles_dir']}")
    if result["owner_claimed"]:
        print(f"  (First enrollment - you're the owner of this Eden.)")
    print("Done. Run `python main.py` and when you talk during an engaged")
    print("session the console will log whether your voice is recognized.")


if __name__ == "__main__":
    main()