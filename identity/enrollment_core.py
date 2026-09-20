"""Shared first-time-owner enrollment logic.

Both the offline CLI (``python -m identity.enroll``) and the voice-driven
first-run bootstrap (main.py) funnel their captured phrase embeddings through
``save_enrollment`` so the "average + persist + claim ownership" logic lives
in exactly one place.
"""

import logging

import numpy as np

from identity.recognition import cosine_similarity
from identity.voiceprint_store import VoiceprintStore

logger = logging.getLogger(__name__)

# Minimum pairwise cosine similarity between captured phrases before an
# enrollment is trusted. Below this, at least two phrases were almost
# certainly not from the same speaker - the average will be a muddled
# embedding that recognizes nobody (mirror of enroll.py's own threshold).
AGREEMENT_MIN_SIMILARITY = 0.6


def enrollment_is_coherent(embeddings: list[np.ndarray]) -> bool:
    """Whether the captured phrases plausibly came from one speaker.

    Returns True for fewer than two embeddings (a single-phrase enrollment
    has nothing to disagree with). Otherwise all phrases must stay within
    ``AGREEMENT_MIN_SIMILARITY`` of each other.
    """
    if len(embeddings) < 2:
        return True
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if cosine_similarity(embeddings[i], embeddings[j]) < AGREEMENT_MIN_SIMILARITY:
                return False
    return True


def save_enrollment(
    name: str,
    phrase_embeddings: list[np.ndarray],
    store: VoiceprintStore | None = None,
    force: bool = False,
) -> dict:
    """Average the phrase embeddings, save the profile, and claim ownership
    when this is the first-ever enrollment.

    Returns a result dict describing what happened: ``saved`` (bool) and
    ``owner_claimed`` (bool). Ownership is claimed ONLY by a genuinely
    first-ever enrollment (no owner marker existed before this save); a
    re-run that already has an owner — even for the same person — never
    changes it.

    ``force`` bypasses the coherence gate for callers (like the CLI) that
    have already surfaced the inconsistency warning and been told to
    continue - otherwise the check would fire twice and the second one
    would raise after the user explicitly proceeded.
    """
    store = store or VoiceprintStore()
    if not phrase_embeddings:
        raise ValueError("no phrase embeddings to enroll")
    if not force and not enrollment_is_coherent(phrase_embeddings):
        raise ValueError(
            "captured phrases scored below the mutual-similarity threshold "
            "and may not be from the same speaker; redo the enrollment"
        )
    owner_was = store.get_owner_name()
    combined = store.average_embeddings(phrase_embeddings)
    store.save_profile(name, combined)
    owner_claimed = False
    if owner_was is None:
        store.set_owner_name(name)
        owner_claimed = True
        logger.info("First-ever enrollment: %r claimed owner", name)
    return {
        "name": name,
        "saved": True,
        "owner_claimed": owner_claimed,
        "profiles_dir": store.profiles_dir,
    }