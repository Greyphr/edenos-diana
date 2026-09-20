"""enrollment_core: coherence gating and first-ever owner claiming."""

import numpy as np
import pytest

from identity.enrollment_core import (
    AGREEMENT_MIN_SIMILARITY,
    enrollment_is_coherent,
    save_enrollment,
)
from identity.voiceprint_store import VoiceprintStore

EMBEDDING = np.array([0.6, 0.8], dtype=np.float32)
ORTHOGONAL = np.array([-0.8, 0.6], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    return VoiceprintStore(
        profiles_dir=str(tmp_path / "profiles"),
        owner_file=str(tmp_path / "owner.json"),
    )


# --- enrollment_is_coherent --------------------------------------------------

def test_single_embedding_is_coherent():
    assert enrollment_is_coherent([EMBEDDING])
    assert enrollment_is_coherent([])


def test_identical_embeddings_are_coherent():
    assert enrollment_is_coherent([EMBEDDING, EMBEDDING, EMBEDDING])


def test_mismatched_embeddings_are_incoherent():
    assert enrollment_is_coherent([EMBEDDING, ORTHOGONAL, EMBEDDING]) is False


def test_marginally_different_embeddings_are_still_coherent():
    near = np.array([0.6001, 0.7999], dtype=np.float32)
    assert enrollment_is_coherent([EMBEDDING, near])


def test_threshold_constant_is_exposed():
    assert AGREEMENT_MIN_SIMILARITY == 0.6


# --- save_enrollment claiming -------------------------------------------------

def test_first_ever_enrollment_claims_owner(store):
    result = save_enrollment("alex", [EMBEDDING, EMBEDDING], store=store)
    assert result["saved"] is True
    assert result["owner_claimed"] is True
    assert store.get_owner_name() == "alex"
    assert store.list_profiles() == ["alex"]


def test_rerun_never_reclaims_owner(store):
    save_enrollment("alex", [EMBEDDING, EMBEDDING], store=store)
    second = save_enrollment("morgan", [EMBEDDING, EMBEDDING], store=store)
    assert second["owner_claimed"] is False
    assert store.get_owner_name() == "alex"
    assert set(store.list_profiles()) == {"alex", "morgan"}


def test_incoherent_enrollment_raises_without_force(store):
    with pytest.raises(ValueError, match="same speaker"):
        save_enrollment("alex", [EMBEDDING, ORTHOGONAL], store=store)
    assert store.list_profiles() == []
    assert store.get_owner_name() is None


def test_force_overrides_coherence_gate(store):
    result = save_enrollment("alex", [EMBEDDING, ORTHOGONAL], store=store, force=True)
    assert result["saved"] is True
    assert store.list_profiles() == ["alex"]


def test_empty_embeddings_raise(store):
    with pytest.raises(ValueError, match="no phrase embeddings"):
        save_enrollment("alex", [], store=store)