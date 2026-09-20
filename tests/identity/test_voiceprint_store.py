"""VoiceprintStore: profile save/list/load/archive round trips and the
owner-marker auto-claim edge cases."""

import json
import os

import numpy as np
import pytest

from identity.voiceprint_store import VoiceprintStore

EMBEDDING = np.array([0.6, 0.8], dtype=np.float32)


@pytest.fixture
def store(tmp_path):
    return VoiceprintStore(
        profiles_dir=str(tmp_path / "profiles"),
        owner_file=str(tmp_path / "owner.json"),
    )


def test_save_list_load_roundtrip(store):
    saved = store.save_profile("alex", EMBEDDING)
    assert saved["name"] == "alex"
    assert store.list_profiles() == ["alex"]
    loaded = store.load_profile("alex")
    assert loaded["name"] == "alex"
    assert loaded["embedding"] == pytest.approx([0.6, 0.8])


def test_load_missing_profile_returns_none(store):
    assert store.load_profile("nobody") is None


def test_load_all_sorted(store):
    store.save_profile("b", EMBEDDING)
    store.save_profile("a", EMBEDDING)
    assert [p["name"] for p in store.load_all()] == ["a", "b"]


def test_archive_moves_profile_aside(store):
    store.save_profile("alex", EMBEDDING)
    dest = store.archive_profile("alex")
    assert dest is not None
    assert os.path.isfile(dest)
    assert "archived" in dest
    assert store.list_profiles() == []
    assert store.load_profile("alex") is None
    assert os.listdir(os.path.join(store.profiles_dir, "archived")) == [os.path.basename(dest)]


def test_archiving_missing_profile_returns_none(store):
    assert store.archive_profile("ghost") is None


def test_two_archives_of_same_name_in_one_run_get_distinct_names(store):
    store.save_profile("alex", EMBEDDING)
    first = store.archive_profile("alex")
    store.save_profile("alex", EMBEDDING)
    second = store.archive_profile("alex")
    assert first != second
    assert os.path.isfile(first)
    assert os.path.isfile(second)
    assert store.list_profiles() == []


# --- Owner auto-claim --------------------------------------------------------

def test_single_profile_auto_claims_and_persists(store):
    store.save_profile("alex", EMBEDDING)
    assert store.get_owner_name() == "alex"
    # The claim is persisted: reading it back does not need the profile.
    with open(store.owner_file, encoding="utf-8") as f:
        assert json.load(f) == {"name": "alex"}


def test_zero_profiles_do_not_claim(store):
    assert store.get_owner_name() is None


def test_two_profiles_do_not_claim_ambiguously(store):
    store.save_profile("alex", EMBEDDING)
    store.save_profile("morgan", EMBEDDING)
    assert store.get_owner_name() is None
    assert not os.path.isfile(store.owner_file)


def test_existing_marker_wins_even_with_no_profiles(store):
    store.set_owner_name("chris")
    assert store.get_owner_name() == "chris"


def test_set_owner_roundtrip(store):
    store.set_owner_name("alex")
    assert store.get_owner_name() == "alex"


# --- average_embeddings ------------------------------------------------------

def test_average_embeddings_is_unit_normalized(store):
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    mean = VoiceprintStore.average_embeddings([a, b])
    assert np.linalg.norm(mean) == pytest.approx(1.0)
    assert mean[0] == pytest.approx(mean[1])


def test_average_embeddings_rejects_empty(store):
    with pytest.raises(ValueError, match="no embeddings"):
        VoiceprintStore.average_embeddings([])