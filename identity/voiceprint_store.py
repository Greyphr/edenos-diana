import json
import logging
import os
import tempfile
import time

import numpy as np

from tools.names import validate_name

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voiceprints"
)

DEFAULT_OWNER_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "owner.json"
)


class VoiceprintStore:
    """Persists enrolled speaker profiles as JSON files.

    A profile is a name plus an embedding vector (list of floats). Profile
    files live under ``identity/voiceprints/`` and are written atomically
    (temp file + os.replace) so a crash never leaves a half-written profile.
    """

    def __init__(
        self, profiles_dir: str = DEFAULT_PROFILES_DIR,
        owner_file: str = DEFAULT_OWNER_FILE,
    ) -> None:
        self.profiles_dir = profiles_dir
        self.owner_file = owner_file
        os.makedirs(self.profiles_dir, exist_ok=True)

    # Owner marker -----------------------------------------------------------
    # Ownership is an explicit marker (identity/owner.json), NOT inferred from
    # profile order: profiles can be added or removed later without that
    # disturbing who the owner is. First-ever enrollment claims it; re-runs
    # never change it.
    #
    # Deliberate edge case: a profile whose owner marker is missing (e.g.
    # voiceprints/ survived a wipe of owner.json) auto-claims ownership when
    # exactly one profile exists. With a single profile nobody could plausibly
    # be the owner but them (and README-level convenience), so a deleted
    # marker shouldn't force a full re-enrollment. With two or more profiles
    # the claim is ambiguous, so None is returned and the owner must be set
    # explicitly (re-enrollment).
    def get_owner_name(self) -> str | None:
        """Owner marker name, or None when nobody is marked.

        Deliberate auto-claim path: a profile whose owner marker is missing
        (e.g. ``voiceprints/`` survived a wipe of ``owner.json``) claims
        ownership when exactly one profile exists — with a single profile
        nobody else could plausibly be the owner, and the marker is persisted
        so the claim survives later profile additions. With two or more
        profiles and no marker the claim is ambiguous, so None is returned
        and the owner must be set explicitly (re-enrollment).
        """
        if os.path.isfile(self.owner_file):
            try:
                with open(self.owner_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                name = data.get("name")
                if name:
                    return name
            except (OSError, json.JSONDecodeError):
                logger.warning("Unreadable owner marker: %s", self.owner_file)
                return None
        profiles = self.list_profiles()
        if len(profiles) == 1:
            name = profiles[0]
            self.set_owner_name(name)
            return name
        return None

    def set_owner_name(self, name: str) -> None:
        validate_name(name, context="owner name")
        data = {"name": name}
        owner_dir = os.path.dirname(self.owner_file)
        fd, tmp_path = tempfile.mkstemp(
            dir=owner_dir, prefix=".tmp-owner-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.owner_file)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise

    def save_profile(self, name: str, embedding: np.ndarray) -> dict:
        validate_name(name, context="profile name")
        profile = {
            "name": name,
            "embedding": [float(x) for x in embedding],
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path = os.path.join(self.profiles_dir, f"{name}.json")
        fd, tmp_path = tempfile.mkstemp(
            dir=self.profiles_dir, prefix=".tmp-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(profile, f)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise
        return profile

    def load_profile(self, name: str) -> dict | None:
        validate_name(name, context="profile name")
        path = os.path.join(self.profiles_dir, f"{name}.json")
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_all(self) -> list[dict]:
        profiles = []
        for entry in sorted(os.listdir(self.profiles_dir)):
            if entry.endswith(".json") and not entry.startswith(".tmp-"):
                path = os.path.join(self.profiles_dir, entry)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        profiles.append(json.load(f))
                except (OSError, json.JSONDecodeError):
                    logger.warning("Skipping unreadable voiceprint: %s", path)
        return profiles

    def list_profiles(self) -> list[str]:
        return [p.get("name", "") for p in self.load_all()]

    def archive_profile(self, name: str) -> str | None:
        """Move a profile aside with a timestamp suffix instead of deleting it.

        Archived files live under ``identity/voiceprints/archived/`` so the
        active scan (``os.listdir`` of the root, ``.json`` only) never
        re-reads them as live profiles.
        """
        validate_name(name, context="profile name")
        path = os.path.join(self.profiles_dir, f"{name}.json")
        if not os.path.isfile(path):
            return None
        archive_dir = os.path.join(self.profiles_dir, "archived")
        os.makedirs(archive_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(archive_dir, f"{name}.{ts}.json")
        n = 1
        while os.path.exists(dest):
            dest = os.path.join(archive_dir, f"{name}.{ts}-{n}.json")
            n += 1
        os.replace(path, dest)
        return dest

    @staticmethod
    def average_embeddings(embeddings: list[np.ndarray]) -> np.ndarray:
        """Mean of many embeddings, normalized to unit length for scoring."""
        if not embeddings:
            raise ValueError("no embeddings to average")
        mean = np.mean(
            np.stack([np.asarray(e, dtype=np.float32) for e in embeddings]),
            axis=0,
        )
        norm = float(np.linalg.norm(mean))
        if norm == 0:
            raise ValueError("cannot normalize an empty embedding")
        return mean / norm