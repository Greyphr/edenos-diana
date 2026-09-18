import json
import logging
import os
import tempfile
import time

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "voiceprints"
)


class VoiceprintStore:
    """Persists enrolled speaker profiles as JSON files.

    A profile is a name plus an embedding vector (list of floats). Profile
    files live under ``identity/voiceprints/`` and are written atomically
    (temp file + os.replace) so a crash never leaves a half-written profile.
    """

    def __init__(self, profiles_dir: str = DEFAULT_PROFILES_DIR) -> None:
        self.profiles_dir = profiles_dir
        os.makedirs(self.profiles_dir, exist_ok=True)

    def save_profile(self, name: str, embedding: np.ndarray) -> dict:
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