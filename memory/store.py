import json
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


class JsonListStore:
    """Atomically read/write a JSON list at a given path.

    Uses the same temp-file + os.replace pattern as identity/voiceprint_store
    so a crash never leaves a half-written file.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def read(self, default: list | None = None) -> list:
        if not os.path.isfile(self.path):
            return list(default or [])
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else list(default or [])
        except (OSError, json.JSONDecodeError):
            logger.warning("Unreadable JSON list at %s; treating as empty", self.path)
            return list(default or [])

    def write(self, data: list) -> None:
        fd, tmp_path = tempfile.mkstemp(
            dir=os.path.dirname(self.path), prefix=".tmp-", suffix=".json"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            raise