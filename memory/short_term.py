import logging
import os
import time
from datetime import datetime, timedelta

from memory.store import JsonListStore

logger = logging.getLogger(__name__)

PRUNE_AFTER_SECONDS = 24 * 60 * 60


def _data_dir(owner_name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", owner_name
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class ShortTermMemory:
    """Recency signal: entries that fall out of a 24-hour window are pruned
    on every write. Same shape as LongTermMemory, stored under
    ``memory/data/<owner_name>/short_term.json``.
    """

    def __init__(self, owner_name: str) -> None:
        self.owner_name = owner_name
        self._store = JsonListStore(
            os.path.join(_data_dir(owner_name), "short_term.json")
        )

    def add_fact(self, text: str) -> None:
        entries = self._store.read()
        entries.append({"text": text, "recorded_at": _now_iso()})
        self._store.write(self._prune(entries))

    def all_facts(self) -> list[dict]:
        # Most recent first.
        return list(reversed(self._prune(self._store.read())))

    def _prune(self, entries: list[dict]) -> list[dict]:
        cutoff = time.time() - PRUNE_AFTER_SECONDS
        kept = []
        for entry in entries:
            recorded_at = entry.get("recorded_at")
            if not recorded_at:
                continue
            try:
                ts = datetime.fromisoformat(recorded_at).timestamp()
            except ValueError:
                continue
            if ts >= cutoff:
                kept.append(entry)
        return kept