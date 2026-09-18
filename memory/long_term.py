import logging
import os
import time

from memory.store import JsonListStore

logger = logging.getLogger(__name__)


def _data_dir(owner_name: str) -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", owner_name
    )


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class LongTermMemory:
    """Durable facts about the owner, persisted as a JSON list of
    ``{"text": ..., "recorded_at": ...}`` under
    ``memory/data/<owner_name>/long_term.json``.
    """

    def __init__(self, owner_name: str) -> None:
        self.owner_name = owner_name
        self._store = JsonListStore(os.path.join(_data_dir(owner_name), "long_term.json"))

    def add_fact(self, text: str) -> None:
        facts = self._store.read()
        facts.append({"text": text, "recorded_at": _now_iso()})
        self._store.write(facts)

    def all_facts(self) -> list[dict]:
        # Most recent first.
        return list(reversed(self._store.read()))

    def search(self, query: str) -> list[dict]:
        q = query.lower()
        return [f for f in self.all_facts() if q in f["text"].lower()]