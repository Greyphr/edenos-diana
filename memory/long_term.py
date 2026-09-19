import logging
import os
import re
import time

from memory.store import JsonListStore
from tools.names import validate_name

logger = logging.getLogger(__name__)

MAX_FACT_LENGTH = 500
MAX_FACTS = 200

# Small, intentionally short stopword list for recall queries.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "to", "of", "for", "in", "on",
    "is", "are", "was", "do", "does", "i", "you", "me", "my", "your",
    "it", "that", "this", "what", "where", "when", "why", "with",
    "have", "has", "had", "can", "could", "would", "be", "get", "got",
})

_WORD_RE = re.compile(r"[a-z0-9']+")


def _keywords(query: str) -> list[str]:
    """Lowercase query keywords with punctuation and stopwords stripped."""
    words = [w.replace("'", "") for w in _WORD_RE.findall(query.lower())]
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


def _data_dir(owner_name: str) -> str:
    validate_name(owner_name, context="owner name")
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
        # Bounded store: when the cap would be exceeded, drop the oldest
        # entries first so a flood of writes can't silently push out
        # everything real.
        if len(facts) > MAX_FACTS:
            del facts[: len(facts) - MAX_FACTS]
        self._store.write(facts)

    def remove_facts_containing(self, match: str) -> int:
        """Remove every fact whose text contains ``match`` (case-insensitive).

        Returns how many facts were removed.
        """
        needle = match.lower()
        facts = self._store.read()
        kept = [fact for fact in facts if needle not in fact["text"].lower()]
        removed = len(facts) - len(kept)
        if removed:
            self._store.write(kept)
        return removed

    def all_facts(self) -> list[dict]:
        # Most recent first.
        return list(reversed(self._store.read()))

    def search(self, query: str) -> list[dict]:
        """Rank facts by how many query keywords appear in each fact.

        Facts with at least one keyword match are returned, ordered by match
        count descending with most-recent-first as the tiebreaker.
        """
        keywords = _keywords(query)
        if not keywords:
            return []
        scored = []
        for fact in self.all_facts():  # most recent first -> recency tiebreak
            text = fact["text"].lower()
            count = sum(1 for kw in keywords if kw in text)
            if count:
                scored.append((count, fact))
        # Stable sort: equal match counts keep most-recent-first order.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [fact for _, fact in scored]