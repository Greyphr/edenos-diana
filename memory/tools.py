from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory

REMEMBER_DECLARATION = {
    "name": "remember",
    "description": (
        "Store a fact the owner tells you, so it is remembered across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact to remember, as stated by the owner.",
            }
        },
        "required": ["fact"],
    },
}

RECALL_DECLARATION = {
    "name": "recall",
    "description": (
        "Look up a previously remembered fact about the owner. Returns an "
        "empty list when nothing matches."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look up, e.g. 'coffee preferences'.",
            }
        },
        "required": ["query"],
    },
}


def make_memory_handlers(owner_name: str) -> dict:
    """Build the remember/recall handlers closing over one owner's memory.

    ``remember`` writes to both long-term (durable) and short-term (recency
    signal) stores. ``recall`` searches long-term facts and never raises for
    a no-match query.
    """

    long_term = LongTermMemory(owner_name)
    short_term = ShortTermMemory(owner_name)

    async def remember(**args) -> dict:
        fact = str(args.get("fact", "")).strip()
        if not fact:
            return {"status": "error", "reason": "empty fact"}
        long_term.add_fact(fact)
        short_term.add_fact(fact)
        return {"status": "ok"}

    async def recall(**args) -> dict:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"matches": []}
        matches = [
            {"text": f["text"]} for f in long_term.search(query)
        ]
        return {"matches": matches}

    return {"remember": remember, "recall": recall}