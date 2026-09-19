from memory.long_term import MAX_FACT_LENGTH, LongTermMemory
from memory.short_term import ShortTermMemory
from tools.registry import RiskTier, ToolSpec

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

FORGET_DECLARATION = {
    "name": "forget",
    "description": (
        "Remove a previously remembered fact about the owner whose text "
        "contains the given string (case-insensitive). Returns how many "
        "facts were removed."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "match": {
                "type": "string",
                "description": (
                    "Text the note must contain (case-insensitive) to be removed."
                ),
            }
        },
        "required": ["match"],
    },
}


def make_memory_handlers(owner_name: str) -> list[ToolSpec]:
    """Build the remember/recall/forget specs closing over one owner's memory.

    ``remember`` writes to both long-term (durable) and short-term (recency
    signal) stores. ``recall`` searches long-term facts and never raises for
    a no-match query. ``forget`` removes long-term facts whose text contains
    the given string. All three are TRIVIAL — owner-only, confirmation-free —
    so they run through the same policy engine as every other tool.
    """

    long_term = LongTermMemory(owner_name)
    short_term = ShortTermMemory(owner_name)

    async def remember(**args) -> dict:
        fact = str(args.get("fact", "")).strip()
        if not fact:
            return {"status": "error", "reason": "empty fact"}
        if len(fact) > MAX_FACT_LENGTH:
            return {
                "status": "error",
                "reason": (
                    f"fact too long ({len(fact)} chars, max {MAX_FACT_LENGTH})"
                ),
            }
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

    async def forget(**args) -> dict:
        match = str(args.get("match", "")).strip()
        if not match:
            return {"status": "error", "reason": "empty match"}
        return {"removed": long_term.remove_facts_containing(match)}

    return [
        ToolSpec(
            name="remember",
            description=REMEMBER_DECLARATION["description"],
            parameters=REMEMBER_DECLARATION["parameters"],
            risk_tier=RiskTier.TRIVIAL,
            handler=remember,
        ),
        ToolSpec(
            name="recall",
            description=RECALL_DECLARATION["description"],
            parameters=RECALL_DECLARATION["parameters"],
            risk_tier=RiskTier.TRIVIAL,
            handler=recall,
        ),
        ToolSpec(
            name="forget",
            description=FORGET_DECLARATION["description"],
            parameters=FORGET_DECLARATION["parameters"],
            risk_tier=RiskTier.TRIVIAL,
            handler=forget,
        ),
    ]