from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory

MAX_LONG_TERM_FACTS = 20  # cap on facts injected into the system instruction


def build_context_summary(owner_name: str) -> str:
    """Render the owner's known facts as a compact context block.

    Returns an empty string on first run (nothing stored yet).
    """
    long_term = LongTermMemory(owner_name)
    short_term = ShortTermMemory(owner_name)
    facts = long_term.all_facts()[:MAX_LONG_TERM_FACTS]
    recent = short_term.all_facts()

    if not facts and not recent:
        return ""

    lines = []
    if facts:
        lines.append(f"Known facts about the owner:")
        lines.extend(f"- {f['text']}" for f in facts)
    if recent:
        lines.append("Recently discussed:")
        lines.extend(f"- {f['text']}" for f in recent)
    return "\n".join(lines)