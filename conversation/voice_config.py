from pathlib import Path

import yaml

PERSONA = (
    "You are Eden, Christopher's personal AI assistant. "
    "You are conversational and concise. "
    "Respond naturally and keep replies brief unless asked for detail."
)

MEMORY_TOOL_INSTRUCTIONS = (
    "You have two tools for remembering things about the owner across "
    "conversations: remember(fact) and recall(query). Call remember "
    "whenever the owner tells you something worth keeping - a "
    "preference, a fact about their life, something to follow up on. "
    "Call recall when they ask about something you might already know, "
    "before saying you don't know. Don't narrate that you're using a "
    "tool, just do it naturally."
)

TOOL_POLICY_INSTRUCTIONS = (
    "Some of your actions require the owner's explicit confirmation "
    "before they happen, handled through a confirm_action(response) tool. "
    "When an action needs confirmation, ask the owner for the exact "
    "confirmation phrase you're told to expect - a plain 'yes' is not "
    "enough for sensitive actions. Only proceed once they say the right "
    "phrase. Don't narrate how the confirmation mechanism works; just "
    "ask naturally and act on the answer."
)


def load_voice_config(path: str | Path = "config/voice.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_instruction(
    config: dict, memory_context: str = "", confirmation_tools: bool = False
) -> str:
    p = config["personality"]
    personality = (
        f"Speak with a {p['accent']} accent, in a {p['tone']}, {p['style']} "
        f"way, with {p['formality']} formality and {p['humor']} humor, at a "
        f"{p['pacing']} pace, with {p['expressiveness']} expressiveness."
    )
    instruction = f"{PERSONA} {personality}"
    blocks = []
    if memory_context:
        blocks.append(MEMORY_TOOL_INSTRUCTIONS)
    if confirmation_tools:
        blocks.append(TOOL_POLICY_INSTRUCTIONS)
    if memory_context:
        blocks.append(memory_context)
    if blocks:
        instruction += "\n\n" + "\n\n".join(blocks)
    return instruction