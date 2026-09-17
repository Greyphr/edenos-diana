from pathlib import Path

import yaml

PERSONA = (
    "You are Eden, Christopher's personal AI assistant. "
    "You are conversational and concise. "
    "Respond naturally and keep replies brief unless asked for detail."
)


def load_voice_config(path: str | Path = "config/voice.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_system_instruction(config: dict) -> str:
    p = config["personality"]
    personality = (
        f"Speak with a {p['accent']} accent, in a {p['tone']}, {p['style']} "
        f"way, with {p['formality']} formality and {p['humor']} humor, at a "
        f"{p['pacing']} pace, with {p['expressiveness']} expressiveness."
    )
    return f"{PERSONA} {personality}"