"""build_system_instruction block gating: each instruction block appears only
when its flag/context is on, and IDENTITY_INSTRUCTIONS is always present."""

import pytest

from conversation.voice_config import (
    FIRST_RUN_ENROLLMENT_INSTRUCTIONS,
    IDENTITY_INSTRUCTIONS,
    MEMORY_TOOL_INSTRUCTIONS,
    REASONING_INSTRUCTIONS,
    TOOL_POLICY_INSTRUCTIONS,
    WEB_SEARCH_INSTRUCTIONS,
    build_first_run_instruction,
    build_system_instruction,
)


@pytest.fixture
def config():
    return {
        "personality": {
            "accent": "neutral",
            "tone": "warm",
            "style": "direct",
            "formality": "casual",
            "humor": "dry",
            "pacing": "relaxed",
            "expressiveness": "subtle",
        }
    }


def test_identity_always_present(config):
    instruction = build_system_instruction(config)
    assert IDENTITY_INSTRUCTIONS in instruction


def test_no_optional_blocks_by_default(config):
    instruction = build_system_instruction(config)
    for block in (
        MEMORY_TOOL_INSTRUCTIONS,
        TOOL_POLICY_INSTRUCTIONS,
        REASONING_INSTRUCTIONS,
        WEB_SEARCH_INSTRUCTIONS,
    ):
        assert block not in instruction


def test_memory_block_needs_memory_context(config):
    with_context = build_system_instruction(config, memory_context="loves jazz")
    assert MEMORY_TOOL_INSTRUCTIONS in with_context
    assert "loves jazz" in with_context
    without = build_system_instruction(config)
    assert MEMORY_TOOL_INSTRUCTIONS not in without


def test_policy_block_needs_confirmation_tools(config):
    assert TOOL_POLICY_INSTRUCTIONS in build_system_instruction(
        config, confirmation_tools=True
    )
    assert TOOL_POLICY_INSTRUCTIONS not in build_system_instruction(config)


def test_reasoning_block_needs_reasoning_tools(config):
    assert REASONING_INSTRUCTIONS in build_system_instruction(
        config, reasoning_tools=True
    )
    assert REASONING_INSTRUCTIONS not in build_system_instruction(config)


def test_web_search_block_needs_web_search_tools(config):
    assert WEB_SEARCH_INSTRUCTIONS in build_system_instruction(
        config, web_search_tools=True
    )
    assert WEB_SEARCH_INSTRUCTIONS not in build_system_instruction(config)


def test_all_blocks_together(config):
    instruction = build_system_instruction(
        config,
        memory_context="loves jazz",
        confirmation_tools=True,
        reasoning_tools=True,
        web_search_tools=True,
    )
    for block in (
        IDENTITY_INSTRUCTIONS,
        MEMORY_TOOL_INSTRUCTIONS,
        TOOL_POLICY_INSTRUCTIONS,
        REASONING_INSTRUCTIONS,
        WEB_SEARCH_INSTRUCTIONS,
    ):
        assert block in instruction


def test_persona_uses_owner_name_and_fallback():
    from conversation.voice_config import FALLBACK_PERSONA_OWNER

    owned = build_system_instruction({"personality": {"accent": "x", "tone": "x",
        "style": "x", "formality": "x", "humor": "x", "pacing": "x",
        "expressiveness": "x"}}, owner_name="alex")
    assert "alex" in owned.split("\n\n")[0]
    assert FALLBACK_PERSONA_OWNER not in owned.split("\n\n")[0]

    generic = build_system_instruction({"personality": {"accent": "x", "tone": "x",
        "style": "x", "formality": "x", "humor": "x", "pacing": "x",
        "expressiveness": "x"}})
    assert FALLBACK_PERSONA_OWNER in generic.split("\n\n")[0]


def test_first_run_instruction_welds_identity_and_enrollment(config):
    instruction = build_first_run_instruction(config)
    assert IDENTITY_INSTRUCTIONS in instruction
    assert FIRST_RUN_ENROLLMENT_INSTRUCTIONS in instruction
    assert MEMORY_TOOL_INSTRUCTIONS not in instruction
    assert TOOL_POLICY_INSTRUCTIONS not in instruction