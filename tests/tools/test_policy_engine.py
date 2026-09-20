"""Full PolicyEngine lifecycle.

Covers: immediate READ execution, WRITE/SENSITIVE staging + confirmation,
SENSITIVE exact-phrase semantics (a bare "yes" is not enough), expiry via
monotonic-clock injection, the identity-freshness gap (fresh recognition
dropping between propose and confirm denies the action), the single-pending
slot (a second proposal is rejected, never replacing the first), and the two
parallel resolution channels (check_transcript and confirm_action share the
same queue and whichever resolves first leaves the other idle).
"""

import time
from unittest import mock

import pytest

from tools.policy_engine import PENDING_TIMEOUT_SECONDS, PolicyEngine
from tools.registry import RiskTier, ToolRegistry, ToolSpec
from tools.roles import ROLE_OWNER, ROLE_UNKNOWN


class _Actors:
    def __init__(self, recognized="alex", owner="alex", fresh=True) -> None:
        self.recognized = recognized
        self.owner = owner
        self.fresh = fresh

    def recognized_name(self) -> str | None:
        return self.recognized

    def owner_name(self) -> str | None:
        return self.owner

    def fresh_state(self) -> bool:
        return self.fresh


_BASE_PARAMS = {
    "type": "object",
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


def _make_spec(name: str, tier: RiskTier, calls: list[str], phrase: str | None = None) -> ToolSpec:
    async def _handler(x: str, **kwargs) -> dict:
        calls.append(name)
        return {"status": "ok", "name": name, "arg": x}

    return ToolSpec(
        name=name,
        description=f"test tool {name}",
        parameters=_BASE_PARAMS,
        risk_tier=tier,
        handler=_handler,
        confirmation_phrase=phrase,
    )


@pytest.fixture
def toolbox():
    """A (registry, engine, calls) with the four canonical tools, plus the
    actor state you can flip mid-test."""
    actors = _Actors()
    calls: list[str] = []
    registry = ToolRegistry()
    registry.register(_make_spec("lookup", RiskTier.READ, calls))
    registry.register(_make_spec("think", RiskTier.TRIVIAL, calls))
    registry.register(_make_spec("write_note", RiskTier.WRITE, calls))
    registry.register(
        _make_spec("delete_vault", RiskTier.SENSITIVE, calls, phrase="erase everything")
    )
    engine = PolicyEngine(
        registry,
        actors.recognized_name,
        actors.owner_name,
        actors.fresh_state,
    )
    return registry, engine, actors, calls


def _wrap(toolbox, name: str):
    registry, engine, _actors, _calls = toolbox
    return engine.wrap_handler(registry.get(name))


# --- READ executes immediately -------------------------------------------

async def test_read_executes_immediately_for_owner(toolbox):
    handler = _wrap(toolbox, "lookup")
    result = await handler(x="a")
    assert result == {"status": "ok", "name": "lookup", "arg": "a"}


async def test_read_is_permitted_for_unknown_role(toolbox):
    _registry, engine, actors, _calls = toolbox
    actors.recognized = "somebody else"
    handler = engine.wrap_handler(_registry.get("lookup"))
    result = await handler(x="a")
    assert result["name"] == "lookup"


# --- WRITE staging + confirmation -----------------------------------------

async def test_write_creates_pending_and_confirmation_executes(toolbox):
    handler = _wrap(toolbox, "write_note")
    staged = await handler(x="remember this")
    assert staged["status"] == "confirmation_required"
    assert "confirm" in staged["ask"].lower()
    calls = toolbox[3]
    assert calls == []  # handler must NOT have run yet

    _registry, engine, _actors, calls = toolbox
    confirmed = await engine.confirm_action(response="yes")
    assert confirmed["name"] == "write_note"
    assert confirmed["tool_name"] == "write_note"
    assert confirmed["arg"] == "remember this"
    assert calls == ["write_note"]


async def test_second_proposal_while_pending_is_rejected(toolbox):
    await _wrap(toolbox, "write_note")(x="first")
    _registry, engine, _actors, calls = toolbox
    rejected = await _wrap(toolbox, "delete_vault")(x="second")
    assert rejected["error"] == "another action is already pending confirmation"
    assert rejected["pending_tool"] == "write_note"

    await engine.confirm_action(response="yes")
    assert calls == ["write_note"]  # the first survives; the second never ran


# --- SENSITIVE needs the exact phrase -------------------------------------

async def test_sensitive_bare_yes_is_not_accepted(toolbox):
    await _wrap(toolbox, "delete_vault")(x="erase")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes")
    assert outcome["status"] == "cancelled"
    assert calls == []


async def test_sensitive_exact_phrase_confirms(toolbox):
    await _wrap(toolbox, "delete_vault")(x="erase")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes erase everything now")
    assert outcome["status"] == "ok"
    assert calls == ["delete_vault"]


# --- Expiry ---------------------------------------------------------------

async def test_pending_expires_after_timeout(toolbox):
    _registry, engine, _actors, calls = toolbox

    fake_time = mock.Mock()
    fake_time.monotonic.side_effect = [
        100.0,
        100.0 + PENDING_TIMEOUT_SECONDS + 1.0,
    ]
    with mock.patch("tools.policy_engine.time", fake_time):
        await _wrap(toolbox, "write_note")(x="old")  # created_at = 100.0
        outcome = await engine.confirm_action(response="yes")  # now 100+46

    assert outcome["status"] == "cancelled"
    assert calls == []
    assert await engine.confirm_action(response="yes") == {"error": "nothing pending"}


# --- Identity-freshness gap ------------------------------------------------

async def test_stale_recognition_between_propose_and_confirm_denies(toolbox):
    actors = toolbox[2]
    await _wrap(toolbox, "write_note")(x="freshness")
    actors.fresh = False  # recognition lands, then the window lapses

    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes")
    assert outcome["error"] == "not permitted"
    assert "fresh recognition" in outcome.get("reason", "")
    assert calls == []

    # The pending action is kept, so a genuinely fresh recognition still
    # clears it.
    actors.fresh = True
    confirmed = await engine.confirm_action(response="yes")
    assert confirmed["name"] == "write_note"
    assert calls == ["write_note"]


async def test_wrong_role_at_confirmation_is_denied(toolbox):
    await _wrap(toolbox, "write_note")(x="overheard")
    actors = toolbox[2]
    actors.recognized = "someone who overheard the phrase"

    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes")
    assert outcome["error"] == "not permitted"
    assert calls == []


# --- TRIVIAL stays owner-gated --------------------------------------------

async def test_trivial_denied_to_unknown_without_pending(toolbox):
    actors = toolbox[2]
    actors.recognized = "stranger"
    outcome = await _wrap(toolbox, "think")(x="deep")
    assert outcome == {"error": "not permitted"}
    _registry, engine, _actors, _calls = toolbox
    assert await engine.confirm_action(response="yes") == {"error": "nothing pending"}


# --- Parallel resolution channels -----------------------------------------

async def test_check_transcript_resolves_the_same_queue(toolbox):
    await _wrap(toolbox, "write_note")(x="spoken")
    _registry, engine, _actors, calls = toolbox

    resolved = await engine.check_transcript(transcript="yes")
    assert resolved is not None
    assert resolved["tool_name"] == "write_note"
    assert calls == ["write_note"]

    # The tool-call channel now finds nothing pending.
    assert await engine.confirm_action(response="yes") == {"error": "nothing pending"}


async def test_confirm_action_resolving_first_leaves_transcript_idle(toolbox):
    await _wrap(toolbox, "write_note")(x="tool path")
    _registry, engine, _actors, calls = toolbox

    await engine.confirm_action(response="yes")
    assert calls == ["write_note"]
    assert await engine.check_transcript(transcript="yes") is None


async def test_check_transcript_without_pending_returns_none(toolbox):
    _registry, engine, _actors, _calls = toolbox
    assert await engine.check_transcript(transcript="yes") is None


# --- Argument validation ---------------------------------------------------

async def test_missing_required_argument_raises(toolbox):
    handler = _wrap(toolbox, "lookup")
    with pytest.raises(ValueError, match="missing required argument"):
        await handler()


async def test_wrong_argument_type_raises(toolbox):
    handler = _wrap(toolbox, "lookup")
    with pytest.raises(ValueError, match="must be a string"):
        await handler(x=123)


# --- Idle cleanup ----------------------------------------------------------

async def test_clear_pending_drops_unconfirmed_action(toolbox):
    await _wrap(toolbox, "write_note")(x="stale")
    _registry, engine, _actors, calls = toolbox
    engine.clear_pending()
    assert await engine.confirm_action(response="yes") == {"error": "nothing pending"}
    assert calls == []


# --- Declarations -----------------------------------------------------------

def test_declarations_include_confirm_action(toolbox):
    _registry, engine, _actors, _calls = toolbox
    names = {decl["name"] for decl in engine.declarations()}
    assert "confirm_action" in names


def test_all_four_tools_declared(toolbox):
    _registry, engine, _actors, _calls = toolbox
    names = {decl["name"] for decl in engine.declarations()}
    assert names == {"lookup", "think", "write_note", "delete_vault", "confirm_action"}