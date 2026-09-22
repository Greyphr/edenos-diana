"""Full PolicyEngine lifecycle.

Covers: immediate READ execution, WRITE/SENSITIVE staging + confirmation,
SENSITIVE exact-phrase semantics (a bare "yes" is not enough), expiry via
monotonic-clock injection, the identity-freshness gap (a strong recognition
that is NOT newer than the proposal denies the action — F-01), the
single-pending slot, the confirmation race (two parallel resolution channels
claim the slot exactly once — F-03), and F-04 word-boundary/no/ambiguous
response classification against both WRITE and SENSITIVE tiers.
"""

from unittest import mock

import pytest
import asyncio

from tools.policy_engine import PENDING_TIMEOUT_SECONDS, PolicyEngine
from tools.registry import RiskTier, ToolRegistry, ToolSpec
from tools.roles import ROLE_OWNER, ROLE_UNKNOWN


class _Actors:
    """Simulates main.py's RecognitionTrust for the policy engine: the last
    strong recognition's monotonic timestamp (inf by default so ordinary tests
    are always fresh) plus the recognized/owner names."""

    def __init__(
        self, recognized="alex", owner="alex", last_strong: float | None = float("inf")
    ) -> None:
        self.recognized = recognized
        self.owner = owner
        self.last_strong = last_strong

    def recognized_name(self) -> str | None:
        return self.recognized

    def owner_name(self) -> str | None:
        return self.owner

    def last_strong_recognition_time(self) -> float | None:
        return self.last_strong

    def record_strong(self, when: float) -> None:
        self.last_strong = when


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
        actors.last_strong_recognition_time,
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

async def test_sensitive_bare_yes_leaves_pending(toolbox):
    await _wrap(toolbox, "delete_vault")(x="erase")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes")
    # "yes" is neither the exact phrase nor an explicit no: ambiguous, left
    # pending so the owner can say the real phrase (which then completes it).
    assert outcome["status"] == "pending"
    assert "not understood" in outcome.get("reason", "")
    assert calls == []

    confirmed = await engine.confirm_action(response="erase everything")
    assert confirmed["status"] == "ok"
    assert calls == ["delete_vault"]


async def test_sensitive_exact_phrase_confirms(toolbox):
    await _wrap(toolbox, "delete_vault")(x="erase")
    _registry, engine, _actors, calls = toolbox
    # Exact equality after normalization: trailing punctuation is stripped.
    outcome = await engine.confirm_action(response="erase everything.")
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


# --- Genuinely-NEW evidence (F-01) -----------------------------------------

async def test_recognition_no_newer_than_proposal_is_denied(toolbox):
    """F-01: the owner's ORIGINAL request that triggered the proposal is not
    new evidence. Immediately confirming with no recognition in between must
    be denied, and only a genuinely later strong recognition completes it."""
    _registry, engine, actors, calls = toolbox
    fake_time = mock.Mock()
    fake_time.monotonic.return_value = 0.0  # proposal created_at = 0.0
    with mock.patch("tools.policy_engine.time", fake_time):
        await _wrap(toolbox, "write_note")(x="immediate")
        actors.record_strong(0.0)  # last strong == proposal time, not after
        denied = await engine.confirm_action(response="yes")

        assert denied["error"] == "not permitted"
        assert "fresh recognition" in denied.get("reason", "")
        assert calls == []

        # A genuine new affirmative utterance lands AFTER the proposal.
        actors.record_strong(1.0)
        confirmed = await engine.confirm_action(response="yes.")

    assert confirmed["name"] == "write_note"
    assert confirmed["tool_name"] == "write_note"
    assert calls == ["write_note"]


async def test_stale_recognition_between_propose_and_confirm_denies(toolbox):
    actors = toolbox[2]
    fake_time = mock.Mock()
    fake_time.monotonic.return_value = 0.0
    with mock.patch("tools.policy_engine.time", fake_time):
        await _wrap(toolbox, "write_note")(x="freshness")  # created_at = 0.0
        actors.record_strong(0.0)  # nothing new landed since the proposal
        _registry, engine, _actors, calls = toolbox
        outcome = await engine.confirm_action(response="yes")
        assert outcome["error"] == "not permitted"
        assert "fresh recognition" in outcome.get("reason", "")
        assert calls == []

        # The pending action was restored (not cleared) by the denial, so a
        # genuinely new recognition still clears it.
        actors.record_strong(1.0)
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


# --- Confirmation race (F-03) ----------------------------------------------

async def test_concurrent_resolutions_execute_handler_exactly_once(toolbox):
    """confirm_action and check_transcript racing (the designed behavior)
    must claim the pending slot exactly once. A slow wait_for_pending forces
    the first attempt to suspend mid-resolution; the second must find the
    slot already claimed and sit out — the handler runs once, never twice."""
    registry, _engine, actors, calls = toolbox

    async def _slow_wait() -> None:
        await asyncio.sleep(0.05)

    engine = PolicyEngine(
        registry,
        actors.recognized_name,
        actors.owner_name,
        actors.last_strong_recognition_time,
        wait_for_pending=_slow_wait,
    )

    fake_time = mock.Mock()
    fake_time.monotonic.return_value = 0.0  # proposal created_at = 0.0
    with mock.patch("tools.policy_engine.time", fake_time):
        await engine.wrap_handler(registry.get("write_note"))(x="race")

        # The confirmation utterance is still scoring: nobody recognized and
        # no new strong recognition yet — so the first resolution attempt
        # hits the wait. The recognition lands mid-wait.
        actors.recognized = None
        actors.record_strong(0.0)

        async def _land_recognition() -> None:
            await asyncio.sleep(0.02)
            actors.recognized = "alex"
            actors.record_strong(1.0)  # genuinely new, > created_at

        _, transcript_out, _ = await asyncio.gather(
            engine.confirm_action(response="yes"),
            engine.check_transcript(transcript="yes"),
            _land_recognition(),
        )

    assert calls == ["write_note"]  # exactly once
    assert transcript_out is None  # the transcript side found nothing pending


# --- F-04 classification, WRITE tier ---------------------------------------

async def test_write_yes_with_punctuation_confirms(toolbox):
    await _wrap(toolbox, "write_note")(x="punct")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes.")
    assert outcome["name"] == "write_note"
    assert calls == ["write_note"]


async def test_write_yes_please_confirms(toolbox):
    await _wrap(toolbox, "write_note")(x="polite")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes please")
    assert outcome["name"] == "write_note"
    assert calls == ["write_note"]


async def test_write_explicit_negative_cancels(toolbox):
    await _wrap(toolbox, "write_note")(x="neg")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="no, don't confirm")
    assert outcome["status"] == "cancelled"
    assert calls == []
    # The slot was cancelled outright: a later yes finds nothing pending.
    assert await engine.confirm_action(response="yes") == {"error": "nothing pending"}


async def test_write_ambiguous_question_leaves_pending(toolbox):
    await _wrap(toolbox, "write_note")(x="ambig")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="wait, which one")
    assert outcome["status"] == "pending"
    assert "not understood" in outcome.get("reason", "")
    assert calls == []
    # Ambiguous does not cancel: a real yes afterwards completes it.
    confirmed = await engine.confirm_action(response="yes")
    assert confirmed["name"] == "write_note"
    assert calls == ["write_note"]


# --- F-04 classification, SENSITIVE tier -----------------------------------

async def test_sensitive_yes_please_leaves_pending(toolbox):
    await _wrap(toolbox, "delete_vault")(x="polite")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="yes please")
    # A polite yes is not the exact phrase — ambiguous, left pending.
    assert outcome["status"] == "pending"
    assert calls == []


async def test_sensitive_explicit_negative_cancels(toolbox):
    await _wrap(toolbox, "delete_vault")(x="neg")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="no, don't confirm")
    assert outcome["status"] == "cancelled"
    assert calls == []
    assert await engine.confirm_action(response="erase everything") == {
        "error": "nothing pending"
    }


async def test_sensitive_ambiguous_question_leaves_pending(toolbox):
    await _wrap(toolbox, "delete_vault")(x="ambig")
    _registry, engine, _actors, calls = toolbox
    outcome = await engine.confirm_action(response="wait, which one")
    assert outcome["status"] == "pending"
    assert calls == []
    # Still pending: the exact phrase resolves it afterwards.
    confirmed = await engine.confirm_action(response="erase everything")
    assert confirmed["status"] == "ok"
    assert calls == ["delete_vault"]


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