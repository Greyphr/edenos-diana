"""
Execution policy tests: ported from eden-os-main's
tests/unit/test_coding_engine.py (execution-policy section), scoped to the
two pieces shipped here. The clarification gate is unwired in this project, so
a FakeGate shaped like ClarificationGateLike stands in to prove the §11
ordering (gate first, budget second) in BOTH modes.
"""

from typing import Any

import pytest

from coding_engine.execution_policy import (
    ClarificationGateLike,
    CodingEngineError,
    ExecutionLimits,
    ExecutionMode,
    ExecutionPolicy,
    LimitExceededError,
)


class _ContentNeeded(Exception):
    pass


class FakeGate(ClarificationGateLike):
    """Enforces a made-up contract: intents need a 'content' argument."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def enforce(self, intent: Any) -> None:
        self.calls.append(intent)
        if isinstance(intent, dict) and not intent.get("content"):
            raise _ContentNeeded("content is required - never assume")


def _intent(**args) -> dict[str, Any]:
    return args


# ---- modes ----------------------------------------------------------------

def test_modes_default_to_approval_gated():
    assert ExecutionPolicy().requires_approval() is True
    assert ExecutionPolicy().autonomous is False
    assert ExecutionPolicy(mode=ExecutionMode.AUTONOMOUS).autonomous is True
    assert ExecutionPolicy(mode=ExecutionMode.AUTONOMOUS).requires_approval() is False
    assert ExecutionPolicy(mode="autonomous").autonomous is True


# ---- before_step: gate first, then budget, in both modes ------------------

def test_before_step_enforces_gate_in_autonomous_mode():
    policy = ExecutionPolicy(
        mode=ExecutionMode.AUTONOMOUS,
        clarification_gate=FakeGate(),
    )
    with pytest.raises(_ContentNeeded):
        policy.before_step(_intent(path="a.py"))  # content missing


def test_before_step_enforces_gate_in_approval_gated_mode():
    policy = ExecutionPolicy(
        mode=ExecutionMode.APPROVAL_GATED,
        clarification_gate=FakeGate(),
    )
    with pytest.raises(_ContentNeeded):
        policy.before_step(_intent(path="a.py"))


def test_before_step_passes_complete_intent_and_counts_in_both_modes():
    steps_gated = []
    steps_auto = []
    gate = FakeGate()

    gated = ExecutionPolicy(mode=ExecutionMode.APPROVAL_GATED, clarification_gate=gate)
    gated.before_step(_intent(path="a.py", content="x"))
    steps_gated.append(gated.budget.steps)

    auto = ExecutionPolicy(mode=ExecutionMode.AUTONOMOUS, clarification_gate=gate)
    auto.before_step(_intent(path="a.py", content="x"))
    steps_auto.append(auto.budget.steps)

    assert steps_gated == [1]
    assert steps_auto == [1]
    assert len(gate.calls) == 2


def test_approval_gating_differs_from_autonomous_only_by_mode():
    gated = ExecutionPolicy(mode=ExecutionMode.APPROVAL_GATED, clarification_gate=FakeGate())
    auto = ExecutionPolicy(mode=ExecutionMode.AUTONOMOUS, clarification_gate=FakeGate())
    # The operational difference: who decides the step runs.
    assert gated.requires_approval() is True and gated.autonomous is False
    assert auto.requires_approval() is False and auto.autonomous is True
    # But §11 still holds: neither mode skips the gate. Give both a complete
    # intent and confirm both advance the budget once each.
    for policy in (gated, auto):
        policy.before_step(_intent(path="a.py", content="x"))
        assert policy.budget.steps == 1


# ---- budget ceilings ------------------------------------------------------

def test_max_steps_ceiling_raises_at_the_right_count():
    policy = ExecutionPolicy(limits=ExecutionLimits(max_steps=2))
    intent = _intent(path="a.py", content="x")
    policy.before_step(intent)
    policy.before_step(intent)
    with pytest.raises(LimitExceededError):
        policy.before_step(intent)


def test_max_runtime_ceiling_raises_immediately_when_exhausted():
    policy = ExecutionPolicy(limits=ExecutionLimits(max_runtime_seconds=0.0))
    with pytest.raises(LimitExceededError):
        policy.before_step(_intent(path="a.py", content="x"))


def test_max_actions_ceiling_raises_at_the_right_count():
    policy = ExecutionPolicy(limits=ExecutionLimits(max_actions=3))
    policy.record_action()
    policy.record_action()
    with pytest.raises(LimitExceededError):
        policy.record_action()


def test_max_retries_ceiling_raises_at_the_right_count():
    policy = ExecutionPolicy(limits=ExecutionLimits(max_retries=2))
    policy.record_retry()
    policy.record_retry()
    with pytest.raises(LimitExceededError):
        policy.record_retry()


def test_max_cost_ceiling_raises_when_reached():
    policy = ExecutionPolicy(limits=ExecutionLimits(max_cost=5.0))
    policy.record_action(cost=3.0)
    policy.record_action(cost=1.0)  # 4.0 < 5.0 still fine
    with pytest.raises(LimitExceededError):
        policy.record_action(cost=1.0)  # 5.0 >= 5.0


def test_limit_exceeded_is_a_coding_engine_error():
    assert issubclass(LimitExceededError, CodingEngineError)


def test_limits_configurable_per_policy():
    small = ExecutionPolicy(limits=ExecutionLimits(max_steps=1))
    large = ExecutionPolicy(limits=ExecutionLimits(max_steps=999))
    assert small.describe()["limits"]["max_steps"] == 1
    assert large.describe()["limits"]["max_steps"] == 999
    assert small.describe()["mode"] == "approval_gated"


# ---- escalate() ------------------------------------------------------------

class FakeWorkspaceManager:
    def __init__(self) -> None:
        self.preserved: list[Any] = []

    def preserve(self, workspace: Any) -> None:
        if getattr(workspace, "fail_preserve", False):
            raise RuntimeError("simulated preserve failure")
        self.preserved.append(workspace)
        workspace.ephemeral = False


class FakeWorkspace:
    def __init__(self) -> None:
        self.ephemeral = True
        self.fail_preserve = False


def test_escalate_preserves_workspace_and_reports():
    wm = FakeWorkspaceManager()
    ws = FakeWorkspace()
    policy = ExecutionPolicy(mode=ExecutionMode.AUTONOMOUS)
    report = policy.escalate("max steps hit", workspace_manager=wm, workspace=ws, task_id="t-1")
    assert report["escalated"] is True
    assert report["preserved_workspace"] is True
    assert report["reason"] == "max steps hit"
    assert report["task_id"] == "t-1"
    assert report["mode"] == "autonomous"
    assert "budget" in report
    assert wm.preserved == [ws]
    assert ws.ephemeral is False


def test_escalate_survives_a_preservation_failure():
    wm = FakeWorkspaceManager()
    ws = FakeWorkspace()
    ws.fail_preserve = True
    policy = ExecutionPolicy()
    report = policy.escalate("something broke", workspace_manager=wm, workspace=ws)
    assert report["escalated"] is True
    assert report["preserved_workspace"] is False
    assert report["reason"] == "something broke"


def test_escalate_reports_without_workspace_refs():
    policy = ExecutionPolicy()
    report = policy.escalate("budget blown")
    assert report["escalated"] is True
    assert report["preserved_workspace"] is False
    assert report["task_id"] is None
    assert report["budget"]["steps"] == 0