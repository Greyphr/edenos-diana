"""
coding_engine/execution_policy.py

Ported from eden-os-main's action_layer/coding_engine/execution_policy.py.
The eden-os-main imports (core.policy_engine.EdenError) are replaced by a
local CodingEngineError so this package stands alone. Design semantics are
kept as-is.

Dual-mode guard for a task's execution loop: APPROVAL_GATED (every step goes
back for a confirmation) or AUTONOMOUS (steps run within hard ceilings). The
mode is per task, never a global switch — a policy instance is created for the
task's loop, not shared system-wide.

Clarification gate (§ the coding engine's autonomous mode — "even as a human
you need to ask before making assumptions"): before_step() enforces the
clarification gate in BOTH modes, before any budget check. This project has no
clarification gate yet, so ClarificationGateLike is defined as a Protocol and
left unwired (pass None) — that wiring is follow-up work, not something to
stub badly. Autonomous mode never means "guess".

Hitting a ceiling raises LimitExceededError (a CodingEngineError, so
orchestrator code can catch the whole family). escalate() is the stop path:
preserve the workspace, report what/why, hand back to the Owner.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class CodingEngineError(Exception):
    """Base error for the coding engine. Standalone by design: this package
    must never import the app's policy engine."""


class ExecutionMode(str, Enum):
    AUTONOMOUS = "autonomous"
    APPROVAL_GATED = "approval_gated"


@dataclass
class ExecutionLimits:
    """Hard ceilings for one run. `max_cost` is optional; the rest always
    apply. None of these is a soft warning — each is a stop condition."""

    max_steps: int = 100
    max_runtime_seconds: float = 3600.0
    max_retries: int = 3
    max_actions: int = 50
    max_cost: float | None = None


class LimitExceededError(CodingEngineError):
    """An autonomous-run ceiling was hit — stop, preserve state, escalate."""


class ClarificationGateLike(Protocol):
    """Anything with an ``enforce(intent)`` that raises on missing/ambiguous
    caller intent. Defined here so before_step() can type-check a real gate
    later; currently no gate exists in this project and None is passed."""

    def enforce(self, intent: Any) -> None: ...


@dataclass
class ExecutionBudget:
    """Counters for one run. check() raises LimitExceededError with the
    specific ceiling that was hit."""

    limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    steps: int = 0
    retries: int = 0
    actions: int = 0
    cost: float = 0.0
    _started_at: float = field(default_factory=time.monotonic, repr=False)

    def check(self) -> None:
        if self.steps >= self.limits.max_steps:
            raise LimitExceededError(
                f"max steps exceeded: {self.steps} >= {self.limits.max_steps}"
            )
        if self.retries > self.limits.max_retries:
            raise LimitExceededError(
                f"max retries exceeded: {self.retries} > {self.limits.max_retries}"
            )
        if self.actions >= self.limits.max_actions:
            raise LimitExceededError(
                f"max actions exceeded: {self.actions} >= {self.limits.max_actions}"
            )
        if self.limits.max_cost is not None and self.cost >= self.limits.max_cost:
            raise LimitExceededError(
                f"max cost exceeded: {self.cost} >= {self.limits.max_cost}"
            )
        elapsed = time.monotonic() - self._started_at
        if elapsed >= self.limits.max_runtime_seconds:
            raise LimitExceededError(
                f"max runtime exceeded: {elapsed:.1f}s >= {self.limits.max_runtime_seconds}s"
            )

    def begin_step(self) -> None:
        # Hard ceiling, not a warning: check first so `max_steps` means the
        # loop may run exactly `max_steps` steps and is stopped by the next
        # before_step.
        self.check()
        self.steps += 1

    def record_action(self, cost: float = 0.0) -> None:
        self.actions += 1
        self.cost += cost
        self.check()

    def record_retry(self) -> None:
        self.retries += 1
        self.check()

    def report(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "retries": self.retries,
            "actions": self.actions,
            "cost": round(self.cost, 4),
            "elapsed_seconds": round(time.monotonic() - self._started_at, 3),
        }


class ExecutionPolicy:
    """Dual-mode guard for a task's execution loop.

    `clarification_gate` is anything with an `enforce(intent)` (e.g. a future
    clarification gate). It runs FIRST in before_step, in both modes — the one
    thing autonomous mode never waives.
    """

    def __init__(
        self,
        mode: ExecutionMode | str = ExecutionMode.APPROVAL_GATED,
        limits: ExecutionLimits | None = None,
        clarification_gate: ClarificationGateLike | None = None,
    ) -> None:
        self.mode = ExecutionMode(mode) if not isinstance(mode, ExecutionMode) else mode
        self.limits = limits or ExecutionLimits()
        self.budget = ExecutionBudget(self.limits)
        self._gate = clarification_gate

    @property
    def autonomous(self) -> bool:
        return self.mode is ExecutionMode.AUTONOMOUS

    def requires_approval(self) -> bool:
        return self.mode is ExecutionMode.APPROVAL_GATED

    def before_step(self, intent: Any) -> None:
        """The clarification gate runs in BOTH modes before the budget ceiling
        is checked. Raises whatever the gate raises for missing/ambiguous
        intent, LimitExceededError when a ceiling is hit."""
        if self._gate is not None:
            self._gate.enforce(intent)
        self.budget.begin_step()

    def check_limits(self) -> None:
        self.budget.check()

    def record_action(self, cost: float = 0.0) -> None:
        self.budget.record_action(cost)

    def record_retry(self) -> None:
        self.budget.record_retry()

    def escalate(
        self,
        reason: str,
        workspace_manager: Any | None = None,
        workspace: Any | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        """Stop path: preserve the workspace (flip it to non-ephemeral so
        cleanup keeps it), report what happened and why, hand back to the
        Owner. Returns the escalation report for the audit trail. Never
        crashes on a preservation failure — the report carries the outcome."""
        preserved = False
        if workspace is not None and workspace_manager is not None:
            try:
                workspace_manager.preserve(workspace)
                preserved = True
            except Exception:  # noqa: BLE001 — escalation never crashes on a preservation failure
                preserved = False
        return {
            "escalated": True,
            "reason": reason,
            "preserved_workspace": preserved,
            "task_id": task_id,
            "mode": self.mode.value,
            "budget": self.budget.report(),
        }

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "limits": {
                key: value for key, value in vars(self.limits).items() if not key.startswith("_")
            },
        }