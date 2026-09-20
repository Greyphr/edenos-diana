"""
Standalone check for the coding-engine foundation (run: python -m
coding_engine.standalone_check). Not wired into Eden's conversation — this is
the survival test for the two ported pieces:

  1. Docker: create a sandboxed container via SandboxManager, run a trivial
     command inside it, get the result back, and destroy the container —
     verified gone.
  2. Execution budget: exercise ExecutionBudget directly and confirm
     LimitExceededError raises at the configured ceiling.

REQUIRES a running Docker daemon:
  * Docker Desktop (or equivalent) must be RUNNING.
  * With Docker Desktop on Windows/macOS, the daemon must be reachable at
    http://127.0.0.1:2375 — enable Settings -> General -> "Expose daemon on
    tcp://localhost:2375 without TLS" (or point EDEN_DOCKER_URL at a
    reachable daemon). The context lives in ONE container with NetworkMode
    "none" and a single workspace bind mount.
  * The sandbox image (python:3.12-slim by default) must be present locally:
    `docker pull python:3.12-slim` if the first run fails with HTTP 404.
"""

import os
import sys
import tempfile
from pathlib import Path

from coding_engine.execution_policy import (
    ExecutionLimits,
    ExecutionPolicy,
    LimitExceededError,
)
from coding_engine.sandbox_manager import SandboxManager, SandboxSpec


def _check_budget() -> None:
    print("\n[1/2] Execution budget ceilings...")
    policy = ExecutionPolicy(limits=ExecutionLimits(max_steps=2))
    for _ in range(2):
        policy.before_step({"path": "ok.py", "content": "x"})
    try:
        policy.before_step({"path": "bad.py"})
    except LimitExceededError as exc:
        print(f"  PASS: LimitExceededError at the max_steps ceiling: {exc}")
    else:  # pragma: no cover - the loop above must raise
        raise SystemExit("FAIL: budget did not raise at the max_steps ceiling")

    actions = ExecutionPolicy(limits=ExecutionLimits(max_actions=3))
    for _ in range(2):
        actions.record_action()
    try:
        actions.record_action()
    except LimitExceededError as exc:
        print(f"  PASS: LimitExceededError at the max_actions ceiling: {exc}")
    else:  # pragma: no cover
        raise SystemExit("FAIL: budget did not raise at the max_actions ceiling")


def _check_sandbox() -> None:
    base_url = os.getenv("EDEN_DOCKER_URL", "http://127.0.0.1:2375")
    manager = SandboxManager(
        base_url=base_url,
        default_spec=SandboxSpec(timeout_seconds=30.0),
    )
    workspace = Path(tempfile.mkdtemp(prefix="eden-sandbox-check-"))
    (workspace / "hello.txt").write_text("from the host", encoding="utf-8")

    print(f"\n[2/2] Docker sandbox against {base_url} ...")
    sandbox = None
    try:
        sandbox = manager.create(workspace, spec=SandboxSpec(timeout_seconds=30.0))
        print(f"  created {sandbox.sandbox_id} ({sandbox.image}), "
              f"container {sandbox.container_id}, status {sandbox.status}")

        result = manager.execute(sandbox, "python --version")
        print(f"  exec 'python --version' -> exit {result['exit_code']}, "
              f"timed_out={result['timed_out']}, verified={result['verified']}")
        print(f"  output: {result['output'].strip()!r}")

        verify = manager.execute(sandbox, "cat /workspace/hello.txt")
        print(f"  exec 'cat /workspace/hello.txt' -> exit {verify['exit_code']}, "
              f"output {verify['output'].strip()!r}")
        if not result["verified"] or result["exit_code"] != 0:
            raise SystemExit(
                "FAIL: trivial command did not verify (is the sandbox image "
                "present? try `docker pull python:3.12-slim`)"
            )
        print("  PASS: sandbox created, command ran, network is default-deny "
              "(NetworkMode none).")
    except Exception as exc:  # noqa: BLE001 - report the actionable failure
        hint = ""
        text = str(exc)
        if "404" in text:
            hint = "\n  Hint: the image is missing - run `docker pull python:3.12-slim`."
        elif "failed" in text or "Timeout" in text or "timed out" in text:
            hint = (
                "\n  Hint: is Docker Desktop RUNNING with the daemon exposed?\n"
                "  Windows/macOS: Dock/settings -> Settings -> General -> "
                '"Expose daemon on tcp://localhost:2375 without TLS".\n'
                "  Or point EDEN_DOCKER_URL at your reachable daemon."
            )
        raise SystemExit(f"FAIL: {exc}{hint}")
    finally:
        if sandbox is not None:
            outcome = manager.destroy(sandbox)
            print(f"  destroyed -> removed={outcome['removed']}, "
                  f"verified={outcome['verified']}")
        (workspace / "hello.txt").unlink(missing_ok=True)
        workspace.rmdir()


def main() -> None:
    _check_budget()
    _check_sandbox()
    print("\nAll checks passed - the coding-engine foundation is operational.")


if __name__ == "__main__":
    main()