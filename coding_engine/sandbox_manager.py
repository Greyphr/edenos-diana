"""
coding_engine/sandbox_manager.py

Ported from eden-os-main's action_layer/coding_engine/sandbox_manager.py.
The eden-os-main imports (core.policy_engine.IntegrationError,
action_layer.tool_system.http_client.DefaultHTTPClient) are replaced by the
local coding_engine.CodingEngineError and coding_engine.http_client. The
Docker orchestration logic — create/start/exec/poll/destroy against the
Docker Engine HTTP API — is kept as-is.

Isolation guarantees, structural not aspirational:

  * default-deny network — NetworkMode "none" (no outbound access at all)
    unless an explicit network allowlist is configured; egress rule
    enforcement beyond that is a network-layer concern outside this module;
  * default-deny credentials — the container Env is exactly what the injected
    credential_provider returns (default: nothing). This manager NEVER
    receives the boot-wide secret store: a vault:// URI or a real secret
    value can never reach a container from here;
  * the workspace reaches the container through exactly one bind mount; the
    rest of the host filesystem stays outside.

The Docker Engine API client is injected (tests use a fake and never touch a
daemon), and the default client is created lazily so this module imports
cleanly — same pattern as the rest of this project.
"""

import os
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from coding_engine.execution_policy import CodingEngineError
from coding_engine.http_client import DefaultHTTPClient

DEFAULT_DOCKER_URL = "http://127.0.0.1:2375"

_EXEC_POLL_INTERVAL = 0.2


@dataclass
class SandboxSpec:
    """What one sandbox container gets. `network_allowlist` is the explicit
    egress allowlist: empty (the default) means NetworkMode "none" — no
    outbound access. When non-empty, the first entry names the Docker network
    to attach; the actual allow/deny rules for that network are enforced at
    the network layer, outside this module."""

    image: str = "python:3.12-slim"
    workspace_mount_path: str = "/workspace"
    memory_mb: int = 2048
    cpu_shares: int = 512
    disk_mb: int = 2048
    timeout_seconds: float = 300.0
    network_allowlist: list[str] = field(default_factory=list)


@dataclass
class Sandbox:
    """One running container backing a thing being worked on. `workspace_root`
    is the bind-mounted host path — the key the manager uses to find a
    sandbox."""

    sandbox_id: str
    container_id: str
    image: str
    workspace_root: str
    status: str
    created_at: float
    spec: SandboxSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "container_id": self.container_id,
            "image": self.image,
            "workspace_root": self.workspace_root,
            "status": self.status,
            "created_at": self.created_at,
        }


class SandboxManager:
    """Docker-container sandboxes (default-deny isolation).

    `credential_provider` returns the ONLY env the container ever sees — by
    default nothing. Pass nothing from the real secret store; this class has
    no notion of vault URIs.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_DOCKER_URL,
        client: Any | None = None,
        credential_provider: Callable[[], dict[str, str]] | None = None,
        default_spec: SandboxSpec | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client or DefaultHTTPClient()
        self.credential_provider = credential_provider or (lambda: {})
        self.default_spec = default_spec or SandboxSpec()
        self._sandboxes: dict[str, Sandbox] = {}

    # ---- internal ----------------------------------------------------------

    def _api(
        self,
        method: str,
        path: str,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any] | list[Any]:
        resp = self.client.request(
            method=method,
            url=self.base_url + path,
            headers={"Content-Type": "application/json"},
            json_payload=json_body,
            params=params,
        )
        if not resp.ok:
            raise CodingEngineError(
                f"sandbox {method} {path} failed (HTTP {resp.status_code}): {resp.text}"
            )
        body = resp.json()
        return body if isinstance(body, (dict, list)) else {}

    # ---- lifecycle ----------------------------------------------------------

    def create(self, workspace_root: str | os.PathLike, spec: SandboxSpec | None = None) -> Sandbox:
        """Create + start one container for a workspace root. Verified: the
        container must be Running after start, else the sandbox fails loudly."""
        spec = spec or self.default_spec
        root = str(Path(workspace_root).resolve())
        env = [f"{k}={v}" for k, v in self.credential_provider().items()]
        host_config: dict[str, Any] = {
            "Binds": [f"{root}:{spec.workspace_mount_path}"],
            "NetworkMode": "none" if not spec.network_allowlist else spec.network_allowlist[0],
            "Resources": {
                "Memory": spec.memory_mb * 1024 * 1024,
                "CpuShares": spec.cpu_shares,
                "StorageOpt": {"size": f"{spec.disk_mb}m"},
            },
        }
        created = self._api(
            "POST",
            "/containers/create",
            json_body={
                "Image": spec.image,
                "Cmd": ["sleep", "infinity"],
                "Env": env,
                "HostConfig": host_config,
            },
        )
        container_id = created.get("Id", "") if isinstance(created, dict) else ""
        if not container_id:
            raise CodingEngineError("docker container create returned no container id")
        sandbox = Sandbox(
            sandbox_id=f"sandbox-{uuid.uuid4().hex[:12]}",
            container_id=container_id,
            image=spec.image,
            workspace_root=root,
            status="created",
            created_at=time.time(),
            spec=spec,
        )
        self._sandboxes[sandbox.sandbox_id] = sandbox
        self._api("POST", f"/containers/{container_id}/start", json_body={})
        state = self.status(sandbox)
        if not state["running"]:
            raise CodingEngineError(
                f"sandbox '{sandbox.sandbox_id}' did not enter running state"
            )
        sandbox.status = "running"
        return sandbox

    def status(self, sandbox: Sandbox) -> dict[str, Any]:
        raw = self._api("GET", f"/containers/{sandbox.container_id}/json")
        state = (raw.get("State") or {}) if isinstance(raw, dict) else {}
        return {
            "ok": True,
            "sandbox_id": sandbox.sandbox_id,
            "running": bool(state.get("Running")),
            "status": state.get("Status", "unknown"),
        }

    def execute(
        self,
        sandbox: Sandbox,
        command: str,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Run one command inside the sandbox. The result carries the exit
        code, captured output, and a `verified` flag (exit code present and no
        timeout) — the execution contract's verify() reads exactly that. A
        timeout is a timed-out result, never a raised crash."""
        try:
            cmd = shlex.split(command)
        except ValueError as exc:
            raise CodingEngineError(
                f"sandbox command is not parseable: {command!r}"
            ) from exc
        exec_created = self._api(
            "POST",
            f"/containers/{sandbox.container_id}/exec",
            json_body={"AttachStdout": True, "AttachStderr": True, "Cmd": cmd},
        )
        exec_id = exec_created.get("Id", "") if isinstance(exec_created, dict) else ""
        if not exec_id:
            raise CodingEngineError("docker exec create returned no exec id")
        start_resp = self.client.request(
            method="POST",
            url=self.base_url + f"/exec/{exec_id}/start",
            headers={"Content-Type": "application/json"},
            json_payload={"Detach": False, "Tty": False},
        )
        output = start_resp.text
        deadline = time.monotonic() + (timeout_seconds or sandbox.spec.timeout_seconds)
        exit_code: int | None = None
        timed_out = False
        while True:
            exec_state = self._api("GET", f"/exec/{exec_id}/json")
            if isinstance(exec_state, dict) and not exec_state.get("Running"):
                exit_code = exec_state.get("ExitCode")
                break
            if time.monotonic() >= deadline:
                timed_out = True
                try:
                    self._api("POST", f"/exec/{exec_id}/kill", json_body={})
                except CodingEngineError:
                    pass
                break
            time.sleep(_EXEC_POLL_INTERVAL)
        return {
            "ok": True,
            "sandbox_id": sandbox.sandbox_id,
            "exit_code": exit_code,
            "output": output,
            "timed_out": timed_out,
            "verified": exit_code is not None and not timed_out,
        }

    def for_workspace(self, workspace_root: str | os.PathLike) -> Sandbox | None:
        """Find the sandbox bound to a workspace root, if one exists."""
        root = str(Path(workspace_root).resolve())
        return next(
            (s for s in self._sandboxes.values() if s.workspace_root == root),
            None,
        )

    def destroy(self, sandbox: Sandbox) -> dict[str, Any]:
        """Force-remove the container and confirm it is actually gone (404 on
        inspect = removed) — verification, not just a delete call."""
        try:
            self._api("DELETE", f"/containers/{sandbox.container_id}", params={"force": "1"})
        except CodingEngineError as exc:
            if "404" not in str(exc):
                raise
        try:
            gone = self.client.request(
                "GET", self.base_url + f"/containers/{sandbox.container_id}/json"
            )
            removed = not gone.ok or gone.status_code == 404
        except CodingEngineError:
            removed = True
        sandbox.status = "removed"
        return {
            "ok": True,
            "sandbox_id": sandbox.sandbox_id,
            "removed": removed,
            "verified": removed,
        }

    def cleanup(self) -> list[str]:
        """Destroy every sandbox the manager created (shutdown hygiene)."""
        removed: list[str] = []
        for sandbox in list(self._sandboxes.values()):
            if sandbox.status != "removed":
                self.destroy(sandbox)
                removed.append(sandbox.sandbox_id)
        return removed