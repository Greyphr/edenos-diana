"""
Sandbox manager tests, ported from eden-os-main's test_coding_engine.py
(sandbox section): everything runs against an injected FakeDocker talking the
Docker Engine HTTP API shape — no daemon, no network.
"""

import pytest

from coding_engine.execution_policy import CodingEngineError
from coding_engine.http_client import HTTPResponse
from coding_engine.sandbox_manager import SandboxManager, SandboxSpec

DOCKER = "http://127.0.0.1:2375"


class FakeDocker:
    """Injected Docker Engine API client: records calls, plays canned
    responses, 200 {} by default."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self.responses: dict[tuple[str, str, tuple[tuple[str, str], ...]], HTTPResponse] = {}
        self.default = HTTPResponse(200, b"{}")

    def on(self, method, url, params=None, status=200, content=b"{}"):
        key = (method, url, tuple(sorted((params or {}).items())))
        self.responses[key] = HTTPResponse(status, content)

    def request(self, method, url, headers=None, json_payload=None, params=None):
        self.calls.append((method, url, json_payload, params))
        key = (method, url, tuple(sorted((params or {}).items())))
        return self.responses.get(key, self.default)


def _running_container(docker, container_id="c1"):
    docker.on(
        "POST", f"{DOCKER}/containers/create", content=f'{{"Id": "{container_id}"}}'.encode()
    )
    docker.on("POST", f"{DOCKER}/containers/{container_id}/start")
    docker.on(
        "GET",
        f"{DOCKER}/containers/{container_id}/json",
        content=b'{"State": {"Running": true, "Status": "running"}}',
    )


def _create_call(docker) -> dict:
    for call in docker.calls:
        if call[0] == "POST" and call[1].endswith("/containers/create"):
            return call[2] or {}
    raise AssertionError("no container create call was made")


def test_sandbox_create_is_network_none_by_default():
    docker = FakeDocker()
    _running_container(docker)
    sm = SandboxManager(client=docker)
    sandbox = sm.create("/tmp/sandbox-root")
    payload = _create_call(docker)
    # default-deny network: no outbound access unless an allowlist is given.
    assert payload["HostConfig"]["NetworkMode"] == "none"
    # default-deny credentials: no env injected when no provider is wired.
    assert payload["Env"] == []
    assert payload["HostConfig"]["Resources"]["Memory"] == 2048 * 1024 * 1024
    assert payload["HostConfig"]["Binds"][0].endswith(":/workspace")
    assert sandbox.status == "running"


def test_sandbox_create_with_allowlist_selects_network():
    docker = FakeDocker()
    _running_container(docker)
    sm = SandboxManager(client=docker)
    sm.create("C:/sandbox-root", spec=SandboxSpec(network_allowlist=["eden-bridge"]))
    visible = _create_call(docker)
    assert visible["HostConfig"]["NetworkMode"] == "eden-bridge"


def test_sandbox_gets_only_scoped_credentials():
    docker = FakeDocker()
    _running_container(docker)
    sm = SandboxManager(
        client=docker,
        credential_provider=lambda: {"CODING_CRED": "scoped-value"},
    )
    sm.create("C:/sandbox-root")
    visible = _create_call(docker)
    assert visible["Env"] == ["CODING_CRED=scoped-value"]


def test_sandbox_execute_returns_verified_exit_code():
    docker = FakeDocker()
    _running_container(docker)
    docker.on("POST", f"{DOCKER}/containers/c1/exec", content=b'{"Id": "e1"}')
    docker.on("POST", f"{DOCKER}/exec/e1/start", content=b"hello output")
    docker.on("GET", f"{DOCKER}/exec/e1/json", content=b'{"Running": false, "ExitCode": 0}')
    sm = SandboxManager(client=docker)
    sandbox = sm.create("C:/sandbox-root")
    result = sm.execute(sandbox, "python -c 'print(1)'")
    assert result["exit_code"] == 0
    assert result["output"] == "hello output"
    assert result["timed_out"] is False
    assert result["verified"] is True


def test_sandbox_execute_timeout_is_not_verified():
    docker = FakeDocker()
    _running_container(docker)
    docker.on("POST", f"{DOCKER}/containers/c1/exec", content=b'{"Id": "e1"}')
    docker.on("POST", f"{DOCKER}/exec/e1/start", content=b"")
    docker.on("GET", f"{DOCKER}/exec/e1/json", content=b'{"Running": true}')
    sm = SandboxManager(client=docker)
    sandbox = sm.create("C:/sandbox-root")
    result = sm.execute(sandbox, "sleep 999", timeout_seconds=0.01)
    assert result["timed_out"] is True
    assert result["exit_code"] is None
    assert result["verified"] is False


def test_sandbox_unparseable_command_raises_coding_engine_error():
    docker = FakeDocker()
    _running_container(docker)
    sm = SandboxManager(client=docker)
    sandbox = sm.create("C:/sandbox-root")
    with pytest.raises(CodingEngineError, match="not parseable"):
        sm.execute(sandbox, "unclosed ' quote")


def test_sandbox_destroy_confirms_removed():
    docker = FakeDocker()
    _running_container(docker)
    docker.on("DELETE", f"{DOCKER}/containers/c1", params={"force": "1"})
    sm = SandboxManager(client=docker)
    sandbox = sm.create("C:/sandbox-root")
    docker.on("GET", f"{DOCKER}/containers/c1/json", status=404)
    result = sm.destroy(sandbox)
    assert result["removed"] is True
    assert result["verified"] is True
    assert sandbox.status == "removed"


def test_sandbox_for_workspace_lookup(tmp_path):
    docker = FakeDocker()
    _running_container(docker)
    sm = SandboxManager(client=docker)
    root = str(tmp_path / "ws-root")
    sandbox = sm.create(root)
    found = sm.for_workspace(root)
    assert found is not None
    assert found.sandbox_id == sandbox.sandbox_id
    assert sm.for_workspace(str(tmp_path / "other")) is None


def test_sandbox_cleanup_destroys_all():
    docker = FakeDocker()
    _running_container(docker, "c1")
    docker.on("DELETE", f"{DOCKER}/containers/c1", params={"force": "1"})
    sm = SandboxManager(client=docker)
    sandbox = sm.create("C:/sandbox-root")
    docker.on("GET", f"{DOCKER}/containers/c1/json", status=404)
    removed = sm.cleanup()
    assert removed == [sandbox.sandbox_id]
    assert sandbox.status == "removed"


def test_sandbox_create_fails_loudly_if_never_running():
    docker = FakeDocker()
    docker.on(
        "POST", f"{DOCKER}/containers/create", content=b'{"Id": "c1"}'
    )
    docker.on("POST", f"{DOCKER}/containers/c1/start")
    docker.on(
        "GET",
        f"{DOCKER}/containers/c1/json",
        content=b'{"State": {"Running": false, "Status": "created"}}',
    )
    sm = SandboxManager(client=docker)
    with pytest.raises(CodingEngineError, match="did not enter running state"):
        sm.create("C:/sandbox-root")