"""
coding_engine/http_client.py

Ported from eden-os-main's action_layer/tool_system/http_client.py, adapted
from `requests` to `httpx` (already a project dependency). The single HTTP
surface the sandbox manager talks through: adapters never bind to a specific
HTTP library, and tests always inject a fake client so nothing here ever
touches a Docker daemon.
"""

import json
from typing import Any

from coding_engine.execution_policy import CodingEngineError


class HTTPResponse:
    """The minimal HTTP surface adapters need, without binding to any HTTP
    library."""

    def __init__(self, status_code: int, content: bytes = b"", headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        try:
            return self.content.decode("utf-8")
        except UnicodeDecodeError:
            return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        if not self.content:
            return {}
        try:
            return json.loads(self.content)
        except ValueError:
            return {}


class HTTPClient:
    """Protocol for the only HTTP method the sandbox manager is allowed to
    use. `json_payload` sends a JSON body; `data` sends a form-encoded body.
    Never pass both."""

    def request(
        self,
        method: str,
        url: str,
        headers: dict | None = None,
        json_payload: dict | None = None,
        params: dict | None = None,
        data: dict | None = None,
    ) -> HTTPResponse:
        ...


class DefaultHTTPClient:
    """httpx-based client. Transport failures (no daemon, refused connection,
    DNS) raise CodingEngineError so the package's error model applies instead
    of a raw exception. Injected by default in SandboxManager, so it is
    created lazily and this module imports cleanly with nothing installed."""

    def request(
        self,
        method: str,
        url: str,
        headers: dict | None = None,
        json_payload: dict | None = None,
        params: dict | None = None,
        data: dict | None = None,
    ) -> HTTPResponse:
        try:
            import httpx
        except ImportError as exc:
            raise CodingEngineError(
                "HTTP transport not available: the 'httpx' package is not installed"
            ) from exc

        try:
            resp = httpx.request(
                method=method,
                url=url,
                headers=headers,
                json=json_payload,
                params=params,
                data=data,
                timeout=15,
            )
        except httpx.HTTPError as exc:
            raise CodingEngineError(f"HTTP {method} {url} failed: {exc}") from exc

        return HTTPResponse(
            status_code=resp.status_code,
            content=resp.content,
            headers=dict(resp.headers),
        )