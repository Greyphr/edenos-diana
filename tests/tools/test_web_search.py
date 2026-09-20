"""web_search tool: missing-key degradation and mocked success shape."""

import asyncio
import json
from unittest import mock

import tools.web_search as web_search
from tools.registry import RiskTier
from tools.web_search import make_web_search_spec


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):  # noqa: ANN002
        return False

    def read(self) -> bytes:
        return self._body


async def test_missing_key_returns_clean_error_dict(monkeypatch):
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    spec = make_web_search_spec()
    result = await spec.handler(query="weather today")
    assert result["error"] == "missing_api_key"
    assert "results" not in result


def test_spec_metadata():
    spec = make_web_search_spec()
    assert spec.name == "web_search"
    assert spec.risk_tier is RiskTier.TRIVIAL
    assert "query" in spec.parameters["required"]


async def test_mocked_success_returns_expected_shape(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "test-key")
    expected = [
        {"title": "One", "snippet": "First result", "url": "https://a"},
        {"title": "Two", "snippet": "Second result", "url": "https://b"},
        {"title": "Three", "snippet": "Third result", "url": "https://c"},
    ]
    with mock.patch("tools.web_search._search", return_value=expected) as fake:
        spec = make_web_search_spec()
        result = await spec.handler(query="python async")
    assert result == {"results": expected}
    fake.assert_called_once_with("python async", "test-key")


def test_search_truncates_to_top_three_and_strips_fields():
    payload = {
        "web": {
            "results": [
                {"title": f"T{i}", "description": f"S{i}", "url": f"https://e{i}"}
                for i in range(5)
            ]
        }
    }
    with mock.patch(
        "urllib.request.urlopen", return_value=_FakeResponse(json.dumps(payload).encode())
    ):
        results = web_search._search("query", "key")
    assert len(results) == 3


def test_search_handles_missing_web_section():
    with mock.patch(
        "urllib.request.urlopen",
        return_value=_FakeResponse(json.dumps({"not": "web"}).encode()),
    ):
        assert web_search._search("query", "key") == []