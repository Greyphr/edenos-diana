"""Web search tool (via the Brave Search API, free tier).

One spec, ``web_search(query)`` — an owned, read-only lookup through
Brave's REST ``/res/v1/web/search`` endpoint using the
``X-Subscription-Token`` header. Returns the top 3 results as a compact
``{"results": [{title, snippet, url}, ...]}`` payload — titles and
snippets only, never full page content.

This follows the exact pattern of ``integrations/spotify/tools.py``:
a factory returns ``ToolSpec`` instances that register into the same
``ToolRegistry`` and policy engine. It lives in ``tools/`` (not an
``integrations/`` package) because it's a plain owned REST call that
needs no client object — only the API key. TRIVIAL tier, owner-only,
runs immediately with no confirmation prompt, same reasoning as the
Spotify playback tools. The key is read from ``BRAVE_SEARCH_API_KEY``;
if it's missing (or the request fails) the handler returns a clear
error dict, never raises.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from tools.registry import RiskTier, ToolSpec

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_TOP_N = 3


def _search(query: str, api_key: str) -> list[dict]:
    """Run the Brave web-search request synchronously.

    Returns the top ``_TOP_N`` results each as ``{title, snippet, url}``.
    Raises :exc:`urllib.error.URLError`/``HTTPError`` on request failure so
    the caller can turn it into a plain error dict.
    """
    params = urllib.parse.urlencode({"q": query, "count": str(_TOP_N), "freshness": "all"})
    request = urllib.request.Request(
        f"{_BRAVE_SEARCH_ENDPOINT}?{params}",
        headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        payload = response.read()
    data = json.loads(payload)
    web = data.get("web") or {}
    results = (web.get("results") or [])[: _TOP_N]
    return [
        {
            "title": (item.get("title") or "").strip(),
            "snippet": (item.get("description") or "").strip(),
            "url": (item.get("url") or "").strip(),
        }
        for item in results
    ]


async def _web_search_handler(*, query: str) -> dict:
    """Async handler: runs the Brave request off the event loop."""
    api_key = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
    if not api_key:
        return {
            "error": "missing_api_key",
            "message": (
                "No BRAVE_SEARCH_API_KEY set. Add one to .env.example and "
                "get a key at https://api.search.brave.com to enable web search."
            ),
        }
    try:
        results = await asyncio.to_thread(_search, query, api_key)
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
        logger.warning("web_search failed: %r", exc)
        return {
            "error": "search_failed",
            "message": f"Web search failed: {exc}",
        }
    return {"results": results}


def make_web_search_spec() -> ToolSpec:
    """Build the ``web_search`` ToolSpec (mirrors Spotify's factory)."""
    return ToolSpec(
        name="web_search",
        description=(
            "Search the web for up-to-date information using the Brave Search "
            "API. Returns the top 3 results as titles, one-line snippets, and "
            "URLs. Use this when the owner asks for current facts, news, "
            "answers, or things you don't already know. Owner-only, runs "
            "immediately - no confirmation prompt."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up.",
                }
            },
            "required": ["query"],
        },
        risk_tier=RiskTier.TRIVIAL,
        handler=_web_search_handler,
    )