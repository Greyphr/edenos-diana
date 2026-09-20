"""Google Calendar client + auth helpers: refresh-once-under-concurrency,
event normalization, create-event body, and flow helpers."""

import pytest
from cryptography.fernet import Fernet
from httpx import AsyncClient, MockTransport, Response

from integrations.calendar.auth import (
    build_authorize_url,
    store_refresh_token,
)
from integrations.calendar.client import (
    CalendarClient,
    CalendarError,
)
from tools.vault import Vault

DATED_EVENT = {
    "id": "e1",
    "summary": "Standup",
    "start": {"dateTime": "2026-09-21T09:30:00+02:00", "timeZone": "Europe/Paris"},
    "end": {"dateTime": "2026-09-21T09:45:00+02:00", "timeZone": "Europe/Paris"},
    "htmlLink": "https://calendar.google.com/event?eid=e1",
    "location": "Zoom",
    "status": "confirmed",
}
ALL_DAY_EVENT = {
    "id": "e2",
    "summary": "Holiday",
    "start": {"date": "2026-12-25"},
    "end": {"date": "2026-12-26"},
    "htmlLink": "https://calendar.google.com/event?eid=e2",
    "status": "confirmed",
}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "client-secret")


@pytest.fixture
def vault(tmp_path):
    return Vault(
        key=Fernet.generate_key().decode(),
        file_path=str(tmp_path / "vault" / "secrets.enc.json"),
    )


def _token_response(request) -> Response:
    if request.url.path == "/token":
        return Response(200, json={"access_token": "access-1", "expires_in": 3600})
    return Response(200, json={"items": []})


def _client_from(transport, vault):
    vault.set("calendar_refresh_token", "refresh-token-1")
    return CalendarClient(vault=vault, transport=transport)


# --- Auth flow helpers --------------------------------------------------------

def test_build_authorize_url_includes_scope_offline_and_state():
    from urllib.parse import parse_qs, urlparse

    url = build_authorize_url("cid", "http://127.0.0.1:8889/callback", state="abc")
    parts = urlparse(url)
    query = parse_qs(parts.query)
    assert parts.scheme == "https"
    assert parts.path == "/o/oauth2/v2/auth"
    assert query["access_type"] == ["offline"]
    assert query["prompt"] == ["consent"]
    assert query["scope"] == ["https://www.googleapis.com/auth/calendar"]
    assert query["client_id"] == ["cid"]
    assert query["redirect_uri"] == ["http://127.0.0.1:8889/callback"]
    assert query["state"] == ["abc"]


def test_store_refresh_token_roundtrip(vault):
    stored = store_refresh_token({"refresh_token": "rt-1", "access_token": "x"}, vault)
    assert stored == "rt-1"
    assert vault.get("calendar_refresh_token") == "rt-1"


def test_store_refresh_token_raises_without_refresh_token(vault):
    with pytest.raises(RuntimeError, match="no refresh_token"):
        store_refresh_token({"error": "invalid_grant"}, vault)


async def test_client_requires_stored_refresh_token(env, tmp_path):
    empty_vault = Vault(
        key=Fernet.generate_key().decode(),
        file_path=str(tmp_path / "empty" / "secrets.enc.json"),
    )
    with pytest.raises(RuntimeError, match="No Google Calendar refresh token"):
        CalendarClient(vault=empty_vault)


async def test_client_requires_google_env(env, monkeypatch, vault):
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    vault.set("calendar_refresh_token", "rt-1")
    with pytest.raises(RuntimeError, match="GOOGLE_CLIENT_ID"):
        CalendarClient(vault=vault)


# --- get_upcoming_events ------------------------------------------------------

async def test_get_upcoming_events_normalizes_and_sets_list_params(env, vault):
    seen = {}

    def handler(request):
        seen["url"] = request.url
        token = _token_response(request)
        if token.status_code == 200 and "access_token" in token.json():
            return token
        return Response(200, json={"items": [DATED_EVENT, ALL_DAY_EVENT]})

    client = _client_from(MockTransport(handler), vault)
    try:
        events = await client.get_upcoming_events(max_results=3)
    finally:
        await client.aclose()

    assert seen["url"].path == "/calendar/v3/calendars/primary/events"
    assert seen["url"].params["orderBy"] == "startTime"
    assert seen["url"].params["singleEvents"] == "true"
    assert seen["url"].params["maxResults"] == "3"
    assert "timeMin" in seen["url"].params  # only events from now onward

    assert events[0]["summary"] == "Standup"
    assert events[0]["start"] == "2026-09-21T09:30:00+02:00"
    assert events[0]["all_day"] is False
    assert events[0]["location"] == "Zoom"
    assert events[0]["status"] == "confirmed"
    assert events[1]["summary"] == "Holiday"
    assert events[1]["start"] == "2026-12-25"
    assert events[1]["all_day"] is True


async def test_get_upcoming_empty_calendar(env, vault):
    client = _client_from(MockTransport(_token_response), vault)
    try:
        assert await client.get_upcoming_events() == []
    finally:
        await client.aclose()


# --- create_event --------------------------------------------------------------

async def test_create_event_posts_expected_body(env, vault):
    posted = {}

    def handler(request):
        posted["method"] = request.method
        posted["path"] = request.url.path
        posted["body"] = request.read()
        token = _token_response(request)
        if token.status_code == 200 and "access_token" in token.json():
            return token
        created = {
            "id": "new-1",
            "summary": "Dentist",
            "start": {"dateTime": "2026-09-22T14:00:00+02:00"},
            "end": {"dateTime": "2026-09-22T14:30:00+02:00"},
            "htmlLink": "https://calendar.google.com/event?eid=new-1",
            "status": "confirmed",
        }
        return Response(200, json=created)

    client = _client_from(MockTransport(handler), vault)
    try:
        event = await client.create_event(
            "Dentist", "2026-09-22T14:00:00+02:00", "2026-09-22T14:30:00+02:00",
            description="annual checkup",
        )
    finally:
        await client.aclose()

    import json

    body = json.loads(posted["body"])
    assert posted["method"] == "POST"
    assert posted["path"] == "/calendar/v3/calendars/primary/events"
    assert body["summary"] == "Dentist"
    assert body["start"] == {"dateTime": "2026-09-22T14:00:00+02:00"}
    assert body["end"] == {"dateTime": "2026-09-22T14:30:00+02:00"}
    assert body["description"] == "annual checkup"
    assert event["summary"] == "Dentist"
    assert event["start"] == "2026-09-22T14:00:00+02:00"


async def test_create_event_rejects_empty_summary(env, vault):
    client = _client_from(MockTransport(lambda r: Response(200, json={})), vault)
    try:
        with pytest.raises(CalendarError, match="summary must not be empty"):
            await client.create_event("   ", "2026-09-22T14:00:00+02:00", "2026-09-22T14:30:00+02:00")
    finally:
        await client.aclose()


# --- Token refresh serialization ----------------------------------------------

async def test_token_refresh_fires_once_under_concurrent_calls(env, vault):
    refresh_hits = 0

    def handler(request):
        nonlocal refresh_hits
        if request.url.path == "/token":
            refresh_hits += 1
            return Response(200, json={"access_token": "access-1", "expires_in": 3600})
        return Response(200, json={"items": []})

    client = _client_from(MockTransport(handler), vault)
    try:
        import asyncio

        await asyncio.gather(
            client.get_upcoming_events(), client.get_upcoming_events()
        )
        assert refresh_hits == 1
        assert client._access_token == "access-1"
    finally:
        await client.aclose()