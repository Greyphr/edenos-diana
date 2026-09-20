"""Google Calendar client used by the calendar_* tools.

The refresh token lives in the vault (written by `python -m
integrations.calendar.auth`). Access tokens are kept in memory only, refreshed
on demand with slack a bit before real expiry — and re-validated by a single
401-retry so we never blindly trust the cached expiry (same pattern as the
Spotify client).
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone

import httpx

from tools.vault import Vault

logger = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
API_BASE = "https://www.googleapis.com"
VAULT_KEY_NAME = "calendar_refresh_token"
EXPIRY_SLACK_SECONDS = 30  # refresh slightly before the token actually dies
# 429 rate-limit sleep: honor Retry-After, but cap it so a misbehaving header
# can't stall the conversation for minutes.
RETRY_AFTER_CAP_SECONDS = 10.0
RETRY_AFTER_DEFAULT_SECONDS = 1.0


class CalendarError(Exception):
    pass


def _normalize_start(raw: object) -> str | None:
    """Turn a Google event start/end payload into a single string.

    ``{"date": "2026-09-21"}`` (all-day) becomes "2026-09-21"; a
    ``{"dateTime": "2026-09-21T15:00:00+02:00", ...}`` stays as its dateTime.
    """
    if isinstance(raw, dict):
        if raw.get("dateTime"):
            return str(raw["dateTime"])
        if raw.get("date"):
            return str(raw["date"])
    return None


def _now_rfc3339() -> str:
    """Current UTC time as an RFC 3339 string for the ``timeMin`` filter."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _normalize_event(item: dict) -> dict:
    """Keep only what the calendar tools / oral responses actually use."""
    start = _normalize_start(item.get("start"))
    end = _normalize_start(item.get("end"))
    return {
        "id": item.get("id"),
        "summary": item.get("summary"),
        "start": start,
        "end": end,
        "all_day": isinstance(item.get("start"), dict) and bool(
            item.get("start", {}).get("date")
        ),
        "html_link": item.get("htmlLink"),
        "location": item.get("location"),
        "description": item.get("description"),
        "status": item.get("status"),
    }


class CalendarClient:
    def __init__(self, vault: Vault | None = None, transport=None):
        self._vault = vault or Vault()
        self._refresh_token = self._vault.get(VAULT_KEY_NAME)
        if not self._refresh_token:
            raise RuntimeError(
                "No Google Calendar refresh token stored. Run "
                "`python -m integrations.calendar.auth` to authorize first."
            )
        if not os.getenv("GOOGLE_CLIENT_ID") or not os.getenv(
            "GOOGLE_CLIENT_SECRET"
        ):
            raise RuntimeError(
                "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET must be set in .env "
                "to refresh the access token."
            )
        self._access_token: str | None = None
        self._access_expires_at: float = 0.0
        self._refresh_lock = asyncio.Lock()
        self._refresh_client = httpx.AsyncClient(timeout=30, transport=transport)
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"User-Agent": "Eden/1.0"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._refresh_client.aclose()

    async def _refresh_access_token(self, *, force: bool = False) -> None:
        async with self._refresh_lock:
            # Serialize refreshes: concurrent callers that all saw a stale
            # token wait on the lock, then re-check and skip when a peer
            # already refreshed. `force` bypasses the re-check so a 401
            # always round-trips to the token endpoint even if the cached
            # expiry still looks valid.
            if (
                not force
                and self._access_token is not None
                and time.time() < self._access_expires_at
            ):
                return
            form = {
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
                "client_id": os.getenv("GOOGLE_CLIENT_ID"),
                "client_secret": os.getenv("GOOGLE_CLIENT_SECRET"),
            }
            response = await self._refresh_client.post(TOKEN_URL, data=form)
            response.raise_for_status()
            data = response.json()
            self._access_token = data["access_token"]
            self._access_expires_at = (
                time.time() + int(data.get("expires_in", 3600)) - EXPIRY_SLACK_SECONDS
            )
            rotated = data.get("refresh_token")
            if rotated and rotated != self._refresh_token:
                logger.info("Google rotated the refresh token; updating the vault")
                self._refresh_token = rotated
                self._vault.set(VAULT_KEY_NAME, rotated)

    async def _ensure_access_token(self) -> None:
        if self._access_token is None or time.time() >= self._access_expires_at:
            await self._refresh_access_token()

    def _retry_after_seconds(self, response: httpx.Response) -> float:
        """Delay to honor a 429 Retry-After, bounded so a hostile header
        can't stall the conversation for minutes."""
        raw = response.headers.get("Retry-After")
        if not raw:
            return RETRY_AFTER_DEFAULT_SECONDS
        try:
            seconds = float(raw)
        except ValueError:
            logger.warning(
                "Unparseable Retry-After %r; defaulting to %.0fs",
                raw,
                RETRY_AFTER_DEFAULT_SECONDS,
            )
            return RETRY_AFTER_DEFAULT_SECONDS
        if seconds < 0:
            seconds = 0.0
        return min(seconds, RETRY_AFTER_CAP_SECONDS)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> httpx.Response:
        await self._ensure_access_token()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        response = await self._client.request(
            method, path, params=params, json=json_body, headers=headers
        )
        if response.status_code == 429:
            # Rate-limited: honor Retry-After (bounded) and retry once.
            delay = self._retry_after_seconds(response)
            logger.warning("Google Calendar API rate-limited (429); retrying in %.1fs", delay)
            await asyncio.sleep(delay)
            response = await self._client.request(
                method, path, params=params, json=json_body, headers=headers
            )
        if response.status_code == 401:
            logger.info(
                "Google Calendar API returned 401; refreshing access token and retrying"
            )
            await self._refresh_access_token(force=True)
            headers = {"Authorization": f"Bearer {self._access_token}"}
            response = await self._client.request(
                method, path, params=params, json=json_body, headers=headers
            )
        return response

    async def get_upcoming_events(self, max_results: int = 5) -> list[dict]:
        """Upcoming events on the primary calendar, soonest start first.

        Only events that start at-or-after right now are returned; all-day
        events whose date is today or later count as upcoming. Each entry is
        the normalized payload from :func:`_normalize_event`.
        """
        params = {
            "timeMin": _now_rfc3339(),
            "maxResults": max(1, int(max_results)),
            "orderBy": "startTime",
            "singleEvents": True,
        }
        response = await self._request(
            "GET", "/calendar/v3/calendars/primary/events", params=params
        )
        response.raise_for_status()
        return [
            _normalize_event(item)
            for item in response.json().get("items", [])
        ]

    async def create_event(
        self,
        summary: str,
        start: str,
        end: str,
        description: str = "",
    ) -> dict:
        """Create an event on the primary calendar, returning it normalized.

        ``start``/``end`` are passed through as RFC 3339 ``dateTime`` values;
        ``summary`` is required and non-empty.
        """
        if not summary or not str(summary).strip():
            raise CalendarError("Event summary must not be empty.")
        body = {
            "summary": str(summary).strip(),
            "start": {"dateTime": str(start)},
            "end": {"dateTime": str(end)},
        }
        if description:
            body["description"] = str(description).strip()
        response = await self._request(
            "POST",
            "/calendar/v3/calendars/primary/events",
            json_body=body,
        )
        response.raise_for_status()
        return _normalize_event(response.json())