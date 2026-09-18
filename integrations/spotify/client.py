"""Spotify client used by the spotify_* tools.

The refresh token lives in the vault (written by `python -m
integrations.spotify.auth`). Access tokens are kept in memory only, refreshed
on demand with slack a bit before real expiry — and re-validated by a single
401-retry so we never blindly trust the cached expiry.

Two playback paths:
- ``launch_track_locally`` opens a specific track through the OS protocol
  handler (Spotify app), working even with Spotify fully closed. This is the
  primary path for "play <song>".
- The Web API play/pause/skip/volume calls steer an *existing active device*;
  a successful local launch has just created one, so those tools rely on it.
"""

import logging
import os
import platform
import re
import subprocess
import time

import httpx

from tools.vault import Vault

logger = logging.getLogger(__name__)

TOKEN_URL = "https://accounts.spotify.com/api/token"
API_BASE = "https://api.spotify.com"
VAULT_KEY_NAME = "spotify_refresh_token"
EXPIRY_SLACK_SECONDS = 30  # refresh slightly before the token actually dies

_TRACK_LINK_RE = re.compile(r"https?://open\.spotify\.com/track/([A-Za-z0-9]+)")


def normalize_track_uri(value: str) -> str:
    """Normalize whatever search_track() returns into ``spotify:track:<id>``.
    Accepts an existing spotify: URI or a full open.spotify.com/track/<id>
    link (with or without query params).
    """
    value = (value or "").strip()
    if value.startswith("spotify:track:"):
        return value
    match = _TRACK_LINK_RE.match(value)
    if match:
        return f"spotify:track:{match.group(1)}"
    raise SpotifyError(f"Unrecognized track URI/URL: {value!r}")


class SpotifyError(Exception):
    pass


class NoActiveDeviceError(SpotifyError):
    """A playback command hit a 404: Spotify has no device steering playback.

    ``devices`` carries the normalized /v1/me/player/devices payload (each
    entry has name/type/is_active) when the client could fetch it — so the
    tool handler can tell the owner whether Spotify sees no devices at all or
    devices that simply aren't active. ``None`` means the device list itself
    couldn't be fetched.
    """

    def __init__(self, message: str, devices: list[dict] | None = None):
        super().__init__(message)
        self.devices = devices


class SpotifyClient:
    def __init__(self, vault: Vault | None = None, transport=None):
        self._vault = vault or Vault()
        self._refresh_token = self._vault.get(VAULT_KEY_NAME)
        if not self._refresh_token:
            raise RuntimeError(
                "No Spotify refresh token stored. Run "
                "`python -m integrations.spotify.auth` to authorize first."
            )
        if not os.getenv("SPOTIFY_CLIENT_ID") or not os.getenv(
            "SPOTIFY_CLIENT_SECRET"
        ):
            raise RuntimeError(
                "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env "
                "to refresh the access token."
            )
        self._access_token: str | None = None
        self._access_expires_at: float = 0.0
        self._refresh_client = httpx.AsyncClient(timeout=30, transport=transport)
        self._client = httpx.AsyncClient(
            base_url=API_BASE,
            headers={"User-Agent": "Eden/1.0"},
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._refresh_client.aclose()

    async def _refresh_access_token(self) -> None:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": os.getenv("SPOTIFY_CLIENT_ID"),
            "client_secret": os.getenv("SPOTIFY_CLIENT_SECRET"),
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
            logger.info("Spotify rotated the refresh token; updating the vault")
            self._refresh_token = rotated
            self._vault.set(VAULT_KEY_NAME, rotated)

    async def _ensure_access_token(self) -> None:
        if self._access_token is None or time.time() >= self._access_expires_at:
            await self._refresh_access_token()

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
        if response.status_code == 401:
            logger.info("Spotify API returned 401; refreshing access token and retrying")
            await self._refresh_access_token()
            headers = {"Authorization": f"Bearer {self._access_token}"}
            response = await self._client.request(
                method, path, params=params, json=json_body, headers=headers
            )
        return response

    async def _playback(
        self,
        method: str,
        path: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> None:
        response = await self._request(method, path, params=params, json_body=json_body)
        if response.status_code == 404:
            raise NoActiveDeviceError(
                "No active Spotify device - open Spotify on a device and try again."
            )
        response.raise_for_status()

    async def get_current_track(self) -> dict | None:
        """Current play context; None when nothing is playing."""
        response = await self._request("GET", "/v1/me/player/currently-playing")
        if response.status_code == 204:
            return None
        response.raise_for_status()
        data = response.json()
        item = data.get("item")
        if not item:
            return None
        album = (item.get("album") or {})
        return {
            "name": item.get("name"),
            "artist": ", ".join(a.get("name", "") for a in item.get("artists", [])),
            "album": album.get("name") if isinstance(album, dict) else None,
            "is_playing": bool(data.get("is_playing")),
        }

    async def search_track(self, query: str) -> str | None:
        """Best-track URI for the query, or None when nothing matched."""
        response = await self._request(
            "GET",
            "/v1/search",
            params={"q": query, "type": "track", "limit": 1},
        )
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        return items[0].get("uri") if items else None

    async def launch_track_locally(self, uri: str) -> None:
        """Open a specific track through the OS protocol handler so it plays
        even when Spotify is fully closed. The Web API path can't start
        playback without an existing active device; this launch creates one.
        """
        track_uri = normalize_track_uri(uri)
        system = platform.system()
        if system == "Windows":
            try:
                os.startfile(track_uri)  # type: ignore[attr-defined]
            except OSError as exc:
                raise SpotifyError(
                    f"Could not open Spotify locally (os.startfile failed): {exc}"
                ) from exc
            return
        if system == "Darwin":
            opener = "open"
        elif system == "Linux":
            opener = "xdg-open"
        else:
            raise SpotifyError(
                f"No local protocol opener for platform {system!r} - "
                "can't launch a track outside the Web API path."
            )
        try:
            subprocess.Popen(
                [opener, track_uri],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise SpotifyError(
                f"Could not open Spotify locally with {opener!r}: {exc}"
            ) from exc

    async def play(self, uri: str | None = None) -> None:
        response = await self._request(
            "PUT", "/v1/me/player/play", json_body={"uris": [uri]} if uri else {}
        )
        if response.status_code == 404:
            # 404 means nothing is steering playback. Distinguish "Spotify
            # sees no device at all" from "devices exist but none is active"
            # by asking /v1/me/player/devices, and deliver that list in the
            # error so the model can tell the owner exactly which case it is.
            raise await self._build_no_active_device_error()
        response.raise_for_status()

    async def _list_devices(self) -> list[dict] | None:
        """Normalized device list {name, type, is_active}, or None if the
        list itself couldn't be fetched."""
        try:
            response = await self._request("GET", "/v1/me/player/devices")
        except httpx.HTTPError as exc:
            logger.warning("Could not fetch Spotify devices: %s", exc)
            return None
        if not (200 <= response.status_code < 300):
            logger.warning(
                "Could not fetch Spotify devices (HTTP %s)", response.status_code
            )
            return None
        return [
            {
                "name": device.get("name"),
                "type": device.get("type"),
                "is_active": bool(device.get("is_active")),
            }
            for device in response.json().get("devices", [])
        ]

    async def _build_no_active_device_error(self) -> NoActiveDeviceError:
        devices = await self._list_devices()
        if devices is None:
            return NoActiveDeviceError(
                "No active Spotify device - could not list available devices."
            )
        if not devices:
            return NoActiveDeviceError(
                "no Spotify devices found - open Spotify and start playing "
                "something first",
                devices=devices,
            )
        listing = ", ".join(
            f"{device.get('name') or 'unnamed'} ({device.get('type')})"
            + (" [active]" if device.get("is_active") else "")
            for device in devices
        )
        if any(device.get("is_active") for device in devices):
            return NoActiveDeviceError(
                f"Spotify reported an active device but playback could not "
                f"start - found: {listing}",
                devices=devices,
            )
        return NoActiveDeviceError(
            f"No active Spotify device - found inactive devices: {listing}",
            devices=devices,
        )

    async def pause(self) -> None:
        await self._playback("PUT", "/v1/me/player/pause")

    async def next_track(self) -> None:
        await self._playback("POST", "/v1/me/player/next")

    async def previous_track(self) -> None:
        await self._playback("POST", "/v1/me/player/previous")

    async def set_volume(self, percent: int) -> None:
        await self._playback(
            "PUT",
            "/v1/me/player/volume",
            params={"volume_percent": max(0, min(100, int(percent)))},
        )