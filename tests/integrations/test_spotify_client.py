"""Spotify client: URI normalization and the refresh-once-under-concurrency
guarantee (asyncio.Lock serialization)."""

import pytest
from cryptography.fernet import Fernet
from httpx import AsyncClient, MockTransport, Response

from integrations.spotify.client import (
    SpotifyClient,
    SpotifyError,
    normalize_track_uri,
)
from tools.vault import Vault


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPOTIFY_CLIENT_ID", "client-id")
    monkeypatch.setenv("SPOTIFY_CLIENT_SECRET", "client-secret")


@pytest.fixture
def vault(tmp_path):
    return Vault(
        key=Fernet.generate_key().decode(),
        file_path=str(tmp_path / "vault" / "secrets.enc.json"),
    )


# --- normalize_track_uri ------------------------------------------------------

def test_plain_uri_passes_through():
    assert normalize_track_uri("spotify:track:abc123") == "spotify:track:abc123"


def test_full_open_spotify_url_is_normalized():
    assert (
        normalize_track_uri("https://open.spotify.com/track/abc123")
        == "spotify:track:abc123"
    )


def test_url_with_query_params_is_normalized():
    assert (
        normalize_track_uri("https://open.spotify.com/track/abc123?si=xyz&x=1")
        == "spotify:track:abc123"
    )


def test_whitespace_is_stripped():
    assert normalize_track_uri("  spotify:track:abc123  ") == "spotify:track:abc123"


@pytest.mark.parametrize(
    "garbage",
    [
        "",
        "not a uri",
        "https://open.spotify.com/album/xyz",
        "spotify:album:xyz",
        "https://youtube.com/watch?v=xyz",
    ],
)
def test_garbage_is_rejected(garbage):
    with pytest.raises(SpotifyError, match="Unrecognized track URI"):
        normalize_track_uri(garbage)


# --- Token refresh serialization ----------------------------------------------

async def test_token_refresh_fires_once_under_concurrent_calls(env, vault):
    vault.set("spotify_refresh_token", "refresh-token-1")
    refresh_hits = 0

    def handler(request):
        nonlocal refresh_hits
        if request.url.path == "/api/token":
            refresh_hits += 1
            return Response(200, json={"access_token": "access-1", "expires_in": 3600})
        return Response(200, json={})

    client = SpotifyClient(
        vault=vault,
        transport=MockTransport(handler),
    )
    try:
        # Two callers both see a stale/no access token and race to refresh.
        # The asyncio.Lock means exactly one round-trips to the token endpoint.
        import asyncio

        await asyncio.gather(
            client._ensure_access_token(), client._ensure_access_token()
        )
        assert refresh_hits == 1
        assert client._access_token == "access-1"
    finally:
        await client.aclose()


async def test_client_requires_stored_refresh_token(env, tmp_path):
    empty_vault = Vault(
        key=Fernet.generate_key().decode(),
        file_path=str(tmp_path / "empty" / "secrets.enc.json"),
    )
    with pytest.raises(RuntimeError, match="No Spotify refresh token"):
        SpotifyClient(vault=empty_vault)