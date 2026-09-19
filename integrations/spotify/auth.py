"""One-time Spotify authorization (stands alone from the runtime).

Run:  python -m integrations.spotify.auth

1. Opens the Spotify authorization page in your browser.
2. Serves a tiny local callback server bound to SPOTIFY_REDIRECT_URI's
   host:port, waiting for the redirect with the authorization code.
3. Exchanges the code for a refresh token.
4. Stores the *refresh* token in the vault only; the access token is never
   persisted.

Mirrors identity/enroll.py: a deliberately separate, one-shot CLI so the
always-on main loop never needs browser interactivity.
"""

import json
import logging
import os
import secrets
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

from dotenv import load_dotenv

from tools.vault import Vault

logger = logging.getLogger(__name__)

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = (
    "user-read-playback-state "
    "user-modify-playback-state "
    "user-read-currently-playing"
)
VAULT_KEY_NAME = "spotify_refresh_token"


def build_authorize_url(
    client_id: str, redirect_uri: str, scopes: str = SCOPES, state: str | None = None
) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
    }
    # OAuth state: echoed back by Spotify in the callback so we can tell the
    # redirect came from a request we initiated, not a spoofed one.
    if state:
        params["state"] = state
    query = urllib.parse.urlencode(params)
    return f"{AUTHORIZE_URL}?{query}"


def exchange_code_for_tokens(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    urlopen=urllib.request.urlopen,
) -> dict:
    """POST the authorization code to the token endpoint. Returns JSON dict."""
    form = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode()
    request = urllib.request.Request(TOKEN_URL, data=form)
    with urlopen(request, timeout=30) as response:  # type: ignore[arg-type]
        return json.load(response)


def store_refresh_token(tokens: dict, vault: Vault) -> str:
    """Persist the refresh token from a token response. Raises if absent."""
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        # Never dump the raw dict: it may contain a live access_token.
        error = tokens.get("error") or "unknown_error"
        description = tokens.get("error_description") or "no error description returned"
        raise RuntimeError(
            f"Token response had no refresh_token: {error}: {description}"
        )
    vault.set(VAULT_KEY_NAME, refresh_token)
    return refresh_token


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures the code Spotify redirects to the local callback URI."""

    captured_code: str | None = None
    expected_state: str | None = None
    reject_reason: str | None = None

    def _reject(self, reason: str) -> None:
        self.__class__.reject_reason = reason
        body = f"{reason}: {self.path}".encode()
        self.send_response(400)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]

        if self.__class__.expected_state is not None:
            if state != self.__class__.expected_state:
                self._reject(
                    "State parameter is missing or does not match - callback rejected"
                )
            elif not code:
                self._reject("Missing code in callback")
            else:
                self.__class__.captured_code = code
                body = b"Authorization complete - you can close this tab."
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)
        elif code:
            self.__class__.captured_code = code
            body = b"Authorization complete - you can close this tab."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(body)
        else:
            self._reject("Missing code in callback")

        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        return  # keep the console quiet during the callback


def main() -> None:
    load_dotenv()
    client_id = os.getenv("SPOTIFY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback")

    missing = [
        name
        for name, value in [
            ("SPOTIFY_CLIENT_ID", client_id),
            ("SPOTIFY_CLIENT_SECRET", client_secret),
        ]
        if not value
    ]
    if missing:
        print(
            "Missing required Spotify env var(s): " + ", ".join(missing)
            + "\nAdd them to .env (see .env.example), then re-run."
        )
        sys.exit(1)

    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        print(
            f"SPOTIFY_REDIRECT_URI must be an http://host:port/path URL "
            f"(the callback server binds to it). Got: {redirect_uri!r}"
        )
        sys.exit(1)

    _CallbackHandler.captured_code = None
    _CallbackHandler.reject_reason = None
    state = secrets.token_urlsafe(16)
    _CallbackHandler.expected_state = state
    authorize_url = build_authorize_url(client_id, redirect_uri, state=state)
    print("Open this URL in your browser (it should have opened):")
    print("  " + authorize_url)
    webbrowser.open(authorize_url)

    try:
        server = HTTPServer((parsed.hostname, parsed.port), _CallbackHandler)
    except OSError as exc:
        print(
            f"Could not bind the callback server to "
            f"{parsed.hostname}:{parsed.port}: {exc}"
        )
        sys.exit(1)

    print(
        f"Waiting for the Spotify callback on "
        f"http://{parsed.hostname}:{parsed.port} ..."
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAuthorization cancelled.")
        sys.exit(1)
    finally:
        server.server_close()

    code = _CallbackHandler.captured_code
    if code is None:
        reason = _CallbackHandler.reject_reason
        if reason:
            print(
                f"Callback rejected: {reason}. "
                "Make sure the authorization page came from this run's URL "
                "and try again."
            )
        else:
            print("No authorization code received. Try again.")
        sys.exit(1)

    print("Exchanging the code for a refresh token...")
    tokens = exchange_code_for_tokens(code, client_id, client_secret, redirect_uri)
    store_refresh_token(tokens, Vault())
    print("Stored the Spotify refresh token in the vault.")
    print("Done - start Eden with `python main.py` to enable the spotify_* tools.")


if __name__ == "__main__":
    main()