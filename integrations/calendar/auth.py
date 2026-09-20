"""One-time Google Calendar authorization (reusable flow + standalone CLI).

Run:  python -m integrations.calendar.auth

1. Builds the Google OAuth authorize URL and opens it in your browser.
2. Serves a tiny local callback server bound to GOOGLE_REDIRECT_URI's
   host:port, waiting for Google to redirect the authorization code back.
3. Exchanges that code for tokens and stores the *refresh* token in the
   vault (the access token is never persisted).

The whole dance lives in :func:`run_calendar_auth_flow`, a reusable
``bool``-returning function that never calls ``sys.exit``. ``main()`` is a
thin CLI wrapper around it (it does the env-var checks / redirect-URI
validation and turns a failed run into ``sys.exit(1)``), so ``python -m
integrations.calendar.auth`` keeps working exactly as before while Eden's
startup (``main.py``) can call :func:`run_calendar_auth_flow` inline when the
owner authorizes from an interactive terminal.
"""

import asyncio
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
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from tools.vault import Vault

logger = logging.getLogger(__name__)

AUTHORIZE_URL = (
    "https://accounts.google.com/o/oauth2/v2/auth"
    "?access_type=offline&prompt=consent"
)
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPES = "https://www.googleapis.com/auth/calendar"
VAULT_KEY_NAME = "calendar_refresh_token"


class _CallbackHandler(BaseHTTPRequestHandler):
    """Captures the callback Google redirects to the local callback URI."""

    captured_code: str | None = None
    reject_reason: str | None = None
    expected_state: str | None = None

    def _reject(self, reason: str) -> None:
        self.__class__.reject_reason = reason
        body = f"{reason}: {self.path}".encode()
        self.send_response(400)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
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


def build_authorize_url(
    client_id: str,
    redirect_uri: str,
    scopes: str = SCOPES,
    state: str | None = None,
) -> str:
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": scopes,
    }
    # OAuth state: echoed back by Google in the callback so we can tell the
    # redirect came from a request we initiated, not a spoofed one.
    if state:
        params["state"] = state
    query = urllib.parse.urlencode(params)
    return f"{AUTHORIZE_URL}&{query}"


def exchange_code_for_tokens(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    urlopen=urllib.request.urlopen,
) -> dict:
    """POST the authorization code to Google's token endpoint. Returns JSON."""
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


async def run_calendar_auth_flow(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    vault: Vault | None = None,
) -> bool:
    """Run the full Google Calendar OAuth dance, returning whether it succeeded.

    This mirrors ``run_spotify_auth_flow`` exactly: it builds the authorize
    URL (with ``access_type=offline`` and ``prompt=consent`` so Google always
    returns a refreshable token), opens it in the browser, binds a local
    callback server, waits for Google to redirect the code back, exchanges
    that code for a refresh token and stores it in the vault.

    It returns ``True`` only when the refresh token actually landed in the
    vault (which is what later makes ``CalendarClient()`` succeed). It returns
    ``False`` on every failure path -- a bind failure on the callback server,
    a KeyboardInterrupt while waiting, a rejected/missing callback, a failed
    token exchange, or a failed vault write -- and it *never* calls
    ``sys.exit``. Callers decide what a failure means: the standalone
    ``python -m integrations.calendar.auth`` entry point treats any ``False``
    as fatal, while Eden's startup treats it as "keep going without the
    calendar_* tools". The callback server runs in a thread via
    ``asyncio.to_thread`` so this async flow never blocks Eden's event loop.
    """

    # Validate before doing anything side-effect-y so a bad config returns
    # False (main() keeps the fatal env-var checks; this is the safe net).
    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        print(
            f"GOOGLE_REDIRECT_URI must be an http://host:port/path URL "
            f"(the callback server binds to it). Got: {redirect_uri!r}"
        )
        return False

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
    except (OSError, TypeError, ValueError) as exc:
        print(
            f"Could not bind the callback server to "
            f"{parsed.hostname}:{parsed.port}: {exc}"
        )
        return False

    print(
        f"Waiting for the Google callback on "
        f"http://{parsed.hostname}:{parsed.port} ..."
    )
    try:
        await asyncio.to_thread(server.serve_forever)
    except KeyboardInterrupt:
        print("\nAuthorization cancelled.")
        return False
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
        return False

    print("Exchanging the code for a refresh token...")
    tokens = exchange_code_for_tokens(code, client_id, client_secret, redirect_uri)
    if vault is None:
        vault = Vault()
    try:
        store_refresh_token(tokens, vault)
    except RuntimeError as exc:
        print(f"Could not store the Google Calendar refresh token: {exc}")
        return False
    print("Stored the Google Calendar refresh token in the vault.")
    return True


def main() -> None:
    """One-shot Google Calendar authorization (standalone CLI entry point)."""
    load_dotenv()
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET", "")
    redirect_uri = os.getenv(
        "GOOGLE_REDIRECT_URI", "http://127.0.0.1:8889/callback"
    )

    missing = [
        name
        for name, value in [
            ("GOOGLE_CLIENT_ID", client_id),
            ("GOOGLE_CLIENT_SECRET", client_secret),
        ]
        if not value
    ]
    if missing:
        print(
            "Missing required Google env var(s): " + ", ".join(missing)
            + "\nAdd them to .env (see .env.example), then re-run."
        )
        sys.exit(1)

    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        print(
            f"GOOGLE_REDIRECT_URI must be an http://host:port/path URL "
            f"(the callback server binds to it). Got: {redirect_uri!r}"
        )
        sys.exit(1)

    if not asyncio.run(
        run_calendar_auth_flow(client_id, client_secret, redirect_uri)
    ):
        sys.exit(1)
    print("Done - start Eden with `python main.py` to enable the calendar_* tools.")


if __name__ == "__main__":
    main()