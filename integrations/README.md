## Spotify (block 5, integrations)

- One-time setup: `python -m integrations.spotify.auth` (opens the browser,
  exchanges the authorization code, stores `spotify_refresh_token` in the
  vault). Requires `VAULT_KEY`, `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`,
  `SPOTIFY_REDIRECT_URI` in `.env`.
- Runtime: `main.py` constructs `SpotifyClient`; if the refresh token is not
  in the vault yet it logs "Spotify not connected — run
  `python -m integrations.spotify.auth`" and skips registering the tools.
- Tools: `spotify_play`, `spotify_resume`, `spotify_pause`, `spotify_skip`,
  `spotify_previous`, `spotify_set_volume` (WRITE tier, confirm-flow) and
  `spotify_current_track` (READ tier).