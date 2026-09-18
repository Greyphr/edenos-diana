"""Spotify Web API integration.

- integrations/spotify/auth.py   one-time browser authorization (stores the
                                 refresh token in the vault, never the access token)
- integrations/spotify/client.py async Web API client for the spotify_* tools
- integrations/spotify/tools.py  ToolSpecs wired into main.py's tool registry
"""