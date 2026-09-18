"""Spotify tool specs for the block 4 registry + policy engine.

Seven specs, same wiring path as reenroll_voice: they register into the same
ToolRegistry. Rich playback commands are TRIVIAL tier — owner-gated (denied
for unrecognized actors) but execute immediately with no confirmation prompt,
unlike WRITE/SENSITIVE which go through confirm_action. spotify_current_track
stays READ: it changes nothing and stays open to unrecognized voices too.

No-active-device (client raises NoActiveDeviceError) is an ordinary *result*,
never a raised error from a handler; an unmatched search is likewise a normal
outcome the model should read and respond to.
"""

from tools.registry import RiskTier, ToolSpec

from .client import NoActiveDeviceError

_PLAY_PARAMETERS = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
_VOLUME_PARAMETERS = {
    "type": "object",
    "properties": {"percent": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["percent"],
}
_NO_PARAMETERS = {"type": "object", "properties": {}}


def _no_active_device_result(exc: NoActiveDeviceError) -> dict:
    """Turn the client's no-device error into a normal tool result, keeping
    its specific message (no devices vs. inactive devices) and the device
    list so the model can tell the owner which situation it is."""
    result = {
        "error": "no_active_device",
        "message": str(exc)
        or "No active Spotify device - open Spotify on a device and try again.",
    }
    if getattr(exc, "devices", None):
        result["devices"] = exc.devices
    return result


def make_spotify_specs(client) -> list[ToolSpec]:
    async def _play(**args) -> dict:
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "missing_query", "message": "No song query provided."}
        try:
            uri = await client.search_track(query)
            if uri is None:
                return {
                    "status": "not_found",
                    "message": f"No Spotify track found for {query!r}.",
                }
            # Primary path: launch locally via the OS protocol handler. This
            # works with Spotify fully closed — the launch itself creates the
            # active device that resume/pause/skip/volume then steer. We do
            # NOT additionally call the Web API play() here.
            await client.launch_track_locally(uri)
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        return {
            "status": "ok",
            "message": f"Opening {query!r} in the Spotify app.",
            "track_uri": uri,
        }

    async def _resume(**args) -> dict:
        try:
            await client.play()
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        return {"status": "ok", "message": "Resumed playback."}

    async def _pause(**args) -> dict:
        try:
            await client.pause()
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        return {"status": "ok", "message": "Paused playback."}

    async def _skip(**args) -> dict:
        try:
            await client.next_track()
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        return {"status": "ok", "message": "Skipped to the next track."}

    async def _previous(**args) -> dict:
        try:
            await client.previous_track()
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        return {"status": "ok", "message": "Went to the previous track."}

    async def _set_volume(**args) -> dict:
        percent = args.get("percent")
        try:
            await client.set_volume(percent)
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        return {"status": "ok", "message": f"Volume set to {percent}%."}

    async def _current_track(**args) -> dict:
        try:
            track = await client.get_current_track()
        except NoActiveDeviceError as exc:
            return _no_active_device_result(exc)
        if track is None:
            return {"status": "nothing_playing", "message": "Nothing is playing right now."}
        album = f" (album {track['album']})" if track.get("album") else ""
        return {
            "status": "ok",
            "message": f"{track.get('name')} by {track.get('artist')}{album}",
            "name": track.get("name"),
            "artist": track.get("artist"),
            "album": track.get("album"),
            "is_playing": track.get("is_playing"),
        }

    return [
        ToolSpec(
            name="spotify_play",
            description=(
                "Play a specific song on Spotify. The query should be a song "
                "title (usually plus the artist) so the search finds one "
                "clear match. The matched track is opened through the OS "
                "protocol handler, which starts playback in the Spotify app "
                "even if it was fully closed (this creates the active device "
                "the other Spotify playback tools then use). Changes the "
                "owner's music, so it is owner-gated, but runs immediately "
                "without a confirmation prompt."
            ),
            parameters=_PLAY_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_play,
        ),
        ToolSpec(
            name="spotify_resume",
            description=(
                "Resume paused Spotify playback on the active device. Unlike "
                "spotify_play there is no specific track to launch, so this "
                "uses the Web API and requires a device to already be active "
                "(e.g. after a spotify_play launch). Owner-gated but runs "
                "immediately, no confirmation prompt."
            ),
            parameters=_NO_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_resume,
        ),
        ToolSpec(
            name="spotify_pause",
            description=(
                "Pause Spotify playback on the active device. Owner-gated "
                "but runs immediately, no confirmation prompt."
            ),
            parameters=_NO_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_pause,
        ),
        ToolSpec(
            name="spotify_skip",
            description=(
                "Skip to the next track on Spotify. Owner-gated but runs "
                "immediately, no confirmation prompt."
            ),
            parameters=_NO_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_skip,
        ),
        ToolSpec(
            name="spotify_previous",
            description=(
                "Go back to the previous track on Spotify. Owner-gated but "
                "runs immediately, no confirmation prompt."
            ),
            parameters=_NO_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_previous,
        ),
        ToolSpec(
            name="spotify_set_volume",
            description=(
                "Set the Spotify playback volume to a percentage (0-100) on "
                "the active device. Owner-gated but runs immediately, no "
                "confirmation prompt."
            ),
            parameters=_VOLUME_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_set_volume,
        ),
        ToolSpec(
            name="spotify_current_track",
            description=(
                "Read what is currently playing on Spotify (track, artist, "
                "album, and whether playback is active). Does not change "
                "anything, so it runs without confirmation and without "
                "requiring recognition."
            ),
            parameters=_NO_PARAMETERS,
            risk_tier=RiskTier.READ,
            handler=_current_track,
        ),
    ]