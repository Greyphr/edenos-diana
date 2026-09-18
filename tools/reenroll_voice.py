from identity.voiceprint_store import VoiceprintStore
from tools.registry import RiskTier, ToolSpec

REENROLL_CONFIRMATION_PHRASE = "yes, replace my voice profile"


def make_reenroll_voice_spec(
    store: VoiceprintStore | None = None,
) -> ToolSpec:
    """Voice-triggered re-enrollment: archive the current voiceprint so the
    next ``python -m identity.enroll`` run starts fresh. Does not capture
    audio itself.
    """
    store = store or VoiceprintStore()

    async def _reenroll_handler(**args) -> dict:
        profiles = store.list_profiles()
        if not profiles:
            return {
                "status": "ok",
                "message": (
                    "No voice profile exists yet - run 'python -m "
                    "identity.enroll' to record one."
                ),
            }
        archived = []
        for name in profiles:
            moved = store.archive_profile(name)
            if moved:
                archived.append(moved)
        return {
            "status": "ok",
            "message": (
                "Voice profile archived - run 'python -m identity.enroll' "
                "to record a new one."
            ),
            "archived": archived,
        }

    return ToolSpec(
        name="reenroll_voice",
        description=(
            "Redo the owner's voice enrollment because their voice keeps "
            "being misrecognized or was recorded poorly. Archives the "
            "current voice profile so the next enrollment starts fresh, and "
            "tells the owner to run 'python -m identity.enroll'."
        ),
        parameters={"type": "object", "properties": {}},
        risk_tier=RiskTier.SENSITIVE,
        handler=_reenroll_handler,
        confirmation_phrase=REENROLL_CONFIRMATION_PHRASE,
    )