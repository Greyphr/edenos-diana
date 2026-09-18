"""Pre-flight checks run before the event loop starts.

Each result is a tuple ``(check_name, passed, detail, severity)`` — the
first three fields are as specified; ``severity`` is either
``SEVERITY_CRITICAL`` or ``SEVERITY_WARNING`` so the caller can decide
whether a failure should abort startup.
"""
import logging
import os

logger = logging.getLogger(__name__)

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"


def run_startup_checks() -> list[tuple[str, bool, str, str]]:
    checks = []

    # GEMINI_API_KEY (critical) -------------------------------------------
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        checks.append(
            (
                "GEMINI_API_KEY",
                True,
                f"set (length {len(api_key)})",
                SEVERITY_CRITICAL,
            )
        )
    else:
        checks.append(
            (
                "GEMINI_API_KEY",
                False,
                "not set - add GEMINI_API_KEY=... to your .env "
                "(copy .env.example)",
                SEVERITY_CRITICAL,
            )
        )

    # Audio devices (critical) --------------------------------------------
    try:
        import sounddevice as sd

        try:
            output = sd.query_devices(kind="output")
            checks.append(
                (
                    "default output audio device",
                    True,
                    f"{output['name']}",
                    SEVERITY_CRITICAL,
                )
            )
        except Exception as e:
            checks.append(
                (
                    "default output audio device",
                    False,
                    f"none found: {e}",
                    SEVERITY_CRITICAL,
                )
            )
        try:
            input_ = sd.query_devices(kind="input")
            checks.append(
                (
                    "default input audio device",
                    True,
                    f"{input_['name']}",
                    SEVERITY_CRITICAL,
                )
            )
        except Exception as e:
            checks.append(
                (
                    "default input audio device",
                    False,
                    f"none found: {e}",
                    SEVERITY_CRITICAL,
                )
            )
    except Exception as e:
        for name in ("default output audio device", "default input audio device"):
            checks.append(
                (name, False, f"sounddevice unavailable: {e}", SEVERITY_CRITICAL)
            )

    # config/voice.yaml (critical) ----------------------------------------
    try:
        from conversation.voice_config import load_voice_config

        config = load_voice_config()
        provider = config.get("provider")
        voice_name = config.get("voice_name")
        checks.append(
            (
                "config/voice.yaml",
                True,
                f"loaded (provider={provider}, voice_name={voice_name})",
                SEVERITY_CRITICAL,
            )
        )
    except Exception as e:
        checks.append(
            (
                "config/voice.yaml",
                False,
                f"failed to load: {e}",
                SEVERITY_CRITICAL,
            )
        )

    # Enrolled voiceprint (warning only) ----------------------------------
    try:
        from identity.voiceprint_store import VoiceprintStore

        profiles = VoiceprintStore().list_profiles()
        if profiles:
            checks.append(
                (
                    "enrolled voiceprint",
                    True,
                    f"{len(profiles)} profile(s): {', '.join(profiles)}",
                    SEVERITY_WARNING,
                )
            )
        else:
            checks.append(
                (
                    "enrolled voiceprint",
                    False,
                    "none - run 'python -m identity.enroll' to add one "
                    "(warning only; Eden runs without it)",
                    SEVERITY_WARNING,
                )
            )
    except Exception as e:
        checks.append(
            (
                "enrolled voiceprint",
                False,
                f"could not inspect: {e}",
                SEVERITY_WARNING,
            )
        )

    return checks