import asyncio
import logging
import sys
import time

import numpy as np
from dotenv import load_dotenv

from conversation.audio_io import AudioIO
from conversation.voice_config import build_system_instruction, load_voice_config
from conversation.voice_provider import get_voice_provider
from conversation.wake_word import WakeWordDetector
from identity.recognition import CONFIDENCE_RECOGNIZED, SpeakerRecognizer
from identity.voiceprint_store import VoiceprintStore
from integrations.spotify.client import SpotifyClient
from integrations.spotify.tools import make_spotify_specs
from memory.context import build_context_summary
from memory.tools import (
    RECALL_DECLARATION,
    REMEMBER_DECLARATION,
    make_memory_handlers,
)
from tools.policy_engine import PolicyEngine
from tools.reenroll_voice import make_reenroll_voice_spec
from tools.registry import ToolRegistry

IDLE_TIMEOUT_SECONDS = 8.0
# How long a single strong recognition keeps the actor treated as the owner.
# Short, weak utterances don't each get scored as a fresh recognition; this
# window spans the gap between them within one continuous engaged session.
# Never carried across idle transitions (cleared in go_idle/reset).
RECOGNITION_TRUST_WINDOW_SECONDS = 25
SPEECH_RMS_THRESHOLD = 400

STATE_IDLE = "idle"
STATE_ENGAGED = "engaged"


class RecognitionTrust:
    """Owner trust state for the policy engine: a time-based window keyed by
    the last genuinely confident recognition (confidence >=
    CONFIDENCE_RECOGNIZED) within the current engaged session.

    A strong hit keeps treating the actor as the owner for
    ``window_seconds`` regardless of how many weak/short results follow it,
    and ``reset()`` (called on the way back to idle) guarantees trust never
    carries across sessions.
    """

    def __init__(
        self, window_seconds: float = RECOGNITION_TRUST_WINDOW_SECONDS
    ) -> None:
        self.window_seconds = window_seconds
        # Timestamp of the last strong recognition, or None before any.
        self.last_strong_at: float | None = None

    def record(self, recognized: bool, confidence: float) -> None:
        # Only a genuinely confident result renews trust, not every result.
        if recognized and confidence >= CONFIDENCE_RECOGNIZED:
            self.last_strong_at = time.time()

    def is_recognized(self) -> bool:
        if self.last_strong_at is None:
            return False
        return (time.time() - self.last_strong_at) <= self.window_seconds

    def reset(self) -> None:
        self.last_strong_at = None


def chunk_rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


async def run():
    config = load_voice_config()

    provider = get_voice_provider(config)
    audio = AudioIO()
    wake_detector = WakeWordDetector()
    recognizer = SpeakerRecognizer()

    # Assume the sole enrolled voiceprint belongs to the owner; otherwise
    # fall back to the default owner name. Mirror of the wake-first design.
    profiles = VoiceprintStore().list_profiles()
    owner_name = profiles[0] if len(profiles) == 1 else "christopher"
    memory_context = build_context_summary(owner_name)
    memory_handlers = make_memory_handlers(owner_name)
    for name, handler in memory_handlers.items():
        provider.register_tool(name, handler)

    # Recognition state driving the policy engine. The actor stays treated
    # as the owner for RECOGNITION_TRUST_WINDOW_SECONDS after the last
    # strong recognition (confidence >= CONFIDENCE_RECOGNIZED), so several
    # short, weak utterances in a row don't each flip the gate — a time-based
    # trust window, not a count-based rolling window. Cleared on the way back
    # to STATE_IDLE so trust never carries across sessions.
    recognition_state = RecognitionTrust()

    def record_recognition(recognized: bool, confidence: float) -> None:
        recognition_state.record(recognized, confidence)

    def get_recognized() -> bool:
        return recognition_state.is_recognized()

    tool_registry = ToolRegistry()
    tool_registry.register(make_reenroll_voice_spec())

    # Spotify integration is optional at startup: without a stored refresh
    # token (or with VAULT_KEY/creds missing) we skip registering the tools
    # with a one-line notice. When present, the specs go through the exact
    # same registry + PolicyEngine as block 4 (WRITE tier confirms, READ runs).
    try:
        spotify_client = SpotifyClient()
    except Exception as exc:
        spotify_client = None
        print("Spotify not connected - run `python -m integrations.spotify.auth` to enable it.")
        print(f"(reason: {exc})")
    if spotify_client is not None:
        for spec in make_spotify_specs(spotify_client):
            tool_registry.register(spec)

    policy = PolicyEngine(tool_registry, get_recognized)
    for spec in tool_registry.all_specs():
        provider.register_tool(spec.name, policy.wrap_handler(spec))
    provider.register_tool("confirm_action", policy.confirm_action)

    mic_task = None
    speaker_task = None
    state_task = None
    session_task = None

    try:
        state = {"value": STATE_IDLE}
        last_activity = [0.0]
        wake_event = asyncio.Event()

        def mark_activity():
            last_activity[0] = time.monotonic()

        def go_idle(reason: str):
            audio.set_engaged(False)
            state["value"] = STATE_IDLE
            recognition_state.reset()
            print(reason)

        def on_model_audio(data: bytes):
            mark_activity()
            audio.enqueue_audio(data)

        async def on_forward_chunk(chunk: bytes):
            if chunk_rms(chunk) >= SPEECH_RMS_THRESHOLD:
                mark_activity()
            recognizer.feed(chunk)
            await provider.send_audio(chunk)

        def print_recognition(name, confidence, recognized, duration_s):
            record_recognition(recognized, confidence)
            dur = f", {duration_s:.1f}s" if duration_s is not None else ""
            if recognized:
                print(f"Recognized: {name} ({confidence:.2f}{dur})")
            else:
                print(f"Unrecognized speaker ({confidence:.2f}{dur})")

        recognizer.on_result(print_recognition)

        def on_wake():
            if state["value"] == STATE_IDLE:
                wake_event.set()

        def on_disconnected():
            go_idle(
                "Connection lost - going quiet; reconnecting in the background..."
            )

        def on_reconnected():
            print("Reconnected - say the wake word to continue.")

        async def state_machine():
            while True:
                if state["value"] == STATE_IDLE:
                    print("Idle - waiting for wake word...")
                    await wake_event.wait()
                    wake_event.clear()
                    state["value"] = STATE_ENGAGED
                    audio.set_engaged(True)
                    mark_activity()
                    audio.play_chime()
                    print("Engaged - Eden is listening...")
                else:
                    await asyncio.sleep(0.25)
                    if time.monotonic() - last_activity[0] > IDLE_TIMEOUT_SECONDS:
                        go_idle(
                            f"No speech or response for {IDLE_TIMEOUT_SECONDS}s "
                            "- going quiet..."
                        )

        provider.on_audio_response(on_model_audio)
        provider.on_interrupted(audio.clear_queue)
        provider.on_disconnected(on_disconnected)
        provider.on_reconnected(on_reconnected)
        await provider.start_session(
            build_system_instruction(config, memory_context, confirmation_tools=True),
            config,
            tool_declarations=[
                REMEMBER_DECLARATION,
                RECALL_DECLARATION,
                *policy.declarations(),
            ],
        )
        mic_task = await audio.start_mic(wake_detector, on_wake, on_forward_chunk)
        speaker_task = await audio.start_speaker()
        state_task = asyncio.create_task(state_machine())

        # Watch the long-running tasks; if any fails (session reconnect
        # exhausted, mic/speaker device lost) surface the exception so it
        # crashes the process and daemon.py restarts us.
        session_task = asyncio.create_task(provider.wait_for_session())
        done, _ = await asyncio.wait(
            [session_task, mic_task, speaker_task, state_task],
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            if task.cancelled():
                continue
            exc = task.exception()
            if exc is not None:
                raise exc
    except asyncio.CancelledError:
        pass
    finally:
        for task in (mic_task, speaker_task, state_task, session_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await provider.stop_session()
        wake_detector.close()
        audio.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    load_dotenv()

    from startup_checks import SEVERITY_CRITICAL, run_startup_checks

    checks = run_startup_checks()
    print("\nStartup checks:")
    for name, passed, detail, severity in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {detail} ({severity})")

    failed_critical = [
        c for c in checks if not c[1] and c[3] == SEVERITY_CRITICAL
    ]
    if failed_critical:
        print("\nStartup failed - resolve these critical issues and try again:")
        for name, passed, detail, severity in failed_critical:
            print(f"  - {name}: {detail}")
        sys.exit(1)

    warnings = [c for c in checks if not c[1]]
    if warnings:
        print("\nWarnings (continuing anyway):")
        for name, passed, detail, severity in warnings:
            print(f"  - {name}: {detail}")

    print()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("Exited cleanly.")
    except Exception as e:
        print(f"Fatal: {e}")
        # Non-zero so daemon.py's outer restart kicks in.
        sys.exit(1)


if __name__ == "__main__":
    main()