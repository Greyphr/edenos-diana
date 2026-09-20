import asyncio
import logging
import os
import re
import sys
import time

import numpy as np
from dotenv import load_dotenv

from conversation.audio_io import AudioIO
from conversation.voice_config import (
    build_first_run_instruction,
    build_system_instruction,
    load_voice_config,
)
from conversation.voice_provider import get_voice_provider
from conversation.wake_word import WakeWordDetector
from identity.embeddings import EmbeddingExtractor
from identity.enrollment_core import save_enrollment
from identity.recognition import (
    CONFIDENCE_RECOGNIZED,
    CONFIDENCE_UNCERTAIN,
    SpeakerRecognizer,
)
from identity.voiceprint_store import VoiceprintStore
from integrations.spotify.client import SpotifyClient
from integrations.spotify.tools import make_spotify_specs
from memory.context import build_context_summary
from memory.tools import make_memory_handlers
from tools.policy_engine import PolicyEngine
from tools.reasoning_delegate import (
    GeminiReasoningProvider,
    make_reasoning_delegate_spec,
)
from tools.reenroll_voice import make_reenroll_voice_spec
from tools.registry import ToolRegistry
from tools.web_search import make_web_search_spec

_LOG = logging.getLogger(__name__)

# Dedicated conversation transcript log: its own FileHandler writes ONLY the
# YOU:/EDEN: dialogue lines to logs/conversation.log, fully isolated from the
# general app logger (own handler, no propagation) so the file stays a clean
# readable transcript with nothing else mixed in. Plain-message format only.
_voicelog = logging.getLogger("eden.voicelog")


def _ensure_voicelog_handler() -> None:
    if _voicelog.handlers:
        return
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    handler = logging.FileHandler(
        os.path.join(logs_dir, "conversation.log"), encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    _voicelog.addHandler(handler)
    _voicelog.setLevel(logging.INFO)
    _voicelog.propagate = False


IDLE_TIMEOUT_SECONDS = 8.0
# How long a single strong recognition keeps the actor treated as the owner.
# Short, weak utterances don't each get scored as a fresh recognition; this
# window spans the gap between them within one continuous engaged session.
# A confident non-match (confidence < CONFIDENCE_UNCERTAIN) cuts through it
# immediately. Never carried across idle transitions (cleared in
# go_idle/reset).
RECOGNITION_TRUST_WINDOW_SECONDS = 25
# Confirmation-grade freshness, deliberately stricter: approving a pending
# WRITE/SENSITIVE action requires a recognition within the last few seconds,
# not just stale trust from earlier in the session.
CONFIRMATION_TRUST_WINDOW_SECONDS = 5
SPEECH_RMS_THRESHOLD = 400

STATE_IDLE = "idle"
STATE_ENGAGED = "engaged"


class RecognitionTrust:
    """Owner trust state for the policy engine: a time-based window keyed by
    the last genuinely confident recognition (confidence >=
    CONFIDENCE_RECOGNIZED) within the current engaged session.

    A strong hit keeps treating the actor as the owner for
    ``window_seconds`` regardless of uncertain results that follow it. A
    confident non-match (confidence < CONFIDENCE_UNCERTAIN) invalidates
    trust immediately, and the ambiguous 0.70-0.79 band in between is where
    short/weak-but-plausibly-owner utterances land - the window exists to
    tolerate exactly those, so they change nothing. ``reset()`` (called on
    the way back to idle) guarantees trust never carries across sessions.
    """

    def __init__(
        self, window_seconds: float = RECOGNITION_TRUST_WINDOW_SECONDS
    ) -> None:
        self.window_seconds = window_seconds
        # Timestamp of the last strong recognition, or None before any or
        # after a confident non-match.
        self.last_strong_at: float | None = None
        # Name of the most recent strong recognition (None before any or
        # after a confident non-match / reset). Drives role_for_actor's
        # name-based ownership comparison.
        self._last_name: str | None = None

    def record(self, name: str | None, recognized: bool, confidence: float) -> None:
        if recognized and confidence >= CONFIDENCE_RECOGNIZED:
            # Genuinely confident hit: renew the trust window and keep who
            # it matched.
            self.last_strong_at = time.time()
            self._last_name = name
        elif confidence < CONFIDENCE_UNCERTAIN:
            # Confident non-match: someone who clearly isn't the owner just
            # spoke, so invalidate trust immediately regardless of how
            # recent the last strong hit was.
            self.last_strong_at = None
            self._last_name = None

    def recognized_name(self) -> str | None:
        return self._last_name if self.is_recognized() else None

    def is_recognized(self) -> bool:
        return self._within(self.window_seconds)

    def is_freshly_recognized(self, window_seconds: float) -> bool:
        """Confirmation-grade freshness. Same single timestamp as
        :meth:`is_recognized`, but a much shorter window — meant to gate
        confirm_action, which should require evidence from right now, not
        stale trust earned earlier in the session."""
        return self._within(window_seconds)

    def _within(self, window_seconds: float) -> bool:
        if self.last_strong_at is None:
            return False
        return (time.time() - self.last_strong_at) <= window_seconds

    def reset(self) -> None:
        self.last_strong_at = None
        self._last_name = None


def chunk_rms(chunk: bytes) -> float:
    samples = np.frombuffer(chunk, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float32) ** 2)))


def _candidate_profile_names(raw: str) -> list[str]:
    """Profile-name variants that pass identity.names path-safety validation.

    A transcribed name is loose text ('Alex Johnson', 'Deirdre O'Brien') but a
    profile name is a single path component (alphanumerics, '_', '-'). Try the
    raw text and progressively-sanitized forms, ending with a guaranteed-safe
    fallback, so first-run enrollment can't crash over a label.
    """
    name = (raw or "").strip()
    variants = [
        name,
        name.lower(),
        re.sub(r"[^A-Za-z]", "", name),
        re.sub(r"[^A-Za-z0-9]", "", name).lower(),
        re.sub(r"[^A-Za-z0-9]", "-", name).strip("-").lower(),
        "owner",
    ]
    return list(dict.fromkeys(v for v in variants if v))


async def run():
    config = load_voice_config()

    provider = get_voice_provider(config)
    audio = AudioIO()
    wake_detector = WakeWordDetector()
    recognizer = SpeakerRecognizer()
    voice_store = VoiceprintStore()

    # Dynamic, named ownership: the owner is whoever the store's marker says
    # (first-ever enrollment claims it; re-runs never change it). This is a
    # mutable slot populated below rather than a hardcoded name anywhere, and
    # re-read through the named getter per call so a first-run enrollment
    # takes effect immediately. The store also auto-claims a single existing
    # profile when its owner marker is missing.
    owner = [voice_store.get_owner_name()]

    def get_owner_name() -> str | None:
        return owner[0]

    is_bootstrap: list[bool] = [not voice_store.list_profiles()]
    memory_context = build_context_summary(get_owner_name() or "")

    # Recognition state driving the policy engine. The actor stays treated
    # as the owner for RECOGNITION_TRUST_WINDOW_SECONDS after the last
    # strong recognition (confidence >= CONFIDENCE_RECOGNIZED), so several
    # short, weak (uncertain-band, 0.70-0.79) utterances in a row don't each
    # flip the gate - a time-based trust window, not a count-based rolling
    # window. A confident non-match (confidence < CONFIDENCE_UNCERTAIN)
    # invalidates it immediately. Cleared on the way back to STATE_IDLE so
    # trust never carries across sessions.
    recognition_state = RecognitionTrust()

    def record_recognition(
        name: str | None, recognized: bool, confidence: float
    ) -> None:
        recognition_state.record(name, recognized, confidence)

    def get_recognized_name() -> str | None:
        return recognition_state.recognized_name()

    # Confirmation needs fresher evidence than ordinary tool gating: a
    # recognition within the (much shorter) confirmation window.
    def get_freshly_recognized_name() -> bool:
        return recognition_state.is_freshly_recognized(CONFIRMATION_TRUST_WINDOW_SECONDS)

    tool_registry = ToolRegistry()
    tool_registry.register(make_reenroll_voice_spec())

    # Deep-thinking delegate: same registry + PolicyEngine path as every other
    # tool. TRIVIAL tier (owner-only, confirmation-free) - it just thinks and
    # returns text. The provider reuses GEMINI_API_KEY and the configured
    # reasoning_model (config/voice.yaml), so no new credential is needed.
    tool_registry.register(
        make_reasoning_delegate_spec(
            GeminiReasoningProvider(model=config.get("reasoning_model"))
        )
    )

    # Web search (Brave Search API): TRIVIAL tier, owner-only, runs
    # immediately with no confirmation prompt. Always registered - if
    # BRAVE_SEARCH_API_KEY is missing the handler returns a clear error dict
    # instead of raising, so Eden can report that search is unavailable.
    tool_registry.register(make_web_search_spec())

    # Memory tools go through the exact same registry + PolicyEngine as
    # everything else: TRIVIAL tier, owner-only, confirmation-free (same
    # reasoning as the Spotify playback tools). Scoped by whatever name owns
    # this Eden; with no owner yet (first-run bootstrap) there is no memory
    # scope, and the tools are still registered (they simply won't pass the
    # owner gate).
    memory_scope = get_owner_name() or ""
    for spec in make_memory_handlers(memory_scope):
        tool_registry.register(spec)

    # Spotify integration is optional at startup: without a stored refresh
    # token (or with VAULT_KEY/creds missing) we skip registering the tools
    # with a one-line notice. When present, the specs go through the exact
    # same registry + PolicyEngine as block 4 (WRITE tier confirms, READ runs).
    try:
        spotify_client = SpotifyClient()
    except Exception as exc:
        spotify_client = None
        if sys.stdin.isatty() and input(
            "\nSpotify isn't connected - authorize now? [y/N] "
        ).strip().lower() in ("y", "yes"):
            from integrations.spotify.auth import run_spotify_auth_flow

            if await run_spotify_auth_flow(
                os.getenv("SPOTIFY_CLIENT_ID", ""),
                os.getenv("SPOTIFY_CLIENT_SECRET", ""),
                os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            ):
                try:
                    spotify_client = SpotifyClient()
                except Exception as exc2:
                    spotify_client = None
                    print("Spotify not connected - run `python -m integrations.spotify.auth` to enable it.")
                    print(f"(reason: {exc2})")
        if spotify_client is None:
            print("Spotify not connected - run `python -m integrations.spotify.auth` to enable it.")
            print(f"(reason: {exc})")
    if spotify_client is not None:
        for spec in make_spotify_specs(spotify_client):
            tool_registry.register(spec)

    policy = PolicyEngine(
        tool_registry,
        get_recognized_name,
        get_owner_name,
        get_freshly_recognized_name,
        # Let the policy engine wait briefly for an in-flight recognition —
        # otherwise the very first command of a session is judged before the
        # utterance that carries it has been scored, and gets denied.
        wait_for_pending=lambda: recognizer.wait_for_pending(timeout=1.5),
    )
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
            # Drop any half-finished utterance so its buffer can't be stitched
            # onto the first audio of the next engaged session.
            recognizer.reset()
            # An unconfirmed proposal (e.g. a WRITE/SENSITIVE action the owner
            # never got to confirm) must not outlive the session: otherwise it
            # would reject the next session's identical request as "already
            # pending" before it's even presented for confirmation.
            policy.clear_pending()
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
            record_recognition(name, recognized, confidence)
            dur = f", {duration_s:.1f}s" if duration_s is not None else ""
            if recognized:
                print(f"Recognized: {name} ({confidence:.2f}{dur})")
            else:
                print(f"Unrecognized speaker ({confidence:.2f}{dur})")

        recognizer.on_result(print_recognition)

        def on_wake():
            if not is_bootstrap[0] and state["value"] == STATE_IDLE:
                wake_event.set()

        def on_disconnected():
            go_idle(
                "Connection lost - going quiet; reconnecting in the background..."
            )

        def on_reconnected(resumed: bool):
            if resumed:
                # Resumption carried the conversation over the ~10-minute
                # connection cap (or a mid-session drop): stay/get back to
                # engaged without the wake word. If a hard drop pushed us
                # idle, wake the state machine; on a graceful GoAway reconnect
                # we never left engaged, so just refresh the activity timer.
                if state["value"] == STATE_IDLE:
                    wake_event.set()
                else:
                    wake_event.clear()
                    mark_activity()
                print("Reconnected - conversation resumed.")
            else:
                print("Reconnected - say the wake word to continue.")

        # First-run bootstrap: with no profiles enrolled there is no one to
        # wake-word nor to recognize, so the session opens unprompted and the
        # recognizer collects voice samples instead of matching profiles.
        # Mutable slot for the name captured via the input-transcription path.
        enrolled_name: list[str | None] = [None]
        enroll_count = [0]
        enroll_target = [5]

        def on_enrollment_sample(check_count: int, target: int) -> None:
            enroll_count[0] = check_count
            if is_bootstrap[0]:
                if enrolled_name[0]:
                    print(f"  Voice sample {check_count}/{target} captured.")
                else:
                    print(f"  Voice sample {check_count}/{target} captured (awaiting name).")

        async def _on_enroll_complete(embeddings) -> None:
            raw_name = (enrolled_name[0] or "owner").strip()
            claimed = None
            last_error = None
            for candidate in _candidate_profile_names(raw_name):
                try:
                    claimed = save_enrollment(
                        candidate, embeddings, store=voice_store
                    )
                    break
                except ValueError as exc:
                    last_error = exc
            if claimed is None:
                # Degenerate: no name variant passes filename safety (and the
                # fallback was unavailable). Keep the session alive rather
                # than crashing the whole boot over a label.
                print(f"  Could not save voiceprint: {last_error}")
                is_bootstrap[0] = False
                owner[0] = None
                print()
                print("  First-run setup did not complete - please check the")
                print("  console and retry enrollment.")
                audio.play_chime()
                return
            is_bootstrap[0] = False
            owner[0] = claimed["name"]
            print()
            print("=" * 60)
            print("FIRST-RUN SETUP COMPLETE")
            print("=" * 60)
            print(f"Voiceprint for '{claimed['name']}' saved to:")
            print(f"  {claimed['profiles_dir']}")
            if claimed["owner_claimed"]:
                print(f"  '{claimed['name']}' is now the owner of this Eden.")
            print("From now on the wake word is required to start a session.")
            print("=" * 60)
            print()
            audio.play_chime()

        recognizer.collect_enrollment(
            enroll_target[0], on_enrollment_sample, _on_enroll_complete
        )

        async def state_machine():
            while True:
                if state["value"] == STATE_IDLE:
                    if is_bootstrap[0]:
                        # First-run bootstrap: no wake word - nobody is
                        # recognized yet, so Eden talks unprompted until the
                        # enrollment completes, at which point the completion
                        # handler drops it normally into idle.
                        state["value"] = STATE_ENGAGED
                        audio.set_engaged(True)
                        mark_activity()
                        print("Bootstrap - Eden is speaking unprompted...")
                        await asyncio.sleep(0.5)
                        continue
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

        # Finalized input speech feeds the *same* pending-confirmation queue the
        # explicit confirm_action tool call does, via the ABC's parallel
        # transcript path. Registration mirrors the other callback registers so
        # the two confirmation channels (spoken vs. tool call) race on equal
        # footing; whichever resolves the pending action first wins.
        #
        # NOTE: the handlers are defined *first* (below) and registered only
        # after both ``def``s have run, so the closure name is bound by the
        # time the provider stores the callback. Registering an as-yet-def'd
        # name would raise NameError on the very line that wires the path.
        def on_owner_input(transcript: str):
            # First-run bootstrap: a finalized transcript in bootstrap mode
            # carries the new owner's *name* (the input-transcription path),
            # not a confirmation — nothing can be pending before an owner
            # exists. Capture it exactly once, then fall through only for
            # confirmation resolution after setup completes.
            if is_bootstrap[0]:
                if enrolled_name[0] is None and transcript.strip():
                    enrolled_name[0] = transcript.strip()
                    print(f"  First-run name heard: {enrolled_name[0]}")
                return
            # In normal (non-bootstrap) mode the transcript is always the
            # owner's words, so the conversation log records it regardless of
            # whether check_transcript finds anything pending below.
            if transcript:
                _voicelog.info("YOU: %s", transcript)
            # Second, parallel confirmation path: a *finalized* transcript of
            # what the owner actually said either carries a pending action's
            # confirmation phrase or settles it. Deliberately dispatched off
            # the receive hot path so a slow policy decision can never stall
            # audio/interruption handling. ``check_transcript`` resolves the
            # same pending queue ``confirm_action`` does, so whichever channel
            # confirms first wins and the other correctly reports nothing
            # pending.
            asyncio.create_task(resolve_from_transcript(transcript))

        async def resolve_from_transcript(transcript: str):
            result = await policy.check_transcript(transcript)
            if result is None:
                # Nothing was pending (or no finalized path needed us).
                return
            # Surface the outcome out loud via the session's own channel, so
            # the owner hears the same ack as a tool-confirmation reply instead
            # of wondering why the model fell silent.
            status = result.get("status") or (
                "done" if "error" not in result else "failed"
            )
            if result.get("tool_name"):
                await provider.send_status_note(
                    f"{result['tool_name']} - {status}"
                )
        provider.on_input_transcript(on_owner_input)

        # Echo back the finalized *output* transcript the same way, so the
        # owner's own voice loop sees what Eden actually said (driven by the
        # provider's server-side output transcription config). Mirrors the
        # input registration pattern exactly: only settled, finished
        # transcripts are surfaced, dispatched off the receive hot path.
        def on_owner_output(transcript: str):
            if transcript:
                _voicelog.info("EDEN: %s", transcript)
            _LOG.info("Eden said: %s", transcript)
        provider.on_output_transcript(on_owner_output)

        await provider.start_session(
            build_first_run_instruction(config)
            if is_bootstrap[0]
            else build_system_instruction(
                config,
                memory_context,
                confirmation_tools=True,
                reasoning_tools=True,
                web_search_tools=True,
                owner_name=get_owner_name(),
            ),
            config,
            tool_declarations=list(policy.declarations()),
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
        if spotify_client is not None:
            await spotify_client.aclose()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _ensure_voicelog_handler()
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
        # Exit 2 = configuration problem (as opposed to a runtime crash,
        # which uses 1): daemon.py recognizes this code and refuses to
        # restart, so a missing GEMINI_API_KEY stops letting the child
        # respawn every 30 seconds forever.
        sys.exit(2)

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