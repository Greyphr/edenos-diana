import asyncio
import logging
import time

import numpy as np
from dotenv import load_dotenv

from conversation.audio_io import AudioIO
from conversation.voice_config import build_system_instruction, load_voice_config
from conversation.voice_provider import get_voice_provider
from conversation.wake_word import WakeWordDetector
from identity.recognition import SpeakerRecognizer
from identity.voiceprint_store import VoiceprintStore
from memory.context import build_context_summary
from memory.tools import (
    RECALL_DECLARATION,
    REMEMBER_DECLARATION,
    make_memory_handlers,
)

IDLE_TIMEOUT_SECONDS = 8.0
SPEECH_RMS_THRESHOLD = 400

STATE_IDLE = "idle"
STATE_ENGAGED = "engaged"


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

    mic_task = None
    speaker_task = None
    state_task = None

    try:
        state = {"value": STATE_IDLE}
        last_activity = [0.0]
        wake_event = asyncio.Event()

        def mark_activity():
            last_activity[0] = time.monotonic()

        def on_model_audio(data: bytes):
            mark_activity()
            audio.enqueue_audio(data)

        async def on_forward_chunk(chunk: bytes):
            if chunk_rms(chunk) >= SPEECH_RMS_THRESHOLD:
                mark_activity()
            recognizer.feed(chunk)
            await provider.send_audio(chunk)

        def print_recognition(name, confidence, recognized, duration_s):
            dur = f", {duration_s:.1f}s" if duration_s is not None else ""
            if recognized:
                print(f"Recognized: {name} ({confidence:.2f}{dur})")
            else:
                print(f"Unrecognized speaker ({confidence:.2f}{dur})")

        recognizer.on_result(print_recognition)

        def on_wake():
            if state["value"] == STATE_IDLE:
                wake_event.set()

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
                        audio.set_engaged(False)
                        state["value"] = STATE_IDLE
                        print(
                            f"No speech or response for {IDLE_TIMEOUT_SECONDS}s "
                            "- going quiet..."
                        )

        provider.on_audio_response(on_model_audio)
        provider.on_interrupted(audio.clear_queue)
        await provider.start_session(
            build_system_instruction(config, memory_context),
            config,
            tool_declarations=[REMEMBER_DECLARATION, RECALL_DECLARATION],
        )
        mic_task = await audio.start_mic(wake_detector, on_wake, on_forward_chunk)
        speaker_task = await audio.start_speaker()
        state_task = asyncio.create_task(state_machine())

        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        for task in (mic_task, speaker_task, state_task):
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
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal: {e}")
    print("Exited cleanly.")


if __name__ == "__main__":
    main()