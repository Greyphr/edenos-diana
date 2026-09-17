import asyncio
import os

from dotenv import load_dotenv

from audio_io import AudioIO
from realtime_session import RealtimeSession


async def run():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
        )

    session = RealtimeSession(api_key=api_key)
    audio = AudioIO()
    mic_task = None
    speaker_task = None

    try:
        async def on_mic_chunk(chunk: bytes):
            await session.send_audio(chunk)

        def on_model_audio(data: bytes):
            audio.enqueue_audio(data)

        await session.connect(on_audio=on_model_audio)
        mic_task = await audio.start_mic(on_mic_chunk)
        speaker_task = await audio.start_speaker()

        print("Eden is listening. Ctrl+C to exit.")
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        pass
    finally:
        for task in (mic_task, speaker_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await session.close()
        audio.close()


def main():
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