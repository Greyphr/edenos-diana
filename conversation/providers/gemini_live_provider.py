# Model: gemini-3.1-flash-live-preview
# (current low-latency Live/native-audio model as of Sept 2026, per Google docs)
import asyncio
import logging
import os
from collections.abc import Callable

from google import genai
from google.genai import types

from conversation.voice_provider import VoiceProvider

logger = logging.getLogger(__name__)

MODEL = "gemini-3.1-flash-live-preview"

# Initial-connection smoothing: a transient blip at startup shouldn't fail the
# whole boot, so retry a few times with a short fixed delay before giving up.
INITIAL_CONNECT_ATTEMPTS = 3
INITIAL_CONNECT_DELAY_SECONDS = 2.0

# Mid-session reconnect backoff, same shape as daemon.py's supervisor.
RECONNECT_START_BACKOFF = 2.0
RECONNECT_MAX_BACKOFF = 30.0
# Total elapsed retry window; when exceeded, give up and raise so the daemon's
# outer restart takes over.
RECONNECT_MAX_TOTAL_SECONDS = 120.0
# Per-attempt cap on how long a single connect may take. A hung TCP connect
# must not sit there unbounded (the reconnect budget above only counts the
# sleep() calls between attempts, not time spent inside a hanging connect).
CONNECT_TIMEOUT_SECONDS = 20.0


class GeminiLiveProvider(VoiceProvider):
    def __init__(self, client=None):
        """``client`` is injectable for tests; production builds from the env."""
        if client is not None:
            self._client = client
        else:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in."
                )
            self._client = genai.Client(
                api_key=api_key, http_options={"api_version": "v1alpha"}
            )
        self._session = None
        self._session_ctx = None
        self._audio_callback: Callable[[bytes], None] | None = None
        self._interrupted_callback: Callable[[], None] | None = None
        self._disconnected_callback: Callable[[], None] | None = None
        self._reconnected_callback: Callable[[], None] | None = None
        self._tool_handlers: dict[str, Callable] = {}
        self._receive_task: asyncio.Task | None = None
        # In-flight tool-call tasks, dispatched off the receive loop so one
        # slow tool can't stall audio/interruption handling. Kept so they can
        # be cancelled on barge-in (a tool shouldn't finish after being
        # talked over) and so done tasks are dropped from tracking.
        self._active_tool_tasks: set[asyncio.Task] = set()
        # Resumption handle from the server's session_resumption_update
        # messages. Passed into the next connect so the model re-attaches to
        # the previous session instead of starting fresh (Live connections
        # end after ~10 minutes, which shouldn't cost the whole conversation).
        self._resumption_handle: str | None = None
        # Whether the connected API mode supports server-side session
        # resumption. Gemini Enterprise mode does; the Developer API rejects
        # the transparent parameter outright, so _open_session flips this off
        # for the whole process on first connect and falls back to fresh
        # sessions (which matches pre-resumption behavior exactly).
        self._resumption_transparent: bool = True
        # Set by the GoAway handler to tell the receive loop to stop iterating
        # the dying session iterator after the proactive reconnect finishes.
        self._reconnect_now = False
        # start_session params cached so a reconnect can replay the same setup
        # without the caller needing to call start_session again.
        self._session_args: dict | None = None

    def register_tool(self, name: str, handler) -> None:
        self._tool_handlers[name] = handler

    async def start_session(
        self,
        system_instruction: str,
        voice_config: dict,
        tool_declarations: list[dict] | None = None,
    ) -> None:
        self._session_args = {
            "system_instruction": system_instruction,
            "voice_config": voice_config,
            "tool_declarations": tool_declarations,
        }
        last_error: Exception | None = None
        for attempt in range(1, INITIAL_CONNECT_ATTEMPTS + 1):
            try:
                if self._session is not None:
                    await self._close_session()
                await asyncio.wait_for(
                    self._open_session(), timeout=CONNECT_TIMEOUT_SECONDS
                )
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = e
                if attempt < INITIAL_CONNECT_ATTEMPTS:
                    logger.warning(
                        "Initial Live connection failed (attempt %d/%d): %r",
                        attempt,
                        INITIAL_CONNECT_ATTEMPTS,
                        e,
                    )
                    if self._resumption_transparent:
                        # A config-level rejection (e.g. resumption-mode
                        # unsupported on the Developer API) is instant and
                        # already fixed by _open_session's retry flag - don't
                        # add a backoff stall before the retry.
                        await asyncio.sleep(INITIAL_CONNECT_DELAY_SECONDS)
        else:
            raise ConnectionError(
                f"Could not connect to the Live session after "
                f"{INITIAL_CONNECT_ATTEMPTS} attempts: {last_error}"
            ) from last_error

        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _open_session(self) -> None:
        if self._session_args is None:
            raise RuntimeError("start_session must be called before opening a session")
        args = self._session_args
        tools = None
        if args["tool_declarations"]:
            tools = [{"function_declarations": list(args["tool_declarations"])}]
        config_kwargs = {
            "response_modalities": ["AUDIO"],
            "system_instruction": args["system_instruction"],
            "tools": tools,
            "speech_config": types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=args["voice_config"]["voice_name"]
                    )
                ),
                language_code=args["voice_config"]["language_code"],
            ),
        }
        if self._resumption_transparent:
            # Resume the previous session when we hold a handle (None handle
            # starts a fresh session). transparent=True makes the server send
            # last_consumed_client_message_index so reconnections can resume
            # mid-turn seamlessly. Only supported on Gemini Enterprise Agent
            # Platform mode; the Developer API rejects it and _open_session
            # detects that on first connect, disabling resumption from then on.
            config_kwargs["session_resumption"] = types.SessionResumptionConfig(
                handle=self._resumption_handle, transparent=True
            )
        config = types.LiveConnectConfig(**config_kwargs)
        try:
            self._session_ctx = self._client.aio.live.connect(model=MODEL, config=config)
            self._session = await self._session_ctx.__aenter__()
        except ValueError as exc:
            message = str(exc)
            if self._resumption_transparent and (
                "transparent" in message or "Enterprise Agent Platform" in message
            ):
                logger.warning(
                    "Live session resumption unsupported on this API mode "
                    "(%s); continuing with fresh sessions", message,
                )
                self._resumption_transparent = False
                self._resumption_handle = None
                ctx = self._session_ctx
                self._session_ctx = None
                if ctx is not None:
                    try:
                        await ctx.__aexit__(None, None, None)
                    except Exception:
                        pass
            raise

    async def _close_session(self) -> None:
        ctx = self._session_ctx
        self._session_ctx = None
        self._session = None
        if ctx is not None:
            await ctx.__aexit__(None, None, None)

    async def _receive_loop(self):
        while True:
            try:
                self._reconnect_now = False
                async for response in self._session.receive():
                    await self._handle_response(response)
                    if self._reconnect_now:
                        # GoAway handling already swapped in a fresh session;
                        # stop iterating the dying one's stream.
                        break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await self._handle_disconnect(e)

    async def _handle_response(self, response) -> None:
        if response.session_resumption_update:
            update = response.session_resumption_update
            if update and update.resumable:
                self._resumption_handle = (
                    update.new_handle or self._resumption_handle
                )
            else:
                # Not resumable at this point: any earlier handle is dead, so
                # a later reconnect must not try to resume from it.
                self._resumption_handle = None
            return

        if response.go_away:
            # The server is ending this connection (session duration cap).
            # Proactively reconnect while the resumption handle is still
            # fresh, instead of waiting for the hard drop and losing context.
            self._reconnect_now = True
            await self._handle_disconnect(None, notify_disconnect=False)
            return

        if response.server_content and response.server_content.interrupted:
            # Barge-in: anything we were waiting on a tool for is being talked
            # over. Cancel in-flight tool tasks before telling the caller, so
            # the interruption registers immediately instead of waiting for a
            # slow tool to finish.
            self._cancel_active_tool_tasks()
            if self._interrupted_callback:
                self._interrupted_callback()
            return

        if response.server_content and response.server_content.model_turn:
            for part in response.server_content.model_turn.parts:
                if part.inline_data and isinstance(part.inline_data.data, bytes):
                    if self._audio_callback:
                        self._audio_callback(part.inline_data.data)
            return

        if response.tool_call:
            # Run the tool off the receive hot path: awaiting the call inline
            # would block the loop from reading further events (audio,
            # interruptions) until the tool finishes.
            task = asyncio.create_task(self._handle_tool_call(response.tool_call))
            self._active_tool_tasks.add(task)
            task.add_done_callback(self._active_tool_tasks.discard)

    def _cancel_active_tool_tasks(self) -> None:
        for task in list(self._active_tool_tasks):
            task.cancel()

    async def _handle_disconnect(
        self, error: Exception | None, notify_disconnect: bool = True
    ) -> None:
        """Reconnect after a connection-level failure (or a proactive GoAway).

        When ``notify_disconnect`` (a real drop), fire the disconnect callback
        first so the caller can fall back to idle while we retry. Either way,
        reconnect tries to resume the previous session via the resumption
        handle; the reconnected callback is told whether the new connection
        actually carried the handle, so the caller can stay engaged when
        context survived or go back to the wake word when it didn't. Raises
        ``error`` (or a GoAway-specific one) once the retry window is
        exhausted.
        """
        if notify_disconnect and self._disconnected_callback:
            self._disconnected_callback()
        if error is not None:
            logger.error("Realtime session disconnected: %r", error, exc_info=True)

        delay = RECONNECT_START_BACKOFF
        elapsed = 0.0
        attempt = 0
        resumed = False
        while True:
            if elapsed >= RECONNECT_MAX_TOTAL_SECONDS:
                if error is not None:
                    logger.error(
                        "Could not reconnect within %.0fs; raising so the daemon "
                        "restarts Eden",
                        elapsed,
                    )
                    raise error
                logger.error(
                    "GoAway reconnect (with resumption) failed within %.0fs; "
                    "raising so the daemon restarts Eden",
                    elapsed,
                )
                raise ConnectionError(
                    "Could not re-establish the Live session after a GoAway"
                )
            attempt += 1
            logger.warning(
                "Reconnecting to the Live session (attempt %d) in %.0fs...",
                attempt,
                delay,
            )
            await asyncio.sleep(delay)
            elapsed += delay
            delay = min(delay * 2, RECONNECT_MAX_BACKOFF)
            try:
                await self._close_session()
                await asyncio.wait_for(
                    self._open_session(), timeout=CONNECT_TIMEOUT_SECONDS
                )
                # We sent the handle if we had one; anything sent and accepted
                # by the server resumes the conversation. A handle that the
                # server can't honor silently starts fresh, and the next
                # session_resumption_update will hand us a new one.
                resumed = self._resumption_handle is not None
                break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("Reconnect attempt %d failed: %r", attempt, e)

        logger.info("Live session reconnected%s", " (resumed)" if resumed else "")
        if self._reconnected_callback:
            try:
                self._reconnected_callback(resumed)
            except Exception:
                logger.exception("on_reconnected callback raised")

    async def _handle_tool_call(self, tool_call) -> None:
        if not tool_call.function_calls:
            return
        # Run handlers off the receive hot path; each is created as a task so
        # one slow tool can't stall audio/interruption handling.
        tasks = [asyncio.create_task(self._run_tool(fc)) for fc in tool_call.function_calls]
        responses = await asyncio.gather(*tasks)
        if self._session is not None:
            await self._session.send_tool_response(function_responses=responses)

    async def _run_tool(self, fc) -> types.FunctionResponse:
        handler = self._tool_handlers.get(fc.name)
        if handler is None:
            return types.FunctionResponse(
                id=fc.id,
                name=fc.name,
                response={"error": f"No handler registered for {fc.name!r}"},
            )
        try:
            result = await handler(**(fc.args or {}))
            if not isinstance(result, dict):
                result = {"result": result}
        except asyncio.CancelledError:
            raise
        except Exception:
            # Never ship internal exception text/stack traces off to the
            # cloud model - it's logged in full server-side, but the model
            # only gets a generic failure marker.
            logger.exception("Tool handler %r failed", fc.name)
            result = {"error": "tool execution failed"}
        return types.FunctionResponse(id=fc.id, name=fc.name, response=result)

    async def send_audio(self, chunk: bytes) -> None:
        if self._session is None:
            return
        try:
            await self._session.send_realtime_input(
                audio=types.Blob(
                    data=chunk,
                    mime_type="audio/pcm;rate=16000",
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Transient: the session may just have dropped (the receive loop is
            # reconnecting). Don't take the mic task down with it.
            logger.warning("Dropping audio chunk, session unavailable: %r", e)

    def on_audio_response(self, callback: Callable[[bytes], None]) -> None:
        self._audio_callback = callback

    def on_interrupted(self, callback: Callable[[], None]) -> None:
        self._interrupted_callback = callback

    def on_disconnected(self, callback: Callable[[], None]) -> None:
        self._disconnected_callback = callback

    def on_reconnected(self, callback: Callable[[], None]) -> None:
        self._reconnected_callback = callback

    async def wait_for_session(self) -> None:
        if self._receive_task is None:
            return
        await self._receive_task

    async def stop_session(self) -> None:
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        await self._close_session()