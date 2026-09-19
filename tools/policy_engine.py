import logging
import time
from collections.abc import Awaitable, Callable

from tools.registry import RiskTier, ToolRegistry, ToolSpec
from tools.roles import has_permission, role_for_actor

logger = logging.getLogger(__name__)

PENDING_TIMEOUT_SECONDS = 45

# Fixed set of affirmative responses accepted for WRITE-tier confirmation.
_AFFIRMATIVES = frozenset({"yes", "yeah", "confirm", "do it", "go ahead"})

# Lightweight basic-type checks for declared parameter types. This is not a
# full JSON Schema validator - just enough to catch a model that omits a
# required argument or passes a plainly wrong kind of value, so those mistakes
# fail loudly instead of surfacing deep inside a handler.
_BASIC_TYPE_CHECKS: dict[str, Callable[[object], bool]] = {
    "string": lambda value: isinstance(value, str),
    "boolean": lambda value: isinstance(value, bool),
    "integer": lambda value: isinstance(value, int) and not isinstance(value, bool),
    "number": lambda value: isinstance(value, (int, float)) and not isinstance(value, bool),
}


def _check_args(spec: ToolSpec, args: dict) -> None:
    """Reject clearly-invalid tool arguments before a handler runs.

    Raises ``ValueError`` when a required key is missing or a present value
    fails its declared basic type check.
    """
    parameters = spec.parameters or {}
    properties = parameters.get("properties") or {}
    for key in parameters.get("required") or []:
        if key not in args:
            raise ValueError(
                f"Tool {spec.name!r} is missing required argument {key!r}"
            )
    for key, value in args.items():
        type_info = properties.get(key) or {}
        declared_type = type_info.get("type")
        check = (
            _BASIC_TYPE_CHECKS.get(declared_type)
            if isinstance(declared_type, str)
            else None
        )
        if check is not None and not check(value):
            raise ValueError(
                f"Tool {spec.name!r} argument {key!r} must be a "
                f"{declared_type}, got {type(value).__name__}"
            )

CONFIRM_ACTION_DECLARATION = {
    "name": "confirm_action",
    "description": (
        "Confirm or cancel a previously requested action that needs the "
        "owner's explicit confirmation. Call this with the owner's actual "
        "response to the confirmation request."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "response": {
                "type": "string",
                "description": "The owner's spoken response to the confirmation request.",
            }
        },
        "required": ["response"],
    },
}


class PolicyEngine:
    """Gate every tool call through actor role + risk tier, and funnel
    WRITE/SENSITIVE actions through a single pending-confirmation flow.

    ``get_recognized`` is a ``Callable[[], bool]`` supplied by main.py,
    backed by the most recent speaker recognition result. ``get_freshly_recognized``
    is the same signal with a much shorter window: confirming a pending action
    requires evidence from right now, not stale trust from earlier in the session.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        get_recognized: Callable[[], bool],
        get_freshly_recognized: Callable[[], bool],
        wait_for_pending: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._registry = registry
        self._get_recognized = get_recognized
        self._get_freshly_recognized = get_freshly_recognized
        self._wait_for_pending = wait_for_pending
        self._pending: dict | None = None

    async def _dispatch(self, spec: ToolSpec, args: dict) -> dict:
        """Run the handler directly (READ/TRIVIAL) or stage a confirmation
        (WRITE/SENSITIVE), after the permission check has already passed."""
        _check_args(spec, args)
        logger.info("Tool %r called", spec.name)
        if spec.risk_tier in (RiskTier.READ, RiskTier.TRIVIAL):
            # Both tiers run the handler directly — no confirmation pending
            # state. The permission check above still gates them: unknown is
            # denied for TRIVIAL (owner-only, confirmation-free), as well as
            # WRITE/SENSITIVE.
            return await spec.handler(**args)
        return self._store_pending(spec, args)

    def wrap_handler(self, spec: ToolSpec):
        """Return the function registered with the voice provider for spec.name."""

        async def handler(**args) -> dict:
            role = role_for_actor(self._get_recognized())
            if not has_permission(role, spec.risk_tier):
                if (
                    spec.risk_tier != RiskTier.READ
                    and self._wait_for_pending is not None
                ):
                    # A recognition for this very utterance may still be
                    # scoring in the background - most likely with the first
                    # thing said in a session. Give it exactly one bounded
                    # wait, then re-check before finalizing the denial. One
                    # wait, not a retry loop: bounded latency, not
                    # unbounded patience.
                    await self._wait_for_pending()
                    role = role_for_actor(self._get_recognized())
                    if has_permission(role, spec.risk_tier):
                        return await self._dispatch(spec, args)
                logger.info(
                    "Tool %r denied for role %r (tier=%s)",
                    spec.name, role, spec.risk_tier.value,
                )
                return {"error": "not permitted"}
            return await self._dispatch(spec, args)

        return handler

    def _store_pending(self, spec: ToolSpec, args: dict) -> dict:
        if self._pending is not None:
            # Never silently discard a proposed action for a new one: that
            # would let a second request wipe out the first before the owner
            # ever gets a chance to confirm it. The slot must be cleared
            # through confirm_action (a deny/cancel response) first.
            logger.info(
                "Refusing to replace pending action %r with %r request",
                self._pending["tool_name"], spec.name,
            )
            return {
                "error": "another action is already pending confirmation",
                "pending_tool": self._pending["tool_name"],
            }
        if spec.risk_tier == RiskTier.SENSITIVE:
            phrase = spec.confirmation_phrase or "confirm"
            self._pending = {
                "tool_name": spec.name,
                "args": args,
                "risk_tier": spec.risk_tier,
                "required_confirmation": phrase,
                "created_at": time.monotonic(),
            }
            return {
                "status": "confirmation_required",
                "ask": f"Ask the owner to say exactly: '{phrase}' to confirm.",
            }
        self._pending = {
            "tool_name": spec.name,
            "args": args,
            "risk_tier": spec.risk_tier,
            "required_confirmation": "yes",
            "created_at": time.monotonic(),
        }
        return {
            "status": "confirmation_required",
            "ask": "Ask the owner to confirm before doing this.",
        }

    def clear_pending(self) -> None:
        """Drop any action still waiting for confirmation.

        Called on the way back to idle, so an unconfirmed proposal can't
        outlive its session and reject the next session's identical request
        as "already pending". Logs only when something was actually cleared,
        keeping the common already-empty case quiet.
        """
        if self._pending is not None:
            logger.info(
                "Cleared pending action %r (idle timeout fired before confirmation)",
                self._pending["tool_name"],
            )
            self._pending = None

    async def _try_resolve_pending(self, response_text: str) -> dict | None:
        """Shared confirmation-resolution core for the policy engine.

        ``response_text`` is the owner's actual response to a pending
        confirmation — either the explicit phrase passed into
        ``confirm_action`` or a finalized input transcript that carried the
        confirmation words (the parallel transcript path). Returns ``None``
        when there is nothing pending, and a resolution dict otherwise (a
        denial/expiry/cancel marker or the awaited tool-handler result).

        The permissions being re-checked are identical whichever path got us
        here: whoever confirms — spoken or via tool call — must still be an
        authorized actor *right now* and back it with a genuinely fresh
        recognitionate confirmation window.
        """
        if self._pending is None:
            return None
        response_text = (response_text or "").strip().lower()
        pending = self._pending
        if time.monotonic() - pending["created_at"] > PENDING_TIMEOUT_SECONDS:
            self._pending = None
            logger.info(
                "Pending action %r expired without confirmation", pending["tool_name"]
            )
            return {
                "status": "cancelled",
                "tool_name": pending["tool_name"],
            }

        # Re-check the actor's permission now, not just when the action was
        # first proposed: the person confirming must still be authorized.
        # Someone who merely overheard the phrase shouldn't complete it.
        # Confirmation additionally demands a recognition that is genuinely
        # FRESH (get_freshly_recognized) - stale trust from earlier in the
        # session isn't good enough to approve a write. Either way, give an
        # in-flight recognition for the confirmation utterance one bounded
        # wait before finalizing a denial (one wait, not a retry loop).
        role = role_for_actor(self._get_recognized())
        fresh = self._get_freshly_recognized()
        if not has_permission(role, pending["risk_tier"]) or not fresh:
            if self._wait_for_pending is not None:
                await self._wait_for_pending()
            role = role_for_actor(self._get_recognized())
            fresh = self._get_freshly_recognized()

        if not has_permission(role, pending["risk_tier"]):
            self._pending = None
            logger.info(
                "Pending action %r denied for role %r (tier=%s) at confirmation",
                pending["tool_name"], role, pending["risk_tier"].value,
            )
            return {
                "error": "not permitted",
                "tool_name": pending["tool_name"],
            }

        if not fresh:
            # Keep the pending action in place so a fresh confirmation
            # attempt can still complete it once a new recognition lands.
            logger.info(
                "Pending action %r denied: confirmation requires a fresh recognition",
                pending["tool_name"],
            )
            return {
                "error": "not permitted",
                "tool_name": pending["tool_name"],
                "reason": "confirmation requires a fresh recognition",
            }

        if pending["risk_tier"] == RiskTier.SENSITIVE:
            affirmed = pending["required_confirmation"].lower() in response_text
        else:
            affirmed = response_text in _AFFIRMATIVES

        self._pending = None
        if not affirmed:
            logger.info(
                "Pending action %r denied by owner response %r",
                pending["tool_name"], response_text,
            )
            return {
                "status": "cancelled",
                "tool_name": pending["tool_name"],
            }

        spec = self._registry.get(pending["tool_name"])
        if spec is None:
            logger.error("Pending action %r has no registered spec", pending["tool_name"])
            return {"error": "no such tool", "tool_name": pending["tool_name"]}
        logger.info("Confirmed executing %r", spec.name)
        _check_args(spec, pending["args"])
        result = await spec.handler(**pending["args"])
        return {**result, "tool_name": spec.name}

    async def confirm_action(self, response: str = "", **extra) -> dict:
        """Explicit tool-call path to confirming a pending action.

        Thin wrapper over the shared resolution core: the tool-call surface
        keeps its ``{"error": "nothing pending"}`` channel while delegating
        the actual permission/freshness/affirmation logic to
        ``_try_resolve_pending`` so the transcript path stays identical.
        """
        result = await self._try_resolve_pending(response)
        if result is None:
            return {"error": "nothing pending"}
        return result

    async def check_transcript(self, transcript: str) -> dict | None:
        """Resolve a pending action from a finalized input transcript.

        The second, parallel path to confirming: instead of the model issuing
        an explicit ``confirm_action`` tool call, the owner's spoken
        confirmation is transcribed by the voice session and checked here
        against whatever is pending. Returns ``None`` when there was nothing
        to resolve (or the transcript side had no pending action); otherwise
        the resolution dict from ``_try_resolve_pending`` — the caller may
        then acknowledge the outcome out loud.
        """
        return await self._try_resolve_pending(transcript)

    def declarations(self) -> list[dict]:
        declarations = self._registry.all_declarations()
        declarations.append(dict(CONFIRM_ACTION_DECLARATION))
        return declarations