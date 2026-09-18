import logging
import time
from collections.abc import Callable

from tools.registry import RiskTier, ToolRegistry, ToolSpec
from tools.roles import has_permission, role_for_actor

logger = logging.getLogger(__name__)

PENDING_TIMEOUT_SECONDS = 45

# Fixed set of affirmative responses accepted for WRITE-tier confirmation.
_AFFIRMATIVES = frozenset({"yes", "yeah", "confirm", "do it", "go ahead"})

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
    backed by the most recent speaker recognition result.
    """

    def __init__(
        self, registry: ToolRegistry, get_recognized: Callable[[], bool]
    ) -> None:
        self._registry = registry
        self._get_recognized = get_recognized
        self._pending: dict | None = None

    def wrap_handler(self, spec: ToolSpec):
        """Return the function registered with the voice provider for spec.name."""

        async def handler(**args) -> dict:
            role = role_for_actor(self._get_recognized())
            if not has_permission(role, spec.risk_tier):
                logger.info(
                    "Tool %r denied for role %r (tier=%s)",
                    spec.name, role, spec.risk_tier.value,
                )
                return {"error": "not permitted"}

            if spec.risk_tier == RiskTier.READ:
                return await spec.handler(**args)

            return self._store_pending(spec, args)

        return handler

    def _store_pending(self, spec: ToolSpec, args: dict) -> dict:
        if self._pending is not None:
            logger.info(
                "Replacing pending action %r with new %r request",
                self._pending["tool_name"], spec.name,
            )
        if spec.risk_tier == RiskTier.SENSITIVE:
            phrase = spec.confirmation_phrase or "confirm"
            self._pending = {
                "tool_name": spec.name,
                "args": args,
                "risk_tier": spec.risk_tier,
                "required_confirmation": phrase,
                "created_at": time.time(),
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
            "created_at": time.time(),
        }
        return {
            "status": "confirmation_required",
            "ask": "Ask the owner to confirm before doing this.",
        }

    async def confirm_action(self, response: str = "", **extra) -> dict:
        response_text = (response or "").strip().lower()
        pending = self._pending
        if pending is None:
            return {"error": "nothing pending"}
        if time.time() - pending["created_at"] > PENDING_TIMEOUT_SECONDS:
            self._pending = None
            logger.info(
                "Pending action %r expired without confirmation", pending["tool_name"]
            )
            return {"status": "cancelled"}

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
            return {"status": "cancelled"}

        spec = self._registry.get(pending["tool_name"])
        if spec is None:
            logger.error("Pending action %r has no registered spec", pending["tool_name"])
            return {"error": "no such tool"}
        logger.info("Confirmed executing %r", spec.name)
        return await spec.handler(**pending["args"])

    def declarations(self) -> list[dict]:
        declarations = self._registry.all_declarations()
        declarations.append(dict(CONFIRM_ACTION_DECLARATION))
        return declarations