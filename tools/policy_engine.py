import logging
import re
import time
from collections.abc import Awaitable, Callable

from tools.registry import RiskTier, ToolRegistry, ToolSpec
from tools.roles import has_permission, role_for_actor

logger = logging.getLogger(__name__)

PENDING_TIMEOUT_SECONDS = 45

# Fixed set of affirmative responses accepted for WRITE-tier confirmation.
# Matched on word boundaries within the (normalized) response, so "yes, please"
# and "yes." both count while "yesterday" doesn't.
_AFFIRMATIVES = frozenset({"yes", "yeah", "confirm", "do it", "go ahead"})

# Explicit-negative word set for WRITE-tier (and SENSITIVE) confirmation,
# checked BEFORE the affirmative set: only a recognized no can cancel a pending
# action. A response that is neither a recognized no nor a recognized yes
# stays pending for another attempt.
_NEGATIVES = frozenset({"no", "don't", "do not", "cancel", "stop", "nope", "wait"})

# Clarifying-question markers: a response that asks for detail ("wait, which
# one?") is ambiguous, not a denial — even when it leads with a negative word
# like "wait".
_QUESTION_WORDS = frozenset(
    {"which", "what", "when", "where", "why", "how", "who"}
)

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


def _normalize_response(text: str | None) -> str:
    """Normalize a confirmation response once, near the top of resolution:
    casefold, strip surrounding whitespace, and strip trailing sentence
    punctuation (".!?"). Keeps "yes.", "YES  ", "please." comparable to the
    affirmative/phrase sets while exact SENSITIVE phrase matching stays exact
    against the normalized form."""
    text = (text or "").casefold().strip()
    return text.strip(" .!?").strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    """Word-boundary containment: ``no`` matches inside "no, don't confirm"
    but not inside "nope" or "notebook"; ``do it`` matches "let's do it"."""
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def _matches_any(text: str, phrases: frozenset[str]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _looks_like_question(text: str) -> bool:
    """Ambiguity test: a trailing question mark or a wh-word ("which", "what",
    ...) means the owner asked for clarification rather than deciding. Such a
    reply must leave the action pending, never cancel it."""
    if "?" in text:
        return True
    words = re.findall(r"[a-z0-9']+", text)
    return any(word in _QUESTION_WORDS for word in words)


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

    ``get_recognized_name`` is a ``Callable[[], str | None]`` supplied by
    main.py, returning the currently recognized speaker name (None when nobody
    is recognized). ``get_owner_name`` is a ``Callable[[], str | None]``
    returning the current owner marker's name. ``get_last_strong_recognition_time``
    returns the raw (monotonic) timestamp of the most recent strong
    recognition; confirming a pending action requires genuinely NEW evidence —
    a strong recognition that landed AFTER the action was created — not just
    recent trust from earlier in the session. Ownership is dynamic — the
    getters are consulted per call, so a first-run enrollment that claims the
    owner name takes effect immediately.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        get_recognized_name: Callable[[], str | None],
        get_owner_name: Callable[[], str | None],
        get_last_strong_recognition_time: Callable[[], float | None] | None = None,
        wait_for_pending: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._registry = registry
        self._get_recognized_name = get_recognized_name
        self._get_owner_name = get_owner_name
        self._get_last_strong_recognition_time = get_last_strong_recognition_time
        self._wait_for_pending = wait_for_pending
        self._pending: dict | None = None

    def _fresh_ok(self, pending: dict) -> bool:
        """Genuinely-NEW-evidence gate for confirmation: True only when a
        strong recognition landed strictly AFTER the pending action was
        created. The owner's original request that triggered the proposal does
        not count — recent-but-older-than-the-proposal trust doesn't approve a
        write. ``pending["created_at"]`` already uses ``time.monotonic()`` and
        so does main.py's RecognitionTrust, making the comparison valid. None
        getter stays permissive (callers that don't supply one accept any
        timing)."""
        if self._get_last_strong_recognition_time is None:
            return True
        last = self._get_last_strong_recognition_time()
        return last is not None and last > pending["created_at"]

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
            role = role_for_actor(
                self._get_recognized_name(), self._get_owner_name()
            )
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
                    role = role_for_actor(
                        self._get_recognized_name(), self._get_owner_name()
                    )
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
            # Registration guarantees a SENSITIVE tool declares an explicit
            # confirmation_phrase (ToolRegistry.register raises otherwise), so
            # there is no weak default to fall back on here.
            phrase = spec.confirmation_phrase
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
        authorized actor *right now* and back it with genuinely new evidence —
        a strong recognition that landed AFTER the action was proposed.
        """
        if self._pending is None:
            return None

        response_text = _normalize_response(response_text)
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

        # Atomic claim of the pending slot, before any await. Two concurrent
        # resolution attempts (confirm_action and check_transcript racing is
        # the designed behavior) could otherwise both pass the initial check,
        # suspend on the wait below, and both execute the handler. Whoever
        # claims first clears the slot; the loser sees None above and sits
        # out. The expiry check stays above this line so an expired action is
        # still reported "cancelled" rather than silently claimed.
        pending, self._pending = self._pending, None

        # Re-check the actor's permission now, not just when the action was
        # first proposed: the person confirming must still be authorized.
        # Someone who merely overheard the phrase shouldn't complete it.
        # Confirmation additionally demands a recognition that is genuinely
        # NEW (a strong recognition AFTER the action was created, not stale
        # trust from earlier in the session). Either way, give an in-flight
        # recognition for the confirmation utterance one bounded wait before
        # finalizing a denial (one wait, not a retry loop).
        role = role_for_actor(
            self._get_recognized_name(), self._get_owner_name()
        )
        fresh = self._fresh_ok(pending)
        if not has_permission(role, pending["risk_tier"]) or not fresh:
            if self._wait_for_pending is not None:
                await self._wait_for_pending()
            role = role_for_actor(
                self._get_recognized_name(), self._get_owner_name()
            )
            fresh = self._fresh_ok(pending)

        if not has_permission(role, pending["risk_tier"]):
            logger.info(
                "Pending action %r denied for role %r (tier=%s) at confirmation",
                pending["tool_name"], role, pending["risk_tier"].value,
            )
            return {
                "error": "not permitted",
                "tool_name": pending["tool_name"],
            }

        if not fresh:
            # Not permanently denied: a genuinely new recognition can still
            # complete it later, so hand the claimed slot back (F-03).
            self._pending = pending
            logger.info(
                "Pending action %r denied: confirmation requires a fresh recognition",
                pending["tool_name"],
            )
            return {
                "error": "not permitted",
                "tool_name": pending["tool_name"],
                "reason": "confirmation requires a fresh recognition",
            }

        # Affirmation classification (F-04). A clarifying question is
        # ambiguous, never a denial — even when it leads with a negative word
        # like "wait". Explicit negatives cancel; WRITE affirmatives match on
        # word boundaries; SENSITIVE requires the exact (normalized) phrase.
        # Anything else leaves the action pending so the owner can try again
        # before the timeout.
        if _looks_like_question(response_text):
            self._pending = pending
            logger.info(
                "Pending action %r kept pending for ambiguous response %r",
                pending["tool_name"], response_text,
            )
            return {
                "status": "pending",
                "tool_name": pending["tool_name"],
                "reason": "response not understood as yes or no",
            }

        if _matches_any(response_text, _NEGATIVES):
            logger.info(
                "Pending action %r denied by owner response %r",
                pending["tool_name"], response_text,
            )
            return {
                "status": "cancelled",
                "tool_name": pending["tool_name"],
            }

        if pending["risk_tier"] == RiskTier.SENSITIVE:
            affirmed = response_text == _normalize_response(
                pending["required_confirmation"]
            )
        else:
            affirmed = _matches_any(response_text, _AFFIRMATIVES)

        if not affirmed:
            # Ambiguous: neither an explicit no nor a recognized yes/phrase.
            self._pending = pending
            logger.info(
                "Pending action %r kept pending for unrecognized response %r",
                pending["tool_name"], response_text,
            )
            return {
                "status": "pending",
                "tool_name": pending["tool_name"],
                "reason": "response not understood as yes or no",
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