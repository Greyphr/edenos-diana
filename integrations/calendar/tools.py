"""Google Calendar tool specs for the block 4 registry + policy engine.

Two specs, same wiring path as the Spotify tools: they register into the same
ToolRegistry and flow through the same PolicyEngine. calendar_get_upcoming is
TRIVIAL tier — owner-gated (denied for unrecognized actors) but executes
immediately with no confirmation prompt. calendar_create_event is WRITE tier:
it creates something persistent on the owner's calendar, so it goes through
confirm_action before the handler runs.

An unconnected client is an ordinary *startup* condition: main.py only
registers these specs when CalendarClient() successfully constructs. API
failures (network, HTTP status, auth) surface as normal error dicts from the
handlers — never as raised exceptions — so the model reads the message and
tells the owner what happened.
"""

from tools.registry import RiskTier, ToolSpec

from .client import CalendarClient, CalendarError

_UPCOMING_PARAMETERS = {
    "type": "object",
    "properties": {
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 50,
            "description": (
                "How many upcoming events to return (default 5)."
            ),
        }
    },
}
_CREATE_PARAMETERS = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "start": {
            "type": "string",
            "description": (
                "RFC 3339 dateTime for the event start, e.g. "
                "2026-09-21T15:00:00+02:00 (include the timezone "
                "offset so Google creates it at the right wall-clock time)."
            ),
        },
        "end": {
            "type": "string",
            "description": (
                "RFC 3339 dateTime for the event end, same format as start."
            ),
        },
        "description": {"type": "string"},
    },
    "required": ["summary", "start", "end"],
}


def make_calendar_specs(client: CalendarClient) -> list[ToolSpec]:
    async def _get_upcoming(**args) -> dict:
        max_results = args.get("max_results") or 5
        try:
            events = await client.get_upcoming_events(max_results=max_results)
        except Exception as exc:  # network/HTTP/auth failures -> normal result
            return {
                "error": "calendar_error",
                "message": f"Could not fetch the calendar: {exc}",
            }
        if not events:
            return {
                "status": "no_events",
                "message": "Nothing is on the calendar from now onward.",
            }
        lines = []
        for event in events:
            when = event.get("start") or "unscheduled"
            label = f"{when} - {event.get('summary') or 'untitled event'}"
            if event.get("location"):
                label += f" ({event['location']})"
            lines.append(label)
        return {
            "status": "ok",
            "message": "; ".join(lines),
            "events": events,
        }

    async def _create_event(**args) -> dict:
        summary = (args.get("summary") or "").strip()
        start = (args.get("start") or "").strip()
        end = (args.get("end") or "").strip()
        description = (args.get("description") or "").strip()
        if not summary or not start or not end:
            return {
                "error": "missing_fields",
                "message": "summary, start, and end are all required to create an event.",
            }
        try:
            event = await client.create_event(
                summary, start, end, description=description
            )
        except CalendarError as exc:
            return {"error": "invalid_event", "message": str(exc)}
        except Exception as exc:  # network/HTTP/auth failures -> normal result
            return {
                "error": "calendar_error",
                "message": f"Could not create the event: {exc}",
            }
        return {
            "status": "ok",
            "message": (
                f"Created calendar event {event.get('summary')!r} "
                f"at {event.get('start')}."
            ),
            "event": event,
        }

    return [
        ToolSpec(
            name="calendar_get_upcoming",
            description=(
                "Read the next upcoming events on the owner's Google Calendar "
                "(soonest first, only events that have not already started). "
                "Returns the event summaries with their start times. Creates "
                "nothing, so it runs immediately without a confirmation prompt."
            ),
            parameters=_UPCOMING_PARAMETERS,
            risk_tier=RiskTier.TRIVIAL,
            handler=_get_upcoming,
        ),
        ToolSpec(
            name="calendar_create_event",
            description=(
                "Create a new event on the owner's Google Calendar with the "
                "given summary, start and end (RFC 3339 dateTime values, "
                "including a timezone offset). Permanently modifies the "
                "owner's calendar, so it must be confirmed by the owner "
                "before it runs."
            ),
            parameters=_CREATE_PARAMETERS,
            risk_tier=RiskTier.WRITE,
            handler=_create_event,
        ),
    ]