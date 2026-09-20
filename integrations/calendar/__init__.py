"""Google Calendar integration.

- integrations/calendar/auth.py    one-time browser authorization (stores the
  refresh token in the vault under "calendar_refresh_token")
- integrations/calendar/client.py  async API client used by the calendar_* tools
- integrations/calendar/tools.py   calendar_get_upcoming (TRIVIAL) and
  calendar_create_event (WRITE) ToolSpecs wired into main.py's registry
"""