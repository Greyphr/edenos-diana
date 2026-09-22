from enum import Enum


class RiskTier(Enum):
    READ = "read"
    TRIVIAL = "trivial"
    WRITE = "write"
    SENSITIVE = "sensitive"


class ToolSpec:
    """A tool plus the policy metadata that governs how it may run.

    ``handler`` is an async ``callable(**args) -> dict``. ``risk_tier`` and
    ``confirmation_phrase`` never leave this layer — only ``name``,
    ``description``, and ``parameters`` appear in the Gemini declarations.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        risk_tier: RiskTier,
        handler,
        confirmation_phrase: str | None = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.risk_tier = risk_tier
        self.handler = handler
        # Required exact phrase for SENSITIVE tier confirmation.
        self.confirmation_phrase = confirmation_phrase


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.risk_tier is RiskTier.SENSITIVE and not spec.confirmation_phrase:
            # A SENSITIVE tool with no real phrase is a design mistake, not a
            # runtime case to handle gracefully: without it the confirmation
            # would fall back on a single weak word, and F-04's exact-phrase
            # matching makes that unworkable. Reject it here so it fails at
            # registration, not mid-session.
            raise ValueError(
                f"SENSITIVE tool {spec.name!r} must declare an explicit "
                "confirmation_phrase"
            )
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def all_declarations(self) -> list[dict]:
        """Gemini function_declarations (name/description/parameters only)."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in self.all_specs()
        ]