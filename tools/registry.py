from enum import Enum


class RiskTier(Enum):
    READ = "read"
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