"""ToolRegistry behavior: registration, lookup, and the Gem-visible surface."""

from tools.registry import RiskTier, ToolRegistry, ToolSpec


def _spec(name: str, tier: RiskTier = RiskTier.READ) -> ToolSpec:
    async def _handler(**args) -> dict:
        return {"status": "ok"}

    return ToolSpec(
        name=name,
        description=f"test tool {name}",
        parameters={"type": "object", "properties": {}},
        risk_tier=tier,
        handler=_handler,
        confirmation_phrase="open sesame",
    )


def test_register_and_get_roundtrip():
    registry = ToolRegistry()
    spec = _spec("lookup")
    registry.register(spec)
    assert registry.get("lookup") is spec
    assert registry.get("missing") is None


def test_all_specs_reflect_registration_order():
    registry = ToolRegistry()
    first = _spec("first")
    second = _spec("second")
    registry.register(first)
    registry.register(second)
    assert registry.all_specs() == [first, second]


def test_duplicate_name_replaces_previous_spec():
    registry = ToolRegistry()
    registry.register(_spec("dup"))
    replacement = _spec("dup", RiskTier.WRITE)
    registry.register(replacement)
    assert registry.get("dup") is replacement


def test_all_declarations_expose_only_name_description_parameters():
    """Policy metadata (risk_tier, confirmation_phrase, handler) must never
    leak into the Gemini function declarations."""
    registry = ToolRegistry()
    registry.register(_spec("search", RiskTier.SENSITIVE))
    (declaration,) = registry.all_declarations()
    assert set(declaration) == {"name", "description", "parameters"}
    assert declaration["name"] == "search"
    assert declaration["description"] == "test tool search"
    assert "risk_tier" not in declaration
    assert "confirmation_phrase" not in declaration
    assert "handler" not in declaration


def test_spec_metadata_roundtrip_preserved():
    spec = _spec("secret", RiskTier.SENSITIVE)
    assert spec.risk_tier is RiskTier.SENSITIVE
    assert spec.confirmation_phrase == "open sesame"