from tools.registry import RiskTier

ROLE_OWNER = "owner"
ROLE_UNKNOWN = "unknown"


def role_for_actor(recognized_name: str | None, owner_name: str | None) -> str:
    """Map a recognized speaker (or None) and the current owner to a role.

    A match between the recognized speaker and the owner marker grants the
    owner role; anything else — a different enrolled name, nobody recognized
    (None), or no owner set yet — is unknown.
    """
    if recognized_name is not None and recognized_name == owner_name:
        return ROLE_OWNER
    return ROLE_UNKNOWN


def has_permission(role: str, risk_tier: RiskTier) -> bool:
    """Role-based permission: owner may act at every tier; unknown may only
    use READ. That includes explicitly NOT TRIVIAL: TRIVIAL runs with no
    confirmation step, so it must stay owner-gated like WRITE/SENSITIVE."""
    if role == ROLE_OWNER:
        return True
    if role == ROLE_UNKNOWN:
        return risk_tier == RiskTier.READ
    return False