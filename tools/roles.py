from tools.registry import RiskTier

ROLE_OWNER = "owner"
ROLE_UNKNOWN = "unknown"


def role_for_actor(recognized: bool) -> str:
    """Map a current recognition state to a role name."""
    return ROLE_OWNER if recognized else ROLE_UNKNOWN


def has_permission(role: str, risk_tier: RiskTier) -> bool:
    """Role-based permission: owner may act at every tier; unknown only READ."""
    if role == ROLE_OWNER:
        return True
    if role == ROLE_UNKNOWN:
        return risk_tier == RiskTier.READ
    return False