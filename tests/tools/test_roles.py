"""role_for_actor and the full has_permission (role x risk_tier) matrix."""

from tools.registry import RiskTier
from tools.roles import ROLE_OWNER, ROLE_UNKNOWN, has_permission, role_for_actor

ALL_TIERS = list(RiskTier)


def test_role_for_actor_matches_recognized_to_owner():
    assert role_for_actor("alex", "alex") == ROLE_OWNER
    assert role_for_actor(None, None) == ROLE_UNKNOWN
    assert role_for_actor(None, "alex") == ROLE_UNKNOWN
    assert role_for_actor("alex", None) == ROLE_UNKNOWN
    assert role_for_actor("bob", "alex") == ROLE_UNKNOWN
    assert role_for_actor("", "alex") == ROLE_UNKNOWN


def test_owner_can_act_at_every_tier():
    for tier in ALL_TIERS:
        assert has_permission(ROLE_OWNER, tier), f"owner denied at {tier}"


def test_unknown_can_only_use_read():
    for tier in ALL_TIERS:
        expected = tier is RiskTier.READ
        assert has_permission(ROLE_UNKNOWN, tier) is expected, (
            f"unknown permission divergence at {tier}: expected {expected}"
        )


def test_trivial_explicitly_denied_to_unknown():
    """TRIVIAL is confirmation-free, so it must stay owner-gated even though
    it is read-only: this is the deliberate divergence from READ."""
    assert has_permission(ROLE_UNKNOWN, RiskTier.TRIVIAL) is False


def test_unknown_role_string_is_denied_everywhere():
    for tier in ALL_TIERS:
        assert has_permission("other", tier) is False