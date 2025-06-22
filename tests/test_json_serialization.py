#!/usr/bin/env python3
"""Test script to verify JSON serialization of SimulationInput."""

from polymcsim.schemas import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    SimulationInput,
    SiteDef,
)


def test_json_serialization():
    """Test that SimulationInput can be serialized to and from JSON."""
    # Create a simple test configuration
    monomers = [
        MonomerDef(
            name="A",
            count=100,
            molar_mass=100.0,
            sites=[SiteDef(type="A_site", status="ACTIVE")],
        ),
        MonomerDef(
            name="B",
            count=100,
            molar_mass=150.0,
            sites=[SiteDef(type="B_site", status="ACTIVE")],
        ),
    ]

    reactions = {frozenset(["A_site", "B_site"]): ReactionSchema(rate=1.0)}

    params = SimParams(name="test_sim", max_reactions=1000)

    # Create SimulationInput using the new API
    config = SimulationInput(monomers=monomers, reactions=reactions, params=params)

    # Serialize to JSON
    json_str = config.model_dump_json(indent=2)

    # Deserialize from JSON
    config_restored = SimulationInput.model_validate_json(json_str)

    # Verify the restored config is equivalent
    assert len(config.reactions) == len(config_restored.reactions)
    assert len(config.reactions) > 0
    for key in config.reactions:
        assert key in config_restored.reactions
        assert config.reactions[key].rate == config_restored.reactions[key].rate
        assert type(config.reactions[key]) is type(config_restored.reactions[key])
        assert isinstance(key, frozenset)


if __name__ == "__main__":
    test_json_serialization()
