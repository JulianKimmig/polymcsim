"""Test the max_conversion functionality in PolyMCsim."""

import pytest

from polymcsim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
)


def test_max_conversion_stops_simulation() -> None:
    """Test that simulation stops when max_conversion is reached."""
    # Create a simple AA + BB step-growth system
    monomers = [
        MonomerDef(
            name="A-Monomer", count=100, sites=[SiteDef(type="A", status="ACTIVE")]
        ),
        MonomerDef(
            name="B-Monomer", count=100, sites=[SiteDef(type="B", status="ACTIVE")]
        ),
    ]

    reactions = {frozenset(["A", "B"]): ReactionSchema(rate=1.0)}

    # Test different max_conversion values
    conversion_targets = [0.2, 0.5, 0.8, 0.95]

    for target_conversion in conversion_targets:
        params = SimParams(
            name=f"test_conversion_{target_conversion}",
            max_conversion=target_conversion,
            random_seed=42,
        )

        config = SimulationInput(monomers=monomers, reactions=reactions, params=params)

        sim = Simulation(config)
        result = sim.run()
        graph, metadata = result.graph, result.metadata

        # Check that final conversion is close to but not exceeding max_conversion
        final_conversion = metadata["final_conversion"]
        assert (
            final_conversion <= target_conversion + 0.01
        )  # Small tolerance for numerical precision
        assert (
            final_conversion >= target_conversion * 0.9
        )  # Should be reasonably close to target

        # Check that we have the expected number of edges (reactions)
        # Each reaction connects 2 monomers, so conversion ≈ 2 * edges / total_monomers
        expected_edges = int(target_conversion * 100)  # Approximate
        actual_edges = graph.number_of_edges()
        assert actual_edges >= expected_edges * 0.8  # Within reasonable range
        assert actual_edges <= expected_edges * 1.2


def test_max_conversion_default() -> None:
    """Test that default max_conversion allows full reaction."""
    monomers = [
        MonomerDef(
            name="A-Monomer",
            count=50,
            sites=[
                SiteDef(type="A", status="ACTIVE"),
                SiteDef(type="A", status="ACTIVE"),
            ],
        )
    ]

    reactions = {frozenset(["A"]): ReactionSchema(rate=1.0)}

    # Use default params (max_conversion = 1.0)
    params = SimParams(name="test_full_conversion")

    config = SimulationInput(monomers=monomers, reactions=reactions, params=params)

    sim = Simulation(config)
    result = sim.run()
    metadata = result.metadata

    # With AA polymerization, should reach high conversion
    final_conversion = metadata["final_conversion"]
    assert final_conversion > 0.9  # Should achieve high conversion


def test_max_conversion_zero() -> None:
    """Test that max_conversion=0 prevents any reactions."""
    monomers = [
        MonomerDef(
            name="A-Monomer", count=10, sites=[SiteDef(type="A", status="ACTIVE")]
        )
    ]

    reactions = {frozenset(["A"]): ReactionSchema(rate=1.0)}

    params = SimParams(name="test_zero_conversion", max_conversion=0.0, random_seed=42)

    config = SimulationInput(monomers=monomers, reactions=reactions, params=params)

    sim = Simulation(config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Should have no reactions
    assert metadata["final_conversion"] == 0.0
    assert metadata["reactions_completed"] == 0
    assert graph.number_of_edges() == 0


def test_max_conversion_validation() -> None:
    """Test that max_conversion parameter validation works."""
    # Test invalid values
    with pytest.raises(ValueError):
        SimParams(max_conversion=-0.1)

    with pytest.raises(ValueError):
        SimParams(max_conversion=1.1)

    # Test valid edge cases
    params1 = SimParams(max_conversion=0.0)
    assert params1.max_conversion == 0.0

    params2 = SimParams(max_conversion=1.0)
    assert params2.max_conversion == 1.0
