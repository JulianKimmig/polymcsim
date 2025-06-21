"""Tests for PolyMCsim schema validation."""

import pytest
from pydantic import ValidationError

from polymcsim.schemas import (
    MonomerDef,
    ReactionSchema,
    SimulationInput,
    SiteDef,
)


def test_monomer_def_negative_count() -> None:
    """Test that a negative monomer count raises a validation error."""
    with pytest.raises(ValidationError):
        MonomerDef(name="A", count=-1, sites=[SiteDef(type="A")])


def test_simulation_input_good() -> None:
    """Test that a valid SimulationInput model can be created."""
    config = SimulationInput(
        monomers=[MonomerDef(name="Monomer", count=100, sites=[SiteDef(type="A")])],
        reactions={frozenset(["A", "A"]): ReactionSchema(rate=1.0)},
    )
    assert len(config.monomers) == 1
    assert config.params.max_reactions == 1_000_000_000
