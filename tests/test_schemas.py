import pytest
from pydantic import ValidationError
from polysim.schemas import MonomerDef, SiteDef, SimulationInput, SimParams, ReactionSchema

def test_monomer_def_negative_count():
    """Tests that a negative monomer count raises a validation error."""
    with pytest.raises(ValidationError):
        MonomerDef(name="A", count=-1, sites=[SiteDef(type="A")])

def test_simulation_input_good():
    """Tests that a valid SimulationInput model can be created."""
    config = SimulationInput(
        monomers=[
            MonomerDef(name="Monomer", count=100, sites=[SiteDef(type="A")])
        ],
        rate_matrix={frozenset(["A", "A"]): 1.0},
        reaction_schema={
            frozenset(["A", "A"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED"
            )
        }
    )
    assert len(config.monomers) == 1
    assert config.params.max_reactions == 1_000_000_000

def test_rate_matrix_key_coerced_to_frozenset():
    """
    Tests that a non-frozenset key in rate_matrix is correctly coerced by Pydantic V2.
    """
    input_data = {
        "monomers": [
            {"name": "Monomer", "count": 100, "sites": [{"type": "A"}]}
        ],
        "rate_matrix": {
            ("A", "A"): 1.0  # Using a tuple instead of frozenset
        },
        "reaction_schema": {
            frozenset(["A", "A"]): {
                "site1_final_status": "CONSUMED",
                "site2_final_status": "CONSUMED"
            }
        }
    }
    model = SimulationInput.model_validate(input_data)
    
    # Check that the tuple key was converted to a frozenset
    assert isinstance(list(model.rate_matrix.keys())[0], frozenset)
    assert frozenset(["A", "A"]) in model.rate_matrix

def test_reaction_schema_mismatch():
    """Tests that a mismatch between rate_matrix and reaction_schema keys is allowed by Pydantic but should be caught in logic."""
    # Pydantic itself won't enforce that keys match between these two dicts.
    # This test documents that behavior. The simulation's logic should handle this.
    config_data = {
        "monomers": [
            {"name": "Monomer", "count": 100, "sites": [{"type": "A"}]}
        ],
        "rate_matrix": {
            frozenset(["A", "B"]): 1.0
        },
        "reaction_schema": {
            frozenset(["A", "C"]): { # Mismatched key
                "site1_final_status": "CONSUMED",
                "site2_final_status": "CONSUMED"
            }
        }
    }
    # This will likely fail later in the simulation logic, not here.
    # We expect this to validate successfully.
    sim_input = SimulationInput.model_validate(config_data)
    assert frozenset(["A", "B"]) in sim_input.rate_matrix
    assert frozenset(["A", "C"]) in sim_input.reaction_schema 