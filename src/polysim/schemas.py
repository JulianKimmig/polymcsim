from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Literal

# --- 1. Pydantic Models for User Input and Validation ---

class SiteDef(BaseModel):
    """Defines a single reactive site on a monomer."""
    type: str = Field(..., description="The type of the reactive site (e.g., 'A', 'Radical').")
    status: Literal['ACTIVE', 'DORMANT'] = Field('ACTIVE', description="Initial status of the site.")

class MonomerDef(BaseModel):
    """Defines a type of monomer in the system."""
    name: str
    count: int = Field(..., gt=0, description="Number of these monomers to add to the system.")
    sites: List[SiteDef]

class ReactionSchema(BaseModel):
    """Describes the outcome of a reaction between two site types."""
    site1_final_status: Literal['CONSUMED']
    site2_final_status: Literal['CONSUMED']
    # Describes which site on the SECOND monomer is activated, and what it becomes.
    # Format: {"original_dormant_type": "new_active_type"}
    activation_map: Dict[str, str] = Field(default_factory=dict)

class SimParams(BaseModel):
    """Parameters to control the simulation execution."""
    name: str = "simulation"
    max_time: float = Field(default=float('inf'), description="Maximum simulation time to run.")
    max_reactions: int = Field(default=1_000_000_000, description="Maximum number of reaction events.")
    random_seed: int = 42

class SimulationInput(BaseModel):
    """The complete input configuration for a PolySim simulation."""
    monomers: List[MonomerDef]
    # Rate constants k for each reactive pair. The key is a frozenset to be order-independent.
    rate_matrix: Dict[frozenset[str], float]
    # Defines what happens to sites after a reaction.
    reaction_schema: Dict[frozenset[str], ReactionSchema]
    params: SimParams = Field(default_factory=SimParams) 