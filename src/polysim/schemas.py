"""Pydantic schemas for PolySim configuration and validation."""

from typing import Dict, List, Literal
from pydantic import BaseModel, Field

# --- 1. Pydantic Models for User Input and Validation ---

class SiteDef(BaseModel):
    """Define a single reactive site on a monomer.
    
    Attributes:
        type: The type of the reactive site (e.g., 'A', 'Radical').
        status: Initial status of the site ('ACTIVE' or 'DORMANT').
    """
    type: str = Field(..., description="The type of the reactive site (e.g., 'A', 'Radical').")
    status: Literal['ACTIVE', 'DORMANT'] = Field(
        'ACTIVE', 
        description="Initial status of the site."
    )

class MonomerDef(BaseModel):
    """Define a type of monomer in the system.
    
    Attributes:
        name: Unique name for this monomer type.
        count: Number of these monomers to add to the system.
        molar_mass: Molar mass of the monomer unit (g/mol).
        sites: List of reactive sites on this monomer.
    """
    name: str = Field(..., description="Unique name for this monomer type.")
    count: int = Field(..., gt=0, description="Number of these monomers to add to the system.")
    molar_mass: float = Field(default=100.0, gt=0, description="Molar mass of the monomer unit (g/mol).")
    sites: List[SiteDef] = Field(..., description="List of reactive sites on this monomer.")

class ReactionSchema(BaseModel):
    """Describe the outcome of a reaction between two site types.
    
    Attributes:
        site1_final_status: Final status of the first site after reaction.
        site2_final_status: Final status of the second site after reaction.
        activation_map: Maps dormant site types to new active types on the second monomer.
        rate: Rate constant for the reaction.
    """
    site1_final_status: Literal['CONSUMED'] = Field(
        ..., 
        description="Final status of the first site after reaction."
    )
    site2_final_status: Literal['CONSUMED'] = Field(
        ..., 
        description="Final status of the second site after reaction."
    )
    activation_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps dormant site types to new active types on the second monomer."
    )
    rate: float = Field(..., description="Rate constant for the reaction.")

class SimParams(BaseModel):
    """Parameters to control the simulation execution.
    
    Attributes:
        name: Name for this simulation run.
        max_time: Maximum simulation time to run.
        max_reactions: Maximum number of reaction events.
        max_conversion: Maximum fraction of monomers that can be reacted (0.0 to 1.0).
        random_seed: Random seed for reproducible results.
    """
    name: str = Field(default="simulation", description="Name for this simulation run.")
    max_time: float = Field(
        default=float('inf'), 
        description="Maximum simulation time to run."
    )
    max_reactions: int = Field(
        default=1_000_000_000, 
        description="Maximum number of reaction events."
    )
    max_conversion: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of monomers that can be reacted (0.0 to 1.0)."
    )
    random_seed: int = Field(default=42, description="Random seed for reproducible results.")

class SimulationInput(BaseModel):
    """Complete input configuration for a PolySim simulation.
    
    Attributes:
        monomers: List of monomer definitions.
        reactions: Dictionary mapping site type pairs to reaction schemas.
        params: Simulation parameters.
    """
    monomers: List[MonomerDef] = Field(..., description="List of monomer definitions.")
    reactions: Dict[frozenset[str], ReactionSchema] = Field(
        ..., 
        description="Dictionary mapping site type pairs to reaction schemas."
    )
    params: SimParams = Field(
        default_factory=SimParams, 
        description="Simulation parameters."
    ) 