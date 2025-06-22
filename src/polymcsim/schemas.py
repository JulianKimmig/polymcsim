"""Pydantic schemas for PolyMCsim configuration and validation."""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import networkx as nx
from pydantic import BaseModel, ConfigDict, Field

# --- 1. Pydantic Models for User Input and Validation ---


class SiteDef(BaseModel):
    """Defines a reactive site on a monomer."""

    type: str
    status: Literal["ACTIVE", "DORMANT", "CONSUMED"] = "ACTIVE"


class MonomerDef(BaseModel):
    """Define a type of monomer in the system.

    Attributes:
        name: Unique name for this monomer type.
        count: Number of these monomers to add to the system.
        molar_mass: Molar mass of the monomer unit (g/mol).
        sites: List of reactive sites on this monomer.

    """

    name: str = Field(..., description="Unique name for this monomer type.")
    count: int = Field(
        ..., gt=0, description="Number of these monomers to add to the system."
    )
    molar_mass: float = Field(
        default=100.0, gt=0, description="Molar mass of the monomer unit (g/mol)."
    )
    sites: List[SiteDef] = Field(
        ..., description="List of reactive sites on this monomer."
    )


class ReactionSchema(BaseModel):
    """Defines the outcome and rate of a reaction between two site types."""

    rate: float = Field(..., gt=0, description="Reaction rate constant.")
    activation_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Maps a dormant site type to the active type it becomes.",
    )

    model_config = ConfigDict(extra="forbid")


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
        default=float("inf"), description="Maximum simulation time to run."
    )
    max_reactions: int = Field(
        default=1_000_000_000, description="Maximum number of reaction events."
    )
    max_conversion: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Maximum fraction of monomers that can be reacted (0.0 to 1.0).",
    )
    random_seed: int = Field(
        default=42, description="Random seed for reproducible results."
    )


class SimulationInput(BaseModel):
    """Complete input configuration for a PolyMCsim simulation.

    Attributes:
        monomers: List of monomer definitions.
        reactions: Dictionary mapping site type pairs to reaction schemas.
        params: Simulation parameters.

    """

    monomers: List[MonomerDef] = Field(..., description="List of monomer definitions.")
    reactions: Dict[frozenset[str], ReactionSchema] = Field(
        ..., description="Dictionary mapping site type pairs to reaction schemas."
    )
    params: SimParams = Field(
        default_factory=SimParams, description="Simulation parameters."
    )


# --- 2. Pydantic Models for Simulation Output ---


class SimulationResult(BaseModel):
    """Holds the results of a single simulation run.

    Attributes:
        graph: The final polymer network structure.
        metadata: A dictionary of metadata about the simulation run.
        config: The input configuration that generated this result.
        error: An error message if the simulation failed.

    """

    graph: nx.Graph | None = None
    metadata: Dict[str, Any] | None = None
    config: "SimulationInput"
    error: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
