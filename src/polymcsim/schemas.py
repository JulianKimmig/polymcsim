"""Pydantic schemas for PolyMCsim configuration and validation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Union

import networkx as nx
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
)

__all__ = [
    "SiteDef",
    "MonomerDef",
    "ReactionSchema",
    "SimParams",
    "SimulationInput",
    "Polymer",
    "SimulationResult",
]

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

    @field_serializer("max_time")
    def serialize_max_time(self, max_time: float) -> Union[float, str]:
        """Serialize inf values as 'inf' string for JSON compatibility."""
        if max_time == float("inf"):
            return "inf"
        return max_time

    @field_validator("max_time", mode="before")
    @classmethod
    def validate_max_time(cls, v: Any) -> float:
        """Convert 'inf' string back to float('inf') during validation."""
        if v == "inf":
            return float("inf")
        return v


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

    @field_serializer("reactions")
    def serialize_reactions(
        self, reactions: Dict[frozenset[str], ReactionSchema]
    ) -> Dict[str, ReactionSchema]:
        """Serialize frozenset keys into strings for JSON compatibility."""
        return {"|".join(k): v for k, v in reactions.items()}

    @field_validator("reactions", mode="before")
    @classmethod
    def validate_reactions(cls, v: Any) -> Dict[frozenset[str], ReactionSchema]:
        """Validate and convert string keys back to frozensets."""
        return {
            (frozenset(k.split("|")) if isinstance(k, str) else k): v
            for k, v in v.items()
        }


# --- 2. Pydantic Models for Simulation Output ---


class Polymer(BaseModel):
    """Represents a single polymer chain.

    This is a connected component in the simulation graph.
    """

    id: int = Field(..., description="A unique identifier for this polymer chain.")
    graph: nx.Graph = Field(
        ..., description="The graph structure of this specific polymer chain."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _molar_mass_cache: float | None = PrivateAttr(default=None)

    @property
    def num_monomers(self) -> int:
        """The number of monomer units in this polymer (degree of polymerization)."""
        return self.graph.number_of_nodes()

    @property
    def molecular_weight(self) -> float:
        """The total molar mass of this polymer chain (g/mol)."""
        if self._molar_mass_cache is not None:
            return self._molar_mass_cache

        self._molar_mass_cache = sum(
            data.get("molar_mass", 100.0) for _, data in self.graph.nodes(data=True)
        )
        return self._molar_mass_cache

    @property
    def branch_points(self) -> int:
        """The number of branch points (nodes with degree > 2)."""
        return sum(1 for _, degree in self.graph.degree() if degree > 2)

    @property
    def is_linear(self) -> bool:
        """Checks if the polymer is linear (maximum degree is 2 or less)."""
        if not self.graph:
            return True
        degrees = [d for _, d in self.graph.degree()]
        return max(degrees) <= 2 if degrees else True

    @property
    def composition(self) -> Dict[str, int]:
        """Return the monomer composition of the polymer as a dictionary."""
        types = [data["monomer_type"] for _, data in self.graph.nodes(data=True)]
        return dict(Counter(types))

    def get_nodes_by_type(self) -> Dict[str, Any]:
        """Return nodes of the polymer graph grouped by monomer type."""
        nodes = {}
        for node, data in self.graph.nodes(data=True):
            if data["monomer_type"] not in nodes:
                nodes[data["monomer_type"]] = []
            nodes[data["monomer_type"]].append(node)
        return nodes


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
    final_state: Optional[Dict[str, Any]] = None

    _polymers_cache: List[Polymer] | None = PrivateAttr(default=None)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_polymers(self) -> List["Polymer"]:
        """Extract individual polymer chains from the main simulation graph.

        This method identifies connected components in the simulation graph, filters
        out unreacted monomers (components with a single node), and returns a
        list of `Polymer` objects. The result is cached after the first call.

        Returns:
            A list of `Polymer` objects, sorted from largest to smallest.

        """
        if self._polymers_cache is not None:
            return self._polymers_cache

        if self.graph is None:
            self._polymers_cache = []
            return self._polymers_cache

        # Find all connected components
        components = list(nx.connected_components(self.graph))

        # Create Polymer objects for components that are actual polymers (size > 1)
        polymers = []
        for component_nodes in components:
            if len(component_nodes) > 1:
                # Create a subgraph for the component. .copy() is important!
                subgraph = self.graph.subgraph(component_nodes).copy()
                polymers.append(Polymer(id=len(polymers), graph=subgraph))  # temp id

        # Sort polymers by size (number of monomers) in descending order
        polymers.sort(key=lambda p: p.num_monomers, reverse=True)

        # Assign final IDs based on sorted order
        for i, p in enumerate(polymers):
            p.id = i

        self._polymers_cache = polymers
        return self._polymers_cache

    def get_largest_polymer(self) -> "Polymer" | None:
        """Return the largest polymer chain from the simulation result.

        Returns:
            The largest `Polymer` object, or None if no polymers were formed.

        """
        polymers = self.get_polymers()
        return polymers[0] if polymers else None

    def get_unreacted_monomers(self) -> List[Dict[str, Any]]:
        """Identify monomers that have not participated in any reactions.

        An unreacted monomer is represented as an isolated node in the graph
        (i.e., its degree is 0).

        Returns:
            A list of dictionaries, where each dictionary contains the
            attributes of an unreacted monomer node (e.g., `monomer_type`).

        """
        if self.graph is None:
            return []

        isolated_nodes = [node for node, degree in self.graph.degree() if degree == 0]
        return [{**self.graph.nodes[node], "id": node} for node in isolated_nodes]

    def get_unreacted_monomer_composition(self) -> Dict[str, int]:
        """Count the number of unreacted monomers of each type.

        Returns:
            A dictionary mapping monomer type to its unreacted count.

        """
        unreacted = self.get_unreacted_monomers()
        types = [m["monomer_type"] for m in unreacted if "monomer_type" in m]
        return dict(Counter(types))

    def get_average_molecular_weights(self) -> Dict[str, float]:
        """Calculate Mn, Mw, and PDI for the polymer mixture.

        Returns:
            A dictionary with keys 'Mn', 'Mw', and 'PDI'. Returns zero values
            if no polymers were formed.

        """
        polymers = self.get_polymers()
        if not polymers:
            return {"Mn": 0.0, "Mw": 0.0, "PDI": 0.0}

        molar_masses = np.array([p.molecular_weight for p in polymers])

        total_mass = np.sum(molar_masses)
        num_chains = len(molar_masses)

        mn = total_mass / num_chains
        mw = np.sum(molar_masses**2) / total_mass if total_mass > 0 else 0.0
        pdi = mw / mn if mn > 0 else 0.0

        return {"Mn": mn, "Mw": mw, "PDI": pdi}

    def summary(self) -> Dict[str, Any]:
        """Provide a summary of the simulation results.

        Returns:
            A dictionary containing key metrics like conversion, number of
            polymers, and average molecular weights.

        """
        if self.error:
            return {"error": self.error}

        polymers = self.get_polymers()
        mw_data = self.get_average_molecular_weights()

        summary_data = {
            "simulation_name": self.config.params.name,
            "final_conversion": self.metadata.get("final_conversion", 0.0)
            if self.metadata
            else 0.0,
            "reactions_completed": self.metadata.get("reactions_completed", 0)
            if self.metadata
            else 0.0,
            "num_polymers": len(polymers),
            "num_monomers_in_polymers": sum(p.num_monomers for p in polymers),
            "num_unreacted_monomers": len(self.get_unreacted_monomers()),
            "unreacted_composition": self.get_unreacted_monomer_composition(),
            "Mn": mw_data["Mn"],
            "Mw": mw_data["Mw"],
            "PDI": mw_data["PDI"],
            "wall_time_seconds": self.metadata.get("wall_time_seconds", 0.0)
            if self.metadata
            else 0.0,
        }
        return summary_data
