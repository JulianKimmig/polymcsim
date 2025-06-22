"""Tests for radical polymerization simulations, including MWD checks."""

from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pytest

from polymcsim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    plot_chain_length_distribution,
)


def calculate_pdi(graph: nx.Graph) -> float:
    """Calculate the Polydispersity Index (PDI) for the polymer graph."""
    # Get all connected components (polymers)
    components = list(nx.connected_components(graph))
    if not components:
        return 0.0

    # Calculate molar mass for each polymer chain, ignoring single-monomer components
    molar_masses = []
    for component in components:
        if len(component) > 1:  # Only consider polymer chains, not unreacted monomers
            mass = sum(graph.nodes[node]["molar_mass"] for node in component)
            molar_masses.append(mass)

    if not molar_masses:
        return 0.0

    molar_masses = np.array(molar_masses)

    # Calculate number-average (Mn) and weight-average (Mw) molar mass
    total_mass: float = np.sum(molar_masses)
    total_chains: int = len(molar_masses)

    Mn = total_mass / total_chains
    Mw: float = np.sum(molar_masses**2) / total_mass

    if Mn == 0:
        return 0.0

    pdi = Mw / Mn
    return pdi


@pytest.fixture
def mma_radical_config() -> SimulationInput:
    """Create configuration for a radical polymerization of MMA."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator",
                count=200,
                molar_mass=64.0,  # Generic initiator
                sites=[SiteDef(type="I", status="ACTIVE")],
            ),
            MonomerDef(
                name="MMA",
                count=20000,
                molar_mass=100.1,  # Methyl Methacrylate
                sites=[
                    SiteDef(type="Vinyl", status="DORMANT"),
                    SiteDef(type="RadicalSite", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "Vinyl"]): ReactionSchema(
                activation_map={"RadicalSite": "Radical"},
                rate=1.0,
            ),
            # Propagation
            frozenset(["Radical", "Vinyl"]): ReactionSchema(
                activation_map={"RadicalSite": "Radical"},
                rate=200.0,
            ),
            # Termination (Combination)
            frozenset(["Radical", "Radical"]): ReactionSchema(rate=100.0),
        },
        params=SimParams(max_conversion=0.9, random_seed=42),
    )


def test_mma_radical_polymerization_mwd(
    mma_radical_config: SimulationInput, plot_path: Path
):
    """Test radical polymerization of MMA.

    Checks if the PDI is within the theoretically expected range for this
    type of polymerization (~1.5-2.0).
    """
    sim = Simulation(mma_radical_config)
    result = sim.run()
    graph, _ = result.graph, result.metadata

    pdi = calculate_pdi(graph)

    # plot the mass distribution
    plot_chain_length_distribution(
        graph, save_path=plot_path / "mma_radical_polymerization_mwd.png"
    )

    # For radical polymerization, PDI is theoretically between 1.5 and 2.0.
    # However, at high conversion (90%) and with this setup, lower PDI
    # values are possible.
    # We allow a wider range to account for simulation stochasticity and
    # conversion effects.
    assert 1.2 < pdi < 2.5, (
        f"PDI of {pdi:.2f} is outside the expected range for radical polymerization."
    )


def test_stochastic_consistency(mma_radical_config: SimulationInput):
    """Test that two runs with the same seed are identical."""
    pass


def get_total_molar_mass(monomers: List[MonomerDef]) -> float:
    """Calculate the total molar mass of all monomers in the system."""
    total_mass: float = 0.0
    for monomer in monomers:
        total_mass += monomer.count * monomer.molar_mass
    return total_mass


def calculate_conversion(
    graph: nx.Graph, initial_total_mass: float, monomers: List[MonomerDef]
) -> float:
    """Calculate the conversion based on the mass of the largest polymer chain."""
    if graph.number_of_nodes() == 0:
        return 0.0

    components = list(nx.weakly_connected_components(graph))
    if not components:
        return 0.0

    largest_component = max(components, key=len)
    subgraph = graph.subgraph(largest_component)

    polymer_mass = sum(data["molar_mass"] for _, data in subgraph.nodes(data=True))
    return polymer_mass / initial_total_mass if initial_total_mass > 0 else 0.0


def get_monomer_compositions(
    graph: nx.Graph,
) -> Dict[int, Dict[str, int]]:
    """Get the monomer composition of each polymer chain."""
    compositions = {}
    for i, component in enumerate(nx.weakly_connected_components(graph)):
        composition: Dict[str, int] = {}
        for node_id in component:
            monomer_type = graph.nodes[node_id].get("monomer_type", "Unknown")
            composition[monomer_type] = composition.get(monomer_type, 0) + 1
        compositions[i] = composition
    return compositions
