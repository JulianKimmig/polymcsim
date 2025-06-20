"""Tests for radical polymerization simulations, including MWD checks."""

import numpy as np
import networkx as nx
import pytest

from polysim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef
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
            mass = sum(graph.nodes[node]['molar_mass'] for node in component)
            molar_masses.append(mass)

    if not molar_masses:
        return 0.0

    molar_masses = np.array(molar_masses)

    # Calculate number-average (Mn) and weight-average (Mw) molar mass
    total_mass = np.sum(molar_masses)
    total_chains = len(molar_masses)
    
    Mn = total_mass / total_chains
    Mw = np.sum(molar_masses**2) / total_mass
    
    if Mn == 0:
        return 0.0
        
    pdi = Mw / Mn
    return pdi


@pytest.fixture
def mma_radical_config() -> SimulationInput:
    """Configuration for a radical polymerization of MMA."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", 
                count=20, 
                molar_mass=64.0,  # Generic initiator
                sites=[SiteDef(type="I", status="ACTIVE")]
            ),
            MonomerDef(
                name="MMA", 
                count=2000, 
                molar_mass=100.1,  # Methyl Methacrylate
                sites=[
                    SiteDef(type="Vinyl", status="DORMANT"),
                    SiteDef(type="RadicalSite", status="DORMANT"),
                ]
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "Vinyl"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"RadicalSite": "Radical"},
                rate=1.0
            ),
            # Propagation
            frozenset(["Radical", "Vinyl"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"RadicalSite": "Radical"},
                rate=200.0
            ),
            # Termination (Combination)
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=100.0
            )
        },
        params=SimParams(max_reactions=1950, random_seed=42)
    )


def test_mma_radical_polymerization_mwd(mma_radical_config: SimulationInput):
    """
    Test radical polymerization of MMA and check if the PDI is within the
    theoretically expected range for this type of polymerization (~1.5-2.0).
    """
    sim = Simulation(mma_radical_config)
    graph, _ = sim.run()
    
    pdi = calculate_pdi(graph)
    
    # For radical polymerization, PDI is theoretically between 1.5 and 2.0.
    # We allow a wider range to account for simulation stochasticity.
    assert 1.4 < pdi < 2.5, f"PDI of {pdi:.2f} is outside the expected range for radical polymerization." 