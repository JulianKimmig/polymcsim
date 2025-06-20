"""Tests for star polymer simulations."""

import networkx as nx
import pytest
from pathlib import Path

from polysim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    plot_chain_length_distribution,
    visualize_polymer,
    create_analysis_dashboard,
    plot_molecular_weight_distribution,
)
from conftest import cleanup_figure, verify_visualization_outputs


def test_star_polymer_generation(plot_path: Path) -> None:
    """Generate a star polymer using a multifunctional core with linear arms.
    
    Star polymers have a central core with multiple linear arms radiating outward.
    """
    # Star polymer: multifunctional core + linear arms
    # Use living polymerization approach to prevent inter-star connections
    n_core = 20  # Multifunctional cores (4 sites each)
    n_arms = 200  # Linear monomers for arms (2 sites each)
    sim_input = SimulationInput(
        monomers=[
            MonomerDef(
                name="Core", 
                count=n_core, 
                sites=[
                    SiteDef(type="I", status="ACTIVE")  # Initiator sites
                    for _ in range(4)
                ]
            ),
            MonomerDef(
                name="Arm", 
                count=n_arms, 
                sites=[
                    SiteDef(type="M", status="DORMANT"),  # Monomer sites (dormant)
                    SiteDef(type="M", status="DORMANT"),
                ]
            ),
        ],
        reactions={
            frozenset(["I", "M"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"M": "R"},  # Activate the other site as radical
                rate=1.0  # Lower initiation rate
            ),
            frozenset(["R", "M"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"M": "R"},  # Propagate the radical
                rate=200.0  # Higher propagation rate to encourage long arms
            )
        },
        params=SimParams(max_reactions=1000, random_seed=2024, name="star_polymer")
    )

    sim = Simulation(sim_input)
    graph, metadata = sim.run()

    # --- Verification ---
    assert isinstance(graph, nx.Graph), "Simulation did not return a valid graph"
    assert graph.number_of_nodes() > 0, "Graph is empty after simulation"
    
    # Check that a large, single component formed
    components = list(nx.connected_components(graph))
    largest_component_size = len(components[0]) if components else 0
    assert largest_component_size > (n_core + n_arms) * 0.1, "Expected a large star polymer structure"
    
    # Analyze star polymer characteristics
    subgraph = graph.subgraph(components[0])
    
    # Find potential star centers (nodes with degree >= 3)
    potential_stars = [n for n, d in subgraph.degree() if d >= 3]
    
    # Count actual star structures (core with multiple arms)
    star_centers = [
        node for node in potential_stars 
        if subgraph.nodes[node]["monomer_type"] == "Core" 
        and len(list(subgraph.neighbors(node))) >= 2
    ]
    
    # Assertions
    assert len(star_centers) > 0, "Expected at least one star structure to form"

    # --- Visualization ---
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Star Polymer (Core+Arms) Analysis",
        save_path=plot_path / "star_polymer_dashboard.png"
    )
    assert dashboard_fig is not None
    cleanup_figure(dashboard_fig)

    mwd_fig = plot_molecular_weight_distribution(
        graph,
        log_scale=True,
        title="Star Polymer Molecular Weight Distribution",
        save_path=plot_path / "star_polymer_mwd.png"
    )
    assert mwd_fig is not None
    cleanup_figure(mwd_fig)

    structure_fig = visualize_polymer(
        graph,
        component_index=0,
        title="Largest Star Polymer Structure",
        layout='kamada_kawai',
        save_path=plot_path / "star_polymer_structure.png"  
    )
    assert structure_fig is not None
    cleanup_figure(structure_fig)

    verify_visualization_outputs([
        plot_path / "star_polymer_dashboard.png",
        plot_path / "star_polymer_mwd.png",
        plot_path / "star_polymer_structure.png"
    ]) 