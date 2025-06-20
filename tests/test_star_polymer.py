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
    visualize_polymer,
    create_analysis_dashboard,
    plot_molecular_weight_distribution
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
                    SiteDef(type="M_in", status="DORMANT"),  # Monomer sites (dormant)
                    SiteDef(type="M_out", status="DORMANT"),
                ]
            ),
        ],
        reactions={
            frozenset(["I", "M_in"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"M_out": "R_arm"},  # Activate arm monomer
                rate=1.0
            ),
            frozenset(["R_arm", "M_in"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"M_out": "R_arm"},  # Propagate the radical
                rate=200.0
            ),
            frozenset(["R_arm", "R_arm"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=0.1,
            )
        },
        params=SimParams(max_reactions=400, random_seed=2024, name="star_polymer")
    )

    sim = Simulation(sim_input)
    graph, metadata = sim.run()

    # --- Verification ---
    assert isinstance(graph, nx.Graph), "Simulation did not return a valid graph"
    assert graph.number_of_nodes() > 0, "Graph is empty after simulation"
    
    components = sorted(list(nx.connected_components(graph)), key=len, reverse=True)
    assert components, "No components found in the graph."
    largest_component_size = len(components[0])

    # We expect a large structure, but not necessarily everything in one component,
    # as multiple stars can form. Let's check if the average component size is reasonable.
    num_polymers = sum(1 for c in components if len(c) > 1)
    assert num_polymers > 0, "No polymers were formed."
    
    avg_size = sum(len(c) for c in components) / len(components)
    
    assert avg_size > 2, "Polymer chains are not growing."

    # Check that the largest polymer is a star (one core, many arms)
    largest_comp = graph.subgraph(components[0])
    
    core_nodes = [n for n, d in largest_comp.nodes(data=True) if d['monomer_type'] == 'Core']
    assert len(core_nodes) >= 1, "Largest polymer should contain at least one core."

    # In a perfect star, core nodes have high degree. Let's check this.
    degrees = [d for n, d in largest_comp.degree() if largest_comp.nodes[n]['monomer_type'] == 'Core']
    assert degrees, "No core nodes found in the largest component to check degrees."

    # Check that at least one core has multiple arms attached
    has_star_structure = False
    for node in core_nodes:
        arm_neighbors = sum(1 for neighbor in largest_comp.neighbors(node) 
                            if largest_comp.nodes[neighbor]['monomer_type'] == 'Arm')
        if arm_neighbors > 2:
            has_star_structure = True
            break
    
    assert has_star_structure, "No core node found with more than 2 arm neighbors."

    # --- Visualization ---
    dashboard_fig = create_analysis_dashboard(
        graph, 
        metadata, 
        title="Star Polymer Analysis",
        save_path=plot_path / "star_polymer_dashboard.png"
    )
    assert dashboard_fig is not None

    mwd_fig = plot_molecular_weight_distribution(
        graph,
        title="Star Polymer MWD",
        save_path=plot_path / "star_polymer_mwd.png"
    )
    assert mwd_fig is not None

    structure_fig = visualize_polymer(
        graph,
        title="Star Polymer Structure",
        save_path=plot_path / "star_polymer_structure.png"
    )
    assert structure_fig is not None

    verify_visualization_outputs([
        plot_path / "star_polymer_dashboard.png",
        plot_path / "star_polymer_mwd.png",
        plot_path / "star_polymer_structure.png"
    ]) 