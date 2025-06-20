"""Tests for gradient polymer simulations."""

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


@pytest.fixture
def gradient_polymer_config() -> SimulationInput:
    """Provide a config for a linear gradient copolymer.
    
    Returns:
        Simulation configuration for gradient polymer formation.
    """
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", 
                count=5, 
                sites=[
                    SiteDef(type="I", status="ACTIVE")
                ]
            ),
            MonomerDef(
                name="MonomerA", 
                count=100, 
                sites=[
                    SiteDef(type="A_Head", status="DORMANT"),
                    SiteDef(type="A_Tail", status="DORMANT"),
                ]
            ),
            MonomerDef(
                name="MonomerB", 
                count=100, 
                sites=[
                    SiteDef(type="B_Head", status="DORMANT"),
                    SiteDef(type="B_Tail", status="DORMANT"),
                ]
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"},
                rate=1.0
            ),
            # Propagation A
            frozenset(["Radical", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"},
                rate=100.0
            ),
            # Propagation B
            frozenset(["Radical", "B_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"B_Tail": "Radical"},
                rate=10.0
            ),
            # Termination
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=10.0
            )
        },
        params=SimParams(random_seed=42, name="gradient_polymer")
    )


def test_simulation_run_gradient_polymer(gradient_polymer_config: SimulationInput) -> None:
    """Test that a gradient polymer simulation runs and produces a valid structure.
    
    Args:
        gradient_polymer_config: Gradient polymer configuration.
    """
    sim = Simulation(gradient_polymer_config)
    graph, meta = sim.run()

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert meta["reactions_completed"] <= gradient_polymer_config.params.max_reactions

    # Check for initiator and monomer types
    types = {attrs["monomer_type"] for _, attrs in graph.nodes(data=True)}
    assert "Initiator" in types
    assert "MonomerA" in types
    assert "MonomerB" in types

    # All nodes should have degree <= 2 for this linear case
    degrees = [d for _, d in graph.degree()]
    assert all(d <= 2 for d in degrees)

    # Check that both monomer types are incorporated into the polymer chains
    components = list(nx.connected_components(graph))
    polymer_chains = [c for c in components if len(c) > 1]
    
    has_monomer_a = False
    has_monomer_b = False
    for chain in polymer_chains:
        for node_id in chain:
            if graph.nodes[node_id]["monomer_type"] == "MonomerA":
                has_monomer_a = True
            if graph.nodes[node_id]["monomer_type"] == "MonomerB":
                has_monomer_b = True
    
    assert has_monomer_a
    assert has_monomer_b 


def test_visualization_gradient_polymer(gradient_polymer_config: SimulationInput, plot_path: Path) -> None:
    """Test the visualization functions for a gradient polymer.
    
    Args:
        gradient_polymer_config: Gradient polymer configuration.
    """
    sim = Simulation(gradient_polymer_config)
    graph, metadata = sim.run()

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph, 
        metadata, 
        title="Gradient Copolymer Analysis",
        save_path=plot_path / "gradient_polymer_dashboard.png"
    )
    assert dashboard_fig is not None
    cleanup_figure(dashboard_fig)

    # Test MWD plot
    mwd_fig = plot_molecular_weight_distribution(
        graph, 
        title="Gradient Copolymer MWD",
        save_path=plot_path / "gradient_polymer_mwd.png"
    )
    assert mwd_fig is not None
    cleanup_figure(mwd_fig)

    # Test polymer structure visualization, colored by monomer type
    structure_fig = visualize_polymer(
        graph, 
        title="Largest Gradient Copolymer Chain",
        component_index=0,
        node_color_by='monomer_type',
        save_path=plot_path / "gradient_polymer_structure.png"
    )
    assert structure_fig is not None
    cleanup_figure(structure_fig)

    # Verify files were created
    verify_visualization_outputs([
        plot_path / "gradient_polymer_dashboard.png",
        plot_path / "gradient_polymer_mwd.png",
        plot_path / "gradient_polymer_structure.png",
    ]) 