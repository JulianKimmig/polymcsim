"""Tests for step-growth polymerization simulations."""

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
def step_growth_config() -> SimulationInput:
    """Provide a config for a typical step-growth (A2 + B2) polymerization.
    
    Returns:
        Simulation configuration for step-growth polymerization.
    """
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="A2_Monomer", 
                count=1000, 
                sites=[
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                ]
            ),
            MonomerDef(
                name="B2_Monomer", 
                count=1000, 
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ]
            ),
        ],
        reactions={
            frozenset(["A", "B"]): ReactionSchema(
                site1_final_status="CONSUMED",  
                site2_final_status="CONSUMED",
                rate=1.0
            )
        },
        params=SimParams(max_reactions=1500, random_seed=42)
    )


def test_simulation_run_step_growth(step_growth_config: SimulationInput) -> None:
    """Test that a step-growth simulation runs and produces a graph.
    
    Args:
        step_growth_config: Step-growth polymerization configuration.
    """
    sim = Simulation(step_growth_config)
    graph, meta = sim.run()

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 2000
    assert graph.number_of_edges() > 0
    assert meta["reactions_completed"] <= step_growth_config.params.max_reactions
    
    # Check node attributes
    for node_id, attrs in graph.nodes(data=True):
        assert "monomer_type" in attrs
        assert attrs["monomer_type"] in ["A2_Monomer", "B2_Monomer"]

    # All nodes should have degree <= 2 for this linear case
    degrees = [d for _, d in graph.degree()]
    assert all(d <= 2 for d in degrees)


def test_visualization_step_growth(step_growth_config: SimulationInput, plot_path: Path) -> None:
    """Test the visualization functions for a step-growth polymer.
    
    Args:
        step_growth_config: Step-growth polymerization configuration.
    """
    sim = Simulation(step_growth_config)
    graph, metadata = sim.run()

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph, 
        metadata, 
        title="Step-Growth Polymer Analysis",
        save_path=plot_path / "step_growth_dashboard.png"
    )
    assert dashboard_fig is not None
    cleanup_figure(dashboard_fig)

    # Test MWD plot
    mwd_fig = plot_molecular_weight_distribution(
        graph, 
        title="Step-Growth Molecular Weight Distribution",
        save_path=plot_path / "step_growth_mwd.png"
    )
    assert mwd_fig is not None
    cleanup_figure(mwd_fig)

    # Test polymer structure visualization of the largest polymer
    structure_fig = visualize_polymer(
        graph, 
        component_index=0,
        title="Largest Step-Growth Polymer",
        save_path=plot_path / "step_growth_structure.png"
    )
    assert structure_fig is not None
    cleanup_figure(structure_fig)

    # Verify files were created
    verify_visualization_outputs([
        plot_path / "step_growth_dashboard.png",
        plot_path / "step_growth_mwd.png",
        plot_path / "step_growth_structure.png",
    ]) 