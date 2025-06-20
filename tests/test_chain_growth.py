"""Tests for chain-growth polymerization simulations."""

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
def chain_growth_config() -> SimulationInput:
    """Provide a config for a typical chain-growth radical polymerization.
    
    Returns:
        Simulation configuration for chain-growth polymerization.
    """
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", 
                count=1000, 
                sites=[
                    SiteDef(type="I", status="ACTIVE")
                ]
            ),
            MonomerDef(
                name="Monomer", 
                count=20000, 
                sites=[
                    SiteDef(type="Head", status="DORMANT"),
                    SiteDef(type="Tail", status="DORMANT"),
                ]
            ),
        ],
        reactions={
            frozenset(["I", "Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"Tail": "Radical"},
                rate=1.0
            ),
            frozenset(["Radical", "Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"Tail": "Radical"},
                rate=100.0
            ),
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=50.0
            )
        },
        params=SimParams(max_reactions=18000, random_seed=101)
    )


def test_simulation_run_chain_growth(chain_growth_config: SimulationInput) -> None:
    """Test that a chain-growth simulation runs and produces a graph.
    
    Args:
        chain_growth_config: Chain-growth polymerization configuration.
    """
    sim = Simulation(chain_growth_config)
    graph, meta = sim.run()

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == sum(m.count for m in chain_growth_config.monomers)
    assert graph.number_of_edges() > 0
    assert meta["reactions_completed"] <= chain_growth_config.params.max_reactions
    
    # Check for initiator and monomer types
    types = {attrs["monomer_type"] for _, attrs in graph.nodes(data=True)}
    assert "Initiator" in types
    assert "Monomer" in types

    # Initiators should have degree 1 (or 0 if unreacted)
    initiator_nodes = [
        n for n, d in graph.nodes(data=True) 
        if d['monomer_type'] == 'Initiator'
    ]
    for i_node in initiator_nodes:
        assert graph.degree(i_node) <= 1


def test_visualization_chain_growth(chain_growth_config: SimulationInput, plot_path: Path) -> None:
    """Test the visualization functions for a chain-growth polymer.
    
    Args:
        chain_growth_config: Chain-growth polymerization configuration.
    """
    sim = Simulation(chain_growth_config)
    graph, metadata = sim.run()

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph, 
        metadata, 
        title="Chain-Growth Polymer Analysis",
        save_path=plot_path / "chain_growth_dashboard.png"
    )
    assert dashboard_fig is not None
    cleanup_figure(dashboard_fig)

    # Test MWD plot
    mwd_fig = plot_molecular_weight_distribution(
        graph, 
        title="Chain-Growth Molecular Weight Distribution",
        save_path=plot_path / "chain_growth_mwd.png"
    )
    assert mwd_fig is not None
    cleanup_figure(mwd_fig)

    # Test polymer structure visualization for the largest component
    structure_fig = visualize_polymer(
        graph, 
        title="Largest Chain-Growth Polymer",
        component_index=0,
        save_path=plot_path / "chain_growth_structure_largest.png"
    )
    assert structure_fig is not None
    cleanup_figure(structure_fig)

    # Verify files were created
    verify_visualization_outputs([
        plot_path / "chain_growth_dashboard.png",
        plot_path / "chain_growth_mwd.png",
        plot_path / "chain_growth_structure_largest.png",
    ]) 