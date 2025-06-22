"""Integration tests for PolyMCsim polymer simulation."""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pytest

# Use non-interactive backend for tests
matplotlib.use("Agg")

from polymcsim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    create_analysis_dashboard,
    plot_molecular_weight_distribution,
    visualize_polymer,
)

# Create a directory for test outputs


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
                count=100,
                sites=[
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="B2_Monomer",
                count=100,
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ],
            ),
        ],
        reactions={frozenset(["A", "B"]): ReactionSchema(rate=1.0)},
        params=SimParams(max_reactions=150, random_seed=42),
    )


@pytest.fixture
def chain_growth_config() -> SimulationInput:
    """Provide a config for a typical chain-growth radical polymerization.

    Returns:
        Simulation configuration for chain-growth polymerization.

    """
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", count=10, sites=[SiteDef(type="I", status="ACTIVE")]
            ),
            MonomerDef(
                name="Monomer",
                count=200,
                sites=[
                    SiteDef(type="Head", status="DORMANT"),
                    SiteDef(type="Tail", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            frozenset(["I", "Head"]): ReactionSchema(
                activation_map={"Tail": "Radical"},
                rate=1.0,
            ),
            frozenset(["Radical", "Head"]): ReactionSchema(
                activation_map={"Tail": "Radical"},
                rate=100.0,
            ),
            frozenset(["Radical", "Radical"]): ReactionSchema(rate=50.0),
        },
        params=SimParams(max_reactions=180, random_seed=101),
    )


def test_simulation_run_step_growth(step_growth_config: SimulationInput) -> None:
    """Test that a step-growth simulation runs and produces a graph.

    Args:
        step_growth_config: Step-growth polymerization configuration.

    """
    sim = Simulation(step_growth_config)
    result = sim.run()
    graph, meta = result.graph, result.metadata

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 200
    assert graph.number_of_edges() > 0
    assert meta["reactions_completed"] <= step_growth_config.params.max_reactions

    # Check node attributes
    for node_id, attrs in graph.nodes(data=True):
        assert "monomer_type" in attrs
        assert attrs["monomer_type"] in ["A2_Monomer", "B2_Monomer"]

    # All nodes should have degree <= 2 for this linear case
    degrees = [d for _, d in graph.degree()]
    assert all(d <= 2 for d in degrees)


def test_simulation_run_chain_growth(chain_growth_config: SimulationInput) -> None:
    """Test that a chain-growth simulation runs and produces a graph.

    Args:
        chain_growth_config: Chain-growth polymerization configuration.

    """
    sim = Simulation(chain_growth_config)
    result = sim.run()
    graph, meta = result.graph, result.metadata

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 210
    assert graph.number_of_edges() > 0
    assert meta["reactions_completed"] <= chain_growth_config.params.max_reactions

    # Check for initiator and monomer types
    types = {attrs["monomer_type"] for _, attrs in graph.nodes(data=True)}
    assert "Initiator" in types
    assert "Monomer" in types

    # Initiators should have degree 1 (or 0 if unreacted)
    initiator_nodes = [
        n for n, d in graph.nodes(data=True) if d["monomer_type"] == "Initiator"
    ]
    for i_node in initiator_nodes:
        assert graph.degree(i_node) <= 1


def test_visualization_step_growth(
    step_growth_config: SimulationInput, plot_path: Path
) -> None:
    """Test visualization of a step-growth polymer.

    Args:
        step_growth_config: Configuration for a step-growth polymer.
        plot_path: Path to save the plot.

    """
    sim = Simulation(step_growth_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Step-Growth Polymer Analysis",
        save_path=plot_path / "step_growth_dashboard.png",
    )
    assert dashboard_fig is not None
    plt.close(dashboard_fig)

    # Test MWD plot
    mwd_fig = plot_molecular_weight_distribution(
        graph, title="Step-Growth MWD", save_path=plot_path / "step_growth_mwd.png"
    )
    assert mwd_fig is not None
    plt.close(mwd_fig)

    # Test polymer structure visualization
    structure_fig = visualize_polymer(
        graph,
        title="Step-Growth Structure",
        save_path=plot_path / "step_growth_structure.png",
    )
    assert structure_fig is not None
    plt.close(structure_fig)

    # Verify files were created
    assert os.path.exists(plot_path / "step_growth_dashboard.png")
    assert os.path.exists(plot_path / "step_growth_mwd.png")
    assert os.path.exists(plot_path / "step_growth_structure.png")


def test_visualization_chain_growth(
    chain_growth_config: SimulationInput, plot_path: Path
) -> None:
    """Test visualization of a chain-growth polymer.

    Args:
        chain_growth_config: Configuration for a chain-growth polymer.
        plot_path: Path to save the plot.

    """
    sim = Simulation(chain_growth_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Chain-Growth Polymer Analysis",
        save_path=plot_path / "chain_growth_dashboard.png",
    )
    assert dashboard_fig is not None
    plt.close(dashboard_fig)

    # Test MWD plot
    mwd_fig = plot_molecular_weight_distribution(
        graph, title="Chain-Growth MWD", save_path=plot_path / "chain_growth_mwd.png"
    )
    assert mwd_fig is not None
    plt.close(mwd_fig)

    # Test polymer structure visualization for the largest component
    structure_fig = visualize_polymer(
        graph,
        title="Largest Chain-Growth Polymer",
        component_index=0,
        save_path=plot_path / "chain_growth_structure_largest.png",
    )
    assert structure_fig is not None
    plt.close(structure_fig)

    # Verify files were created
    assert os.path.exists(plot_path / "chain_growth_dashboard.png")
    assert os.path.exists(plot_path / "chain_growth_mwd.png")
    assert os.path.exists(plot_path / "chain_growth_structure_largest.png")
