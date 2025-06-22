"""Tests for step-growth polymerization simulations."""

from pathlib import Path

import networkx as nx
import pytest
from conftest import verify_visualization_outputs

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
                ],
            ),
            MonomerDef(
                name="B2_Monomer",
                count=1000,
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ],
            ),
        ],
        reactions={frozenset(["A", "B"]): ReactionSchema(rate=1.0)},
        params=SimParams(max_reactions=1500, random_seed=42),
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
    create_analysis_dashboard(
        graph,
        metadata,
        save_path=str(plot_path / "step_growth_dashboard.png"),
    )

    # Test MWD plot
    plot_molecular_weight_distribution(
        graph, save_path=str(plot_path / "step_growth_mwd.png")
    )

    # Test polymer structure visualization of the largest polymer
    visualize_polymer(
        graph,
        save_path=str(plot_path / "step_growth_structure.png"),
    )

    # Verify files were created
    verify_visualization_outputs(
        [
            str(plot_path / "step_growth_dashboard.png"),
            str(plot_path / "step_growth_mwd.png"),
            str(plot_path / "step_growth_structure.png"),
        ]
    )


def test_linear_polymer_mwd(tmp_path, step_growth_config: SimulationInput):
    """Test the MWD of a linear polymer."""
    step_growth_config.params.max_reactions = 14
    sim = Simulation(step_growth_config)
    result = sim.run()
    graph = result.graph

    # Test MWD plotting
    mwd_path = tmp_path / "mwd.png"
    plot_molecular_weight_distribution(graph, save_path=str(mwd_path))
    assert mwd_path.with_suffix(".png").exists()


def test_visualize_linear_polymer(tmp_path, step_growth_config: SimulationInput):
    """Test the visualization of a linear polymer."""
    step_growth_config.params.max_reactions = 10
    sim = Simulation(step_growth_config)
    result = sim.run()
    graph = result.graph

    # Test visualization
    viz_path = tmp_path / "polymer.png"
    visualize_polymer(
        graph,
        save_path=str(viz_path),
    )
    assert viz_path.with_suffix(".png").exists()
