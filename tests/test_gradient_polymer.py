"""Tests for gradient polymer simulations."""

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
def gradient_polymer_config() -> SimulationInput:
    """Provide a config for a linear gradient copolymer.

    Returns:
        Simulation configuration for gradient polymer formation.

    """
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", count=5, sites=[SiteDef(type="I", status="ACTIVE")]
            ),
            MonomerDef(
                name="MonomerA",
                count=100,
                sites=[
                    SiteDef(type="A_Head", status="DORMANT"),
                    SiteDef(type="A_Tail", status="DORMANT"),
                ],
            ),
            MonomerDef(
                name="MonomerB",
                count=100,
                sites=[
                    SiteDef(type="B_Head", status="DORMANT"),
                    SiteDef(type="B_Tail", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "A_Head"]): ReactionSchema(
                activation_map={"A_Tail": "Radical"},
                rate=1.0,
            ),
            # Propagation A
            frozenset(["Radical", "A_Head"]): ReactionSchema(
                activation_map={"A_Tail": "Radical"},
                rate=100.0,
            ),
            # Propagation B
            frozenset(["Radical", "B_Head"]): ReactionSchema(
                activation_map={"B_Tail": "Radical"},
                rate=10.0,
            ),
            # Termination
            frozenset(["Radical", "Radical"]): ReactionSchema(rate=10.0),
        },
        params=SimParams(random_seed=42, name="gradient_polymer"),
    )


def test_simulation_run_gradient_polymer(
    gradient_polymer_config: SimulationInput,
) -> None:
    """Test that a gradient polymer simulation runs and produces a valid structure.

    Args:
        gradient_polymer_config: Gradient polymer configuration.

    """
    sim = Simulation(gradient_polymer_config)
    result = sim.run()
    graph, meta = result.graph, result.metadata

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


def test_visualization_gradient_polymer(
    gradient_polymer_config: SimulationInput, plot_path: Path
) -> None:
    """Test visualization of a gradient polymer.

    Args:
        gradient_polymer_config: Configuration for a gradient polymer.
        plot_path: Path to save the plot.

    """
    sim = Simulation(gradient_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Gradient Copolymer Analysis",
        save_path=plot_path / "gradient_polymer_dashboard.png",
    )
    assert dashboard_fig is not None

    # Test MWD plot
    mwd_fig = plot_molecular_weight_distribution(
        graph, save_path=plot_path / "gradient_mwd.png"
    )
    assert mwd_fig is not None

    # Test polymer structure visualization, colored by monomer type
    structure_fig = visualize_polymer(
        graph,
        component_index=0,  # Largest
        node_color_by="monomer_type",
        save_path=plot_path / "gradient_structure_colored.png",
    )
    assert structure_fig is not None

    # Verify files were created
    verify_visualization_outputs(
        [
            plot_path / "gradient_polymer_dashboard.png",
            plot_path / "gradient_mwd.png",
            plot_path / "gradient_structure_colored.png",
        ]
    )
