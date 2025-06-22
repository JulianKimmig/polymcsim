"""Tests for star polymer simulations."""

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
def star_polymer_config() -> SimulationInput:
    """Create a star polymer configuration."""
    n_core = 5  # Multifunctional cores (4 sites each)
    n_arms = 200  # Linear monomers for arms (2 sites each)
    sim_input = SimulationInput(
        monomers=[
            MonomerDef(
                name="Core",
                count=n_core,
                sites=[
                    SiteDef(type="I", status="ACTIVE")  # Initiator sites
                    for _ in range(4)
                ],
            ),
            MonomerDef(
                name="Arm",
                count=n_arms,
                sites=[
                    SiteDef(type="M_in", status="DORMANT"),  # Monomer sites (dormant)
                    SiteDef(type="M_out", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            frozenset(["I", "M_in"]): ReactionSchema(
                activation_map={"M_out": "R_arm"},  # Activate arm monomer
                rate=5.0,
            ),
            frozenset(["R_arm", "M_in"]): ReactionSchema(
                activation_map={"M_out": "R_arm"},  # Propagate the radical
                rate=1.0,
            ),
            frozenset(["R_arm", "R_arm"]): ReactionSchema(
                rate=0.1,
            ),
        },
        params=SimParams(max_reactions=400, random_seed=2024, name="star_polymer"),
    )
    return sim_input


def test_star_polymer_generation(
    plot_path: Path, star_polymer_config: SimulationInput
) -> None:
    """Generate a star polymer using a multifunctional core with linear arms.

    Star polymers have a central core with multiple linear arms radiating outward.
    """
    # Star polymer: multifunctional core + linear arms
    # Use living polymerization approach to prevent inter-star connections

    sim = Simulation(star_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # --- Verification ---
    assert isinstance(graph, nx.Graph), "Simulation did not return a valid graph"
    assert graph.number_of_nodes() > 0, "Graph is empty after simulation"

    components = sorted(list(nx.connected_components(graph)), key=len, reverse=True)
    assert components, "No components found in the graph."

    # We expect a large structure, but not necessarily everything in one
    # component, as multiple stars can form. Let's check if the average
    # component size is reasonable.
    num_polymers = sum(1 for c in components if len(c) > 1)
    assert num_polymers > 0, "No polymers were formed."
    total_polymer_nodes = sum(len(c) for c in components if len(c) > 1)

    avg_size = total_polymer_nodes / num_polymers

    assert avg_size > 2, "Polymer chains are not growing."

    # Check that the largest polymer is a star (one core, many arms)
    largest_comp = graph.subgraph(components[0])

    core_nodes = [
        n for n, d in largest_comp.nodes(data=True) if d["monomer_type"] == "Core"
    ]
    assert len(core_nodes) >= 1, "Largest polymer should contain at least one core."

    # In a perfect star, core nodes have high degree. Let's check this.
    degrees = [
        d
        for n, d in largest_comp.degree()
        if largest_comp.nodes[n]["monomer_type"] == "Core"
    ]
    assert degrees, "No core nodes found in the largest component to check degrees."

    # Check that at least one core has multiple arms attached
    has_star_structure = False
    for node in core_nodes:
        arm_neighbors = sum(
            1
            for neighbor in largest_comp.neighbors(node)
            if largest_comp.nodes[neighbor]["monomer_type"] == "Arm"
        )
        if arm_neighbors > 2:
            has_star_structure = True
            break

    assert has_star_structure, "No core node found with more than 2 arm neighbors."

    # --- Visualization ---
    create_analysis_dashboard(
        graph,
        metadata,
        save_path=str(plot_path / "star_polymer_dashboard.png"),
    )

    plot_molecular_weight_distribution(
        graph, save_path=str(plot_path / "star_polymer_mwd.png")
    )

    visualize_polymer(
        graph,
        save_path=str(plot_path / "star_polymer_structure.png"),
    )

    verify_visualization_outputs(
        [
            str(plot_path / "star_polymer_dashboard.png"),
            str(plot_path / "star_polymer_mwd.png"),
            str(plot_path / "star_polymer_structure.png"),
        ]
    )

    assert all(d >= 1 for d in degrees)
    # The core monomer should have a degree equal to its functionality
    assert degrees[0] == 4


def test_star_polymer_mwd(tmp_path, star_polymer_config: SimulationInput):
    """Test the MWD of a star polymer."""
    # ... existing code ...
    sim = Simulation(star_polymer_config)
    result = sim.run()
    graph = result.graph

    # Test MWD plotting
    mwd_path = tmp_path / "mwd.png"
    plot_molecular_weight_distribution(graph, save_path=str(mwd_path))
    assert mwd_path.with_suffix(".png").exists()


def test_visualize_star_polymer(tmp_path, star_polymer_config: SimulationInput):
    """Test the visualization of a star polymer."""
    # ... existing code ...
    sim = Simulation(star_polymer_config)
    result = sim.run()
    graph = result.graph

    # Test visualization
    viz_path = tmp_path / "polymer.png"
    visualize_polymer(
        graph,
        save_path=str(viz_path),
    )
    assert viz_path.with_suffix(".png").exists()


def test_analysis_dashboard_star_polymer(
    tmp_path, star_polymer_config: SimulationInput
):
    """Test the analysis dashboard for a star polymer."""
    # ... existing code ...
    sim = Simulation(star_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Test dashboard generation
    dashboard_path = tmp_path / "dashboard.png"
    create_analysis_dashboard(graph, metadata, save_path=str(dashboard_path))
    assert dashboard_path.exists()
