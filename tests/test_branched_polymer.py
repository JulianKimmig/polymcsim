"""Tests for branched polymer simulations."""

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
    plot_branching_analysis,
    plot_molecular_weight_distribution,
    visualize_polymer,
)


@pytest.fixture
def branched_polymer_config() -> SimulationInput:
    """Provide a config for a branched polymer with trifunctional monomers.

    Returns:
        Simulation configuration for branched polymer formation.

    """
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", count=10, sites=[SiteDef(type="I", status="ACTIVE")]
            ),
            MonomerDef(
                name="LinearMonomer",
                count=200,
                sites=[
                    SiteDef(type="A_Head", status="DORMANT"),
                    SiteDef(type="A_Tail", status="DORMANT"),
                ],
            ),
            MonomerDef(
                name="BranchMonomer",
                count=50,
                sites=[
                    SiteDef(type="B_Head", status="DORMANT"),
                    SiteDef(type="B_Tail", status="DORMANT"),
                    SiteDef(
                        type="B_Branch", status="DORMANT"
                    ),  # Third site for branching
                ],
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "A_Head"]): ReactionSchema(
                activation_map={"A_Tail": "Radical"},
                rate=1.0,
            ),
            # Propagation on linear monomer
            frozenset(["Radical", "A_Head"]): ReactionSchema(
                activation_map={"A_Tail": "Radical"},
                rate=100.0,
            ),
            # Propagation on branch monomer (head)
            frozenset(["Radical", "B_Head"]): ReactionSchema(
                activation_map={"B_Tail": "Radical"},
                rate=80.0,
            ),
            # Branching reaction (branch site)
            frozenset(["Radical", "B_Branch"]): ReactionSchema(
                activation_map={"B_Tail": "Radical"},
                rate=60.0,
            ),
            # Termination
            frozenset(["Radical", "Radical"]): ReactionSchema(rate=20.0),
        },
        params=SimParams(max_reactions=5000, random_seed=123, name="branched_polymer"),
    )


def test_simulation_run_branched_polymer(
    branched_polymer_config: SimulationInput,
) -> None:
    """Test that a branched polymer simulation runs and produces a valid structure.

    Args:
        branched_polymer_config: Branched polymer configuration.

    """
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph, meta = result.graph, result.metadata

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert meta["reactions_completed"] <= branched_polymer_config.params.max_reactions

    # Check for all monomer types
    types = {attrs["monomer_type"] for _, attrs in graph.nodes(data=True)}
    assert "Initiator" in types
    assert "LinearMonomer" in types
    assert "BranchMonomer" in types

    # Check for branching - some nodes should have degree > 2
    degrees = [d for _, d in graph.degree()]
    max_degree = max(degrees)
    assert max_degree > 2, f"Expected branching but max degree was {max_degree}"

    # Count nodes with degree > 2 (branch points)
    branch_points = sum(1 for d in degrees if d > 2)
    assert branch_points > 0, "Expected at least one branch point"

    # Check that branch monomers are incorporated
    components = list(nx.connected_components(graph))
    polymer_chains = [c for c in components if len(c) > 1]

    has_linear_monomer = False
    has_branch_monomer = False
    for chain in polymer_chains:
        for node_id in chain:
            if graph.nodes[node_id]["monomer_type"] == "LinearMonomer":
                has_linear_monomer = True
            if graph.nodes[node_id]["monomer_type"] == "BranchMonomer":
                has_branch_monomer = True

    assert has_linear_monomer
    assert has_branch_monomer


def test_visualization_branched_polymer(
    branched_polymer_config: SimulationInput, plot_path: Path
) -> None:
    """Test visualization of a branched polymer.

    Args:
        branched_polymer_config: Configuration for a branched polymer.
        plot_path: Path to save the plot.

    """
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Create a dashboard for comprehensive analysis
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Branched Polymer Analysis Dashboard",
        save_path=plot_path / "branched_polymer_dashboard.png",
    )
    assert dashboard_fig is not None

    # Test individual plots as well
    structure_fig = visualize_polymer(
        graph,
        title="Largest Branched Polymer",
        component_index=0,
        node_outline_color="darkred",
        save_path=plot_path / "branched_polymer_structure.png",
    )
    assert structure_fig is not None

    mwd_fig = plot_molecular_weight_distribution(
        graph,
        title="Branched Polymer MWD",
        log_scale=True,
        save_path=plot_path / "branched_polymer_mwd.png",
    )
    assert mwd_fig is not None

    branching_fig = plot_branching_analysis(
        graph,
        title="Branched Polymer Branching Analysis",
        save_path=plot_path / "branched_polymer_branching.png",
    )
    assert branching_fig is not None

    # Verify files were created
    verify_visualization_outputs(
        [
            plot_path / "branched_polymer_dashboard.png",
            plot_path / "branched_polymer_structure.png",
            plot_path / "branched_polymer_mwd.png",
            plot_path / "branched_polymer_branching.png",
        ]
    )


def test_hyperbranched_polymer_generation(plot_path: Path) -> None:
    """Generate a hyperbranched polymer using A2 + B4 monomers.

    Checks for high branching and many terminal groups.
    Reference: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.894096/full
    """
    # A2 + B4 system: classic for hyperbranched polymers
    # Use stoichiometric imbalance to ensure terminal groups remain
    n_A2 = 80  # 160 A sites
    n_B4 = 60  # 240 B sites (excess B to create terminal groups)
    sim_input = SimulationInput(
        monomers=[
            MonomerDef(
                name="A2",
                count=n_A2,
                sites=[
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="B4",
                count=n_B4,
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ],
            ),
        ],
        reactions={
            frozenset(["A", "B"]): ReactionSchema(rate=1.0),
            frozenset(["A", "A"]): ReactionSchema(rate=0.2),
        },
        params=SimParams(
            max_reactions=300, random_seed=2024, name="hyperbranched_polymer"
        ),
    )

    sim = Simulation(sim_input)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Check that the largest component is highly branched
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest)
    degrees = [d for _, d in subgraph.degree()]
    n_branch_points = sum(1 for d in degrees if d >= 3)
    n_terminal = sum(1 for d in degrees if d == 1)
    avg_degree = sum(degrees) / len(degrees)

    # Hyperbranched polymers should have many branch points and terminal groups
    assert n_branch_points > 0, "Expected branch points in hyperbranched polymer"
    assert n_terminal > 0, "Expected terminal groups in hyperbranched polymer"
    assert avg_degree > 2.0, f"Expected average degree > 2, got {avg_degree}"

    # Test visualization
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Hyperbranched Polymer Analysis",
        save_path=plot_path / "hyperbranched_dashboard.png",
    )
    assert dashboard_fig is not None

    structure_fig = visualize_polymer(
        graph,
        title="Hyperbranched Structure",
        component_index=0,
        save_path=plot_path / "hyperbranched_structure.png",
    )
    assert structure_fig is not None

    verify_visualization_outputs(
        [
            plot_path / "hyperbranched_dashboard.png",
            plot_path / "hyperbranched_structure.png",
        ]
    )


def test_dendrimer_like_structure(plot_path: Path) -> None:
    """Generate a dendrimer-like structure using a core and branching monomers.

    A -> B-B reactions from a central core.
    """
    init_site_count = 4
    sim_input = SimulationInput(
        monomers=[
            # Central core with 4 reactive 'A' sites
            MonomerDef(
                name="Core",
                count=1,
                sites=[
                    SiteDef(type="A", status="ACTIVE") for _ in range(init_site_count)
                ],
            ),
            # AB2 monomer: one 'A' to react with core, two 'B's for next gen
            MonomerDef(
                name="B1",
                count=init_site_count,
                sites=[
                    SiteDef(type="B1_Head", status="ACTIVE"),
                    SiteDef(type="B1_Tail", status="ACTIVE"),
                    SiteDef(type="B1_Tail", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="B2",
                count=init_site_count * 2,
                sites=[
                    SiteDef(type="B2_Head", status="ACTIVE"),
                    SiteDef(type="B2_Tail", status="ACTIVE"),
                    SiteDef(type="B2_Tail", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="B3",
                count=init_site_count * 2 * 2,
                sites=[
                    SiteDef(type="B3_Head", status="ACTIVE"),
                    SiteDef(type="B3_Tail", status="ACTIVE"),
                    SiteDef(type="B3_Tail", status="ACTIVE"),
                ],
            ),
        ],
        reactions={
            # Core reacts with the head of the AB2 monomer
            frozenset(["A", "B1_Head"]): ReactionSchema(
                rate=10.0,
            ),
            frozenset(["B1_Tail", "B2_Head"]): ReactionSchema(
                rate=10.0,
            ),
            frozenset(["B2_Tail", "B3_Head"]): ReactionSchema(
                rate=10.0,
            ),
        },
        params=SimParams(random_seed=42, name="dendrimer_structure"),
    )

    sim = Simulation(sim_input)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    assert graph.number_of_nodes() > 1
    assert metadata["reactions_completed"] > 0

    # The structure should be one single component centered around the core
    assert nx.number_connected_components(graph) == 1

    # Find the core node
    core_node = [n for n, d in graph.nodes(data=True) if d["monomer_type"] == "Core"][0]
    assert graph.degree(core_node) <= 4

    # Check for generational growth (nodes at distance 1, 2, etc. from core)
    distances = nx.shortest_path_length(graph, source=core_node)
    max_dist = max(distances.values())
    assert max_dist >= 1, "Expected at least one generation of growth"

    # Visualize the resulting structure
    structure_fig = visualize_polymer(
        graph,
        title="Dendrimer-like Structure",
        node_outline_color="gold",
        save_path=plot_path / "dendrimer_structure.png",
    )
    assert structure_fig is not None

    verify_visualization_outputs([plot_path / "dendrimer_structure.png"])
