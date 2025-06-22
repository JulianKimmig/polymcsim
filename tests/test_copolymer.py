"""Tests for various copolymer architectures."""

from pathlib import Path

import networkx as nx
import pytest

from polymcsim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    create_analysis_dashboard,
)


@pytest.fixture
def block_copolymer_config() -> SimulationInput:
    """Create configuration for a block copolymer (e.g., AAAA-BBBB)."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", count=10, sites=[SiteDef(type="I", status="ACTIVE")]
            ),
            MonomerDef(
                name="MonomerA",
                count=1000,
                sites=[
                    SiteDef(type="A_in", status="DORMANT"),
                    SiteDef(type="A_out", status="DORMANT"),
                ],
            ),
            MonomerDef(
                name="MonomerB",
                count=1000,
                sites=[
                    SiteDef(type="B_in", status="DORMANT"),
                    SiteDef(type="B_out", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            # Initiation of A
            frozenset(["I", "A_in"]): ReactionSchema(
                activation_map={"A_out": "RadicalA"},
                rate=1.0,
            ),
            # Propagation of A
            frozenset(["RadicalA", "A_in"]): ReactionSchema(
                activation_map={"A_out": "RadicalA"},
                rate=10.0,
            ),
            # Crossover from A to B
            frozenset(["RadicalA", "B_in"]): ReactionSchema(
                activation_map={"B_out": "RadicalB"},
                rate=1.0,
            ),
            # Propagation of B
            frozenset(["RadicalB", "B_in"]): ReactionSchema(
                activation_map={"B_out": "RadicalB"},
                rate=10.0,
            ),
            # Termination reactions
            frozenset(["RadicalA", "RadicalA"]): ReactionSchema(rate=0.1),
            frozenset(["RadicalB", "RadicalB"]): ReactionSchema(rate=0.1),
            frozenset(["RadicalA", "RadicalB"]): ReactionSchema(rate=0.1),
        },
        params=SimParams(max_conversion=0.8, random_seed=42, name="block_copolymer"),
    )


def test_block_copolymer_formation(
    block_copolymer_config: SimulationInput, plot_path: Path
):
    """Test for the formation of block copolymers."""
    sim = Simulation(block_copolymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    assert metadata["reactions_completed"] > 0
    components = list(nx.connected_components(graph))
    largest_polymer = graph.subgraph(max(components, key=len))

    # Check that both monomers are in the polymer
    types_in_polymer = {
        data["monomer_type"] for _, data in largest_polymer.nodes(data=True)
    }
    assert "MonomerA" in types_in_polymer
    assert "MonomerB" in types_in_polymer

    # A simple check for block structure:
    # Most B monomers should be connected to other B monomers or to A monomers,
    # but rarely should an A monomer be connected to two different B monomers.
    a_nodes = [
        n
        for n, d in largest_polymer.nodes(data=True)
        if d["monomer_type"] == "MonomerA"
    ]
    for node in a_nodes:
        b_neighbors = 0
        for neighbor in largest_polymer.neighbors(node):
            if largest_polymer.nodes[neighbor]["monomer_type"] == "MonomerB":
                b_neighbors += 1
        assert b_neighbors <= 1, "Monomer A should not be connecting two B blocks."

    # Visualization
    dashboard_fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Block Copolymer Analysis",
        save_path=plot_path / "block_copolymer_dashboard.png",
    )
    assert dashboard_fig is not None


@pytest.fixture
def random_copolymer_config() -> SimulationInput:
    """Create configuration for a random copolymer."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator", count=10, sites=[SiteDef(type="I", status="ACTIVE")]
            ),
            MonomerDef(
                name="MonomerA",
                count=100,
                sites=[
                    SiteDef(type="A_in", status="DORMANT"),
                    SiteDef(type="A_out", status="DORMANT"),
                ],
            ),
            MonomerDef(
                name="MonomerB",
                count=100,
                sites=[
                    SiteDef(type="B_in", status="DORMANT"),
                    SiteDef(type="B_out", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "A_in"]): ReactionSchema(
                activation_map={"A_out": "Radical"},
                rate=1.0,
            ),
            frozenset(["I", "B_in"]): ReactionSchema(
                activation_map={"B_out": "Radical"},
                rate=1.0,
            ),
            # Propagation
            frozenset(["Radical", "A_in"]): ReactionSchema(
                activation_map={"A_out": "Radical"},
                rate=5.0,
            ),
            frozenset(["Radical", "B_in"]): ReactionSchema(
                activation_map={"B_out": "Radical"},
                rate=5.0,
            ),
        },
        params=SimParams(max_reactions=1500, random_seed=43, name="random_copolymer"),
    )


def test_random_copolymer_formation(
    random_copolymer_config: SimulationInput, plot_path: Path
):
    """Test for the formation of random copolymers."""
    sim = Simulation(random_copolymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    create_analysis_dashboard(
        graph,
        metadata,
        title="Random Copolymer Analysis",
        save_path=plot_path / "random_copolymer_dashboard.png",
    )

    components = list(nx.connected_components(graph))
    largest_polymer = graph.subgraph(max(components, key=len))

    # Check that monomer sequences are mixed
    a_b_links = 0
    a_a_links = 0
    b_b_links = 0

    for u, v in largest_polymer.edges():
        u_type = largest_polymer.nodes[u]["monomer_type"]
        v_type = largest_polymer.nodes[v]["monomer_type"]
        if u_type == "MonomerA" and v_type == "MonomerB":
            a_b_links += 1
        elif u_type == "MonomerA" and v_type == "MonomerA":
            a_a_links += 1
        elif u_type == "MonomerB" and v_type == "MonomerB":
            b_b_links += 1

    # In a random copolymer, expect a mix of all link types
    assert a_b_links > 0 and a_a_links > 0 and b_b_links > 0


@pytest.fixture
def alternating_copolymer_config() -> SimulationInput:
    """Create configuration for an alternating copolymer (ABABAB)."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="MonomerA",
                count=100,
                sites=[
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="MonomerB",
                count=100,
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ],
            ),
        ],
        reactions={
            frozenset(["A", "B"]): ReactionSchema(rate=10.0),
            # No A-A or B-B reactions allowed
        },
        params=SimParams(
            max_reactions=150, random_seed=44, name="alternating_copolymer"
        ),
    )


def test_alternating_copolymer_formation(
    alternating_copolymer_config: SimulationInput, plot_path: Path
):
    """Test for the formation of strictly alternating copolymers."""
    sim = Simulation(alternating_copolymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    create_analysis_dashboard(
        graph,
        metadata,
        title="Alternating Copolymer Analysis",
        save_path=plot_path / "alternating_copolymer_dashboard.png",
    )

    components = list(nx.connected_components(graph))
    largest_polymer = graph.subgraph(max(components, key=len))

    # Check that no A-A or B-B links exist
    for u, v in largest_polymer.edges():
        u_type = largest_polymer.nodes[u]["monomer_type"]
        v_type = largest_polymer.nodes[v]["monomer_type"]
        assert u_type != v_type, "Found a non-alternating link."
