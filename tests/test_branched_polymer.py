import pytest
import networkx as nx

from polysim import (
    Simulation, 
    SimulationInput, 
    MonomerDef, 
    SiteDef, 
    ReactionSchema, 
    SimParams,
    visualize_polymer,
    plot_chain_length_distribution
)
from conftest import cleanup_figure, verify_visualization_outputs


@pytest.fixture
def branched_polymer_config():
    """Provides a config for a branched polymer with trifunctional monomers."""
    return SimulationInput(
        monomers=[
            MonomerDef(name="Initiator", count=10, sites=[
                SiteDef(type="I", status="ACTIVE")
            ]),
            MonomerDef(name="LinearMonomer", count=200, sites=[
                SiteDef(type="A_Head", status="DORMANT"),
                SiteDef(type="A_Tail", status="DORMANT"),
            ]),
            MonomerDef(name="BranchMonomer", count=50, sites=[
                SiteDef(type="B_Head", status="DORMANT"),
                SiteDef(type="B_Tail", status="DORMANT"),
                SiteDef(type="B_Branch", status="DORMANT"),  # Third site for branching
            ]),
        ],
        
        reactions={
            # Initiation
            frozenset(["I", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"},
                rate=1.0
            ),
            # Propagation on linear monomer
            frozenset(["Radical", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"},
                rate=100.0
            ),
            # Propagation on branch monomer (head)
            frozenset(["Radical", "B_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"B_Tail": "Radical"},
                rate=80.0
            ),
            # Branching reaction (branch site)
            frozenset(["Radical", "B_Branch"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"B_Tail": "Radical"},
                rate=60.0
            ),
            # Termination
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=20.0
            )
        },
        params=SimParams(max_reactions=5000, random_seed=123, name="branched_polymer")
    )


def test_simulation_run_branched_polymer(branched_polymer_config):
    """Tests that a branched polymer simulation runs and produces a valid structure."""
    sim = Simulation(branched_polymer_config)
    graph, meta = sim.run()

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert meta["reactions_completed"] <= branched_polymer_config.params.max_reactions

    # Check for all monomer types
    types = {attrs["monomer_type"] for _, attrs in graph.nodes(data=True)}
    assert "Initiator" in types
    assert "LinearMonomer" in types
    assert "BranchMonomer" in types

    # Check for branching - some nodes should have degree > 2
    degrees = [d for n, d in graph.degree()]
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


def test_visualization_branched_polymer(branched_polymer_config):
    """Tests the visualization functions for a branched polymer."""
    sim = Simulation(branched_polymer_config)
    graph, _ = sim.run()

    # Test polymer structure visualization with outline
    fig_structure = visualize_polymer(
        graph, 
        title="Test Branched Polymer Structure",
        component_index=0,
        node_outline_color='darkred',
        save_path="test_output/branched_polymer_structure.png"
    )
    assert fig_structure is not None
    cleanup_figure(fig_structure)

    # Test chain length distribution plot
    fig_dist = plot_chain_length_distribution(
        graph, 
        title="Test Branched Polymer Distribution",
        save_path="test_output/branched_polymer_dist.png"
    )
    assert fig_dist is not None
    cleanup_figure(fig_dist)

    # Verify files were created
    verify_visualization_outputs([
        "test_output/branched_polymer_structure.png",
        "test_output/branched_polymer_dist.png"
    ])


def test_hyperbranched_polymer_generation():
    """
    Generates a hyperbranched polymer using A2 + B4 monomers and checks for high branching and many terminal groups.
    Reference: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.894096/full
    """
    from polysim import Simulation, SimulationInput, MonomerDef, SiteDef, ReactionSchema, SimParams, visualize_polymer, plot_chain_length_distribution

    # A2 + B4 system: classic for hyperbranched polymers
    # Use stoichiometric imbalance to ensure terminal groups remain
    n_A2 = 80  # 160 A sites
    n_B4 = 60  # 240 B sites (excess B to create terminal groups)
    sim_input = SimulationInput(
        monomers=[
            MonomerDef(name="A2", count=n_A2, sites=[
                SiteDef(type="A", status="ACTIVE"),
                SiteDef(type="A", status="ACTIVE"),
            ]),
            MonomerDef(name="B4", count=n_B4, sites=[
                SiteDef(type="B", status="ACTIVE"),
                SiteDef(type="B", status="ACTIVE"),
                SiteDef(type="B", status="ACTIVE"),
                SiteDef(type="B", status="ACTIVE"),
            ]),
        ],

        reactions={
            frozenset(["A", "B"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=1.0
            ),
            frozenset(["A", "A"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=0.2
            )
        },
        params=SimParams(max_reactions=300, random_seed=2024, name="hyperbranched_polymer")
    )

    sim = Simulation(sim_input)
    graph, meta = sim.run()

    # Check that the largest component is highly branched
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest)
    degrees = [d for n, d in subgraph.degree()]
    n_branch_points = sum(1 for d in degrees if d >= 3)
    n_terminal = sum(1 for d in degrees if d == 1)
    avg_degree = sum(degrees) / len(degrees)

    print(f"Hyperbranched polymer analysis:")
    print(f"  Total nodes: {len(subgraph)}")
    print(f"  Branch points (degree ≥ 3): {n_branch_points} ({100*n_branch_points/len(subgraph):.1f}%)")
    print(f"  Terminal groups (degree = 1): {n_terminal} ({100*n_terminal/len(subgraph):.1f}%)")
    print(f"  Average degree: {avg_degree:.2f}")

    # Hyperbranched: many branch points, reasonable terminals, avg degree > 2
    assert n_branch_points > 0.05 * len(subgraph), f"Expected >5% branch points, got {n_branch_points}/{len(subgraph)}"
    assert n_terminal > 0, f"Expected some terminal groups, got {n_terminal}"
    assert avg_degree > 2.0, f"Expected avg degree > 2, got {avg_degree}"

    # Optionally visualize
    fig = visualize_polymer(
        subgraph,
        title="Hyperbranched Polymer Structure",
        component_index=0,
        node_outline_color='black',
        save_path="test_output/hyperbranched_polymer_structure.png"
    )
    cleanup_figure(fig)
    verify_visualization_outputs(["test_output/hyperbranched_polymer_structure.png"])


