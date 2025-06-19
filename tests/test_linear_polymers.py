import pytest
import networkx as nx
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for tests
import matplotlib.pyplot as plt

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

# Create a directory for test outputs
os.makedirs("test_output", exist_ok=True)

@pytest.fixture
def step_growth_config():
    """Provides a config for a typical step-growth (A2 + B2) polymerization."""
    return SimulationInput(
        monomers=[
            MonomerDef(name="A2_Monomer", count=100, sites=[
                SiteDef(type="A", status="ACTIVE"),
                SiteDef(type="A", status="ACTIVE"),
            ]),
            MonomerDef(name="B2_Monomer", count=100, sites=[
                SiteDef(type="B", status="ACTIVE"),
                SiteDef(type="B", status="ACTIVE"),
            ]),
        ],
        rate_matrix={
            frozenset(["A", "B"]): 1.0,
        },
        reaction_schema={
            frozenset(["A", "B"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
            )
        },
        params=SimParams(max_reactions=150, random_seed=42)
    )

@pytest.fixture
def chain_growth_config():
    """Provides a config for a typical chain-growth radical polymerization."""
    return SimulationInput(
        monomers=[
            MonomerDef(name="Initiator", count=1000, sites=[
                SiteDef(type="I", status="ACTIVE")
            ]),
            MonomerDef(name="Monomer", count=20000, sites=[
                SiteDef(type="Head", status="DORMANT"),
                SiteDef(type="Tail", status="DORMANT"),
            ]),
        ],
        rate_matrix={
            frozenset(["Radical", "Head"]): 100.0, # Propagation
            frozenset(["I", "Head"]): 1.0,         # Initiation
            frozenset(["Radical", "Radical"]): 50.0 # Termination
        },
        reaction_schema={
            frozenset(["I", "Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"Tail": "Radical"}
            ),
            frozenset(["Radical", "Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"Tail": "Radical"}
            ),
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
            )
        },
        params=SimParams(max_reactions=18000, random_seed=101)
    )

def test_simulation_run_step_growth(step_growth_config):
    """Tests that a step-growth simulation runs and produces a graph."""
    sim = Simulation(step_growth_config)
    graph, meta = sim.run()

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 200
    assert graph.number_of_edges() > 0
    assert meta["reactions_completed"] <= step_growth_config.params.max_reactions
    
    # Check node attributes
    for node_id, attrs in graph.nodes(data=True):
        assert "monomer_type" in attrs
        assert attrs["monomer_type"] in ["A2_Monomer", "B2_Monomer"]

    # All nodes should have degree <= 2 for this linear case
    degrees = [d for n, d in graph.degree()]
    assert all(d <= 2 for d in degrees)

def test_simulation_run_chain_growth(chain_growth_config):
    """Tests that a chain-growth simulation runs and produces a graph."""
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
    initiator_nodes = [n for n, d in graph.nodes(data=True) if d['monomer_type'] == 'Initiator']
    for i_node in initiator_nodes:
        assert graph.degree(i_node) <= 1

def test_visualization_step_growth(step_growth_config):
    """Tests the visualization functions for a step-growth polymer."""
    sim = Simulation(step_growth_config)
    graph, _ = sim.run()

    # Test polymer structure visualization
    fig_structure = visualize_polymer(
        graph, 
        title="Test Step-Growth Structure",
        save_path="test_output/step_growth_structure.png"
    )
    assert fig_structure is not None
    plt.close(fig_structure)
    assert os.path.exists("test_output/step_growth_structure.png")

    # Test chain length distribution plot
    fig_dist = plot_chain_length_distribution(
        graph, 
        title="Test Step-Growth Distribution",
        save_path="test_output/step_growth_dist.png"
    )
    assert fig_dist is not None
    plt.close(fig_dist)
    assert os.path.exists("test_output/step_growth_dist.png")

def test_visualization_chain_growth(chain_growth_config):
    """Tests the visualization functions for a chain-growth polymer."""
    sim = Simulation(chain_growth_config)
    graph, _ = sim.run()

    # Test polymer structure visualization for the largest component
    fig_structure = visualize_polymer(
        graph, 
        title="Test Chain-Growth Structure (Largest)",
        component_index=0, # Plot largest component
        save_path="test_output/chain_growth_structure_largest.png"
    )
    assert fig_structure is not None
    plt.close(fig_structure)
    assert os.path.exists("test_output/chain_growth_structure_largest.png")

    # Test chain length distribution plot
    fig_dist = plot_chain_length_distribution(
        graph, 
        title="Test Chain-Growth Distribution",
        save_path="test_output/chain_growth_dist.png"
    )
    assert fig_dist is not None
    plt.close(fig_dist)
    assert os.path.exists("test_output/chain_growth_dist.png")

@pytest.fixture
def gradient_polymer_config():
    """Provides a config for a linear gradient copolymer."""
    return SimulationInput(
        monomers=[
            MonomerDef(name="Initiator", count=5, sites=[
                SiteDef(type="I", status="ACTIVE")
            ]),
            MonomerDef(name="MonomerA", count=100, sites=[
                SiteDef(type="A_Head", status="DORMANT"),
                SiteDef(type="A_Tail", status="DORMANT"),
            ]),
            MonomerDef(name="MonomerB", count=100, sites=[
                SiteDef(type="B_Head", status="DORMANT"),
                SiteDef(type="B_Tail", status="DORMANT"),
            ]),
        ],
        rate_matrix={
            frozenset(["I", "A_Head"]): 1.0,         # Initiation
            frozenset(["Radical", "A_Head"]): 100.0,  # Propagation A
            frozenset(["Radical", "B_Head"]): 10.0,   # Propagation B (slower)
            # frozenset(["Radical", "Radical"]): 10.0,  # Termination disabled for better gradient view
        },
        reaction_schema={
            # Initiation
            frozenset(["I", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"}
            ),
            # Propagation A
            frozenset(["Radical", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"}
            ),
            # Propagation B
            frozenset(["Radical", "B_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"B_Tail": "Radical"}
            ),
            # Termination
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
            )
        },
        params=SimParams(random_seed=42, name="gradient_polymer")
    )

def test_simulation_run_gradient_polymer(gradient_polymer_config):
    """Tests that a gradient polymer simulation runs and produces a valid structure."""
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
    degrees = [d for n, d in graph.degree()]
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


def test_visualization_gradient_polymer(gradient_polymer_config):
    """Tests the visualization functions for a gradient polymer."""
    sim = Simulation(gradient_polymer_config)
    graph, _ = sim.run()

    # Test polymer structure visualization
    fig_structure = visualize_polymer(
        graph, 
        title="Test Gradient Polymer Structure",
        component_index=0,
        save_path="test_output/gradient_polymer_structure.png"
    )
    assert fig_structure is not None
    plt.close(fig_structure)
    assert os.path.exists("test_output/gradient_polymer_structure.png")

    # Test chain length distribution plot
    fig_dist = plot_chain_length_distribution(
        graph, 
        title="Test Gradient Polymer Distribution",
        save_path="test_output/gradient_polymer_dist.png"
    )
    assert fig_dist is not None
    plt.close(fig_dist)
    assert os.path.exists("test_output/gradient_polymer_dist.png")


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
        rate_matrix={
            frozenset(["I", "A_Head"]): 1.0,         # Initiation
            frozenset(["Radical", "A_Head"]): 100.0,  # Propagation on linear monomer
            frozenset(["Radical", "B_Head"]): 80.0,   # Propagation on branch monomer
            frozenset(["Radical", "B_Branch"]): 60.0, # Branching reaction
            frozenset(["Radical", "Radical"]): 20.0,  # Termination
        },
        reaction_schema={
            # Initiation
            frozenset(["I", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"}
            ),
            # Propagation on linear monomer
            frozenset(["Radical", "A_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"A_Tail": "Radical"}
            ),
            # Propagation on branch monomer (head)
            frozenset(["Radical", "B_Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"B_Tail": "Radical"}
            ),
            # Branching reaction (branch site)
            frozenset(["Radical", "B_Branch"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"B_Tail": "Radical"}
            ),
            # Termination
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
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
    plt.close(fig_structure)
    assert os.path.exists("test_output/branched_polymer_structure.png")

    # Test chain length distribution plot
    fig_dist = plot_chain_length_distribution(
        graph, 
        title="Test Branched Polymer Distribution",
        save_path="test_output/branched_polymer_dist.png"
    )
    assert fig_dist is not None
    plt.close(fig_dist)
    assert os.path.exists("test_output/branched_polymer_dist.png")

