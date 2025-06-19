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
            MonomerDef(name="Initiator", count=10, sites=[
                SiteDef(type="I", status="ACTIVE")
            ]),
            MonomerDef(name="Monomer", count=200, sites=[
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
        params=SimParams(max_reactions=180, random_seed=101)
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
    assert graph.number_of_nodes() == 210
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