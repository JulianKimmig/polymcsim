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

def test_star_polymer_generation():
    """
    Generates a star polymer using a multifunctional core with linear arms.
    Star polymers have a central core with multiple linear arms radiating outward.
    """
    from polysim import Simulation, SimulationInput, MonomerDef, SiteDef, ReactionSchema, SimParams, visualize_polymer, plot_chain_length_distribution

    # Star polymer: multifunctional core + linear arms
    # Use living polymerization approach to prevent inter-star connections
    n_core = 20  # Multifunctional cores (4 sites each)
    n_arms = 200  # Linear monomers for arms (2 sites each)
    sim_input = SimulationInput(
        monomers=[
            MonomerDef(name="Core", count=n_core, sites=[
                SiteDef(type="I", status="ACTIVE")  # Initiator sites
                for _ in range(4)
            ]),
            MonomerDef(name="Arm", count=n_arms, sites=[
                SiteDef(type="M", status="DORMANT"),  # Monomer sites (dormant)
                SiteDef(type="M", status="DORMANT"),
            ]),
        ],
        reactions={
            frozenset(["I", "M"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"M": "R"},  # Activate the other site as radical
                rate=10.0
            ),
            frozenset(["R", "M"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"M": "R"},  # Propagate the radical
                rate=0.8
            )
        },
        params=SimParams(max_reactions=400, random_seed=2024, name="star_polymer")
    )

    sim = Simulation(sim_input)
    graph, meta = sim.run()

    # Analyze star polymer characteristics
    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    subgraph = graph.subgraph(largest)
    
    # Find potential star centers (nodes with degree >= 3)
    potential_stars = [n for n, d in subgraph.degree() if d >= 3]
    
    # Count actual star structures (core with multiple arms)
    star_centers = []
    for node in potential_stars:
        if subgraph.nodes[node]["monomer_type"] == "Core":
            neighbors = list(subgraph.neighbors(node))
            # Check if this core has multiple connections (forming a star)
            if len(neighbors) >= 2:
                star_centers.append(node)
    
    # Calculate star polymer metrics
    n_stars = len(star_centers)
    total_nodes = len(subgraph)
    avg_degree = sum(d for n, d in subgraph.degree()) / total_nodes
    
    print(f"Star polymer analysis:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Potential star centers (degree ≥ 3): {len(potential_stars)}")
    print(f"  Actual star centers (Core with ≥2 arms): {n_stars}")
    print(f"  Average degree: {avg_degree:.2f}")
    
    if star_centers:
        # Analyze the largest star
        largest_star = max(star_centers, key=lambda x: len(list(subgraph.neighbors(x))))
        n_arms_largest = len(list(subgraph.neighbors(largest_star)))
        print(f"  Largest star has {n_arms_largest} arms")

    # Star polymer assertions
    assert n_stars > 0, f"Expected at least one star structure, got {n_stars}"
    assert avg_degree > 1.5, f"Expected avg degree > 1.5 for star polymer, got {avg_degree}"
    
    # Check that cores are well-connected (star-like structure)
    if star_centers:
        avg_arms_per_star = sum(len(list(subgraph.neighbors(star))) for star in star_centers) / len(star_centers)
        assert avg_arms_per_star >= 2.0, f"Expected avg arms per star ≥ 2, got {avg_arms_per_star}"

    # Visualize the star polymer
    fig = visualize_polymer(
        subgraph,
        component_index=0,
        title="Star Polymer Structure",
        node_outline_color='darkblue',
        save_path="test_output/star_polymer_structure.png"
    )
    cleanup_figure(fig)
    verify_visualization_outputs(["test_output/star_polymer_structure.png"]) 