"""Tests for cross-linked polymer network formation."""

import networkx as nx
import pytest
from pathlib import Path

from polysim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    visualize_polymer,
    create_analysis_dashboard,
)


@pytest.fixture
def crosslinked_polymer_config() -> SimulationInput:
    """Configuration for a cross-linked polymer network (e.g., from A3 and B2 monomers)."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="A3_Monomer",
                count=100,
                sites=[
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="B2_Monomer",
                count=150,
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ],
            ),
        ],
        reactions={
            frozenset(["A", "B"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                rate=1.0,
            )
        },
        params=SimParams(
            max_reactions=500,  # High number of reactions to ensure network formation
            random_seed=42,
            name="crosslinked_polymer"
        ),
    )


def test_crosslinked_network_formation(crosslinked_polymer_config: SimulationInput, plot_path: Path):
    """Test that a highly cross-linked network forms a giant component."""
    sim = Simulation(crosslinked_polymer_config)
    graph, metadata = sim.run()

    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert metadata["reactions_completed"] > 0

    # Check that a giant component has formed
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    assert len(components) > 0, "No polymer components found."

    giant_component = components[0]
    total_nodes = graph.number_of_nodes()
    
    # Assert that the largest component contains a significant fraction of all nodes,
    # indicating network formation (gelation).
    assert len(giant_component) / total_nodes > 0.8, \
        f"The giant component is too small ({len(giant_component)}/{total_nodes} nodes)."

    # Verify that both monomer types are in the giant component
    component_graph = graph.subgraph(giant_component)
    monomer_types = {data["monomer_type"] for _, data in component_graph.nodes(data=True)}
    assert "A3_Monomer" in monomer_types
    assert "B2_Monomer" in monomer_types

    # Visualize the network
    dashboard_fig = create_analysis_dashboard(
        graph, 
        metadata, 
        title="Cross-linked Network Analysis",
        save_path=plot_path / "crosslinked_dashboard.png"
    )
    assert dashboard_fig is not None

    structure_fig = visualize_polymer(
        graph,
        title="Cross-linked Polymer Network",
        save_path=plot_path / "crosslinked_structure.png"
    )
    assert structure_fig is not None 