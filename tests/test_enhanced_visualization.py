"""Test the enhanced visualization functions."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from polymcsim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    create_analysis_dashboard,
    export_polymer_data,
    plot_branching_analysis,
    plot_conversion_analysis,
    plot_molecular_weight_distribution,
)


@pytest.fixture
def branched_polymer_config() -> SimulationInput:
    """Create configuration for a branched polymer system."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="A3",
                count=100,
                molar_mass=100.0,
                sites=[
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                    SiteDef(type="A", status="ACTIVE"),
                ],
            ),
            MonomerDef(
                name="B2",
                count=150,
                molar_mass=120.0,
                sites=[
                    SiteDef(type="B", status="ACTIVE"),
                    SiteDef(type="B", status="ACTIVE"),
                ],
            ),
        ],
        reactions={frozenset(["A", "B"]): ReactionSchema(rate=1.0)},
        params=SimParams(
            name="enhanced_viz_branched",
            max_reactions=300,
            max_conversion=0.6,
            random_seed=42,
        ),
    )


def test_molecular_weight_distribution(branched_polymer_config, plot_path: Path):
    """Test the molecular weight distribution plot."""
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph = result.graph

    # Test normal scale
    fig = plot_molecular_weight_distribution(
        graph,
        show_pdi=True,
        title="Test MWD - Normal Scale",
        save_path=plot_path / "test_mwd_normal.png",
    )
    assert fig is not None
    assert (plot_path / "test_mwd_normal.png").exists()
    plt.close(fig)

    # Test log scale
    fig = plot_molecular_weight_distribution(
        graph,
        log_scale=True,
        show_pdi=True,
        title="Test MWD - Log Scale",
        save_path=plot_path / "test_mwd_log.png",
    )
    assert fig is not None
    assert (plot_path / "test_mwd_log.png").exists()
    plt.close(fig)


def test_conversion_analysis(branched_polymer_config, plot_path: Path):
    """Test the conversion analysis plot."""
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    # Test without edge data
    fig = plot_conversion_analysis(
        metadata,
        title="Test Conversion Analysis",
        save_path=plot_path / "test_conversion.png",
    )
    assert fig is not None
    assert (plot_path / "test_conversion.png").exists()
    plt.close(fig)

    # Test with mock edge data (normally would come from simulation internals)
    edge_data = [
        (i, j, t)
        for (i, j), t in zip(graph.edges(), np.linspace(0, 1, len(graph.edges())))
    ]
    fig = plot_conversion_analysis(
        metadata,
        edge_data=edge_data,
        title="Test Conversion Analysis with Kinetics",
        save_path=plot_path / "test_conversion_kinetics.png",
    )
    assert fig is not None
    assert (plot_path / "test_conversion_kinetics.png").exists()
    plt.close(fig)


def test_branching_analysis(branched_polymer_config, plot_path: Path):
    """Test the branching analysis plot."""
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph = result.graph

    fig = plot_branching_analysis(
        graph,
        title="Test Branching Analysis",
        save_path=plot_path / "test_branching.png",
    )
    assert fig is not None
    assert (plot_path / "test_branching.png").exists()
    plt.close(fig)


def test_analysis_dashboard(branched_polymer_config, plot_path: Path):
    """Test the comprehensive analysis dashboard."""
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    fig = create_analysis_dashboard(
        graph,
        metadata,
        title="Test Polymer Analysis Dashboard",
        save_path=plot_path / "test_dashboard.png",
    )
    assert fig is not None
    assert (plot_path / "test_dashboard.png").exists()
    plt.close(fig)


def test_export_polymer_data(branched_polymer_config, plot_path: Path):
    """Test data export functionality."""
    sim = Simulation(branched_polymer_config)
    result = sim.run()
    graph, metadata = result.graph, result.metadata

    output_files = export_polymer_data(
        graph, metadata, output_dir=plot_path, prefix="test_polymer"
    )

    # Check that files were created
    assert "chain_data" in output_files
    assert "summary" in output_files

    # Verify files exist
    assert output_files["chain_data"].exists()
    assert output_files["summary"].exists()

    # Check file contents (basic validation)
    import pandas as pd

    # Check chain data
    df_chains = pd.read_csv(output_files["chain_data"])
    assert "chain_id" in df_chains.columns
    assert "molecular_weight" in df_chains.columns
    assert "branch_points" in df_chains.columns
    assert len(df_chains) > 0  # Should have some polymer chains

    # Check summary data
    df_summary = pd.read_csv(output_files["summary"])
    assert "total_monomers" in df_summary.columns
    assert "final_conversion" in df_summary.columns
    assert "PDI" in df_summary.columns
    assert len(df_summary) == 1  # Should have exactly one row


def test_empty_graph_handling(plot_path: Path):
    """Test that visualization functions handle empty graphs gracefully."""
    import networkx as nx

    empty_graph = nx.Graph()
    empty_metadata = {
        "final_conversion": 0.0,
        "reactions_completed": 0,
        "final_simulation_time": 0.0,
        "wall_time_seconds": 0.0,
    }

    # Test each function with empty graph
    fig = plot_molecular_weight_distribution(empty_graph)
    assert fig is not None
    plt.close(fig)

    fig = plot_branching_analysis(empty_graph)
    assert fig is not None
    plt.close(fig)

    fig = create_analysis_dashboard(empty_graph, empty_metadata)
    assert fig is not None
    plt.close(fig)

    # Export should still work but might not create chain_data file
    output_files = export_polymer_data(
        empty_graph, empty_metadata, output_dir=plot_path, prefix="empty_test"
    )
    assert "summary" in output_files  # Summary should always be created
