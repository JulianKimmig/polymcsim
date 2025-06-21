"""Benchmark for radical polymerization simulations."""

from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from polymcsim import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    Simulation,
    SimulationInput,
    SiteDef,
    create_analysis_dashboard,
)


def create_radical_sim_config(n_monomers: int, n_initiators: int) -> SimulationInput:
    """Create a simulation config for radical polymerization."""
    return SimulationInput(
        monomers=[
            MonomerDef(
                name="Initiator",
                count=n_initiators,
                sites=[SiteDef(type="I", status="ACTIVE")],
            ),
            MonomerDef(
                name="Monomer",
                count=n_monomers,
                sites=[
                    SiteDef(type="M_in", status="DORMANT"),
                    SiteDef(type="M_out", status="DORMANT"),
                ],
            ),
        ],
        reactions={
            # Initiation
            frozenset(["I", "M_in"]): ReactionSchema(
                activation_map={"M_out": "R"},
                rate=1.0,
            ),
            # Propagation
            frozenset(["R", "M_in"]): ReactionSchema(
                activation_map={"M_out": "R"},
                rate=100.0,
            ),
            # Termination
            frozenset(["R", "R"]): ReactionSchema(
                rate=10.0,
            ),
        },
        params=SimParams(
            max_reactions=n_initiators + n_monomers,
            random_seed=42,
            name=f"radical_poly_{n_monomers}",
        ),
    )


@pytest.mark.parametrize(
    "n_monomers", [100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
)
def test_radical_polymerization_scaling(
    benchmark: BenchmarkFixture, n_monomers: int, plot_path: Path
) -> None:
    """Benchmark radical polymerization with increasing numbers of monomers."""
    n_initiators = n_monomers // 20  # Keep initiator ratio constant
    config = create_radical_sim_config(n_monomers, n_initiators)
    sim = Simulation(config)

    # The benchmark fixture will run this function multiple times
    graph, metadata = benchmark(sim.run)

    # plot dashboard
    create_analysis_dashboard(
        graph,
        metadata,
        title="Cross-linked Polymer Analysis",
        save_path=plot_path / "crosslinked_polymer_dashboard.png",
    )
