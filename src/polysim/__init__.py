"""PolySim - Monte Carlo Polymer Graph Generation Library."""

from .schemas import SimulationInput, MonomerDef, SiteDef, ReactionSchema, SimParams
from .simulation import Simulation, run_simulation, run_batch
from .visualization import (
    visualize_polymer, 
    plot_chain_length_distribution,
    plot_molecular_weight_distribution,
    plot_conversion_analysis,
    plot_branching_analysis,
    create_analysis_dashboard,
    export_polymer_data
)

__version__ = "0.1.0"

__all__ = [
    "Simulation",
    "run_simulation",
    "run_batch",
    "SimulationInput",
    "MonomerDef",
    "SiteDef",
    "ReactionSchema",
    "SimParams",
    "visualize_polymer",
    "plot_chain_length_distribution",
] 