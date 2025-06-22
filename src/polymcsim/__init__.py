"""PolyMCsim - Monte Carlo Polymer Graph Generation Library."""

from .polymer_generation import (
    PolymerSearchOptimizer,
    VarParam,
    generate_polymers_with_mass,
)
from .schemas import (
    MonomerDef,
    Polymer,
    ReactionSchema,
    SimParams,
    SimulationInput,
    SimulationResult,
    SiteDef,
)
from .simulation import Simulation, run_batch, run_simulation
from .visualization import (
    create_analysis_dashboard,
    export_polymer_data,
    plot_branching_analysis,
    plot_chain_length_distribution,
    plot_conversion_analysis,
    plot_molecular_weight_distribution,
    visualize_polymer,
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
    "plot_molecular_weight_distribution",
    "plot_conversion_analysis",
    "plot_branching_analysis",
    "create_analysis_dashboard",
    "export_polymer_data",
    "Polymer",
    "SimulationResult",
    "generate_polymers_with_mass",
    "VarParam",
    "PolymerSearchOptimizer",
]
