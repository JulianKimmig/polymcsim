import networkx as nx
from typing import Tuple, Dict, Any
from .optim_sim import SimulationInput, run_simulation as run_optim_simulation

class Simulation:
    """
    A wrapper for the optimized PolySim simulation engine.
    """
    def __init__(self, config: SimulationInput):
        """
        Initializes the simulation with a complete configuration.

        Args:
            config (SimulationInput): The detailed simulation configuration object.
        """
        self.config = config
        self.graph: nx.Graph = None
        self.metadata: Dict[str, Any] = None

    def run(self) -> Tuple[nx.Graph, Dict[str, Any]]:
        """
        Executes the simulation.

        This method calls the core Numba-optimized Kinetic Monte Carlo engine
        and runs the simulation to completion based on the provided configuration.

        Returns:
            A tuple containing:
            - nx.Graph: The final polymer network structure.
            - dict: A dictionary of metadata about the simulation run.
        """
        self.graph, self.metadata = run_optim_simulation(self.config)
        return self.graph, self.metadata

    def get_graph(self) -> nx.Graph:
        """
        Returns the resulting polymer graph.

        Returns None if the simulation has not been run.
        """
        return self.graph

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns metadata from the simulation run.

        Returns None if the simulation has not been run.
        """
        return self.metadata 