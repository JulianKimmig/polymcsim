"""Generates and saves homopolymer graphs of a specific molar mass.

This script runs a polymerization simulation and extracts polymer chains
that match a target molar mass within a given tolerance.
"""

import random
from typing import Dict, List, Optional

import networkx as nx
import optuna

from ..schemas import (
    MonomerDef,
    ReactionSchema,
    SimParams,
    SimulationInput,
    SiteDef,
)
from ..simulation import Simulation


class PolymerSearchOptimizer:
    """Optimizes simulation parameters to find polymers of a target molar mass."""

    def __init__(
        self,
        sim_config_template: str,
        var_params: Dict[str, "VarParam"],
        min_num_polymers_to_find: int,
        target_polymer_molar_mass: float,
        min_mass: float,
        max_mass: float,
        max_tries: int,
        raise_on_failure: bool = True,
    ):
        """Initialize the optimizer.

        Args:
            sim_config_template: A JSON string of the simulation input with
                placeholders.
            var_params: A dictionary of variable parameters to optimize.
            min_num_polymers_to_find: The minimum number of polymers to find.
            target_polymer_molar_mass: The target molar mass of the polymer.
            min_mass: The minimum allowed molar mass.
            max_mass: The maximum allowed molar mass.
            max_tries: The maximum number of optimization trials.
            raise_on_failure: Whether to raise an error if not enough polymers
                are found.

        """
        self.sim_config_template = sim_config_template
        self.var_params = var_params
        self.min_num_polymers_to_find = min_num_polymers_to_find
        self.target_polymer_molar_mass = target_polymer_molar_mass
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.max_tries = max_tries
        self.found_polymers: Optional[List[nx.Graph]] = []
        self.raise_on_failure = raise_on_failure

    def _objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna to maximize."""
        _config_dump = self.sim_config_template
        for param_name, param_value in self.var_params.items():
            rstring = f'"{param_value.field}":<{param_name}>'
            sampled_value = None
            if param_value.type is int:
                sampled_value = trial.suggest_int(
                    param_name, param_value.min, param_value.max
                )
            elif param_value.type is float:
                sampled_value = trial.suggest_float(
                    param_name, param_value.min, param_value.max
                )
            _config_dump = _config_dump.replace(
                rstring, f'"{param_value.field}":{sampled_value}'
            )

        sim_config = SimulationInput.model_validate_json(_config_dump)
        sim = Simulation(sim_config)
        result = sim.run()

        polymers = result.get_polymers()
        mn = result.get_average_molecular_weights()["Mn"]
        self.found_polymers.extend(
            [
                p
                for p in polymers
                if self.min_mass <= p.molecular_weight <= self.max_mass
            ]
        )

        if len(self.found_polymers) >= self.min_num_polymers_to_find:
            trial.set_user_attr("success", True)

        # Optuna maximizes this value
        return abs(mn - self.target_polymer_molar_mass)

    def _callback(self, study: optuna.Study, trial: optuna.Trial):
        """Stop the study when enough polymers are found."""
        if trial.user_attrs.get("success", False):
            study.stop()

    def find(self) -> Optional[List[nx.Graph]]:
        """Run the optimization and return the found polymers."""
        study = optuna.create_study(direction="minimize")
        study.optimize(
            self._objective, n_trials=self.max_tries, callbacks=[self._callback]
        )

        if (
            len(self.found_polymers) < self.min_num_polymers_to_find
            and self.raise_on_failure
        ):
            raise ValueError(
                f"Could not find {self.min_num_polymers_to_find} polymers. "
                f"Found {len(self.found_polymers)}."
            )

        self.found_polymers.sort(
            key=lambda x: abs(x.molecular_weight - self.target_polymer_molar_mass)
        )
        return self.found_polymers


class VarParam:
    """A variable parameter for optimization."""

    def __init__(self, type, min, max, field):
        """Initialize a variable parameter.

        Args:
            type: The type of the parameter (int or float).
            min: The minimum value of the parameter.
            max: The maximum value of the parameter.
            field: The field name in the simulation config.

        """
        self.type = type
        self.min = min
        self.max = max
        self.field = field
        self.inital_sample = None

    def sample(self):
        """Sample a random value for the parameter."""
        if self.type is int:
            return random.randint(self.min, self.max)
        elif self.type is float:
            return random.uniform(self.min, self.max)
        else:
            raise ValueError(f"Unsupported type: {self.type}")

    @classmethod
    def sample_initital(cls, params: Dict[str, "VarParam"]):
        """Sample initial values for all parameters."""
        sampled = {}
        for k, v in params.items():
            sampled_value = None
            while sampled_value is None or sampled_value in sampled.values():
                sampled_value = v.sample()
            sampled[k] = sampled_value
            v.inital_sample = sampled_value
        return sampled


def generate_polymers_with_mass(
    sim_config: SimulationInput,
    var_params: Dict[str, VarParam],
    min_num_polymers_to_find: int,
    target_polymer_molar_mass: float,
    tolerance_abs: float | None,
    min_mass: float | None,
    max_mass: float | None,
    tolerance_percent: float = 5.0,
    max_tries: int = 100,
    raise_on_failure: bool = True,
):
    """Run a simulation and save polymer graphs that match the mass criteria.

    Args:
        sim_config: The simulation input object.
        var_params: A dictionary of variable parameters to optimize.
        min_num_polymers_to_find: The number of polymer graphs to generate.
        target_polymer_molar_mass: The target molar mass for the polymer.
        tolerance_abs: Allowed absolute deviation from the target molar mass.
        min_mass: Minimum allowed polymer molar mass.
        max_mass: Maximum allowed polymer molar mass.
        tolerance_percent: Allowed deviation from the target molar mass in percent.
        max_tries: The maximum number of optimization trials.
        raise_on_failure: Whether to raise an error if not enough polymers are found.

    """
    if min_mass is not None:
        pass
    elif tolerance_abs is not None:
        min_mass = target_polymer_molar_mass - tolerance_abs
    else:
        min_mass = target_polymer_molar_mass * (1 - tolerance_percent / 100.0)

    if max_mass is not None:
        pass
    elif tolerance_abs is not None:
        max_mass = target_polymer_molar_mass + tolerance_abs
    else:
        max_mass = target_polymer_molar_mass * (1 + tolerance_percent / 100.0)

    print(
        f"Target mass: {target_polymer_molar_mass:.2f}, "
        f"Searching for polymers with mass in [{min_mass}, {max_mass}]"
    )

    # Estimate number of monomers needed. This is a heuristic.
    # We need enough monomers to form the desired number of polymers.
    config_dump = sim_config.model_dump_json()
    for param_name, param_value in var_params.items():
        rstring = f'"{param_value.field}":{param_value.inital_sample}'
        if rstring not in config_dump:
            raise ValueError(
                f"Parameter {param_name} not found in config dump:\n"
                f"{config_dump}\n{rstring}"
            )
        # if more than one occurence, raise error
        if config_dump.count(rstring) > 1:
            raise ValueError(
                f"Parameter {param_name} found multiple times in config dump:\n"
                f"{config_dump}\n{rstring}"
            )
        config_dump = config_dump.replace(
            rstring, f'"{param_value.field}":<{param_name}>'
        )

    optimizer = PolymerSearchOptimizer(
        sim_config_template=config_dump,
        var_params=var_params,
        min_num_polymers_to_find=min_num_polymers_to_find,
        target_polymer_molar_mass=target_polymer_molar_mass,
        min_mass=min_mass,
        max_mass=max_mass,
        max_tries=max_tries,
        raise_on_failure=raise_on_failure,
    )

    found_polymers = optimizer.find()

    return found_polymers


def main():
    """Generate homopolymer graphs by molar mass."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate homopolymer graphs by molar mass."
    )
    parser.add_argument(
        "-n",
        "--num-polymers",
        type=int,
        default=100,
        help="Number of polymer graphs to generate.",
    )

    parser.add_argument(
        "--polymer-mass",
        type=float,
        default=10000.0,
        help="Target molar mass of the polymer (g/mol).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="Allowed molar mass deviation in percent."
        " Used if other tolerances are not set.",
    )
    parser.add_argument(
        "--tolerance-abs",
        type=float,
        help="Allowed absolute molar mass deviation (g/mol). "
        "Overrides percent tolerance.",
    )
    parser.add_argument(
        "--min-mass",
        type=float,
        help="Minimum polymer molar mass. Overrides other tolerance settings.",
    )
    parser.add_argument(
        "--max-mass",
        type=float,
        help="Maximum polymer molar mass. Overrides other tolerance settings.",
    )
    args = parser.parse_args()

    params = {
        "num_initiators": VarParam(int, 1, 1000, "count"),
        "num_monomers": VarParam(int, 1000, 100_000, "count"),
        "rate_initiator": VarParam(float, 100.0, 1000.0, "rate"),
        "rate_monomer": VarParam(float, 1.0, 10.0, "rate"),
    }

    VarParam.sample_initital(params)

    monomers = [
        MonomerDef(
            name="Initiator",
            count=params["num_initiators"].inital_sample,
            sites=[SiteDef(type="I", status="ACTIVE")],
        ),
        MonomerDef(
            name="Monomer",
            count=params["num_monomers"].inital_sample,
            molar_mass=200,
            sites=[
                SiteDef(type="Head", status="DORMANT"),
                SiteDef(type="Tail", status="DORMANT"),
            ],
        ),
    ]

    reactions = {
        frozenset(["I", "Head"]): ReactionSchema(
            activation_map={"Tail": "Radical"},
            rate=params["rate_initiator"].inital_sample,
        ),
        frozenset(["Radical", "Head"]): ReactionSchema(
            activation_map={"Tail": "Radical"},
            rate=params["rate_monomer"].inital_sample,
        ),
    }

    # High conversion to get larger polymers
    sim_params = SimParams(random_seed=42)

    config = SimulationInput(
        monomers=monomers,
        reactions=reactions,
        params=sim_params,
    )

    generate_polymers_with_mass(
        sim_config=config,
        var_params=params,
        min_num_polymers_to_find=args.num_polymers,
        target_polymer_molar_mass=args.polymer_mass,
        tolerance_percent=args.tolerance,
        tolerance_abs=args.tolerance_abs,
        min_mass=args.min_mass,
        max_mass=args.max_mass,
        raise_on_failure=False,
    )
