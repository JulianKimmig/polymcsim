"""Main simulation interface for PolyMCsim polymer generation."""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
from numba.core import types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList
from tqdm.auto import tqdm

from .core import STATUS_ACTIVE, STATUS_CONSUMED, STATUS_DORMANT, run_kmc_loop
from .schemas import SimulationInput, SimulationResult


def _validate_config(config: SimulationInput) -> bool:
    """Perform runtime validation of the simulation configuration.

    Checks for logical consistency between different parts of the configuration
    beyond Pydantic's scope.

    Args:
        config: The simulation configuration to validate.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If configuration is invalid with a clear error message.

    """
    all_known_types = set()
    initial_site_statuses: Dict[str, str] = {}  # Maps site type string to its status

    for monomer in config.monomers:
        for site in monomer.sites:
            all_known_types.add(site.type)
            if (
                site.type in initial_site_statuses
                and initial_site_statuses[site.type] != site.status
            ):
                raise ValueError(
                    f"Inconsistent Status: Site type '{site.type}' is defined as "
                    "both ACTIVE and DORMANT across different monomers. A site "
                    "type must have a consistent status."
                )
            initial_site_statuses[site.type] = site.status

    for reaction_def in config.reactions.values():
        all_known_types.update(reaction_def.activation_map.values())

    # Validate reaction definitions using the complete set of known types
    for reaction_pair, reaction_def in config.reactions.items():
        pair_list = list(reaction_pair)

        # Check that all reacting sites are known to the system
        for site_type in pair_list:
            if site_type not in all_known_types:
                raise ValueError(
                    f"Undefined Site: The site type '{site_type}' used in "
                    f"reaction {reaction_pair} "
                    "is not defined on any monomer and is "
                    "not created by any activation."
                )

        # Infer status of reacting sites (emergent sites are always ACTIVE)
        type1 = pair_list[0]
        type2 = pair_list[1] if len(pair_list) > 1 else type1
        status1 = initial_site_statuses.get(
            type1, "ACTIVE"
        )  # Default to ACTIVE for emergent types
        status2 = initial_site_statuses.get(type2, "ACTIVE")

        if status1 == "DORMANT" and status2 == "DORMANT":
            raise ValueError(
                f"Invalid Reaction: Reaction pair {reaction_pair} "
                "involves two DORMANT sites. "
                f"At least one site in a reaction must be ACTIVE."
            )

        # Validate activation logic
        if not reaction_def.activation_map:
            continue

        original_dormant_type, new_active_type = list(
            reaction_def.activation_map.items()
        )[0]

        # The new type must be a known site type
        if new_active_type not in all_known_types:
            raise ValueError(
                "Undefined Activation Product: The new site type "
                f"'{new_active_type}' created by activation "
                f"in reaction {reaction_pair} is not defined anywhere in the system."
            )

        if original_dormant_type not in initial_site_statuses:
            raise ValueError(
                f"Undefined Activation Target: The site '{original_dormant_type}' "
                "targeted for activation in reaction "
                f"{reaction_pair} is not defined on any monomer."
            )

        if initial_site_statuses[original_dormant_type] != "DORMANT":
            raise ValueError(
                f"Invalid Activation Target: The site '{original_dormant_type}' "
                "targeted for activation in reaction "
                f"{reaction_pair} must be DORMANT, but it is defined as ACTIVE."
            )

    return True


def _calculate_conversion(
    sites_data: np.ndarray, monomer_data: np.ndarray, total_monomers: int
) -> float:
    """Calculate conversion as fraction of monomers with consumed sites.

    Args:
        sites_data: Array of site data with columns
            [monomer_id, site_type_id, status, monomer_site_idx].
        monomer_data: Array of monomer data with
            columns [monomer_type_id, first_site_idx].
        total_monomers: Total number of monomers in the system.

    Returns:
        Conversion as a fraction between 0 and 1.

    """
    reacted_monomers = set()

    # Find all monomers that have at least one consumed site
    for i in range(len(sites_data)):
        if sites_data[i, 2] == STATUS_CONSUMED:  # Check if site is consumed
            monomer_id = sites_data[i, 0]
            reacted_monomers.add(monomer_id)

    conversion = len(reacted_monomers) / total_monomers if total_monomers > 0 else 0.0
    return conversion


def run_simulation(config: SimulationInput) -> SimulationResult:
    """Run a polymer generation simulation.

    This function acts as a bridge between the user-friendly Pydantic/Python
    configuration and the high-performance Numba-JIT'd core.

    Args:
        config: Complete simulation configuration.

    Returns:
        A `SimulationResult` object containing the polymer graph, simulation
        metadata, and the original input configuration.

    Raises:
        ValueError: If configuration validation fails.

    """
    print("--- PolyMCsim Simulation ---")
    print("0. Validating configuration...")
    _validate_config(config)
    print("1. Translating inputs to Numba-compatible format...")

    np.random.seed(config.params.random_seed)

    # Mappings from string names to integer IDs for Numba
    all_site_types = set()
    for monomer in config.monomers:
        for site in monomer.sites:
            all_site_types.add(site.type)
    # Collect all site types from reactions
    for pair, reaction_def in config.reactions.items():
        all_site_types.update(pair)
        all_site_types.update(reaction_def.activation_map.values())

    # Create deterministic mappings
    site_type_map = {name: i for i, name in enumerate(sorted(list(all_site_types)))}
    monomer_type_map = {monomer.name: i for i, monomer in enumerate(config.monomers)}

    # Flatten data into NumPy arrays
    total_monomers = sum(monomer.count for monomer in config.monomers)
    total_sites = sum(monomer.count * len(monomer.sites) for monomer in config.monomers)

    # sites_data: [monomer_id, site_type_id, status, monomer_site_idx]
    sites_data = np.zeros((total_sites, 4), dtype=np.int64)
    # monomer_data: [monomer_type_id, first_site_idx]
    monomer_data = np.zeros((total_monomers + 1, 2), dtype=np.int64)

    # Pre-populate all possible keys in the NumbaDicts
    int_list_type = types.ListType(types.int64)
    available_sites_active = NumbaDict.empty(
        key_type=types.int64, value_type=int_list_type
    )
    available_sites_dormant = NumbaDict.empty(
        key_type=types.int64, value_type=int_list_type
    )
    site_position_map_active = NumbaDict.empty(
        key_type=types.int64, value_type=types.int64
    )
    site_position_map_dormant = NumbaDict.empty(
        key_type=types.int64, value_type=types.int64
    )
    for site_name, site_id in site_type_map.items():
        available_sites_active[site_id] = NumbaList.empty_list(types.int64)
        available_sites_dormant[site_id] = NumbaList.empty_list(types.int64)

    current_monomer_id = 0
    current_site_idx = 0
    for monomer_def in config.monomers:
        monomer_type_id = monomer_type_map[monomer_def.name]
        for _ in range(monomer_def.count):
            monomer_data[current_monomer_id, 0] = monomer_type_id
            monomer_data[current_monomer_id, 1] = current_site_idx
            for site_idx, site in enumerate(monomer_def.sites):
                site_type_id = site_type_map[site.type]
                status_int = (
                    STATUS_ACTIVE if site.status == "ACTIVE" else STATUS_DORMANT
                )

                sites_data[current_site_idx] = [
                    current_monomer_id,
                    site_type_id,
                    status_int,
                    site_idx,
                ]

                # Populate initial available site lists and position maps
                if status_int == STATUS_ACTIVE:
                    site_list = available_sites_active[site_type_id]
                    site_list.append(current_site_idx)
                    site_position_map_active[current_site_idx] = len(site_list) - 1
                elif status_int == STATUS_DORMANT:
                    site_list = available_sites_dormant[site_type_id]
                    site_list.append(current_site_idx)
                    site_position_map_dormant[current_site_idx] = len(site_list) - 1

                current_site_idx += 1
            current_monomer_id += 1
    monomer_data[total_monomers, 1] = total_sites  # Sentinel for size calculation

    # Translate kinetics with canonical ordering
    # First, create a map of site types to their status for easy lookup
    site_status_map: Dict[str, str] = {}
    for monomer in config.monomers:
        for site in monomer.sites:
            site_status_map.setdefault(site.type, site.status)
    # Ensure types that only appear after activation are marked ACTIVE
    for schema in config.reactions.values():
        for new_type in schema.activation_map.values():
            site_status_map.setdefault(new_type, "ACTIVE")

    # Build the reaction channel list with a guaranteed order
    reaction_channels_list = []
    is_ad_reaction_channel_list = []
    for pair in config.reactions.keys():
        pair_list = list(pair)

        # Handle self-reaction first
        if len(pair_list) == 1:
            reaction_channels_list.append((pair_list[0], pair_list[0]))
            is_ad_reaction_channel_list.append(False)
            continue

        type1, type2 = pair_list[0], pair_list[1]
        status1 = site_status_map.get(type1)
        status2 = site_status_map.get(type2)

        # Enforce canonical order: (Active, Dormant) or sorted(Active, Active)
        if status1 == "ACTIVE" and status2 == "DORMANT":
            reaction_channels_list.append((type1, type2))
            is_ad_reaction_channel_list.append(True)
        elif status1 == "DORMANT" and status2 == "ACTIVE":
            reaction_channels_list.append((type2, type1))  # SWAP to keep Active first
            is_ad_reaction_channel_list.append(True)
        else:  # Both ACTIVE (or both DORMANT, which is a non-reactive channel anyway)
            reaction_channels_list.append(tuple(sorted(pair_list)))
            is_ad_reaction_channel_list.append(False)

    # Test guards
    for pair_tuple in reaction_channels_list:
        pair_fs = frozenset(pair_tuple)
        if pair_fs not in config.reactions:
            raise ValueError(
                f"Configuration Error: The reaction pair {pair_fs} is defined in "
                "`rate_matrix` but is missing from `reaction_schema`. "
                "Every reaction must have a defined outcome."
            )

    # The rest of the translation now works with this canonical ordering
    num_reactions = len(reaction_channels_list)
    reaction_channels = np.array(
        [[site_type_map[p[0]], site_type_map[p[1]]] for p in reaction_channels_list],
        dtype=np.int64,
    )
    rate_constants = np.array(
        [config.reactions[frozenset(p)].rate for p in reaction_channels_list],
        dtype=np.float64,
    )
    is_ad_reaction_channel = np.array(is_ad_reaction_channel_list, dtype=np.bool_)
    is_self_reaction = np.array(
        [p[0] == p[1] for p in reaction_channels_list], dtype=np.bool_
    )

    activation_outcomes = np.full((num_reactions, 2), -1, dtype=np.int64)

    for i, pair_tuple in enumerate(reaction_channels_list):
        pair_fs = frozenset(pair_tuple)
        schema = config.reactions[pair_fs]
        if schema.activation_map:
            original_type, new_type = list(schema.activation_map.items())[0]
            activation_outcomes[i, 0] = site_type_map[original_type]
            activation_outcomes[i, 1] = site_type_map[new_type]

    print("2. Starting KMC simulation loop...")
    start_time = time.time()

    # Run the core simulation in chunks to provide progress updates
    total_reactions_to_run = config.params.max_reactions
    chunk_size = max(1, total_reactions_to_run // 100)  # Update 100 times

    all_edges = []
    reactions_done_total = 0
    final_time = 0.0

    # Only track conversion if max_conversion is less than 1.0
    track_conversion = config.params.max_conversion < 1.0
    current_conversion = 0.0

    with tqdm(total=total_reactions_to_run, desc="Simulating") as pbar:
        # Check initial conversion only if we're tracking it
        if track_conversion:
            current_conversion = _calculate_conversion(
                sites_data, monomer_data, total_monomers
            )
            if current_conversion >= config.params.max_conversion:
                print(
                    f"\nInitial conversion ({current_conversion:.2%}) already meets or "
                    f"exceeds max_conversion ({config.params.max_conversion:.2%})"
                )

        while reactions_done_total < total_reactions_to_run:
            # Check if we need to stop due to max_conversion
            if track_conversion and current_conversion >= config.params.max_conversion:
                print(
                    f"\nMax conversion ({config.params.max_conversion:.2%}) "
                    f"reached at {current_conversion:.2%}"
                )
                pbar.total = reactions_done_total
                pbar.refresh()
                break

            # Adapt chunk size based on how close we are to max_conversion
            if track_conversion:
                # Estimate remaining reactions allowed
                remaining_conversion = config.params.max_conversion - current_conversion
                if remaining_conversion > 0:
                    # Conservative estimate: assume each reaction converts 2 monomers
                    estimated_reactions_left = int(
                        remaining_conversion * total_monomers / 2
                    )
                    reactions_this_chunk = min(
                        max(
                            1, estimated_reactions_left // 10
                        ),  # Do in small chunks near limit
                        chunk_size,
                        total_reactions_to_run - reactions_done_total,
                    )
                else:
                    break
            else:
                reactions_this_chunk = min(
                    chunk_size, total_reactions_to_run - reactions_done_total
                )

            kmc_args = (
                sites_data,
                monomer_data,
                available_sites_active,
                available_sites_dormant,
                site_position_map_active,
                site_position_map_dormant,
                reaction_channels,
                rate_constants,
                is_ad_reaction_channel,
                is_self_reaction,
                activation_outcomes,
                config.params.max_time,
                reactions_this_chunk,
            )
            try:
                edges_chunk, reactions_in_chunk, final_time = run_kmc_loop(*kmc_args)
            except Exception as e:
                print(f"Error in KMC loop: {e}")
                raise e

            if edges_chunk:
                all_edges.extend(edges_chunk)

            reactions_done_total += reactions_in_chunk
            pbar.update(reactions_in_chunk)

            # Check conversion after this chunk only if we're tracking it
            if track_conversion:
                current_conversion = _calculate_conversion(
                    sites_data, monomer_data, total_monomers
                )
                pbar.set_postfix({"conversion": f"{current_conversion:.2%}"})

            if reactions_in_chunk < reactions_this_chunk:
                # KMC loop terminated early (no more reactions)
                pbar.total = reactions_done_total
                pbar.refresh()
                break

    end_time = time.time()

    # Calculate final conversion
    final_conversion = _calculate_conversion(sites_data, monomer_data, total_monomers)

    print(f"3. Simulation finished in {end_time - start_time:.4f} seconds.")
    print(f"   - Reactions: {reactions_done_total}")
    print(f"   - Final Sim Time: {final_time:.4e}")
    print(f"   - Final Conversion: {final_conversion:.2%}")

    # Build user-friendly NetworkX graph output
    print("4. Constructing NetworkX graph...")
    graph = nx.Graph()

    # Add nodes with attributes
    # Create a map from monomer_type_id not just to name but to the whole def
    monomer_def_map = {monomer_type_map[m.name]: m for m in config.monomers}

    for i in range(total_monomers):
        monomer_type_id = monomer_data[i, 0]
        monomer_def = monomer_def_map[monomer_type_id]
        graph.add_node(
            i, monomer_type=monomer_def.name, molar_mass=monomer_def.molar_mass
        )

    # Add edges with attributes
    for u, v, t in all_edges:
        graph.add_edge(int(u), int(v), formation_time=t)

    metadata = {
        "wall_time_seconds": end_time - start_time,
        "reactions_completed": reactions_done_total,
        "final_simulation_time": final_time,
        "final_conversion": final_conversion,
        "num_components": nx.number_connected_components(graph),
    }
    final_state = {
        "sites_data": sites_data,
        "monomer_data": monomer_data,
        "available_sites_active": available_sites_active,
        "available_sites_dormant": available_sites_dormant,
        "site_position_map_active": site_position_map_active,
        "site_position_map_dormant": site_position_map_dormant,
        "reaction_channels": reaction_channels,
        "rate_constants": rate_constants,
        "is_ad_reaction_channel": is_ad_reaction_channel,
        "is_self_reaction": is_self_reaction,
    }

    return SimulationResult(
        graph=graph, metadata=metadata, config=config, final_state=final_state
    )


def run_batch(
    configs: List[SimulationInput], max_workers: Optional[int] = None
) -> Dict[str, SimulationResult]:
    """Run a batch of simulations in parallel using a process pool.

    Args:
        configs: A list of simulation configurations.
        max_workers: The maximum number of worker processes to use.
                    If None, it defaults to the number of CPUs on the machine.

    Returns:
        A dictionary mapping simulation names to their `SimulationResult` objects.
        If a simulation fails, the `error` field of the result will be populated.

    """
    results = {}
    name_to_config = {cfg.params.name: cfg for cfg in configs}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {
            executor.submit(run_simulation, config): config.params.name
            for config in configs
        }

        for future in tqdm(
            as_completed(future_to_name), total=len(configs), desc="Batch Simulations"
        ):
            name = future_to_name[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                print(f"'{name}' generated an exception: {exc}")
                config = name_to_config[name]
                results[name] = SimulationResult(config=config, error=str(exc))
    return results


class Simulation:
    """A wrapper for the optimized PolyMCsim simulation engine."""

    def __init__(self, config: SimulationInput) -> None:
        """Initialize the simulation with a complete configuration.

        Args:
            config: The detailed simulation configuration object.

        """
        self.config = config
        self.result: Optional[SimulationResult] = None

    def run(self) -> SimulationResult:
        """Execute the simulation.

        This method calls the core Numba-optimized Kinetic Monte Carlo engine
        and runs the simulation to completion based on the provided configuration.
        It stores the result internally and also returns it.

        Returns:
            A `SimulationResult` object containing the final polymer network,
            metadata, and the input configuration.

        """
        self.result = run_simulation(self.config)
        return self.result

    def get_graph(self) -> Optional[nx.Graph]:
        """Return the resulting polymer graph.

        Returns:
            The polymer graph, or None if the simulation has not been run or failed.

        """
        return self.result.graph if self.result else None

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Return metadata from the simulation run.

        Returns:
            The metadata dictionary, or None if the simulation has not been
            run or failed.

        """
        return self.result.metadata if self.result else None
