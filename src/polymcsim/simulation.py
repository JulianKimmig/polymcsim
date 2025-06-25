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
    """Run a polymer generation simulation."""
    print("--- PolyMCsim Simulation ---")
    print("0. Validating configuration...")
    _validate_config(config)
    print("1. Translating inputs to Numba-compatible format...")

    np.random.seed(config.params.random_seed)

    # --- Mappings and Data Flattening ---
    all_site_types = set()
    for monomer in config.monomers:
        for site in monomer.sites:
            all_site_types.add(site.type)
    for pair, reaction_def in config.reactions.items():
        all_site_types.update(pair)
        all_site_types.update(reaction_def.activation_map.values())

    site_type_map = {
        name: np.int32(i) for i, name in enumerate(sorted(list(all_site_types)))
    }
    monomer_type_map = {
        monomer.name: np.int32(i) for i, monomer in enumerate(config.monomers)
    }

    total_monomers = sum(monomer.count for monomer in config.monomers)
    total_sites = sum(monomer.count * len(monomer.sites) for monomer in config.monomers)

    sites_data = np.zeros((total_sites, 4), dtype=np.int32)
    monomer_data = np.zeros((total_monomers + 1, 2), dtype=np.int32)

    # --- Vectorized Array Population ---
    monomer_counts = np.array([m.count for m in config.monomers])
    monomer_type_ids = np.array([monomer_type_map[m.name] for m in config.monomers])
    sites_per_monomer_type = np.array([len(m.sites) for m in config.monomers])

    if total_monomers > 0:
        all_monomer_type_ids = np.repeat(monomer_type_ids, monomer_counts)
        monomer_data[:total_monomers, 0] = all_monomer_type_ids
        sites_per_instance = np.repeat(sites_per_monomer_type, monomer_counts)
        monomer_data[:total_monomers, 1] = np.concatenate(
            ([0], np.cumsum(sites_per_instance)[:-1])
        )
    monomer_data[total_monomers, 1] = total_sites

    if total_sites > 0:
        site_props = [
            (
                [site_type_map[s.type] for s in m.sites],
                [
                    STATUS_ACTIVE if s.status == "ACTIVE" else STATUS_DORMANT
                    for s in m.sites
                ],
                list(range(len(m.sites))),
            )
            for m in config.monomers
        ]
        all_site_type_ids = np.concatenate(
            [props[0] * count for props, count in zip(site_props, monomer_counts)]
        )
        all_site_statuses = np.concatenate(
            [props[1] * count for props, count in zip(site_props, monomer_counts)]
        )
        all_monomer_site_indices = np.concatenate(
            [props[2] * count for props, count in zip(site_props, monomer_counts)]
        )
        all_monomer_ids = np.repeat(np.arange(total_monomers), sites_per_instance)
        sites_data[:, 0] = all_monomer_ids
        sites_data[:, 1] = all_site_type_ids
        sites_data[:, 2] = all_site_statuses
        sites_data[:, 3] = all_monomer_site_indices

    # --- Numba Collections Population ---
    int_list_type = types.ListType(types.int32)
    available_sites_active = NumbaDict.empty(
        key_type=types.int32, value_type=int_list_type
    )
    available_sites_dormant = NumbaDict.empty(
        key_type=types.int32, value_type=int_list_type
    )
    site_position_map_active = NumbaDict.empty(
        key_type=types.int32, value_type=types.int32
    )
    site_position_map_dormant = NumbaDict.empty(
        key_type=types.int32, value_type=types.int32
    )

    all_site_indices = np.arange(total_sites, dtype=np.int32)

    for site_type_id in site_type_map.values():
        is_site_type = sites_data[:, 1] == site_type_id

        # Handle ACTIVE sites
        active_mask = is_site_type & (sites_data[:, 2] == STATUS_ACTIVE)
        active_indices = all_site_indices[active_mask]
        # Create a typed list, handling the empty case to avoid TypeError
        if active_indices.size == 0:
            available_sites_active[site_type_id] = NumbaList.empty_list(types.int32)
        else:
            available_sites_active[site_type_id] = NumbaList(active_indices)

        if active_indices.size > 0:
            # pos_map_chunk = _build_position_map_jit(active_indices)
            pos_map_chunk = dict(
                zip(active_indices, np.arange(len(active_indices), dtype=np.int32))
            )
            site_position_map_active.update(pos_map_chunk)

        # Handle DORMANT sites
        dormant_mask = is_site_type & (sites_data[:, 2] == STATUS_DORMANT)
        dormant_indices = all_site_indices[dormant_mask]
        # Create a typed list, handling the empty case to avoid TypeError
        if dormant_indices.size == 0:
            available_sites_dormant[site_type_id] = NumbaList.empty_list(types.int32)
        else:
            available_sites_dormant[site_type_id] = NumbaList(dormant_indices)

        if dormant_indices.size > 0:
            pos_map_chunk = dict(
                zip(dormant_indices, np.arange(len(dormant_indices), dtype=np.int32))
            )
            site_position_map_dormant.update(pos_map_chunk)

    # --- Kinetics Translation ---
    site_status_map: Dict[str, str] = {}
    for monomer in config.monomers:
        for site in monomer.sites:
            site_status_map.setdefault(site.type, site.status)
    for schema in config.reactions.values():
        for new_type in schema.activation_map.values():
            site_status_map.setdefault(new_type, "ACTIVE")

    reaction_channels_list = []
    is_ad_reaction_channel_list = []
    for pair in config.reactions.keys():
        pair_list = list(pair)
        if len(pair_list) == 1:
            reaction_channels_list.append((pair_list[0], pair_list[0]))
            is_ad_reaction_channel_list.append(False)
            continue
        type1, type2 = pair_list[0], pair_list[1]
        status1, status2 = site_status_map.get(type1), site_status_map.get(type2)
        if status1 == "ACTIVE" and status2 == "DORMANT":
            reaction_channels_list.append((type1, type2))
            is_ad_reaction_channel_list.append(True)
        elif status1 == "DORMANT" and status2 == "ACTIVE":
            reaction_channels_list.append((type2, type1))
            is_ad_reaction_channel_list.append(True)
        else:
            reaction_channels_list.append(tuple(sorted(pair_list)))
            is_ad_reaction_channel_list.append(False)

    num_reactions = len(reaction_channels_list)
    reaction_channels = np.array(
        [[site_type_map[p[0]], site_type_map[p[1]]] for p in reaction_channels_list],
        dtype=np.int32,
    )
    rate_constants = np.array(
        [config.reactions[frozenset(p)].rate for p in reaction_channels_list],
        dtype=np.float64,
    )
    is_ad_reaction_channel = np.array(is_ad_reaction_channel_list, dtype=np.bool_)
    is_self_reaction = np.array(
        [p[0] == p[1] for p in reaction_channels_list], dtype=np.bool_
    )
    activation_outcomes = np.full((num_reactions, 2), -1, dtype=np.int32)
    for i, pair_tuple in enumerate(reaction_channels_list):
        schema = config.reactions[frozenset(pair_tuple)]
        if schema.activation_map:
            original_type, new_type = list(schema.activation_map.items())[0]
            activation_outcomes[i, 0] = site_type_map[original_type]
            activation_outcomes[i, 1] = site_type_map[new_type]

    # --- KMC Loop ---
    print("2. Starting KMC simulation loop...")
    start_time = time.time()
    total_reactions_to_run = config.params.max_reactions
    chunk_size = max(1, total_reactions_to_run // config.params.chunk_size)
    all_edges = []
    reactions_done_total = 0
    final_time = 0.0
    track_conversion = config.params.max_conversion < 1.0
    current_conversion = 0.0

    with tqdm(total=total_reactions_to_run, desc="Simulating") as pbar:
        if track_conversion:
            current_conversion = _calculate_conversion(
                sites_data, monomer_data, total_monomers
            )
            if current_conversion >= config.params.max_conversion:
                pbar.total = 0
                pbar.refresh()

        while reactions_done_total < total_reactions_to_run:
            if track_conversion and current_conversion >= config.params.max_conversion:
                pbar.total = reactions_done_total
                pbar.refresh()
                break

            reactions_this_chunk = chunk_size
            if track_conversion:
                remaining_conversion = config.params.max_conversion - current_conversion
                if remaining_conversion > 0:
                    # Estimate reactions needed to reach target, run a fraction of them
                    estimated_reactions_left = int(
                        remaining_conversion * total_monomers / 2
                    )
                    reactions_this_chunk = min(
                        max(1, estimated_reactions_left // 10), chunk_size
                    )

            reactions_this_chunk = min(
                reactions_this_chunk, total_reactions_to_run - reactions_done_total
            )
            if reactions_this_chunk <= 0:
                break

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
            edges_chunk, reactions_in_chunk, final_time = run_kmc_loop(*kmc_args)
            if edges_chunk:
                all_edges.extend(edges_chunk)
            reactions_done_total += reactions_in_chunk
            pbar.update(reactions_in_chunk)

            if track_conversion:
                current_conversion = _calculate_conversion(
                    sites_data, monomer_data, total_monomers
                )
                pbar.set_postfix({"conversion": f"{current_conversion:.2%}"})

            if reactions_in_chunk < reactions_this_chunk:
                pbar.total = reactions_done_total
                pbar.refresh()
                break
    end_time = time.time()

    final_conversion = _calculate_conversion(sites_data, monomer_data, total_monomers)
    print(f"3. Simulation finished in {end_time - start_time:.4f} seconds.")
    print(
        f"   - Reactions: {reactions_done_total}, Final Sim Time: {final_time:.4e}, "
        f"Final Conversion: {final_conversion:.2%}"
    )

    # --- OPTIMIZED Graph Construction ---
    print("4. Constructing NetworkX graph (Optimized)...")
    graph = nx.Graph()
    monomer_def_map = {monomer_type_map[m.name]: m for m in config.monomers}

    # Use add_nodes_from for bulk creation (MUCH faster)
    node_generator = (
        (
            i,
            {
                "monomer_type": monomer_def_map[monomer_data[i, 0]].name,
                "molar_mass": monomer_def_map[monomer_data[i, 0]].molar_mass,
            },
        )
        for i in range(total_monomers)
    )
    graph.add_nodes_from(node_generator)

    # Use add_edges_from for bulk creation (MUCH faster)
    if all_edges:
        edge_generator = (
            (int(u), int(v), {"formation_time": t}) for u, v, t in all_edges
        )
        graph.add_edges_from(edge_generator)

    # --- Final Result Packaging ---
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
