"""Core Kinetic Monte Carlo simulation engine for PolyMCsim."""

from typing import List, Tuple

import numpy as np
from numba import njit
from numba.core import types
from numba.typed import List as NumbaList

# --- Numba-Optimized KMC Simulation Engine ---

# Numba requires integer keys for typed dicts
int_list_type = types.ListType(types.int64)
numba_dict_type = types.DictType(types.int64, int_list_type)
int_to_int_dict_type = types.DictType(types.int64, types.int64)

# Define integer constants for site statuses for Numba
STATUS_ACTIVE = 1
STATUS_DORMANT = 0
STATUS_CONSUMED = -1

# Maximum attempts to find different monomers in a reaction
MAX_MONOMER_SELECTION_ATTEMPTS = 100


@njit(cache=True)
def _select_reaction_channel(propensities: np.ndarray, total_propensity: float) -> int:
    """Select a reaction channel based on its propensity.

    Args:
        propensities: Array of reaction propensities.
        total_propensity: Sum of all propensities.

    Returns:
        Index of the selected reaction channel.

    """
    rand_val = np.random.rand() * total_propensity
    cumulative = 0.0
    for i, propensity in enumerate(propensities):
        cumulative += propensity
        if rand_val < cumulative:
            return i
    return len(propensities) - 1


@njit
def _remove_from_available_sites(
    available_sites: types.DictType(types.int64, int_list_type),
    site_position_map: types.DictType(types.int64, types.int64),
    site_type_id: int,
    site_global_idx_to_remove: int,
):
    """Remove a site from the list of available sites in-place.

    This is an O(1) operation.

    Args:
        available_sites: Dictionary mapping site_type_id to list of global
            site indices.
        site_position_map: Dictionary mapping site_global_idx to its position
            in the list.
        site_type_id: Type ID of the site to remove.
        site_global_idx_to_remove: Global index of the site to remove.

    """
    sites_list = available_sites[site_type_id]
    pos_to_remove = site_position_map[site_global_idx_to_remove]
    del site_position_map[site_global_idx_to_remove]

    # To avoid a slow pop(0) or pop(i), we move the last element
    # to the position of the one we want to remove, and then pop the end.
    last_site_in_list = sites_list[-1]
    sites_list[pos_to_remove] = last_site_in_list
    site_position_map[last_site_in_list] = pos_to_remove
    sites_list.pop()


@njit
def _add_to_available_sites(
    available_sites: types.DictType(types.int64, int_list_type),
    site_position_map: types.DictType(types.int64, types.int64),
    site_type_id: int,
    site_global_idx_to_add: int,
):
    """Add a site to the available list using O(1) swap-and-pop.

    It swaps the element to be added with the last element and pops.
    This requires a map from site_global_idx to its position in the list.

    Args:
        available_sites: Dict mapping site_type_id to list of global site indices.
        site_position_map: Dict mapping site_global_idx to its position in the list.
        site_type_id: Type ID of the site to add.
        site_global_idx_to_add: Global index of the site to add.

    """
    site_list = available_sites[site_type_id]

    # O(1) lookup for the position of the site to remove
    pos_to_remove = site_position_map[site_global_idx_to_add]
    del site_position_map[site_global_idx_to_add]
    last_site_idx_in_list = len(site_list) - 1

    if pos_to_remove != last_site_idx_in_list:
        # If we're not removing the last element, move the last element to this position
        last_site_global_idx = site_list[last_site_idx_in_list]
        site_list[pos_to_remove] = last_site_global_idx
        # Update the position map for the moved element
        site_position_map[last_site_global_idx] = pos_to_remove

    # Remove the last element
    site_list.pop()


def _run_kmc_loop(
    sites_data: np.ndarray,
    monomer_data: np.ndarray,
    available_sites_active: types.DictType(types.int64, int_list_type),
    available_sites_dormant: types.DictType(types.int64, int_list_type),
    site_position_map_active: types.DictType(types.int64, types.int64),
    site_position_map_dormant: types.DictType(types.int64, types.int64),
    reaction_channels: np.ndarray,
    rate_constants: np.ndarray,
    is_ad_reaction_channel: np.ndarray,
    is_self_reaction: np.ndarray,
    activation_outcomes: np.ndarray,
    max_time: float,
    max_reactions: int,
):
    """Run the core Kinetic Monte Carlo simulation loop.

    This function is pure Numba and only operates on NumPy arrays and
    Numba-typed collections.

    Args:
        sites_data: Shape (N_sites, 4). Cols: [monomer_id, site_type_id,
            status, monomer_site_idx].
        monomer_data: Shape (N_monomers, 2). Cols: [monomer_type_id,
            first_site_idx].
        available_sites_active: Numba Dicts mapping site_type_id to a Numba
            List of global site indices.
        available_sites_dormant: Numba Dicts mapping site_type_id to a Numba
            List of global site indices.
        site_position_map_active: Numba Dicts mapping site_global_idx to its
            position in the active list.
        site_position_map_dormant: Numba Dicts mapping site_global_idx to its
            position in the dormant list.
        reaction_channels: Shape (N_reactions, 2). Pairs of reacting
            site_type_ids.
        rate_constants: Shape (N_reactions,). Rate constant for each channel.
        is_ad_reaction_channel: Shape (N_reactions,). Boolean array indicating
            active-dormant reaction channels.
        is_self_reaction: Shape (N_reactions,). Boolean array indicating
            self-reaction channels.
        activation_outcomes: Shape (N_reactions, 2).
            [target_dormant_type, new_active_type].
        max_time: Maximum simulation time to run.
        max_reactions: Maximum number of reaction events.

    Returns:
        Tuple of (edges, reaction_count, sim_time) where edges is a list of
        (u, v, time) tuples.

    """
    sim_time = 0.0
    reaction_count = 0

    # Store edges as (u, v, time)
    edges: List[Tuple[int, int, float]] = NumbaList()

    propensities = np.zeros(len(reaction_channels), dtype=np.float64)

    while sim_time < max_time and reaction_count < max_reactions:
        # 1. Calculate Propensities
        total_propensity = 0.0
        for i in range(len(reaction_channels)):
            type1_id, type2_id = reaction_channels[i]

            n1_list = available_sites_active.get(
                type1_id, NumbaList.empty_list(types.int64)
            )
            n1 = len(n1_list)

            if is_ad_reaction_channel[i]:
                n2_list = available_sites_dormant.get(
                    type2_id, NumbaList.empty_list(types.int64)
                )
                n2 = len(n2_list)
                propensity = rate_constants[i] * n1 * n2
            else:  # Active-Active reaction
                n2_list = available_sites_active.get(
                    type2_id, NumbaList.empty_list(types.int64)
                )
                n2 = len(n2_list)
                if is_self_reaction[i]:
                    # Correction for reacting with the same type
                    propensity = rate_constants[i] * n1 * (n1 - 1) * 0.5
                else:
                    propensity = rate_constants[i] * n1 * n2

            propensities[i] = propensity
            total_propensity += propensity

        if total_propensity == 0:
            print("No more reactions possible. Halting.")
            break

        # 2. Advance Time
        dt = -np.log(np.random.rand()) / total_propensity
        sim_time += dt

        # 3. Select Reaction
        channel_idx = _select_reaction_channel(propensities, total_propensity)
        type1_id, type2_id = reaction_channels[channel_idx]

        # 4. Select Reactants
        is_ad_reaction = is_ad_reaction_channel[channel_idx]

        list1 = available_sites_active[type1_id]
        if is_ad_reaction:
            list2 = available_sites_dormant[type2_id]
        else:
            list2 = available_sites_active[type2_id]

        # Ensure we pick two different monomers
        for _ in range(MAX_MONOMER_SELECTION_ATTEMPTS):
            idx1_in_list = np.random.randint(0, len(list1))
            site1_global_idx = list1[idx1_in_list]
            monomer1_id = sites_data[site1_global_idx, 0]

            if type1_id == type2_id and len(list2) > 1:
                # Avoid picking the same site twice
                idx2_in_list = np.random.randint(0, len(list2))
                while idx1_in_list == idx2_in_list:
                    idx2_in_list = np.random.randint(0, len(list2))
            else:
                idx2_in_list = np.random.randint(0, len(list2))

            site2_global_idx = list2[idx2_in_list]
            monomer2_id = sites_data[site2_global_idx, 0]

            if monomer1_id != monomer2_id:
                break
        else:
            # Could not find two different monomers, skip this step.
            # This can happen in late-stage gelation.
            continue

        # 5. Execute Reaction: Update System State
        # Add edge to graph
        edges.append((monomer1_id, monomer2_id, sim_time))

        # Update site statuses
        sites_data[site1_global_idx, 2] = STATUS_CONSUMED
        sites_data[site2_global_idx, 2] = STATUS_CONSUMED

        # Remove reacted sites from available lists (O(1) swap-and-pop)
        _remove_from_available_sites(
            available_sites_active, site_position_map_active, type1_id, site1_global_idx
        )
        if is_ad_reaction:
            _remove_from_available_sites(
                available_sites_dormant,
                site_position_map_dormant,
                type2_id,
                site2_global_idx,
            )
        else:
            _remove_from_available_sites(
                available_sites_active,
                site_position_map_active,
                type2_id,
                site2_global_idx,
            )

        # Handle activation on the second monomer
        target_dormant_type, new_active_type = activation_outcomes[channel_idx]
        if new_active_type != -1:
            # Find the site on monomer2 that needs activation
            monomer2_first_site = monomer_data[monomer2_id, 1]
            num_sites_on_monomer = (
                monomer_data[monomer2_id + 1, 1] - monomer2_first_site
                if (monomer2_id + 1) < monomer_data.shape[0]
                else sites_data.shape[0] - monomer2_first_site
            )
            for s_offset in range(num_sites_on_monomer):
                site_to_check_idx = monomer2_first_site + s_offset
                # We need to activate a DORMANT site that is NOT the one
                # that just reacted.

                # Condition 1: It must be the correct dormant type.
                is_correct_dormant_type = (
                    sites_data[site_to_check_idx, 1] == target_dormant_type
                )

                # Condition 2: It must not be the site that just participated
                # in the reaction.
                is_not_the_reacted_site = site_to_check_idx != site2_global_idx

                if is_correct_dormant_type and is_not_the_reacted_site:
                    # Activate this site
                    sites_data[site_to_check_idx, 1] = new_active_type
                    sites_data[site_to_check_idx, 2] = STATUS_ACTIVE
                    # Add to active list
                    active_list = available_sites_active[new_active_type]
                    active_list.append(site_to_check_idx)
                    site_position_map_active[site_to_check_idx] = len(active_list) - 1
                    # Remove from dormant list
                    _remove_from_available_sites(
                        available_sites_dormant,
                        site_position_map_dormant,
                        target_dormant_type,
                        site_to_check_idx,
                    )
                    break  # Found and activated, no need to check other sites

        reaction_count += 1

    return edges, reaction_count, sim_time


# JIT-compile the main simulation loop
run_kmc_loop = njit(cache=True)(_run_kmc_loop)
