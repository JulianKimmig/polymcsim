import numpy as np
import numba
from numba import njit
from numba.core import types
from numba.typed import Dict as NumbaDict
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

@njit(cache=True)
def _select_reaction_channel(propensities: np.ndarray, total_propensity: float) -> int:
    """Selects a reaction channel based on its propensity."""
    rand_val = np.random.rand() * total_propensity
    cumulative = 0.0
    for i, p in enumerate(propensities):
        cumulative += p
        if rand_val < cumulative:
            return i
    return len(propensities) - 1

@njit(cache=True)
def _update_available_sites(available_sites, site_position_map, site_type_id, site_global_idx_to_remove):
    """
    Truly O(1) removal of a site from the available list.
    It swaps the element to be removed with the last element and pops.
    This requires a map from site_global_idx to its position in the list.
    """
    site_list = available_sites[site_type_id]
    
    # O(1) lookup for the position of the site to remove
    pos_to_remove = site_position_map[site_global_idx_to_remove]
    del site_position_map[site_global_idx_to_remove]
    last_site_idx_in_list = len(site_list) - 1

    if pos_to_remove != last_site_idx_in_list:
        # If we're not removing the last element, move the last element to this position
        last_site_global_idx = site_list[last_site_idx_in_list]
        site_list[pos_to_remove] = last_site_global_idx
        # Update the position map for the moved element
        site_position_map[last_site_global_idx] = pos_to_remove

    # Remove the last element
    site_list.pop()


@njit(cache=True)
def _run_kmc_loop(
    sites_data: np.ndarray,
    monomer_data: np.ndarray,
    available_sites_active: numba_dict_type,
    available_sites_dormant: numba_dict_type,
    site_position_map_active: int_to_int_dict_type,
    site_position_map_dormant: int_to_int_dict_type,
    reaction_channels: np.ndarray,
    rate_constants: np.ndarray,
    is_ad_reaction_channel: np.ndarray,
    is_self_reaction: np.ndarray,
    activation_outcomes: np.ndarray,
    max_time: float,
    max_reactions: int,
):
    """
    The core, high-performance Kinetic Monte Carlo simulation loop.
    This function is pure Numba and only operates on NumPy arrays and Numba-typed collections.
    
    Args:
        sites_data (np.ndarray): Shape (N_sites, 4). Cols: [monomer_id, site_type_id, status, monomer_site_idx]
        monomer_data (np.ndarray): Shape (N_monomers, 2). Cols: [monomer_type_id, first_site_idx]
        available_sites_*: Numba Dicts mapping site_type_id to a Numba List of global site indices.
        site_position_map_*: Numba Dicts mapping site_global_idx to its position in the corresponding available_sites list.
        reaction_channels (np.ndarray): Shape (N_reactions, 2). Pairs of reacting site_type_ids.
        rate_constants (np.ndarray): Shape (N_reactions,). Rate constant for each channel.
        is_ad_reaction_channel (np.ndarray): Shape (N_reactions,). Boolean array indicating active-dormant reaction channels.
        is_self_reaction (np.ndarray): Shape (N_reactions,). Boolean array indicating self-reaction channels.
        activation_outcomes (np.ndarray): Shape (N_reactions, 2). [target_dormant_type, new_active_type]
    """
    sim_time = 0.0
    reaction_count = 0
    
    # Store edges as (u, v, time)
    edges = NumbaList()
    
    propensities = np.zeros(len(reaction_channels), dtype=np.float64)

    while sim_time < max_time and reaction_count < max_reactions:
        # 1. Calculate Propensities
        total_propensity = 0.0
        for i in range(len(reaction_channels)):
            type1_id, type2_id = reaction_channels[i]
            
            n1 = len(available_sites_active.get(type1_id, NumbaList.empty_list(types.int64)))
            
            if is_ad_reaction_channel[i]:
                n2 = len(available_sites_dormant.get(type2_id, NumbaList.empty_list(types.int64)))
                prop = rate_constants[i] * n1 * n2
            else: # Active-Active reaction
                n2 = len(available_sites_active.get(type2_id, NumbaList.empty_list(types.int64)))
                if is_self_reaction[i]:
                    # Correction for reacting with the same type
                    prop = rate_constants[i] * n1 * (n1 - 1) * 0.5
                else:
                    prop = rate_constants[i] * n1 * n2
            
            propensities[i] = prop
            total_propensity += prop

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
        max_attempts = 100
        for _ in range(max_attempts):
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
        _update_available_sites(available_sites_active, site_position_map_active, type1_id, site1_global_idx)
        if is_ad_reaction:
            _update_available_sites(available_sites_dormant, site_position_map_dormant, type2_id, site2_global_idx)
        else:
            _update_available_sites(available_sites_active, site_position_map_active, type2_id, site2_global_idx)

        # Handle activation on the second monomer
        target_dormant_type, new_active_type = activation_outcomes[channel_idx]
        if new_active_type != -1:
            # Find the site on monomer2 that needs activation
            monomer2_type_id = monomer_data[monomer2_id, 0]
            monomer2_first_site = monomer_data[monomer2_id, 1]
            num_sites_on_monomer = monomer_data[monomer2_id + 1, 1] - monomer2_first_site if monomer2_id + 1 < len(monomer_data) else len(sites_data) - monomer2_first_site

            for s_offset in range(num_sites_on_monomer):
                site_to_check_idx = monomer2_first_site + s_offset
                if sites_data[site_to_check_idx, 1] == target_dormant_type:
                    # Found it. Activate it.
                    sites_data[site_to_check_idx, 1] = new_active_type
                    sites_data[site_to_check_idx, 2] = STATUS_ACTIVE
                    # Add to the active list and update its position map
                    active_list = available_sites_active[new_active_type]
                    active_list.append(site_to_check_idx)
                    site_position_map_active[site_to_check_idx] = len(active_list) - 1
                    # Remove from dormant list
                    _update_available_sites(available_sites_dormant, site_position_map_dormant, target_dormant_type, site_to_check_idx)
                    break
                    
        reaction_count += 1
        
    return edges, reaction_count, sim_time 