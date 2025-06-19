import time
import numpy as np
import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any, Literal

import numba
from numba import njit
from numba.core import types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

# --- 1. Pydantic Models for User Input and Validation ---

class SiteDef(BaseModel):
    """Defines a single reactive site on a monomer."""
    type: str = Field(..., description="The type of the reactive site (e.g., 'A', 'Radical').")
    status: Literal['ACTIVE', 'DORMANT'] = Field('ACTIVE', description="Initial status of the site.")

class MonomerDef(BaseModel):
    """Defines a type of monomer in the system."""
    name: str
    count: int = Field(..., gt=0, description="Number of these monomers to add to the system.")
    sites: List[SiteDef]

class ReactionSchema(BaseModel):
    """Describes the outcome of a reaction between two site types."""
    site1_final_status: Literal['CONSUMED']
    site2_final_status: Literal['CONSUMED']
    # Describes which site on the SECOND monomer is activated, and what it becomes.
    # Format: {"original_dormant_type": "new_active_type"}
    activation_map: Dict[str, str] = Field(default_factory=dict)

class SimParams(BaseModel):
    """Parameters to control the simulation execution."""
    max_time: float = Field(default=float('inf'), description="Maximum simulation time to run.")
    max_reactions: int = Field(default=1_000_000_000, description="Maximum number of reaction events.")
    random_seed: int = 42

class SimulationInput(BaseModel):
    """The complete input configuration for a PolySim simulation."""
    monomers: List[MonomerDef]
    # Rate constants k for each reactive pair. The key is a frozenset to be order-independent.
    rate_matrix: Dict[frozenset[str], float]
    # Defines what happens to sites after a reaction.
    reaction_schema: Dict[frozenset[str], ReactionSchema]
    params: SimParams = Field(default_factory=SimParams)


# --- 2. The Core Numba-Optimized KMC Simulation Engine ---

# Numba requires integer keys for typed dicts
int_list_type = types.ListType(types.int64)
numba_dict_type = types.DictType(types.int64, int_list_type)

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
def _update_available_sites(available_sites, site_type_id, site_global_idx):
    """
    Optimized O(1) removal of a site from the available list.
    It swaps the element to be removed with the last element and pops.
    """
    site_list = available_sites[site_type_id]
    # Find the index of the site_global_idx in the list
    # This is O(N) but N is the number of sites of one type, often manageable.
    # A dict mapping site_idx -> list_idx could optimize this, but adds complexity.
    idx_in_list = -1
    for i in range(len(site_list)):
        if site_list[i] == site_global_idx:
            idx_in_list = i
            break
    
    if idx_in_list != -1:
        last_item = site_list.pop()
        if idx_in_list < len(site_list):
            site_list[idx_in_list] = last_item


@njit(nopython=True, cache=True)
def _run_kmc_loop(
    sites_data: np.ndarray,
    monomer_data: np.ndarray,
    available_sites_active: numba_dict_type,
    available_sites_dormant: numba_dict_type,
    reaction_channels: np.ndarray,
    rate_constants: np.ndarray,
    reaction_outcomes: np.ndarray,
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
        reaction_channels (np.ndarray): Shape (N_reactions, 2). Pairs of reacting site_type_ids.
        rate_constants (np.ndarray): Shape (N_reactions,). Rate constant for each channel.
        reaction_outcomes (np.ndarray): Shape (N_reactions, 2). Final statuses for sites 1 and 2.
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
            
            # Check if this is an Active-Dormant reaction
            is_ad_reaction = type2_id in available_sites_dormant
            
            n1 = len(available_sites_active.get(type1_id, NumbaList.empty_list(types.int64)))
            
            if is_ad_reaction:
                n2 = len(available_sites_dormant.get(type2_id, NumbaList.empty_list(types.int64)))
                prop = rate_constants[i] * n1 * n2
            else: # Active-Active reaction
                n2 = len(available_sites_active.get(type2_id, NumbaList.empty_list(types.int64)))
                if type1_id == type2_id:
                    # Correction for reacting with the same type
                    prop = rate_constants[i] * n1 * (n1 - 1) / 2.0
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
        is_ad_reaction = type2_id in available_sites_dormant

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
        _update_available_sites(available_sites_active, type1_id, site1_global_idx)
        if is_ad_reaction:
            _update_available_sites(available_sites_dormant, type2_id, site2_global_idx)
        else:
            _update_available_sites(available_sites_active, type2_id, site2_global_idx)

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
                    # Add to the active list
                    available_sites_active[new_active_type].append(site_to_check_idx)
                    # Remove from dormant list
                    _update_available_sites(available_sites_dormant, target_dormant_type, site_to_check_idx)
                    break
                    
        reaction_count += 1
        
    return edges, reaction_count, sim_time


# --- 3. The Main Wrapper Function ---

def run_simulation(config: SimulationInput) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Main function to configure and run a polymer generation simulation.

    This function acts as a bridge between the user-friendly Pydantic/Python
    configuration and the high-performance Numba-JIT'd core.
    """
    print("--- PolySim Simulation ---")
    print("1. Translating inputs to Numba-compatible format...")
    
    np.random.seed(config.params.random_seed)

    # Mappings from string names to integer IDs for Numba
    all_site_types = set()

    # 1. From monomer definitions (finds 'Head', 'Tail', etc.)
    for m in config.monomers:
        for s in m.sites:
            all_site_types.add(s.type)

    # 2. From rate matrix keys (finds 'Radical', 'I', etc.)
    for pair in config.rate_matrix.keys():
        all_site_types.update(pair)

    # 3. From reaction schema activation maps (finds what sites become)
    for schema in config.reaction_schema.values():
        for new_type in schema.activation_map.values():
            all_site_types.add(new_type)

    # Now create the map. Sorting makes the mapping deterministic.
    site_type_map = {name: i for i, name in enumerate(sorted(list(all_site_types)))}    
    monomer_type_map = {m.name: i for i, m in enumerate(config.monomers)}
    
    # --- Flatten data into NumPy arrays ---
    total_monomers = sum(m.count for m in config.monomers)
    total_sites = sum(m.count * len(m.sites) for m in config.monomers)

    # sites_data: [monomer_id, site_type_id, status, monomer_site_idx]
    sites_data = np.zeros((total_sites, 4), dtype=np.int64)
    # monomer_data: [monomer_type_id, first_site_idx]
    monomer_data = np.zeros((total_monomers + 1, 2), dtype=np.int64) 
    
    # Pre-populate all possible keys in the NumbaDicts
    available_sites_active = NumbaDict.empty(key_type=types.int64, value_type=int_list_type)
    available_sites_dormant = NumbaDict.empty(key_type=types.int64, value_type=int_list_type)
    for site_name, site_id in site_type_map.items():
        available_sites_active[site_id] = NumbaList.empty_list(types.int64)
        available_sites_dormant[site_id] = NumbaList.empty_list(types.int64)


    current_monomer_id = 0
    current_site_idx = 0
    for m_def in config.monomers:
        monomer_type_id = monomer_type_map[m_def.name]
        for i in range(m_def.count):
            monomer_data[current_monomer_id, 0] = monomer_type_id
            monomer_data[current_monomer_id, 1] = current_site_idx
            for s_idx, site in enumerate(m_def.sites):
                site_type_id = site_type_map[site.type]
                status_int = STATUS_ACTIVE if site.status == 'ACTIVE' else STATUS_DORMANT
                
                sites_data[current_site_idx] = [current_monomer_id, site_type_id, status_int, s_idx]
                
                # Populate initial available site lists
                if status_int == STATUS_ACTIVE:
                    if site_type_id not in available_sites_active:
                        available_sites_active[site_type_id] = NumbaList.empty_list(types.int64)
                    available_sites_active[site_type_id].append(current_site_idx)
                elif status_int == STATUS_DORMANT:
                    if site_type_id not in available_sites_dormant:
                        available_sites_dormant[site_type_id] = NumbaList.empty_list(types.int64)
                    available_sites_dormant[site_type_id].append(current_site_idx)

                current_site_idx += 1
            current_monomer_id += 1
    monomer_data[total_monomers, 1] = total_sites # Sentinel for size calculation

    # --- Translate kinetics with CANONICAL ORDERING ---

    # 1. First, create a map of site types to their status for easy lookup.
    site_status_map = {}
    for m in config.monomers:
        for s in m.sites:
            site_status_map.setdefault(s.type, s.status)
    # Ensure types that only appear after activation are marked ACTIVE
    for schema in config.reaction_schema.values():
        for new_type in schema.activation_map.values():
             site_status_map.setdefault(new_type, 'ACTIVE')
             
    # 2. Now, build the reaction channel list with a guaranteed order.
    reaction_channels_list = []
    for pair in config.rate_matrix.keys():
        pair_list = list(pair)
        
        # Handle self-reaction first
        if len(pair_list) == 1:
            reaction_channels_list.append((pair_list[0], pair_list[0]))
            continue

        type1, type2 = pair_list[0], pair_list[1]
        status1 = site_status_map.get(type1)
        status2 = site_status_map.get(type2)

        # Enforce canonical order: (Active, Dormant) or sorted(Active, Active)
        if status1 == 'ACTIVE' and status2 == 'DORMANT':
            reaction_channels_list.append((type1, type2))
        elif status1 == 'DORMANT' and status2 == 'ACTIVE':
            reaction_channels_list.append((type2, type1)) # SWAP to keep Active first
        else: # Both ACTIVE (or both DORMANT, which is a non-reactive channel anyway)
            reaction_channels_list.append(tuple(sorted(pair_list)))
    
    # 3. The rest of the translation now works with this canonical ordering.
    num_reactions = len(reaction_channels_list)
    reaction_channels = np.array([[site_type_map[p[0]], site_type_map[p[1]]] for p in reaction_channels_list], dtype=np.int64)
    rate_constants = np.array([config.rate_matrix[frozenset(p)] for p in reaction_channels_list], dtype=np.float64)
    
    reaction_outcomes = np.full((num_reactions, 2), STATUS_CONSUMED, dtype=np.int64)
    activation_outcomes = np.full((num_reactions, 2), -1, dtype=np.int64)

    for i, pair_tuple in enumerate(reaction_channels_list):
        pair_fs = frozenset(pair_tuple)
        schema = config.reaction_schema[pair_fs]
        if schema.activation_map:
            original_type, new_type = list(schema.activation_map.items())[0]
            activation_outcomes[i, 0] = site_type_map[original_type]
            activation_outcomes[i, 1] = site_type_map[new_type]
            
    print("2. Starting KMC simulation loop...")
    start_time = time.time()
    
    # --- Run the core simulation ---
    edges, reactions_done, final_time = _run_kmc_loop(
        sites_data,
        monomer_data,
        available_sites_active,
        available_sites_dormant,
        reaction_channels,
        rate_constants,
        reaction_outcomes,
        activation_outcomes,
        config.params.max_time,
        config.params.max_reactions
    )
    
    end_time = time.time()
    print(f"3. Simulation finished in {end_time - start_time:.4f} seconds.")
    print(f"   - Reactions: {reactions_done}")
    print(f"   - Final Sim Time: {final_time:.4e}")

    # --- 4. Build user-friendly NetworkX graph output ---
    print("4. Constructing NetworkX graph...")
    G = nx.Graph()
    
    # Add nodes with attributes
    rev_monomer_map = {i: name for name, i in monomer_type_map.items()}
    for i in range(total_monomers):
        m_type_id = monomer_data[i, 0]
        G.add_node(i, monomer_type=rev_monomer_map[m_type_id])
    
    # Add edges with attributes
    for u, v, t in edges:
        G.add_edge(int(u), int(v), formation_time=t)
        
    metadata = {
        "wall_time_seconds": end_time - start_time,
        "reactions_completed": reactions_done,
        "final_simulation_time": final_time,
        "num_components": nx.number_connected_components(G),
        "config": config.model_dump()
    }
    
    return G, metadata

# --- Example Usage ---

if __name__ == "__main__":

    # --- EXAMPLE 1: Step-Growth Polymerization (A2 + B3 -> Crosslinked Gel) ---
    print("\n" + "="*50)
    print("Running Example 1: Step-Growth Crosslinking (A2 + B3)")
    print("="*50)

    config_step_growth = SimulationInput(
        monomers=[
            MonomerDef(name="A2_Diamine", count=3000, sites=[
                SiteDef(type="A_Amine", status="ACTIVE"),
                SiteDef(type="A_Amine", status="ACTIVE"),
            ]),
            MonomerDef(name="B3_AcidChloride", count=2000, sites=[
                SiteDef(type="B_AcidCl", status="ACTIVE"),
                SiteDef(type="B_AcidCl", status="ACTIVE"),
                SiteDef(type="B_AcidCl", status="ACTIVE"),
            ]),
        ],
        rate_matrix={
            frozenset(["A_Amine", "B_AcidCl"]): 1.0,
        },
        reaction_schema={
            frozenset(["A_Amine", "B_AcidCl"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
            )
        },
        params=SimParams(max_reactions=4950, random_seed=101) # Stop just before gel point for this system
    )

    graph_sg, meta_sg = run_simulation(config_step_growth)
    
    # Basic analysis
    if graph_sg.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(graph_sg), key=len)
        print(f"\nAnalysis (Step-Growth):")
        print(f"  - Total nodes: {graph_sg.number_of_nodes()}")
        print(f"  - Total edges: {graph_sg.number_of_edges()}")
        print(f"  - Number of polymer chains/networks: {meta_sg['num_components']}")
        print(f"  - Size of largest polymer: {len(largest_cc)} monomers")


    # --- EXAMPLE 2: Chain-Growth Radical Polymerization (Styrene) ---
    print("\n" + "="*50)
    print("Running Example 2: Chain-Growth Radical Polymerization")
    print("="*50)
    
    config_chain_growth = SimulationInput(
        monomers=[
            MonomerDef(name="Initiator", count=50, sites=[
                SiteDef(type="I", status="ACTIVE")
            ]),
            MonomerDef(name="Styrene", count=5000, sites=[
                SiteDef(type="Head", status="DORMANT"),
                SiteDef(type="Tail", status="DORMANT"),
            ]),
        ],
        rate_matrix={
            frozenset(["Radical", "Head"]): 1000.0,  # Propagation
            frozenset(["I", "Head"]): 1.0,          # Initiation
            frozenset(["Radical", "Radical"]): 100.0 # Termination
        },
        reaction_schema={
            # Initiation: Initiator radical attacks a monomer head
            frozenset(["I", "Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"Tail": "Radical"} # The tail becomes a new radical
            ),
            # Propagation: Polymer radical attacks a new monomer head
            frozenset(["Radical", "Head"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
                activation_map={"Tail": "Radical"}
            ),
            # Termination by combination
            frozenset(["Radical", "Radical"]): ReactionSchema(
                site1_final_status="CONSUMED",
                site2_final_status="CONSUMED",
            )
        },
        params=SimParams(max_reactions=5000, random_seed=202)
    )

    graph_cg, meta_cg = run_simulation(config_chain_growth)

    # Basic analysis
    if graph_cg.number_of_nodes() > 0:
        components = list(nx.connected_components(graph_cg))
        # Filter out unreacted monomers (isolates)
        polymer_chains = [c for c in components if len(c) > 1]
        print(f"\nAnalysis (Chain-Growth):")
        print(f"  - Total nodes: {graph_cg.number_of_nodes()}")
        print(f"  - Total edges: {graph_cg.number_of_edges()}")
        print(f"  - Number of polymer chains formed: {len(polymer_chains)}")
        if polymer_chains:
             print(f"  - Avg chain length: {np.mean([len(c) for c in polymer_chains]):.2f} monomers")
             print(f"  - Max chain length: {max(len(c) for c in polymer_chains)} monomers")