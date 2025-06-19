import time
import numpy as np
import networkx as nx
from typing import Tuple, Dict, Any, List
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .schemas import SimulationInput, MonomerDef, SiteDef, ReactionSchema, SimParams
from .core import _run_kmc_loop, STATUS_ACTIVE, STATUS_DORMANT, STATUS_CONSUMED
from numba.core import types
from numba.typed import Dict as NumbaDict
from numba.typed import List as NumbaList

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
    int_list_type = types.ListType(types.int64)
    available_sites_active = NumbaDict.empty(key_type=types.int64, value_type=int_list_type)
    available_sites_dormant = NumbaDict.empty(key_type=types.int64, value_type=int_list_type)
    site_position_map_active = NumbaDict.empty(key_type=types.int64, value_type=types.int64)
    site_position_map_dormant = NumbaDict.empty(key_type=types.int64, value_type=types.int64)
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
    is_ad_reaction_channel_list = []
    for pair in config.rate_matrix.keys():
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
        if status1 == 'ACTIVE' and status2 == 'DORMANT':
            reaction_channels_list.append((type1, type2))
            is_ad_reaction_channel_list.append(True)
        elif status1 == 'DORMANT' and status2 == 'ACTIVE':
            reaction_channels_list.append((type2, type1)) # SWAP to keep Active first
            is_ad_reaction_channel_list.append(True)
        else: # Both ACTIVE (or both DORMANT, which is a non-reactive channel anyway)
            reaction_channels_list.append(tuple(sorted(pair_list)))
            is_ad_reaction_channel_list.append(False)
    
    # 3. The rest of the translation now works with this canonical ordering.
    num_reactions = len(reaction_channels_list)
    reaction_channels = np.array([[site_type_map[p[0]], site_type_map[p[1]]] for p in reaction_channels_list], dtype=np.int64)
    rate_constants = np.array([config.rate_matrix[frozenset(p)] for p in reaction_channels_list], dtype=np.float64)
    is_ad_reaction_channel = np.array(is_ad_reaction_channel_list, dtype=np.bool_)
    is_self_reaction = np.array([p[0] == p[1] for p in reaction_channels_list], dtype=np.bool_)
    
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
    # We run the simulation in chunks to provide progress updates.
    
    total_reactions_to_run = config.params.max_reactions
    chunk_size = max(1, total_reactions_to_run // 100) # Update 100 times
    
    all_edges = []
    reactions_done_total = 0
    final_time = 0.0

    with tqdm(total=total_reactions_to_run, desc="Simulating") as pbar:
        while reactions_done_total < total_reactions_to_run:
            
            reactions_this_chunk = min(chunk_size, total_reactions_to_run - reactions_done_total)

            edges_chunk, reactions_in_chunk, final_time = _run_kmc_loop(
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
                reactions_this_chunk
            )
            
            if edges_chunk:
                all_edges.extend(edges_chunk)
            
            reactions_done_total += reactions_in_chunk
            pbar.update(reactions_in_chunk)

            if reactions_in_chunk < reactions_this_chunk:
                # KMC loop terminated early (no more reactions)
                pbar.total = reactions_done_total
                pbar.refresh()
                break
    
    end_time = time.time()
    print(f"3. Simulation finished in {end_time - start_time:.4f} seconds.")
    print(f"   - Reactions: {reactions_done_total}")
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
    for u, v, t in all_edges:
        G.add_edge(int(u), int(v), formation_time=t)
        
    metadata = {
        "wall_time_seconds": end_time - start_time,
        "reactions_completed": reactions_done_total,
        "final_simulation_time": final_time,
        "num_components": nx.number_connected_components(G),
        "config": config.model_dump()
    }
    
    return G, metadata

def run_batch(configs: List[SimulationInput], max_workers: int = None) -> Dict[str, Tuple[nx.Graph, Dict[str, Any]]]:
    """
    Runs a batch of simulations in parallel using a process pool.

    Args:
        configs (List[SimulationInput]): A list of simulation configurations.
        max_workers (int, optional): The maximum number of worker processes to use.
                                     If None, it defaults to the number of CPUs on the machine.

    Returns:
        Dict[str, Tuple[nx.Graph, Dict[str, Any]]]: A dictionary mapping simulation names
                                                     to their (graph, metadata) results.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_name = {executor.submit(run_simulation, config): config.params.name for config in configs}
        
        for future in tqdm(as_completed(future_to_name), total=len(configs), desc="Batch Simulations"):
            name = future_to_name[future]
            try:
                graph, metadata = future.result()
                results[name] = (graph, metadata)
            except Exception as exc:
                print(f'{name} generated an exception: {exc}')
                results[name] = (None, {"error": str(exc)})
    return results

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
        self.graph, self.metadata = run_simulation(self.config)
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