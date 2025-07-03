"""Utility functions for polymer analysis and calculations."""

from collections import Counter

import networkx as nx


def calculate_SHI(polymer_graph: nx.Graph) -> float:
    """Calculate the Sequence Heterogeneity Index (SHI) for a polymer graph.

    Args:
        polymer_graph: An object with 'nodes' and 'edges' attributes.
                       - nodes: A list where polymer_graph.nodes[i] is an object
                                with a 'type' attribute (e.g., monomer ID).
                       - edges: A list of tuples, where each tuple (u, v)
                                represents a bond between node u and node v.

    Returns:
        A float representing the SHI value.

    """
    total_bond_count = len(polymer_graph.edges)

    # Handle the case of a single monomer with no bonds.
    if total_bond_count == 0:
        return 0.0

    hetero_bond_count = 0
    for u, v in polymer_graph.edges:
        type_u = polymer_graph.nodes[u]["monomer_type"]
        type_v = polymer_graph.nodes[v]["monomer_type"]
        if type_u != type_v:
            hetero_bond_count += 1

    # Calculate the final ratio
    shi = hetero_bond_count / total_bond_count

    return shi


def calculate_nSHI(polymer_graph: nx.Graph) -> float:
    """Calculate the Normalized Sequence Heterogeneity Index (nSHI).

    Normalization happens with respect to a random polymer of the same
    composition.
    """
    shi = calculate_SHI(polymer_graph)

    monomer_types = [data["monomer_type"] for _, data in polymer_graph.nodes(data=True)]

    num_monomers = len(monomer_types)

    # Edge case: If there are fewer than 2 monomers, no bonds can be formed.
    # The expected SHI is therefore 0.
    if num_monomers < 2:
        return 0.0

    # Use collections.Counter to efficiently count occurrences of each monomer type.
    # This returns a dictionary-like object: {'A': 3, 'B': 2}
    counts = Counter(monomer_types)

    # --- Calculation Step ---
    sum_of_squared_fractions = 0.0

    # Iterate through the counts of each unique monomer type.
    for monomer_type in counts:
        count = counts[monomer_type]

        # Calculate the mole fraction (f_i) for this monomer type.
        mole_fraction = count / num_monomers

        # Square the mole fraction and add it to the sum.
        sum_of_squared_fractions += mole_fraction**2

    # The expected SHI is 1 minus the sum of the squared fractions.
    expected_shi = 1.0 - sum_of_squared_fractions

    # Return normalized SHI (avoid division by zero)
    if expected_shi == 0.0:
        return 0.0

    return shi / (expected_shi * 2)
