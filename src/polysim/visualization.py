"""Visualization utilities for PolySim polymer graphs."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from typing import Dict, List, Optional, Tuple, Union


def visualize_polymer(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (12, 8),
    layout: str = 'spring',
    node_size: int = 300,
    node_color_by: str = 'monomer_type',
    node_outline_color: str = 'black',
    with_labels: bool = False,
    title: Optional[str] = None,
    seed: Optional[int] = None,
    component_index: Optional[Union[int, str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualize a polymer graph with customizable styling.
    
    Args:
        graph: The NetworkX Graph to visualize.
        figsize: Figure size (width, height) in inches.
        layout: Graph layout algorithm ('spring', 'kamada_kawai', 'circular').
        node_size: Size of nodes in the visualization.
        node_color_by: Node attribute to use for coloring ('monomer_type' or None).
        node_outline_color: Color of the node outline/border.
        with_labels: Whether to show node labels with monomer type.
        title: Optional title for the plot.
        seed: Random seed for layout algorithms.
        component_index: Which component to plot. Can be:
                        - None: Plot all components.
                        - int: Plot the nth largest component (0 = largest).
                        - 'random': Plot a random component from chains with >1 monomer.
        save_path: If provided, saves the figure to this path.
        
    Returns:
        Matplotlib Figure object.
        
    Raises:
        TypeError: If input is not a NetworkX Graph object.
        ValueError: If layout algorithm is unknown or component index is out of range.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph object.")
        
    plot_graph = graph.copy()

    # Get connected components
    components = sorted(list(nx.connected_components(plot_graph)), key=len, reverse=True)
    if not components:
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            plt.title(title)
        plt.axis('off')
        return fig
            
    # Select component to plot
    if component_index is not None:
        if isinstance(component_index, str) and component_index == 'random':
            # Filter for non-monomer components
            polymer_components = [c for c in components if len(c) > 1]
            if not polymer_components:
                print("No polymer chains with more than one monomer to plot.")
                selected_nodes = components[0]  # Fallback to largest component
            else:
                rand_idx = np.random.randint(0, len(polymer_components))
                selected_nodes = polymer_components[rand_idx]
        else:
            try:
                selected_nodes = components[component_index]
            except IndexError:
                raise ValueError(
                    f"Component index {component_index} out of range. "
                    f"Graph has {len(components)} components."
                )
        plot_graph = plot_graph.subgraph(selected_nodes).copy()
        
    if len(plot_graph) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            plt.title(title)
        plt.axis('off')
        return fig
    
    # --- Create Figure ---
    fig, ax = plt.subplots(figsize=figsize)
    
    # --- Layout ---
    pos = None
    if layout == 'spring':
        pos = nx.spring_layout(plot_graph, k=0.1, iterations=50, seed=seed)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(plot_graph)
    elif layout == 'circular':
        pos = nx.circular_layout(plot_graph)
    else:
        raise ValueError(f"Unknown layout: {layout}")
        
    # --- Node Colors ---
    if node_color_by == 'monomer_type':
        node_colors_map = [
            plot_graph.nodes[node].get('monomer_type', 'gray') 
            for node in plot_graph.nodes()
        ]
        unique_types = sorted(list(set(node_colors_map)))
        color_palette = plt.colormaps.get_cmap('tab20')(np.linspace(0, 1, len(unique_types)))
        color_dict = dict(zip(unique_types, color_palette))
        node_colors = [color_dict[c] for c in node_colors_map]
    else:
        node_colors = 'skyblue'
        
    # --- Drawing ---
    nx.draw_networkx_nodes(
        plot_graph, pos, node_color=node_colors, node_size=node_size, 
        edgecolors=node_outline_color, linewidths=1.5, ax=ax
    )
    
    if plot_graph.number_of_edges() > 0:
        edge_pos = np.array([(pos[e[0]], pos[e[1]]) for e in plot_graph.edges()])
        edge_collection = LineCollection(edge_pos, colors='black', linewidths=1.5, alpha=0.8)
        ax.add_collection(edge_collection)
    
    if with_labels:
        labels = {
            node: f"{plot_graph.nodes[node]['monomer_type']}" 
            for node in plot_graph.nodes()
        }
        nx.draw_networkx_labels(
            plot_graph, pos, labels=labels, font_size=8, ax=ax, font_color="black"
        )
    
    # --- Title and Legend ---
    plot_title = title if title else "Polymer Structure"
    if component_index is not None:
        plot_title += f" (Component {component_index}, {len(plot_graph)} monomers)"
    ax.set_title(plot_title, fontsize=16)

    if node_color_by == 'monomer_type' and 'color_dict' in locals():
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label=ctype,
                      markerfacecolor=color, markersize=10) 
            for ctype, color in color_dict.items()
        ]
        ax.legend(handles=legend_elements, title="Monomer Types", loc="best")

    ax.set_aspect('equal')
    plt.tight_layout()
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig


def plot_chain_length_distribution(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (8, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot the distribution of chain lengths in the polymer.
    
    Args:
        graph: The NetworkX Graph to analyze.
        figsize: Figure size (width, height) in inches.
        title: Optional title for the plot.
        save_path: If provided, saves the figure to this path.
            
    Returns:
        Matplotlib Figure object.
        
    Raises:
        TypeError: If input is not a NetworkX Graph object.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph object.")

    components = list(nx.connected_components(graph))
    # We are often interested in actual polymers, not unreacted monomers
    chain_lengths = [len(c) for c in components if len(c) > 1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if chain_lengths:
        ax.hist(chain_lengths, bins='auto', alpha=0.75, color='cornflowerblue', edgecolor='black')
        
        stats_text = (
            f'Number of Chains (>1): {len(chain_lengths)}\n'
            f'Mean Length: {np.mean(chain_lengths):.2f}\n'
            f'Max Length: {max(chain_lengths)}'
        )
        ax.text(
            0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='aliceblue', alpha=0.8)
        )
    else:
        ax.text(
            0.5, 0.5, "No polymer chains formed (all monomers are isolated).",
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes
        )

    ax.set_xlabel('Chain Length (Number of Monomers)')
    ax.set_ylabel('Frequency')
    ax.set_title(title if title else 'Chain Length Distribution')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    return fig 