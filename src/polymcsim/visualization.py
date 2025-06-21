"""Visualization utilities for PolyMCsim polymer graphs."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


def visualize_polymer(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (12, 8),
    layout: str = "spring",
    node_size: int = 300,
    node_color_by: str = "monomer_type",
    node_outline_color: str = "black",
    with_labels: bool = False,
    title: Optional[str] = None,
    seed: Optional[int] = None,
    component_index: Optional[Union[int, str]] = None,
    save_path: Optional[Union[str, Path]] = None,
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
    components = sorted(
        list(nx.connected_components(plot_graph)), key=len, reverse=True
    )
    if not components:
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            plt.title(title)
        plt.axis("off")
        return fig

    # Select component to plot
    if component_index is not None:
        if isinstance(component_index, str) and component_index == "random":
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
        plt.axis("off")
        return fig

    # --- Create Figure ---
    fig, ax = plt.subplots(figsize=figsize)

    # --- Layout ---
    pos = None
    if layout == "spring":
        pos = nx.spring_layout(plot_graph, k=0.1, iterations=50, seed=seed)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(plot_graph)
    elif layout == "circular":
        pos = nx.circular_layout(plot_graph)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # --- Node Colors ---
    if node_color_by == "monomer_type":
        node_colors_map = [
            plot_graph.nodes[node].get("monomer_type", "gray")
            for node in plot_graph.nodes()
        ]
        unique_types = sorted(list(set(node_colors_map)))
        color_palette = plt.colormaps.get_cmap("tab20")(
            np.linspace(0, 1, len(unique_types))
        )
        color_dict = dict(zip(unique_types, color_palette))
        node_colors = [color_dict[c] for c in node_colors_map]
    else:
        node_colors = "skyblue"

    # --- Drawing ---
    nx.draw_networkx_nodes(
        plot_graph,
        pos,
        node_color=node_colors,
        node_size=node_size,
        edgecolors=node_outline_color,
        linewidths=1.5,
        ax=ax,
    )

    if plot_graph.number_of_edges() > 0:
        edge_pos = np.array([(pos[e[0]], pos[e[1]]) for e in plot_graph.edges()])
        edge_collection = LineCollection(
            edge_pos, colors="black", linewidths=1.5, alpha=0.8
        )
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

    if node_color_by == "monomer_type" and "color_dict" in locals():
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=ctype,
                markerfacecolor=color,
                markersize=10,
            )
            for ctype, color in color_dict.items()
        ]
        ax.legend(handles=legend_elements, title="Monomer Types", loc="best")

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_chain_length_distribution(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (8, 5),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
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
        ax.hist(
            chain_lengths,
            bins="auto",
            alpha=0.75,
            color="cornflowerblue",
            edgecolor="black",
        )

        stats_text = (
            f"Number of Chains (>1): {len(chain_lengths)}\n"
            f"Mean Length: {np.mean(chain_lengths):.2f}\n"
            f"Max Length: {max(chain_lengths)}"
        )
        ax.text(
            0.95,
            0.95,
            stats_text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.8),
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No polymer chains formed (all monomers are isolated).",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
        )

    ax.set_xlabel("Chain Length (Number of Monomers)")
    ax.set_ylabel("Frequency")
    ax.set_title(title if title else "Chain Length Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_molecular_weight_distribution(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (10, 6),
    bins: Union[int, str, np.ndarray] = 50,
    log_scale: bool = False,
    show_pdi: bool = True,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot the molecular weight distribution (MWD) of the polymer.

    This is more informative than chain length distribution as it accounts
    for the actual molar masses of different monomer types.

    Args:
        graph: The NetworkX Graph to analyze.
        figsize: Figure size (width, height) in inches.
        bins: Number of bins or bin specification for histogram.
        log_scale: Whether to use log scale for x-axis (useful for broad distributions).
        show_pdi: Whether to calculate and display PDI on the plot.
        title: Optional title for the plot.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object with MWD plot.

    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph object.")

    # Calculate molecular weights for each polymer chain
    components = list(nx.connected_components(graph))
    molar_masses = []

    for component in components:
        if len(component) > 1:  # Only consider actual polymer chains
            mass = sum(graph.nodes[node].get("molar_mass", 100.0) for node in component)
            molar_masses.append(mass)

    if not molar_masses:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No polymer chains formed.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    molar_masses = np.array(molar_masses)

    # Calculate PDI if requested
    pdi_text = ""
    if show_pdi:
        mn = np.mean(molar_masses)  # Number average
        mw = np.sum(molar_masses**2) / np.sum(molar_masses)  # Weight average
        pdi = mw / mn if mn > 0 else 0
        pdi_text = f"\nPDI = {pdi:.3f}"

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use weights to create a proper number distribution
    weights = np.ones_like(molar_masses) / len(molar_masses)

    n, bins_edges, _ = ax.hist(
        molar_masses,
        bins=bins,
        weights=weights,
        alpha=0.75,
        color="darkseagreen",
        edgecolor="black",
        label="Number distribution",
    )

    # Add weight distribution as a line plot
    bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2
    # Calculate weight fraction in each bin
    weight_fractions = []
    for i in range(len(bins_edges) - 1):
        mask = (molar_masses >= bins_edges[i]) & (molar_masses < bins_edges[i + 1])
        weight_in_bin = np.sum(molar_masses[mask])
        weight_fractions.append(weight_in_bin)

    weight_fractions = np.array(weight_fractions) / np.sum(molar_masses)

    ax2 = ax.twinx()
    ax2.plot(
        bin_centers,
        weight_fractions,
        "r-",
        linewidth=2,
        label="Weight distribution",
        alpha=0.8,
    )
    ax2.set_ylabel("Weight Fraction", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    # Statistics text
    stats_text = (
        f"Number of Chains: {len(molar_masses)}\n"
        f"Mn = {np.mean(molar_masses):.0f} g/mol\n"
        f"Mw = {mw:.0f} g/mol"
        f"{pdi_text}"
    )
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.8),
    )

    # Formatting
    ax.set_xlabel("Molecular Weight (g/mol)")
    ax.set_ylabel("Number Fraction", color="darkgreen")
    ax.tick_params(axis="y", labelcolor="darkgreen")

    if log_scale:
        ax.set_xscale("log")

    ax.set_title(title if title else "Molecular Weight Distribution")
    ax.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_conversion_analysis(
    metadata: Dict[str, any],
    edge_data: Optional[List[Tuple[int, int, float]]] = None,
    figsize: Tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot conversion and reaction kinetics analysis.

    Args:
        metadata: Simulation metadata containing conversion and timing info.
        edge_data: Optional list of (node1, node2, time) tuples for kinetic analysis.
        figsize: Figure size (width, height) in inches.
        title: Optional title for the plot.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object with conversion analysis.

    """
    fig = plt.figure(figsize=figsize)

    # Extract data from metadata
    final_conversion = metadata.get("final_conversion", 0)
    final_time = metadata.get("final_simulation_time", 0)
    reactions_completed = metadata.get("reactions_completed", 0)
    max_conversion_param = (
        metadata.get("config", {}).get("params", {}).get("max_conversion", 1.0)
    )

    # If we have edge formation times, we can plot conversion vs time
    if edge_data and len(edge_data) > 0:
        # Sort edges by time
        sorted_edges = sorted(edge_data, key=lambda x: x[2])
        times = [e[2] for e in sorted_edges]

        # Estimate conversion at each reaction
        # Assuming 2 monomers react per edge (simplified)
        total_monomers = metadata.get("config", {}).get(
            "total_monomers", reactions_completed * 2
        )  # fallback estimate
        conversions = np.arange(1, len(times) + 1) * 2 / total_monomers

        # Create subplots
        ax1 = plt.subplot(2, 2, 1)
        ax1.plot(times, conversions, "b-", linewidth=2)
        ax1.axhline(
            y=max_conversion_param,
            color="r",
            linestyle="--",
            label=f"Max conversion limit ({max_conversion_param:.1%})",
        )
        ax1.set_xlabel("Simulation Time")
        ax1.set_ylabel("Conversion")
        ax1.set_title("Conversion vs Time")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Reaction rate vs time
        ax2 = plt.subplot(2, 2, 2)
        # Calculate instantaneous rates
        time_diffs = np.diff(times)
        rates = 1 / time_diffs[time_diffs > 0]
        rate_times = times[1 : len(rates) + 1]

        ax2.semilogy(rate_times, rates, "g-", alpha=0.7, linewidth=1)
        ax2.set_xlabel("Simulation Time")
        ax2.set_ylabel("Reaction Rate (reactions/time)")
        ax2.set_title("Reaction Rate vs Time")
        ax2.grid(True, alpha=0.3)

        # Conversion vs reactions
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(range(len(conversions)), conversions, "purple", linewidth=2)
        ax3.set_xlabel("Number of Reactions")
        ax3.set_ylabel("Conversion")
        ax3.set_title("Conversion vs Number of Reactions")
        ax3.grid(True, alpha=0.3)

    else:
        ax3 = plt.subplot(1, 1, 1)

    # Summary statistics (bottom right or main if no time data)
    if edge_data and len(edge_data) > 0:
        ax4 = plt.subplot(2, 2, 4)
    else:
        ax4 = ax3

    stats_text = (
        f"Final Conversion: {final_conversion:.1%}\n"
        f"Max Conversion Parameter: {max_conversion_param:.1%}\n"
        f"Total Reactions: {reactions_completed:,}\n"
        f"Final Simulation Time: {final_time:.2e}\n"
        f"Wall Time: {metadata.get('wall_time_seconds', 0):.2f} s"
    )

    ax4.text(
        0.5,
        0.5,
        stats_text,
        transform=ax4.transAxes,
        verticalalignment="center",
        horizontalalignment="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=1", facecolor="lightgray", alpha=0.8),
    )
    ax4.set_title("Simulation Summary")
    ax4.axis("off")

    fig.suptitle(title if title else "Polymerization Kinetics Analysis", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def plot_branching_analysis(
    graph: nx.Graph,
    figsize: Tuple[int, int] = (10, 6),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Analyze and visualize the degree of branching in the polymer.

    Args:
        graph: The NetworkX Graph to analyze.
        figsize: Figure size (width, height) in inches.
        title: Optional title for the plot.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object with branching analysis.

    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph object.")

    # Get degree distribution
    degrees = dict(graph.degree())

    # Separate analysis for polymer chains only
    components = list(nx.connected_components(graph))
    polymer_degrees = []

    for component in components:
        if len(component) > 1:  # Only polymer chains
            for node in component:
                polymer_degrees.append(degrees[node])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Degree distribution histogram
    if polymer_degrees:
        unique_degrees, counts = np.unique(polymer_degrees, return_counts=True)
        ax1.bar(unique_degrees, counts, color="coral", edgecolor="black", alpha=0.8)

        # Calculate branching statistics
        linear_units = counts[unique_degrees == 2].sum() if 2 in unique_degrees else 0
        terminal_units = counts[unique_degrees == 1].sum() if 1 in unique_degrees else 0
        branch_points = (
            counts[unique_degrees > 2].sum() if any(unique_degrees > 2) else 0
        )

        total_units = sum(counts)
        branching_freq = branch_points / total_units if total_units > 0 else 0

        stats_text = (
            "Terminal units: "
            f"{terminal_units} ({terminal_units / total_units * 100:.1f}%)\n"
            "Linear units: "
            f"{linear_units} ({linear_units / total_units * 100:.1f}%)\n"
            "Branch points: "
            f"{branch_points} ({branch_points / total_units * 100:.1f}%)\n"
            "Branching frequency: "
            f"{branching_freq:.3f}"
        )

        ax1.text(
            0.95,
            0.95,
            stats_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="aliceblue", alpha=0.8),
        )

    ax1.set_xlabel("Node Degree")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Degree Distribution in Polymer Chains")
    ax1.grid(True, alpha=0.3, axis="y")

    # Branch point distribution by component size
    branch_data = []
    for component in components:
        if len(component) > 1:
            branch_nodes = [n for n in component if degrees[n] > 2]
            branch_data.append((len(component), len(branch_nodes)))

    if branch_data:
        sizes, branches = zip(*branch_data)
        ax2.scatter(sizes, branches, alpha=0.6, s=50, color="darkblue")

        # Add trend line if enough data
        if len(sizes) > 3:
            z = np.polyfit(sizes, branches, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(sizes), max(sizes), 100)
            ax2.plot(
                x_trend,
                p(x_trend),
                "r--",
                alpha=0.8,
                label=f"Trend: y={z[0]:.3f}x+{z[1]:.3f}",
            )
            ax2.legend()

    ax2.set_xlabel("Component Size (# monomers)")
    ax2.set_ylabel("Number of Branch Points")
    ax2.set_title("Branching vs Chain Size")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title if title else "Polymer Branching Analysis", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def create_analysis_dashboard(
    graph: nx.Graph,
    metadata: Dict[str, any],
    figsize: Tuple[int, int] = (16, 12),
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Create a comprehensive dashboard with multiple polymer analysis plots.

    Args:
        graph: The NetworkX Graph to analyze.
        metadata: Simulation metadata.
        figsize: Figure size (width, height) in inches.
        title: Optional title for the dashboard.
        save_path: If provided, saves the figure to this path.

    Returns:
        Matplotlib Figure object with analysis dashboard.

    """
    fig = plt.figure(figsize=figsize)

    # 1. Polymer structure visualization (top left)
    ax1 = plt.subplot(3, 3, 1)
    plt.sca(ax1)

    # Find largest component for visualization
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if components and len(components[0]) > 1:
        largest_polymer = graph.subgraph(components[0])
        pos = nx.spring_layout(largest_polymer, k=0.1, iterations=50)

        # Color by monomer type
        node_colors = []
        monomer_types = list(
            set(nx.get_node_attributes(largest_polymer, "monomer_type").values())
        )
        color_map = dict(zip(monomer_types, plt.cm.tab10(range(len(monomer_types)))))

        for node in largest_polymer.nodes():
            mtype = largest_polymer.nodes[node].get("monomer_type", "unknown")
            node_colors.append(color_map.get(mtype, "gray"))

        nx.draw(
            largest_polymer,
            pos,
            node_color=node_colors,
            node_size=50,
            with_labels=False,
            ax=ax1,
        )
        ax1.set_title(f"Largest Polymer ({len(largest_polymer)} monomers)")
    else:
        ax1.text(0.5, 0.5, "No polymers formed", ha="center", va="center")
        ax1.set_title("Polymer Structure")

    # 2. Chain length distribution (top middle)
    ax2 = plt.subplot(3, 3, 2)
    chain_lengths = [len(c) for c in components if len(c) > 1]
    if chain_lengths:
        ax2.hist(
            chain_lengths,
            bins="auto",
            alpha=0.75,
            color="cornflowerblue",
            edgecolor="black",
        )
        ax2.set_xlabel("Chain Length")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Chain Length Distribution")
        ax2.grid(True, alpha=0.3, axis="y")

    # 3. Molecular weight distribution (top right)
    ax3 = plt.subplot(3, 3, 3)
    molar_masses = []
    for component in components:
        if len(component) > 1:
            mass = sum(graph.nodes[node].get("molar_mass", 100.0) for node in component)
            molar_masses.append(mass)

    if molar_masses:
        ax3.hist(
            molar_masses, bins=30, alpha=0.75, color="darkseagreen", edgecolor="black"
        )

        # Calculate and display PDI
        mn = np.mean(molar_masses)
        mw = np.sum(np.array(molar_masses) ** 2) / np.sum(molar_masses)
        pdi = mw / mn if mn > 0 else 0

        ax3.text(
            0.95,
            0.95,
            f"PDI = {pdi:.3f}",
            transform=ax3.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        ax3.set_xlabel("Molecular Weight (g/mol)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Molecular Weight Distribution")
        ax3.grid(True, alpha=0.3, axis="y")

    # 4. Degree distribution (middle left)
    ax4 = plt.subplot(3, 3, 4)
    degrees = [d for n, d in graph.degree() if graph.nodes[n]]
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    ax4.bar(unique_degrees, counts, color="coral", edgecolor="black", alpha=0.8)
    ax4.set_xlabel("Node Degree")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Connectivity Distribution")
    ax4.grid(True, alpha=0.3, axis="y")

    # 5. Component size distribution (middle middle)
    ax5 = plt.subplot(3, 3, 5)
    component_sizes = [len(c) for c in components]
    size_counts = dict(zip(*np.unique(component_sizes, return_counts=True)))

    sizes = sorted(size_counts.keys())
    counts = [size_counts[s] for s in sizes]

    ax5.bar(
        sizes[:20], counts[:20], color="mediumpurple", edgecolor="black", alpha=0.8
    )  # Limit to first 20
    ax5.set_xlabel("Component Size")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Component Size Distribution")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Simulation statistics (middle right)
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis("off")

    total_monomers = len(graph.nodes())
    num_components = len(components)
    reacted_monomers = len([n for n in graph.nodes() if graph.degree(n) > 0])

    stats_text = (
        f"Total Monomers: {total_monomers:,}\n"
        f"Reacted Monomers: {reacted_monomers:,}\n"
        f"Number of Components: {num_components:,}\n"
        f"Number of Polymers (>1): {len([c for c in components if len(c) > 1]):,}\n"
        f"Largest Component: {len(components[0]) if components else 0:,} monomers\n\n"
        f"Final Conversion: {metadata.get('final_conversion', 0):.1%}\n"
        f"Reactions Completed: {metadata.get('reactions_completed', 0):,}\n"
        f"Simulation Time: {metadata.get('final_simulation_time', 0):.2e}\n"
        f"Wall Time: {metadata.get('wall_time_seconds', 0):.2f} s"
    )

    ax6.text(
        0.5,
        0.5,
        stats_text,
        transform=ax6.transAxes,
        verticalalignment="center",
        horizontalalignment="center",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )
    ax6.set_title("Simulation Summary")

    # 7-9. Bottom row can be used for additional analysis
    # For example, monomer type distribution
    ax7 = plt.subplot(3, 3, 7)
    monomer_type_counts = {}
    for node in graph.nodes():
        mtype = graph.nodes[node].get("monomer_type", "unknown")
        monomer_type_counts[mtype] = monomer_type_counts.get(mtype, 0) + 1

    if monomer_type_counts:
        types = list(monomer_type_counts.keys())
        counts = list(monomer_type_counts.values())
        colors = plt.cm.Set3(range(len(types)))

        ax7.pie(counts, labels=types, autopct="%1.1f%%", colors=colors)
        ax7.set_title("Monomer Type Distribution")

    # Overall title
    fig.suptitle(title if title else "Polymer Analysis Dashboard", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    return fig


def export_polymer_data(
    graph: nx.Graph,
    metadata: Dict[str, any],
    output_dir: Union[str, Path],
    prefix: str = "polymer_analysis",
) -> Dict[str, Path]:
    """Export polymer analysis data to CSV files for external analysis.

    Args:
        graph: The NetworkX Graph to analyze.
        metadata: Simulation metadata.
        output_dir: Directory to save the CSV files.
        prefix: Prefix for the output filenames.

    Returns:
        Dictionary mapping data type to file path.

    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    # 1. Export chain data
    components = list(nx.connected_components(graph))
    chain_data = []

    for i, component in enumerate(components):
        if len(component) > 1:  # Only polymer chains
            mass = sum(graph.nodes[node].get("molar_mass", 100.0) for node in component)

            # Count monomer types
            type_counts = {}
            for node in component:
                mtype = graph.nodes[node].get("monomer_type", "unknown")
                type_counts[mtype] = type_counts.get(mtype, 0) + 1

            # Calculate branching
            degrees = [graph.degree(n) for n in component]
            branch_points = sum(1 for d in degrees if d > 2)

            chain_data.append(
                {
                    "chain_id": i,
                    "length": len(component),
                    "molecular_weight": mass,
                    "branch_points": branch_points,
                    "avg_degree": np.mean(degrees),
                    **{f"monomer_{k}": v for k, v in type_counts.items()},
                }
            )

    if chain_data:
        df_chains = pd.DataFrame(chain_data)
        chain_file = output_dir / f"{prefix}_chain_data.csv"
        df_chains.to_csv(chain_file, index=False)
        output_files["chain_data"] = chain_file

    # 2. Export summary statistics
    summary_data = {
        "total_monomers": len(graph.nodes()),
        "total_chains": len([c for c in components if len(c) > 1]),
        "unreacted_monomers": len([c for c in components if len(c) == 1]),
        "final_conversion": metadata.get("final_conversion", 0),
        "reactions_completed": metadata.get("reactions_completed", 0),
        "simulation_time": metadata.get("final_simulation_time", 0),
        "wall_time_seconds": metadata.get("wall_time_seconds", 0),
    }

    # Add PDI if we have chains
    if chain_data:
        masses = [c["molecular_weight"] for c in chain_data]
        mn = np.mean(masses)
        mw = np.sum(np.array(masses) ** 2) / np.sum(masses)
        summary_data["Mn"] = mn
        summary_data["Mw"] = mw
        summary_data["PDI"] = mw / mn if mn > 0 else 0

    df_summary = pd.DataFrame([summary_data])
    summary_file = output_dir / f"{prefix}_summary.csv"
    df_summary.to_csv(summary_file, index=False)
    output_files["summary"] = summary_file

    print(f"Exported polymer analysis data to {output_dir}")
    for key, path in output_files.items():
        print(f"  - {key}: {path.name}")

    return output_files
