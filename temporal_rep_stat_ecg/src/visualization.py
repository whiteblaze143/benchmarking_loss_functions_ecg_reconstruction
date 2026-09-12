"""Visualization utilities for temporal repSpat on ECG signals.

Faithful adaptation of author visualization.py to 1D temporal biomedical signals:
- plot_temporal_clusters: ECG waveform colored by initial episode / RTP cluster
- plot_similarity_graph: Undirected graph G_sim with edge weights and highlighted maximal cliques
- plot_rtp_feature_distribution: Bar and box plots of morphology features per RTP cluster
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def plot_temporal_clusters(
    timestamps: np.ndarray,
    labels: np.ndarray,
    signal_wave: np.ndarray | None = None,
    figsize: tuple[float, float] = (12, 4),
    title: str = "ECG Temporal Clusters and Repeated Patterns",
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots temporal episode / RTP assignments along the time axis.

    Parameters
    ----------
    timestamps : np.ndarray
        Beat timestamps in seconds [n].
    labels : np.ndarray
        Assigned episode or RTP labels [n].
    signal_wave : np.ndarray, optional
        Continuous ECG waveform samples for background display.
    figsize : tuple
        Figure size (width, height).
    title : str
        Plot title.
    save_path : str, optional
        File path to save the figure.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")

    if signal_wave is not None:
        t_wave = np.linspace(timestamps[0], timestamps[-1], len(signal_wave))
        ax.plot(t_wave, signal_wave, color="darkgray", alpha=0.6, linewidth=0.8, label="ECG")

    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        color = cmap(i % 10)
        ax.scatter(
            timestamps[mask],
            np.zeros(np.sum(mask)),
            s=40,
            color=color,
            label=f"Pattern {lab}",
            zorder=3,
        )
        # Highlight spans
        indices = np.flatnonzero(mask)
        # Group contiguous runs
        runs = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
        for run in runs:
            if len(run) > 0:
                t_start = timestamps[run[0]]
                t_end = timestamps[run[-1]]
                ax.axvspan(t_start, t_end, color=color, alpha=0.2)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude / Cluster")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_similarity_graph(
    G: nx.Graph,
    cliques: list[list[int]] | None = None,
    figsize: tuple[float, float] = (6, 6),
    title: str = "Episode Similarity Graph (Edges: p_adj >= 0.05)",
    save_path: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the episode similarity graph G_sim with edge weights and highlighted cliques.

    Parameters
    ----------
    G : nx.Graph
        Similarity graph constructed by build_similarity_graph.
    cliques : list of list of int, optional
        List of qualifying maximal cliques (size >= 3) to highlight.
    figsize : tuple
        Figure size.
    title : str
        Plot title.
    save_path : str, optional
        Save path.

    Returns
    -------
    fig, ax : tuple
        Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty Graph (No Episodes)", ha="center", va="center")
        return fig, ax

    pos = nx.circular_layout(G)

    # Determine node colors based on clique participation
    clique_nodes = set()
    if cliques:
        for c in cliques:
            clique_nodes.update(c)

    node_colors = [
        "#ff7f0e" if node in clique_nodes else "#1f77b4"
        for node in G.nodes()
    ]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_color="white", font_weight="bold", ax=ax)

    # Draw edges
    edges = G.edges(data=True)
    if edges:
        nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5, alpha=0.7, ax=ax)
        edge_labels = {
            (u, v): f"{d.get('weight', 0.0):.3f}"
            for u, v, d in edges
        }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
