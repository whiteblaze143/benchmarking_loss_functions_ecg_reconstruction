"""Publication-quality visualization utilities for KirkVCG repSpat.

Provides:
1. 3D VCG cardiac dipole loop trajectory colored by discovered REP states.
2. 12-lead ECG timeline overlay with shaded REP segments.
3. Similarity Graph G_sim with highlighted maximal cliques.
4. Markov Transition Dynamics heatmap.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_vcg_3d_trajectory(
    vcg: np.ndarray,
    labels: np.ndarray,
    title: str = "3D VCG Dipole Trajectory with Discovered REPs",
    save_path: str | None = None,
) -> plt.Figure:
    """Plots the 3D VCG cardiac dipole loop colored by REP labels.

    Parameters
    ----------
    vcg : np.ndarray
        Array of shape [n, 3] representing (Vx, Vy, Vz).
    labels : np.ndarray
        Sample-level pattern labels of length n.
    title : str
        Figure title.
    save_path : str, optional
        File path to save the generated figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

    for idx, lab in enumerate(unique_labels):
        mask = (labels == lab)
        ax.scatter(
            vcg[mask, 0],
            vcg[mask, 1],
            vcg[mask, 2],
            label=f"REP {lab}",
            color=cmap(idx),
            s=12,
            alpha=0.7,
        )

    # Plot continuous trajectory line faintly in background
    ax.plot(vcg[:, 0], vcg[:, 1], vcg[:, 2], color="gray", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Vx (Lead X, mV)")
    ax.set_ylabel("Vy (Lead Y, mV)")
    ax.set_zlabel("Vz (Lead Z, mV)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_ecg_timeline_with_reps(
    time: np.ndarray,
    ecg_signal: np.ndarray,
    labels: np.ndarray,
    lead_name: str = "Lead II",
    title: str = "ECG Waveform with REP Label Overlay",
    save_path: str | None = None,
) -> plt.Figure:
    """Plots an ECG lead waveform with shaded segments for each REP state.

    Parameters
    ----------
    time : np.ndarray
        Time in seconds of length n.
    ecg_signal : np.ndarray
        1D voltage signal of length n.
    labels : np.ndarray
        Sample-level pattern labels of length n.
    lead_name : str
        Name of plotted lead.
    title : str
        Figure title.
    save_path : str, optional
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, ecg_signal, color="#1e293b", linewidth=1.2, label=lead_name, zorder=3)

    unique_labels = sorted(np.unique(labels))
    cmap = plt.cm.get_cmap("tab10", max(len(unique_labels), 1))

    # Highlight segments
    for idx, lab in enumerate(unique_labels):
        mask = (labels == lab)
        ax.scatter(
            time[mask],
            ecg_signal[mask],
            color=cmap(idx),
            s=15,
            label=f"REP {lab}",
            zorder=4,
            alpha=0.8,
        )

    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Voltage (mV)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", ncol=min(len(unique_labels) + 1, 6), fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_similarity_graph(
    G: nx.Graph,
    cliques: list[list[int]] | None = None,
    title: str = "Similarity Graph G_sim & Maximal Cliques",
    save_path: str | None = None,
) -> plt.Figure:
    """Visualizes the similarity graph with maximal cliques highlighted.

    Parameters
    ----------
    G : nx.Graph
        Similarity graph.
    cliques : list of list of int, optional
        Extracted maximal cliques.
    title : str
        Title.
    save_path : str, optional
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color="#cbd5e1", node_size=600, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#94a3b8", width=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=11, font_family="sans-serif", ax=ax)

    # Highlight cliques
    if cliques:
        cmap = plt.cm.get_cmap("Set2", max(len(cliques), 1))
        for i, clique in enumerate(cliques):
            color = cmap(i)
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=clique,
                node_color=[color] * len(clique),
                node_size=750,
                ax=ax,
                label=f"Clique {i+1} (Size {len(clique)})",
            )
            # Highlight clique internal edges
            clique_sub = G.subgraph(clique)
            nx.draw_networkx_edges(
                clique_sub, pos, edge_color=color, width=3.0, ax=ax
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    if cliques:
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_transition_matrix(
    trans_matrix: np.ndarray,
    title: str = "REP Markov Transition Dynamics",
    save_path: str | None = None,
) -> plt.Figure:
    """Plots heatmap of the Markov transition dynamics between REP states."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(trans_matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax)

    ax.set_xlabel("To State", fontsize=11)
    ax.set_ylabel("From State", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # Overlay numbers
    for i in range(trans_matrix.shape[0]):
        for j in range(trans_matrix.shape[1]):
            val = trans_matrix[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
