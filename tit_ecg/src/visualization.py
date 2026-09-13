"""Visualization tools for repSpat and Temporal ECG/VCG adaptation.

Provides publication-quality figures:
1. Temporal segmentation comparison: ECG timeline colored by Ground Truth waves,
   initial CAHC temporal intervals, Connected Component reassignment, and Clique motifs.
2. 3D VCG cardiac loops colored by initial vs reassigned cluster labels.
3. Cluster similarity graph G_sim with retained edges and MMD^2 weights.
"""
from __future__ import annotations

import os
from typing import Any
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_temporal_repspat_summary(
    results: dict[str, Any],
    ecg_signal: np.ndarray | None = None,
    ground_truth_waves: np.ndarray | None = None,
    time: np.ndarray | None = None,
    title: str = "Temporal repSpat ECG Analysis",
    save_path: str | None = None,
) -> plt.Figure:
    """Plots multi-panel diagnostic summary of Temporal repSpat on ECG/VCG.

    Panels:
    1. VCG channels (Vx, Vy, Vz) over time.
    2. Ground Truth wave annotations (P, QRS, T) if provided.
    3. Initial contiguous CAHC temporal clusters (C_g).
    4. Reassigned recurrent clusters (Y_clique and Y_CC).
    5. Similarity graph G_sim.
    """
    vcg = results.get("vcg")
    if time is None:
        time = results.get("time", np.arange(len(vcg)) / 500.0 if vcg is not None else None)

    initial_labels = results.get("initial_labels")
    labels_cc = results.get("labels_cc")
    labels_clique = results.get("labels_clique")
    G_sim = results.get("similarity_graph")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 2, width_ratios=[3, 1], hspace=0.35, wspace=0.25)

    # Panel 1: VCG Channels
    ax0 = fig.add_subplot(gs[0, 0])
    if vcg is not None:
        ax0.plot(time, vcg[:, 0], label="Vx (transverse)", color="#1f77b4", lw=1.2)
        ax0.plot(time, vcg[:, 1], label="Vy (longitudinal)", color="#2ca02c", lw=1.2)
        ax0.plot(time, vcg[:, 2], label="Vz (sagittal)", color="#d62728", lw=1.2)
        ax0.set_ylabel("VCG (mV)", fontsize=10)
        ax0.set_title("3D Vectorcardiogram Dipole Components v(t)", fontsize=11, fontweight="bold")
        ax0.legend(loc="upper right", ncol=3, fontsize=8)
        ax0.grid(True, alpha=0.3)
    elif ecg_signal is not None:
        ax0.plot(time, ecg_signal[:, 1] if ecg_signal.ndim > 1 else ecg_signal, color="#1f77b4", lw=1.2)
        ax0.set_ylabel("Lead II (mV)", fontsize=10)
        ax0.set_title("ECG Waveform", fontsize=11, fontweight="bold")
        ax0.grid(True, alpha=0.3)

    # Panel 2: Ground Truth Wave Annotations (if present)
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    if ground_truth_waves is not None:
        cmap_gt = plt.cm.get_cmap("tab10", 5)
        im1 = ax1.imshow(
            ground_truth_waves[np.newaxis, :],
            aspect="auto",
            extent=[time[0], time[-1], 0, 1],
            cmap="Set1",
            interpolation="nearest",
        )
        ax1.set_yticks([])
        ax1.set_ylabel("GT Waves", fontsize=10)
        ax1.set_title("Ground Truth Delineations (0=Iso, 1=P-wave, 2=QRS, 3=T-wave)", fontsize=10)
    else:
        ax1.text(0.5, 0.5, "No ground truth wave annotations provided", ha="center", va="center")
        ax1.set_axis_off()

    # Panel 3: Initial CAHC Contiguous Temporal Intervals
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    if initial_labels is not None:
        n_init = len(np.unique(initial_labels))
        ax2.imshow(
            initial_labels[np.newaxis, :],
            aspect="auto",
            extent=[time[0], time[-1], 0, 1],
            cmap="tab20",
            interpolation="nearest",
        )
        ax2.set_yticks([])
        m_star = results.get("m_star", "?")
        G_star = results.get("G_star", "?")
        ax2.set_ylabel(f"CAHC (G*={G_star})", fontsize=10)
        ax2.set_title(f"Initial CAHC Contiguous Intervals (m*={m_star}, G*={G_star})", fontsize=10)

    # Panel 4: Reassigned Motif Labels (Clique Primary & CC Audit)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    if labels_clique is not None:
        ax3.imshow(
            labels_clique[np.newaxis, :],
            aspect="auto",
            extent=[time[0], time[-1], 0, 1],
            cmap="Accent",
            interpolation="nearest",
        )
        ax3.set_yticks([])
        n_clique = len(np.unique(labels_clique))
        ax3.set_ylabel(f"Clique ({n_clique})", fontsize=10)
        ax3.set_xlabel("Elapsed Time (seconds)", fontsize=10)
        ax3.set_title(f"Final Reassigned Recurrent Motifs (Maximal Cliques, {n_clique} groups)", fontsize=10)

    # Right Column: Similarity Graph G_sim
    ax_graph = fig.add_subplot(gs[1:3, 1])
    if G_sim is not None and len(G_sim.nodes) > 0:
        pos = nx.circular_layout(G_sim)
        node_colors = []
        cluster_to_clique = results.get("cluster_to_clique", {})
        for node in G_sim.nodes():
            node_colors.append(cluster_to_clique.get(node, 0))

        nx.draw_networkx_nodes(
            G_sim, pos, ax=ax_graph, node_color=node_colors, cmap=plt.cm.Accent, node_size=350
        )
        nx.draw_networkx_labels(G_sim, pos, ax=ax_graph, font_size=9, font_weight="bold")

        edges = G_sim.edges(data=True)
        if edges:
            nx.draw_networkx_edges(G_sim, pos, ax=ax_graph, edge_color="#444444", width=1.5, alpha=0.7)
            edge_labels = {(u, v): f"{d.get('weight', 0):.2f}" for u, v, d in edges}
            nx.draw_networkx_edge_labels(G_sim, pos, edge_labels=edge_labels, ax=ax_graph, font_size=7)

        ax_graph.set_title(f"Similarity Graph G_sim\n(Edges = Retained q > 0.05)", fontsize=10)
        ax_graph.axis("off")
    else:
        ax_graph.text(0.5, 0.5, "Empty Graph", ha="center", va="center")
        ax_graph.set_axis_off()

    # Concordance Metrics Box
    concordance = results.get("concordance")
    if concordance:
        ax_info = fig.add_subplot(gs[3, 1])
        ax_info.axis("off")
        info_text = (
            f"Wave Concordance:\n"
            f"Initial CAHC ARI: {concordance['ari_initial']:.3f}\n"
            f"Initial CAHC NMI: {concordance['nmi_initial']:.3f}\n"
            f"Clique ARI:       {concordance['ari_clique']:.3f}\n"
            f"Clique NMI:       {concordance['nmi_clique']:.3f}\n"
            f"CC ARI:           {concordance['ari_cc']:.3f}\n"
            f"CC NMI:           {concordance['nmi_cc']:.3f}"
        )
        ax_info.text(
            0.1, 0.5, info_text, fontsize=9, family="monospace", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", edgecolor="#cccccc")
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_3d_vcg_loops(
    vcg: np.ndarray,
    labels: np.ndarray,
    title: str = "3D VCG Loops by Cluster",
    save_path: str | None = None,
) -> plt.Figure:
    """Plots 3D vectorcardiogram dipole trajectory colored by cluster assignment."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    unique_labels = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx, lbl in enumerate(unique_labels):
        mask = labels == lbl
        ax.scatter(
            vcg[mask, 0],
            vcg[mask, 1],
            vcg[mask, 2],
            label=f"Cluster {lbl}",
            color=cmap(idx),
            s=8,
            alpha=0.6,
        )

    ax.set_xlabel("Vx (mV)", fontsize=9)
    ax.set_ylabel("Vy (mV)", fontsize=9)
    ax.set_zlabel("Vz (mV)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
