"""Similarity Graph Construction, Dual Reassignment (Connected Components & Cliques), and Conflict Handling.

Addresses the paper inconsistency between:
- Algorithm 1 Step 4(b): Reassignment via Connected Components
- Section 2.4 & Section 3.2: Reassignment via Maximal Cliques / fully connected subgraphs

Both labelings (Y_CC and Y_clique) are produced and returned, with Y_clique treated
as the primary scientific interpretation and overlapping-clique ambiguities flagged.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_similarity_graph(
    pairwise_df: pd.DataFrame,
    all_clusters: list[int] | np.ndarray,
    fdr_alpha: float = 0.05,
) -> nx.Graph:
    """Constructs cluster similarity graph G_sim = (V, E) from pairwise test results.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        Pairwise testing results containing 'cluster_1', 'cluster_2', 'q_value', 'obs_mmd2'.
    all_clusters : list of int
        All initial CAHC cluster labels.
    fdr_alpha : float
        FDR significance threshold (default 0.05). [PAPER §2.3]

    Returns
    -------
    G_sim : nx.Graph
        Undirected graph with edge between clusters where null hypothesis was NOT rejected (q > alpha).
        Edge weight is observed MMD^2 [PAPER §2.4].
    """
    G_sim = nx.Graph()
    for c in all_clusters:
        G_sim.add_node(int(c))

    if len(pairwise_df) == 0:
        return G_sim

    # Retain edge when q > alpha (insufficient evidence to conclude distributions differ) [PAPER §2.4]
    retained_edges = pairwise_df[pairwise_df["q_value"] > fdr_alpha]

    for _, row in retained_edges.iterrows():
        u = int(row["cluster_1"])
        v = int(row["cluster_2"])
        w = float(row["obs_mmd2"])
        G_sim.add_edge(u, v, weight=w)

    return G_sim


def reassign_by_connected_components(
    initial_labels: np.ndarray,
    G_sim: nx.Graph,
) -> tuple[np.ndarray, dict[int, int]]:
    """Performs Connected Components reassignment according to Algorithm 1 Step 4(b).

    Parameters
    ----------
    initial_labels : np.ndarray
        Initial CAHC labels.
    G_sim : nx.Graph
        Similarity graph.

    Returns
    -------
    labels_cc : np.ndarray
        Reassigned labels based on connected components.
    cluster_to_cc : dict[int, int]
        Mapping from original cluster ID to component label.
    """
    components = sorted(nx.connected_components(G_sim), key=lambda c: (len(c), min(c)), reverse=True)
    cluster_to_cc = {}

    for new_label, comp in enumerate(components):
        for node in comp:
            cluster_to_cc[node] = new_label

    labels_cc = np.array([cluster_to_cc.get(lbl, lbl) for lbl in initial_labels], dtype=int)
    return labels_cc, cluster_to_cc


def extract_maximal_cliques(
    G_sim: nx.Graph,
    min_size: int = 2,
) -> list[list[int]]:
    """Finds all maximal cliques in the similarity graph of size >= min_size.

    Parameters
    ----------
    G_sim : nx.Graph
        Similarity graph.
    min_size : int
        Minimum clique size (default: 2, mathematical clique) [PAPER §2.4].

    Returns
    -------
    cliques : list of list of int
        Sorted maximal cliques.
    """
    all_cliques = list(nx.find_cliques(G_sim))
    filtered = [sorted(c) for c in all_cliques if len(c) >= min_size]
    # Sort cliques by size descending, then smallest node ascending
    filtered.sort(key=lambda c: (-len(c), c[0]))
    return filtered


def reassign_by_cliques(
    initial_labels: np.ndarray,
    G_sim: nx.Graph,
    min_size: int = 2,
) -> tuple[np.ndarray, dict[int, int], dict]:
    """Performs clique-based reassignment according to Section 2.4 and Section 3.2.

    Clusters that form a maximal clique are assigned a common label.
    Overlapping cliques (where a cluster belongs to multiple maximal cliques) are
    detected, reported as an unresolved graph ambiguity, and resolved deterministically
    by assigning the shared node to the clique with smaller mean internal MMD^2 edge weight.

    Parameters
    ----------
    initial_labels : np.ndarray
        Initial CAHC cluster labels.
    G_sim : nx.Graph
        Similarity graph.
    min_size : int
        Minimum clique size.

    Returns
    -------
    labels_clique : np.ndarray
        Reassigned labels based on cliques.
    cluster_to_clique : dict[int, int]
        Mapping from original cluster ID to clique label.
    audit_info : dict
        Information on maximal cliques, overlaps, and conflicts.
    """
    cliques = extract_maximal_cliques(G_sim, min_size=min_size)
    all_nodes = set(G_sim.nodes())

    # Check for overlaps
    node_to_cliques = {node: [] for node in all_nodes}
    for c_idx, c in enumerate(cliques):
        for node in c:
            node_to_cliques[node].append(c_idx)

    overlapping_nodes = [node for node, clist in node_to_cliques.items() if len(clist) > 1]
    has_overlap = len(overlapping_nodes) > 0

    # Deterministic assignment:
    # Iterate cliques in descending order of size, then min average edge weight
    def clique_avg_weight(clique: list[int]) -> float:
        if len(clique) <= 1:
            return 0.0
        weights = []
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                u, v = clique[i], clique[j]
                if G_sim.has_edge(u, v):
                    weights.append(G_sim[u][v].get("weight", 0.0))
        return float(np.mean(weights)) if weights else 0.0

    cliques_with_score = [
        (c, len(c), clique_avg_weight(c))
        for c in cliques
    ]
    # Sort: larger size first, smaller internal MMD^2 weight first
    cliques_with_score.sort(key=lambda item: (-item[1], item[2]))

    assigned_nodes = set()
    cluster_to_label = {}
    next_label = 0

    selected_cliques = []
    for c, _, _ in cliques_with_score:
        # Check available unassigned nodes in this clique
        unassigned_in_c = [node for node in c if node not in assigned_nodes]
        # If at least min_size nodes in this clique can be merged together
        if len(unassigned_in_c) >= min_size:
            selected_cliques.append(unassigned_in_c)
            for node in unassigned_in_c:
                cluster_to_label[node] = next_label
                assigned_nodes.add(node)
            next_label += 1

    # Any remaining nodes not part of a multi-node clique keep their own distinct label
    for node in sorted(all_nodes):
        if node not in assigned_nodes:
            cluster_to_label[node] = next_label
            assigned_nodes.add(node)
            next_label += 1

    labels_clique = np.array([cluster_to_label.get(lbl, lbl) for lbl in initial_labels], dtype=int)

    audit_info = {
        "all_maximal_cliques": cliques,
        "overlapping_nodes": overlapping_nodes,
        "has_overlap": has_overlap,
        "selected_cliques": selected_cliques,
        "n_maximal_cliques": len(cliques),
        "n_final_clique_groups": len(np.unique(labels_clique)),
        "graph_descriptors": compute_graph_descriptors(G_sim, cliques, overlapping_nodes),
    }

    return labels_clique, cluster_to_label, audit_info


def compute_graph_descriptors(
    G_sim: nx.Graph,
    cliques: list[list[int]] | None = None,
    overlapping_nodes: list[int] | None = None,
) -> dict:
    """Computes first-class topological graph descriptors for similarity graph G_sim."""
    n_nodes = G_sim.number_of_nodes()
    n_edges = G_sim.number_of_edges()
    max_edges = n_nodes * (n_nodes - 1) / 2 if n_nodes > 1 else 1
    density = float(n_edges / max_edges)

    if cliques is None:
        cliques = extract_maximal_cliques(G_sim, min_size=2)
    if overlapping_nodes is None:
        node_to_cliques = {node: [] for node in G_sim.nodes()}
        for c_idx, c in enumerate(cliques):
            for node in c:
                node_to_cliques[node].append(c_idx)
        overlapping_nodes = [node for node, clist in node_to_cliques.items() if len(clist) > 1]

    weights = [d.get("weight", 0.0) for _, _, d in G_sim.edges(data=True)]
    mean_edge_weight = float(np.mean(weights)) if weights else 0.0

    degrees = [d for _, d in G_sim.degree()]
    if sum(degrees) > 0:
        p_deg = np.array(degrees) / sum(degrees)
        p_deg = p_deg[p_deg > 0]
        deg_entropy = float(-np.sum(p_deg * np.log2(p_deg)))
    else:
        deg_entropy = 0.0

    clique_sizes = [len(c) for c in cliques]

    return {
        "n_nodes": int(n_nodes),
        "n_edges": int(n_edges),
        "density": density,
        "n_maximal_cliques": len(cliques),
        "clique_sizes": clique_sizes,
        "max_clique_size": int(max(clique_sizes)) if clique_sizes else 0,
        "overlap_nodes_count": len(overlapping_nodes),
        "overlap_fraction": float(len(overlapping_nodes) / n_nodes) if n_nodes > 0 else 0.0,
        "mean_retained_edge_weight": mean_edge_weight,
        "degree_entropy": deg_entropy,
    }

