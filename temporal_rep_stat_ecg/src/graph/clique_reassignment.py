"""Similarity Graph Construction and Maximal Clique-Based Reassignment.

Implements Step 4 of repSpat adapted to ECG Repeated Temporal Patterns (RTPs):
- Similarity Graph G_sim: Nodes are initial CAHC episodes; edges retained if p_adj >= alpha.
- Maximal Clique Extraction: Finds fully connected subgraphs of size >= 3.
- Reassignment: Merges all episodes in a clique into unified Repeated Temporal Pattern (RTP) labels.
- Isolated episodes or subgraphs of size < 3 retain their individual initial labels.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_similarity_graph(
    pairwise_df: pd.DataFrame,
    all_episodes: list[int] | np.ndarray | None = None,
    alpha: float = 0.05,
) -> nx.Graph:
    """Constructs the undirected similarity graph G_sim.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame from pairwise_episode_testing containing columns:
        ['episode_1', 'episode_2', 'obs_mmd_sq', 'adj_p'].
    all_episodes : list-like, optional
        Complete list of episode node IDs to include in the graph.
    alpha : float
        FDR significance threshold. Edges are added if adj_p >= alpha.

    Returns
    -------
    G : nx.Graph
        Undirected graph with edge attributes 'weight' = obs_mmd_sq and 'adj_p'.
    """
    G = nx.Graph()

    if all_episodes is not None:
        G.add_nodes_from([int(ep) for ep in all_episodes])
    elif not pairwise_df.empty:
        nodes = pd.unique(pairwise_df[["episode_1", "episode_2"]].values.ravel())
        G.add_nodes_from([int(n) for n in nodes])

    if pairwise_df.empty:
        return G

    # Edge condition: adj_p >= alpha (statistically indistinguishable distributions)
    similar_pairs = pairwise_df[pairwise_df["adj_p"] >= alpha]

    for _, row in similar_pairs.iterrows():
        u = int(row["episode_1"])
        v = int(row["episode_2"])
        w = float(row["obs_mmd_sq"])
        p = float(row["adj_p"])
        G.add_edge(u, v, weight=w, adj_p=p)

    return G


def extract_maximal_cliques(
    G: nx.Graph,
    min_size: int = 3,
) -> list[list[int]]:
    """Extracts maximal cliques of size >= min_size from G.

    Parameters
    ----------
    G : nx.Graph
        Similarity graph.
    min_size : int
        Minimum clique size required for reassignment (repSpat default: 3).

    Returns
    -------
    cliques : list of list of int
        List of maximal cliques satisfying len(clique) >= min_size.
    """
    all_cliques = list(nx.find_cliques(G))
    qualifying_cliques = [
        sorted(c) for c in all_cliques if len(c) >= min_size
    ]
    # Sort by size descending
    qualifying_cliques.sort(key=len, reverse=True)
    return qualifying_cliques


def reassign_cliques(
    initial_labels: np.ndarray,
    cliques: list[list[int]],
) -> tuple[np.ndarray, dict[int, int]]:
    """Reassigns initial episode labels into unified Repeated Temporal Pattern (RTP) labels.

    Episodes participating in a maximal clique of size >= 3 are merged into
    a unified RTP label. Episodes not participating in any qualifying clique
    retain their original episode labels.

    Parameters
    ----------
    initial_labels : np.ndarray
        Array of initial CAHC cluster labels [n] (1-indexed).
    cliques : list of list of int
        List of qualifying maximal cliques (each of size >= 3).

    Returns
    -------
    reassigned_labels : np.ndarray
        Updated labels [n] with unified RTP labels.
    mapping : dict
        Mapping from initial episode ID to new RTP/episode ID.
    """
    initial_labels = np.asarray(initial_labels, dtype=int)
    reassigned_labels = initial_labels.copy()

    unique_episodes = sorted(np.unique(initial_labels))
    mapping: dict[int, int] = {ep: ep for ep in unique_episodes}

    # Group overlapping cliques into unified components if needed
    clique_graph = nx.Graph()
    for clique in cliques:
        for u in clique:
            clique_graph.add_node(u)
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                clique_graph.add_edge(clique[i], clique[j])

    # Connected components of qualifying cliques
    rtp_groups = list(nx.connected_components(clique_graph))

    for group in rtp_groups:
        # Use the minimum episode ID in the group as the unified RTP label
        unified_label = min(group)
        for ep in group:
            mapping[ep] = unified_label

    for orig_ep, new_ep in mapping.items():
        reassigned_labels[initial_labels == orig_ep] = new_ep

    return reassigned_labels, mapping
