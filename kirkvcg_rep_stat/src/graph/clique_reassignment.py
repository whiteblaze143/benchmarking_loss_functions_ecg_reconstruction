"""Similarity Graph Construction and Maximal Clique Reassignment.

Implements Step 4 of repSpat mapped to Repeated Electrophysiological Patterns (REPs):
- Builds undirected Similarity Graph G_sim:
  - Nodes: Initial CAHC clusters {1, ..., G}
  - Edges: Retained when p_adj >= alpha (distributions indistinguishable under MMD^2)
  - Edge weights: Empirical observed MMD^2
- Extracts maximal cliques of size >= 3.
- Reassigns clusters forming a maximal clique into a unified Repeated Electrophysiological Pattern (REP) label.
- Clusters not participating in a size >= 3 clique retain individual labels (flagged as unique or anomalous events).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def build_similarity_graph(
    pairwise_df: pd.DataFrame,
    all_clusters: list[int] | np.ndarray | None = None,
    alpha: float = 0.05,
) -> nx.Graph:
    """Constructs the undirected similarity graph G_sim.

    Parameters
    ----------
    pairwise_df : pd.DataFrame
        DataFrame containing ['cluster_1', 'cluster_2', 'obs_mmd_sq', 'adj_p'].
    all_clusters : list-like, optional
        List of all initial cluster IDs.
    alpha : float
        Significance threshold. Edges added where adj_p >= alpha.

    Returns
    -------
    G : nx.Graph
        Undirected graph with edge attributes 'weight' = obs_mmd_sq and 'adj_p'.
    """
    G = nx.Graph()

    if all_clusters is not None:
        G.add_nodes_from([int(c) for c in all_clusters])
    elif not pairwise_df.empty:
        nodes = pd.unique(pairwise_df[["cluster_1", "cluster_2"]].values.ravel())
        G.add_nodes_from([int(n) for n in nodes])

    if pairwise_df.empty:
        return G

    # Edge condition: adj_p >= alpha (statistically indistinguishable distributions)
    similar_pairs = pairwise_df[pairwise_df["adj_p"] >= alpha]
    for _, row in similar_pairs.iterrows():
        u = int(row["cluster_1"])
        v = int(row["cluster_2"])
        w = float(row["obs_mmd_sq"])
        p = float(row["adj_p"])
        G.add_edge(u, v, weight=w, adj_p=p)

    return G


def extract_maximal_cliques(
    G: nx.Graph,
    min_size: int = 3,
) -> list[list[int]]:
    """Extracts all maximal cliques in G having size >= min_size.

    Parameters
    ----------
    G : nx.Graph
        Similarity graph.
    min_size : int
        Minimum clique size (repSpat default: 3).

    Returns
    -------
    cliques : list of list of int
        List of qualifying maximal cliques sorted by size descending.
    """
    all_cliques = list(nx.find_cliques(G))
    qualifying = [sorted(c) for c in all_cliques if len(c) >= min_size]
    qualifying.sort(key=len, reverse=True)
    return qualifying


def reassign_cliques_to_reps(
    initial_labels: np.ndarray,
    cliques: list[list[int]],
) -> tuple[np.ndarray, dict[int, int], list[dict]]:
    """Reassigns initial cluster labels into unified Repeated Electrophysiological Pattern (REP) labels.

    Clusters participating in a maximal clique of size >= 3 are merged into
    a unified REP label. Clusters not participating in any qualifying clique
    retain their original identity (mapped to unique non-overlapping integer IDs).

    Parameters
    ----------
    initial_labels : np.ndarray
        Array of initial cluster labels [n].
    cliques : list of list of int
        List of qualifying maximal cliques.

    Returns
    -------
    rep_labels : np.ndarray
        Updated sample-level labels [n].
    cluster_to_rep : dict
        Mapping from initial cluster ID -> unified REP / unique ID.
    clique_metadata : list of dict
        Summary of each extracted clique and its assigned REP ID.
    """
    initial_arr = np.asarray(initial_labels, dtype=int)
    all_unique_clusters = sorted(np.unique(initial_arr))

    cluster_to_rep: dict[int, int] = {}
    assigned_clusters: set[int] = set()
    clique_metadata: list[dict] = []

    current_rep_id = 1

    # Assign cliques ordered by size descending
    for clique in cliques:
        unassigned = [c for c in clique if c not in assigned_clusters]
        # In repSpat, if >= min_size (3) nodes can be merged, form a REP
        if len(unassigned) >= 2 or (len(unassigned) >= 1 and len(clique) >= 3):
            # Form unified REP label
            for c in clique:
                if c not in cluster_to_rep:
                    cluster_to_rep[c] = current_rep_id
                    assigned_clusters.add(c)

            clique_metadata.append({
                "rep_id": current_rep_id,
                "member_clusters": clique,
                "size": len(clique),
                "is_rep": True,
            })
            current_rep_id += 1

    # Remaining clusters that did not form size >= 3 cliques
    unmerged_clusters = [c for c in all_unique_clusters if c not in cluster_to_rep]
    for c in unmerged_clusters:
        cluster_to_rep[c] = current_rep_id
        clique_metadata.append({
            "rep_id": current_rep_id,
            "member_clusters": [c],
            "size": 1,
            "is_rep": False,
        })
        current_rep_id += 1

    # Map sample-level labels
    rep_labels = np.array([cluster_to_rep.get(val, val) for val in initial_arr], dtype=int)

    return rep_labels, cluster_to_rep, clique_metadata
