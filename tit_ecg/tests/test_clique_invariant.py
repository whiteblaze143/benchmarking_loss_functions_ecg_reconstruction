"""Unit test for Clique Resolver Invariant:
Every non-singleton group in Y_resolved must itself induce a clique in G_sim:
    forall g, h in R, g != h => (g, h) in E_sim

Tests:
1. Complete graph K_4: Single resolved group containing all 4 nodes.
2. 4-cycle C_4: (0-1-2-3-0): Contains no triangles. Cliques are individual edges of size 2.
   Reassignment must NOT merge nodes 0 and 2 (no edge exists).
3. Bow-tie graph: Two triangles sharing node 2: (0, 1, 2) and (2, 3, 4).
   Tests overlap resolution. Node 2 must be assigned to ONE triangle, and the other
   must either remain size 2 (which is an edge) or singletons.
   In all cases, NO non-adjacent nodes can ever share a group!
4. Disconnected graph with isolated singletons: Singletons participate in no maximal clique,
   and must remain distinct isolated singletons in Y_resolved.
5. Real clinical ECG recordings: Verifies the invariant on real VCG records from LUDB, ISP, and PTB-XL.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from tit_ecg.src.graph_reassignment import reassign_by_cliques
from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.pipeline import TemporalRepSpatECG
from tit_ecg.src.dataset_adapters import LUDBAdapter, ISPAdapter, PTBXLAdapter


def verify_clique_invariant_on_graph(G_sim: nx.Graph, initial_labels: np.ndarray | None = None) -> None:
    nodes = sorted(list(G_sim.nodes()))
    if initial_labels is None:
        initial_labels = np.array(nodes, dtype=int)

    labels_clique, cluster_to_label, audit_info = reassign_by_cliques(
        initial_labels=initial_labels,
        G_sim=G_sim,
        min_size=2,
    )

    # 1. Check partition completeness: every node in G_sim is mapped
    assert set(cluster_to_label.keys()) == set(nodes), "Partition does not cover all graph nodes!"

    # 2. Check clique invariant: every non-singleton group must induce a clique in G_sim
    resolved_groups: dict[int, list[int]] = {}
    for node, lbl in cluster_to_label.items():
        resolved_groups.setdefault(lbl, []).append(node)

    for grp_lbl, members in resolved_groups.items():
        if len(members) > 1:
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    u, v = members[i], members[j]
                    assert G_sim.has_edge(u, v), (
                        f"CLIQUE INVARIANT VIOLATION: In resolved group {grp_lbl}, "
                        f"nodes {u} and {v} do not share an edge in G_sim! "
                        f"Members: {members}"
                    )

    # 3. Check that singletons with no edges in G_sim remain singletons
    for node in nodes:
        if G_sim.degree(node) == 0:
            lbl = cluster_to_label[node]
            assert len(resolved_groups[lbl]) == 1, f"Isolated node {node} was incorrectly grouped!"


def test_complete_graph_k4():
    G = nx.complete_graph(4)
    verify_clique_invariant_on_graph(G)


def test_cycle_graph_c4():
    # 4-cycle: 0-1-2-3-0
    G = nx.cycle_graph(4)
    verify_clique_invariant_on_graph(G)


def test_bowtie_graph_overlap():
    # Bow-tie: Triangles (0, 1, 2) and (2, 3, 4) sharing node 2
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
    verify_clique_invariant_on_graph(G)


def test_isolated_singletons_and_clique():
    # Triangle (0, 1, 2) + isolated singletons 3, 4, 5
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    G.add_nodes_from([3, 4, 5])
    verify_clique_invariant_on_graph(G)


def test_real_clinical_ecg_graphs():
    cfg = RepSpatConfig.fast_test_config(
        m_grid=[4, 8],
        G_grid=[4, 6],
        kernel_scale_rule="median_heuristic",
        kernel_param=1.0,
        n_permutations=20,
        random_state=42,
    )

    adapters = [
        ("LUDB", LUDBAdapter(), 1),
        ("ISP", ISPAdapter(), 1),
        ("PTBXL", PTBXLAdapter(), 1),
    ]

    for name, adapter, rid in adapters:
        rec = adapter.load_record(rid, n_samples=500)
        vcg = rec["vcg"]
        fs = rec["fs"]
        model = TemporalRepSpatECG(config=cfg)
        model.fit(vcg, sampling_rate=fs, is_vcg=True)

        G_sim = model.results_["similarity_graph"]
        initial_labels = model.results_["initial_labels"]

        verify_clique_invariant_on_graph(G_sim, initial_labels=initial_labels)
        print(f"PASS: Real ECG record {name} Rec {rid} satisfies the clique resolver invariant.")


if __name__ == "__main__":
    test_complete_graph_k4()
    test_cycle_graph_c4()
    test_bowtie_graph_overlap()
    test_isolated_singletons_and_clique()
    test_real_clinical_ecg_graphs()
    print("\nALL CLIQUE RESOLVER INVARIANT UNIT TESTS PASSED!")
