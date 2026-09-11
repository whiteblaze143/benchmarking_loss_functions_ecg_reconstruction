"""repSpat similarity graph and connected-component reassignment.

Implements Stage 2 of Q-VCG / RepSpat:
- Graph G_R = (V, E) where nodes are contiguous spatial domains G_i.
- Edges connect domains passing:
    1. Sufficient patient overlap support.
    2. Between-domain MMD^2 <= epsilon_MMD (reproducibility-calibrated margin).
    3. Consistency across RBF and IMQ kernels.
- Connected components produce quotient classes Q = V / ~.
- Classifies each motif as REPEATED (>= 2 disconnected spatial components) or LOCAL_ONLY (1 component).
"""
from __future__ import annotations

import networkx as nx
import numpy as np


class QuotientClosure:
    """Manages transitive quotient closure and motif classification."""

    def __init__(
        self,
        n_domains: int,
        epsilon_mmd: float,
        min_paired_patients: int = 8,
    ):
        self.n_domains = n_domains
        self.epsilon_mmd = epsilon_mmd
        self.min_paired_patients = min_paired_patients

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(n_domains))

        self.domain_to_motif: dict[int, int] = {}
        self.motif_to_domains: dict[int, list[int]] = {}
        self.motif_status: dict[int, str] = {}  # "REPEATED" or "LOCAL_ONLY"
        self.pairwise_edges: list[tuple[int, int, float]] = []

    def build_similarity_graph(
        self,
        pairwise_mmd_results: list[dict],
    ) -> nx.Graph:
        """Add an edge exactly when the BH-adjusted IMQ test does not reject."""
        self.graph.remove_edges_from(list(self.graph.edges))
        self.pairwise_edges = []
        for res in pairwise_mmd_results:
            d1, d2 = int(res["domain_1"]), int(res["domain_2"])
            q_value = float(res["q_bh_imq"])
            if not bool(res.get("bh_reject_imq", q_value < 0.05)):
                mmd2 = float(res["mmd2_imq_obs"])
                self.graph.add_edge(d1, d2, weight=mmd2, q_value=q_value)
                self.pairwise_edges.append((d1, d2, mmd2))
        return self.graph

    def build_equivalence_graph(
        self, pairwise_mmd_results: list[dict], min_kernel_agreement: int = 3
    ) -> nx.Graph:
        """Backward-compatible wrapper; new results use BH graph semantics.

        Args:
            pairwise_mmd_results: List of dicts with at minimum:
                - domain_1: int
                - domain_2: int
                - mmd2_gaussian: float   (primary decision kernel)
                - kernel_agreement: int  (# of non-primary kernels also below epsilon)
                - n_shared_patients: int
                - passed_support: bool
            min_kernel_agreement: Minimum number of secondary kernels that must
                also agree (score below epsilon_mmd) for edge creation.
                Out of 5 non-primary kernels; default 3 = simple majority.
        """
        if pairwise_mmd_results and "q_bh_imq" in pairwise_mmd_results[0]:
            return self.build_similarity_graph(pairwise_mmd_results)
        for res in pairwise_mmd_results:
            d1 = int(res["domain_1"])
            d2 = int(res["domain_2"])
            # Primary kernel is Gaussian; fall back to any mmd2 field if key missing
            mmd2 = float(res.get("mmd2_gaussian", res.get("mmd2_rbf", res.get("mmd2", 999.0))))
            passed_support = res.get("passed_support", False)
            kernel_agreement = int(res.get("kernel_agreement", min_kernel_agreement))

            # Equivalence requires:
            # 1. Sufficient paired patient support
            # 2. Primary (Gaussian) MMD below reproducibility-calibrated epsilon
            # 3. Cross-kernel majority: >= min_kernel_agreement secondary kernels also agree
            if passed_support and mmd2 <= self.epsilon_mmd and kernel_agreement >= min_kernel_agreement:
                self.graph.add_edge(d1, d2, weight=mmd2)
                self.pairwise_edges.append((d1, d2, mmd2))

        return self.graph


    def compute_quotient_classes(
        self,
        domain_adjacency: dict[int, set[int]] | None = None,
    ) -> dict[int, int]:
        """Computes connected components and categorizes REPEATED vs LOCAL_ONLY.

        Args:
            domain_adjacency: Optional graph of physical domain adjacency.
                              If provided, determines whether merged domains form
                              disconnected spatial components.

        Returns:
            Mapping from domain_id in {0, ..., M-1} to motif_id in {0, ..., K-1}.
        """
        components = list(nx.connected_components(self.graph))
        # Sort components by size (largest first), then by minimum domain ID
        components.sort(key=lambda c: (-len(c), min(c)))

        self.domain_to_motif = {}
        self.motif_to_domains = {}
        self.motif_status = {}
        self.motif_is_clique: dict[int, bool] = {}

        for motif_id, comp in enumerate(components):
            comp_list = sorted(list(comp))
            self.motif_to_domains[motif_id] = comp_list
            for d in comp_list:
                self.domain_to_motif[d] = motif_id

            subgraph = self.graph.subgraph(comp_list)
            is_clique = len(comp_list) <= 1 or subgraph.number_of_edges() == len(comp_list) * (len(comp_list) - 1) // 2
            self.motif_is_clique[motif_id] = is_clique

            if not is_clique:
                self.motif_status[motif_id] = "AMBIGUOUS_NON_CLIQUE"
            elif len(comp_list) == 1:
                self.motif_status[motif_id] = "LOCAL_ONLY"
            else:
                if domain_adjacency is not None:
                    # Check whether domains in comp form disconnected spatial graph
                    sub_adj = nx.Graph()
                    sub_adj.add_nodes_from(comp_list)
                    for d in comp_list:
                        for neighbor in domain_adjacency.get(d, set()):
                            if neighbor in comp:
                                sub_adj.add_edge(d, neighbor)
                    num_sub_components = nx.number_connected_components(sub_adj)
                    if num_sub_components >= 2:
                        self.motif_status[motif_id] = "REPEATED"
                    else:
                        self.motif_status[motif_id] = "LOCAL_ONLY"
                else:
                    # Default: merging 2 or more discrete domains is treated as repeated
                    self.motif_status[motif_id] = "REPEATED"

        return self.domain_to_motif

    @property
    def num_motifs(self) -> int:
        return len(self.motif_to_domains)

    @property
    def compression_ratio(self) -> float:
        """C_Q = 1 - K / M."""
        if self.n_domains == 0:
            return 0.0
        return 1.0 - (self.num_motifs / self.n_domains)

    @property
    def num_repeated_motifs(self) -> int:
        return sum(1 for status in self.motif_status.values() if status == "REPEATED")
