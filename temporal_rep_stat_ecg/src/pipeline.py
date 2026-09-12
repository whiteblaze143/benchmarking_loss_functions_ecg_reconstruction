"""End-to-end TemporalRepSpat pipeline for ECG Repeated Temporal Pattern discovery.

Translates repSpat (Senanayake & Jeganathan, Spatial Statistics, 2026) to 1D temporal signals:
- Step 1: Attribute Dissimilarity (Euclidean / Jaccard) & Temporal Links L
- Step 2: Constrained Agglomerative Hierarchical Clustering (CAHC) & Modified Silhouette Optimization
- Step 3: Pairwise MMD^2 with IMQ Kernel, k-means Block Permutation & BH FDR Multiple Testing
- Step 4: Similarity Graph & Maximal Clique Reassignment into Repeated Temporal Patterns (RTPs)
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .clustering.distance import compute_attribute_distances
from .clustering.cahc import construct_temporal_adjacency, temporal_constrained_hac
from .clustering.silhouette import temporal_silhouette_analysis, compute_modified_silhouette
from .testing.mmd import compute_mmd_sq
from .testing.block_permutation import create_feature_blocks
from .testing.multiple_testing import pairwise_episode_testing
from .graph.clique_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_cliques,
)


class TemporalRepSpat:
    """Nonparametric pipeline for discovering Repeated Temporal Patterns (RTPs) in ECG signals.

    Parameters
    ----------
    metric : str
        Attribute distance metric: 'euclidean' (continuous) or 'jaccard' (binary).
    m_neighbors : int or list of int, default=4
        Number of temporal neighbors. If list, grid search is performed.
    n_clusters : int or list of int, default=5
        Number of initial temporal episodes G. If list/range, grid search is performed.
    kernel : str, default='IMQ'
        Kernel for MMD^2 ('IMQ' or 'Gaussian').
    kernel_param : float, default=1.0
        Parameter c for IMQ kernel or bandwidth for Gaussian.
    n_permutations : int, default=200
        Number of block permutations B for two-sample null distribution.
    alpha : float, default=0.05
        FDR significance threshold for multiple testing correction.
    min_clique_size : int, default=3
        Minimum maximal clique size required to declare a unified RTP.
    random_state : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        metric: str = "euclidean",
        m_neighbors: int | list[int] = 4,
        n_clusters: int | list[int] | range = 5,
        kernel: str = "IMQ",
        kernel_param: float = 1.0,
        n_permutations: int = 200,
        alpha: float = 0.05,
        min_clique_size: int = 3,
        random_state: int = 42,
    ):
        self.metric = metric.lower()
        self.m_neighbors = m_neighbors
        self.n_clusters = n_clusters
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.min_clique_size = min_clique_size
        self.random_state = random_state

        # Fitted attributes
        self.is_fitted: bool = False
        self.best_m_: int | None = None
        self.best_G_: int | None = None
        self.silhouette_df_: pd.DataFrame | None = None
        self.dist_matrix_: np.ndarray | None = None
        self.adjacency_: sp.csr_matrix | None = None
        self.initial_labels_: np.ndarray | None = None
        self.block_ids_: np.ndarray | None = None
        self.pairwise_df_: pd.DataFrame | None = None
        self.similarity_graph_: nx.Graph | None = None
        self.maximal_cliques_: list[list[int]] | None = None
        self.rtp_labels_: np.ndarray | None = None
        self.label_mapping_: dict[int, int] | None = None

    def fit(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> TemporalRepSpat:
        """Fits the complete Temporal repSpat pipeline.

        Parameters
        ----------
        X : np.ndarray
            Beat attribute matrix of shape [n, p].
        timestamps : np.ndarray, optional
            1D timestamps of beats [n]. If None, uniform indices are assumed.

        Returns
        -------
        self : TemporalRepSpat
            Fitted estimator.
        """
        X = np.asarray(X, dtype=float)
        n = len(X)
        if timestamps is None:
            timestamps = np.arange(n, dtype=float)
        else:
            timestamps = np.asarray(timestamps, dtype=float)

        # -------------------------------------------------------------
        # STEP 1: Attribute Dissimilarity & Temporal Adjacency
        # -------------------------------------------------------------
        self.dist_matrix_ = compute_attribute_distances(X, metric=self.metric)

        # -------------------------------------------------------------
        # STEP 2: CAHC & Modified Silhouette Optimization
        # -------------------------------------------------------------
        is_grid_m = isinstance(self.m_neighbors, (list, tuple))
        is_grid_G = isinstance(self.n_clusters, (list, tuple, range))

        if is_grid_m or is_grid_G:
            m_list = list(self.m_neighbors) if is_grid_m else [self.m_neighbors]
            g_list = list(self.n_clusters) if is_grid_G else [self.n_clusters]

            self.silhouette_df_, best_params, _ = temporal_silhouette_analysis(
                X=X,
                timestamps=timestamps,
                dist_matrix=self.dist_matrix_,
                m_neighbors_list=m_list,
                n_clusters_range=g_list,
                metric=self.metric,
            )
            self.best_m_, self.best_G_ = best_params
        else:
            self.best_m_ = int(self.m_neighbors)
            self.best_G_ = int(self.n_clusters)
            self.silhouette_df_ = None

        self.initial_labels_, _, self.adjacency_ = temporal_constrained_hac(
            X=X,
            timestamps=timestamps,
            n_clusters=self.best_G_,
            m_neighbors=self.best_m_,
            metric=self.metric,
        )

        # -------------------------------------------------------------
        # STEP 3: Pairwise MMD^2 & Block Permutation Testing
        # -------------------------------------------------------------
        self.block_ids_ = create_feature_blocks(
            X=X,
            labels=self.initial_labels_,
            m_neighbors=self.best_m_,
            random_state=self.random_state,
        )

        self.pairwise_df_ = pairwise_episode_testing(
            labels=self.initial_labels_,
            dist_matrix=self.dist_matrix_,
            block_ids=self.block_ids_,
            kernel=self.kernel,
            kernel_param=self.kernel_param,
            n_permutations=self.n_permutations,
            alpha=self.alpha,
            random_state=self.random_state,
        )

        # -------------------------------------------------------------
        # STEP 4: Similarity Graph & Maximal Clique Reassignment
        # -------------------------------------------------------------
        unique_eps = sorted(np.unique(self.initial_labels_))
        self.similarity_graph_ = build_similarity_graph(
            pairwise_df=self.pairwise_df_,
            all_episodes=unique_eps,
            alpha=self.alpha,
        )

        self.maximal_cliques_ = extract_maximal_cliques(
            G=self.similarity_graph_,
            min_size=self.min_clique_size,
        )

        self.rtp_labels_, self.label_mapping_ = reassign_cliques(
            initial_labels=self.initial_labels_,
            cliques=self.maximal_cliques_,
        )

        self.is_fitted = True
        return self

    def fit_predict(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fits pipeline and returns unified RTP labels."""
        self.fit(X, timestamps=timestamps)
        return self.rtp_labels_

    def get_summary(self) -> dict:
        """Returns structured summary of discovery results."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted first.")

        n_initial = len(np.unique(self.initial_labels_))
        n_rtp = len(np.unique(self.rtp_labels_))
        n_cliques = len(self.maximal_cliques_)

        # Ensure JSON-serializable standard Python types
        cliques_clean = [[int(x) for x in c] for c in self.maximal_cliques_]
        mapping_clean = {int(k): int(v) for k, v in self.label_mapping_.items()}

        return {
            "best_m": int(self.best_m_) if self.best_m_ is not None else None,
            "best_G": int(self.best_G_) if self.best_G_ is not None else None,
            "initial_episodes_count": int(n_initial),
            "rtp_clusters_count": int(n_rtp),
            "maximal_cliques_count": int(n_cliques),
            "maximal_cliques": cliques_clean,
            "label_mapping": mapping_clean,
            "edges_in_similarity_graph": int(self.similarity_graph_.number_of_edges()),
            "nodes_in_similarity_graph": int(self.similarity_graph_.number_of_nodes()),
        }
