"""KirkVCG repSpat: Master Nonparametric ECG Trajectory Pipeline.

Implements the complete 4-step pipeline mapped from repSpat to 3D VCG dipole space:
1. Matrix Representation:
   - Attribute Dissimilarity D (Continuous Euclidean or Binary Jaccard)
   - Spatial Adjacency Matrix L (m-NN in 3D VCG space R^3, optional hybrid)
2. Constrained HAC & Parameter Optimization:
   - Contiguity constraint in 3D VCG space
   - Ward.D2 Lance-Williams recursive distance updates
   - Grid search over (m, G) maximizing VCG-informed modified silhouette
3. Pairwise Distributional Testing:
   - Empirical MMD^2 with Inverse Multiquadratic (IMQ) kernel
   - Feature-space k-means blocking and permutation resampling
   - Benjamini-Hochberg FDR multiple testing at alpha = 0.05
4. Similarity Graph & Maximal Clique Reassignment:
   - Undirected graph G_sim with edges where p_adj >= 0.05
   - Extraction of maximal cliques of size >= 3
   - Merging clique clusters into unified Repeated Electrophysiological Patterns (REPs)
5. Interpretable Encoding:
   - Occupancy, Markov transition dynamics, within-REP variability, anomaly ratio.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .vcg.transform import ecg_to_vcg_kors
from .features.ecg_features import extract_sample_features, standardize_features
from .clustering.distance import compute_attribute_distances
from .clustering.vcg_adjacency import construct_vcg_adjacency, construct_hybrid_adjacency
from .clustering.cahc import vcg_constrained_hac
from .clustering.silhouette import (
    compute_vcg_modified_silhouette,
    optimize_vcg_hyperparameters,
)
from .testing.block_permutation import pairwise_cluster_testing
from .testing.multiple_testing import apply_fdr_to_pairwise_results
from .graph.clique_reassignment import (
    build_similarity_graph,
    extract_maximal_cliques,
    reassign_cliques_to_reps,
)
from .encoder.encoder import KirkVCGRepSpatEncoder


class KirkVCGRepSpat:
    """Master estimator for 3D VCG Repeated Electrophysiological Pattern discovery and encoding."""

    def __init__(
        self,
        metric: str = "euclidean",
        m_neighbors: int = 4,
        n_clusters: int = 6,
        auto_tune: bool = False,
        m_grid: list[int] | range = (3, 5, 8),
        G_grid: list[int] | range = range(4, 9),
        kernel: str = "imq",
        kernel_param: float = 1.0,
        n_permutations: int = 200,
        alpha: float = 0.05,
        min_clique_size: int = 3,
        use_hybrid: bool = False,
        temporal_window: int = 1,
        random_state: int = 42,
    ):
        self.metric = metric.lower()
        self.m_neighbors = m_neighbors
        self.n_clusters = n_clusters
        self.auto_tune = auto_tune
        self.m_grid = m_grid
        self.G_grid = G_grid
        self.kernel = kernel
        self.kernel_param = kernel_param
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.min_clique_size = min_clique_size
        self.use_hybrid = use_hybrid
        self.temporal_window = temporal_window
        self.random_state = random_state

        # Fitted state attributes
        self.is_fitted_ = False
        self.initial_labels_: np.ndarray | None = None
        self.rep_labels_: np.ndarray | None = None
        self.dist_matrix_: np.ndarray | None = None
        self.adjacency_: sp.csr_matrix | None = None
        self.cahc_model_ = None
        self.best_m_: int = m_neighbors
        self.best_G_: int = n_clusters
        self.tuning_results_: pd.DataFrame | None = None
        self.pairwise_df_: pd.DataFrame | None = None
        self.similarity_graph_: nx.Graph | None = None
        self.maximal_cliques_: list[list[int]] = []
        self.cluster_to_rep_: dict[int, int] = {}
        self.clique_metadata_: list[dict] = []
        self.encoder_ = KirkVCGRepSpatEncoder()
        self.encoding_: dict | None = None

    def fit(
        self,
        X: np.ndarray,
        vcg: np.ndarray,
        ecg: np.ndarray | None = None,
    ) -> KirkVCGRepSpat:
        """Fits the VCG repSpat pipeline on feature matrix X and 3D VCG trajectory.

        Parameters
        ----------
        X : np.ndarray
            Electrophysiological attribute matrix [n, p].
        vcg : np.ndarray
            3D VCG coordinates [n, 3].
        ecg : np.ndarray, optional
            Original multi-lead ECG array [n, 12].

        Returns
        -------
        self : KirkVCGRepSpat
        """
        X_arr = np.asarray(X, dtype=np.float64)
        vcg_arr = np.asarray(vcg, dtype=np.float64)

        if len(X_arr) != len(vcg_arr):
            raise ValueError(f"Length mismatch: X has {len(X_arr)}, vcg has {len(vcg_arr)} samples.")

        # Step 1: Matrix Representation
        self.dist_matrix_ = compute_attribute_distances(
            X_arr, metric=self.metric, standardize=(self.metric == "euclidean")
        )

        # Step 2: CAHC & Optional Hyperparameter Optimization
        if self.auto_tune:
            best_m, best_G, tuning_df = optimize_vcg_hyperparameters(
                X_arr,
                vcg_arr,
                m_neighbors_list=self.m_grid,
                n_clusters_range=self.G_grid,
                metric=self.metric,
                use_hybrid=self.use_hybrid,
            )
            self.best_m_ = best_m
            self.best_G_ = best_G
            self.tuning_results_ = tuning_df
        else:
            self.best_m_ = self.m_neighbors
            self.best_G_ = self.n_clusters

        self.initial_labels_, self.cahc_model_, self.adjacency_ = vcg_constrained_hac(
            X_arr,
            vcg_arr,
            n_clusters=self.best_G_,
            m_neighbors=self.best_m_,
            use_hybrid=self.use_hybrid,
            temporal_window=self.temporal_window,
            metric=self.metric,
        )

        # Step 3: Pairwise MMD^2 & Block Permutation Testing
        raw_pairwise = pairwise_cluster_testing(
            X_arr,
            self.initial_labels_,
            m_neighbors=self.best_m_,
            kernel=self.kernel,
            kernel_param=self.kernel_param,
            n_permutations=self.n_permutations,
            random_state=self.random_state,
        )
        self.pairwise_df_ = apply_fdr_to_pairwise_results(raw_pairwise, alpha=self.alpha)

        # Step 4: Similarity Graph & Maximal Clique Reassignment
        initial_clusters = sorted(np.unique(self.initial_labels_))
        self.similarity_graph_ = build_similarity_graph(
            self.pairwise_df_,
            all_clusters=initial_clusters,
            alpha=self.alpha,
        )

        self.maximal_cliques_ = extract_maximal_cliques(
            self.similarity_graph_,
            min_size=self.min_clique_size,
        )

        self.rep_labels_, self.cluster_to_rep_, self.clique_metadata_ = reassign_cliques_to_reps(
            self.initial_labels_,
            self.maximal_cliques_,
        )

        # Step 5: Encoding & Feature Extraction
        self.encoding_ = self.encoder_.encode(
            self.rep_labels_,
            X_arr,
            vcg=vcg_arr,
            clique_metadata=self.clique_metadata_,
        )

        self.is_fitted_ = True
        return self

    def fit_predict(
        self,
        X: np.ndarray,
        vcg: np.ndarray,
        ecg: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fits model and returns sample-level REP labels."""
        self.fit(X, vcg, ecg=ecg)
        return self.rep_labels_

    def fit_from_ecg(
        self,
        ecg: np.ndarray,
        fs: float = 500.0,
    ) -> KirkVCGRepSpat:
        """Convenience method to fit directly from raw multi-lead ECG signals."""
        feat_dict = extract_sample_features(ecg, fs=fs)
        X = feat_dict["continuous"] if self.metric == "euclidean" else feat_dict["binary"]
        vcg = feat_dict["vcg"]
        return self.fit(X, vcg, ecg=ecg)

    def get_summary(self) -> dict:
        """Returns diagnostic summary of the fitted pipeline."""
        if not self.is_fitted_:
            raise RuntimeError("Pipeline must be fitted before calling get_summary().")

        n_initial = len(np.unique(self.initial_labels_))
        n_final = len(np.unique(self.rep_labels_))
        n_cliques = len(self.maximal_cliques_)

        return {
            "best_m": self.best_m_,
            "best_G": self.best_G_,
            "initial_clusters_count": n_initial,
            "final_rep_labels_count": n_final,
            "maximal_cliques": self.maximal_cliques_,
            "n_cliques": n_cliques,
            "clique_metadata": self.clique_metadata_,
            "n_pairwise_tests": len(self.pairwise_df_) if self.pairwise_df_ is not None else 0,
            "similar_pairs_count": int((self.pairwise_df_["is_similar"]).sum())
            if self.pairwise_df_ is not None
            else 0,
            "anomaly_fraction": self.encoding_.get("anomaly_fraction", 0.0)
            if self.encoding_
            else 0.0,
            "encoding": self.encoding_,
        }
