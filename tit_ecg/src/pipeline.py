"""End-to-End Pipeline for Exact repSpat and Minimal-Change ECG/VCG Adaptation.

Implements:
1. ExactRepSpat: Canonical reference implementation for arbitrary spatial/attribute data (S, X).
2. TemporalRepSpatECG: Minimal-change adaptation where:
       s_t = (tau_t, 0) in R^2 (temporal domain embedded in R^2)
       X(s_t) = v(t) in R^3 (Kors 3D VCG trajectory)
   Preserves paper roles: s supplies domain contiguity, X supplies multivariate attributes.
3. VCGStateRepSpatResidual: Secondary state-space adaptation:
       s_t = v(t) in R^3 (electrical state-space domain)
       X(s_t) = r(t) (residual non-dipolar attribute)
   Explicitly separated to prevent conflating temporal and state-space formulations.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .cahc import (
    compute_attribute_dissimilarity,
    construct_domain_links,
    run_cahc,
    search_optimal_cahc,
)
from .config import RepSpatConfig
from .graph_reassignment import (
    build_similarity_graph,
    reassign_by_cliques,
    reassign_by_connected_components,
)
from .mmd_test import run_all_pairwise_mmd_tests
from .vcg_transform import KORS_REGRESSION_MATRIX, ecg_to_vcg_kors


class ExactRepSpat:
    """Exact, reproducible reference implementation of repSpat (Senanayake & Jeganathan, 2026).

    Parameters
    ----------
    config : RepSpatConfig, optional
        Algorithmic configuration and provenance options.
    """

    def __init__(self, config: RepSpatConfig | None = None) -> None:
        self.config = config or RepSpatConfig()
        self.results_: dict[str, Any] = {}

    def fit(self, S: np.ndarray, X: np.ndarray) -> ExactRepSpat:
        """Executes full 4-stage repSpat procedure on domain coordinates S and attributes X.

        Stage 1: CAHC with Lance-Williams Ward update and modified silhouette grid search.
        Stage 2: Pairwise biased empirical MMD^2 (Eq. 6) with IMQ kernel (c=1).
        Stage 3: Attribute k-means block permutation test with strict '>' stopping rule.
        Stage 4: Benjamini-Hochberg FDR correction and dual (CC & Clique) graph reassignment.

        Parameters
        ----------
        S : np.ndarray of shape [n, d]
            Domain coordinates (e.g. physical 2D coordinates [x, y], or (tau_t, 0)).
        X : np.ndarray of shape [n, p]
            Multivariate attribute vectors.

        Returns
        -------
        self : ExactRepSpat
        """
        S = np.asarray(S, dtype=float)
        X = np.asarray(X, dtype=float)
        n_samples = len(S)
        if len(X) != n_samples:
            raise ValueError(f"S and X must have equal sample counts: {len(S)} vs {len(X)}")

        # 1. Contiguity & CAHC grid search [PAPER §2.1 & §2.1.1]
        m_star, G_star, scores_df, partition_cache = search_optimal_cahc(
            S=S,
            X=X,
            m_grid=self.config.m_grid,
            G_grid=self.config.G_grid,
            metric=self.config.metric,
            linkage=self.config.linkage,
            symmetrize_links=self.config.symmetrize_links,
        )

        initial_labels = partition_cache.get((m_star, G_star))
        if initial_labels is None:
            # Recompute if not cached
            D = compute_attribute_dissimilarity(X, metric=self.config.metric)
            L = construct_domain_links(S, m=m_star, symmetrize=self.config.symmetrize_links)
            initial_labels = run_cahc(D=D, L=L, n_clusters=G_star, linkage=self.config.linkage, X=X)
        else:
            D = compute_attribute_dissimilarity(X, metric=self.config.metric)
            L = construct_domain_links(S, m=m_star, symmetrize=self.config.symmetrize_links)

        # 2 & 3. Pairwise Biased MMD^2 and Whole-Block Permutations [PAPER §2.2, §2.3, App. A.2]
        pairwise_df = run_all_pairwise_mmd_tests(
            X=X,
            labels=initial_labels,
            dist_matrix=D,
            m_star=m_star,
            kernel=self.config.kernel,
            kernel_param=self.config.kernel_param,
            kernel_scale_rule=self.config.kernel_scale_rule,
            block_mode=self.config.block_permutation_mode,
            n_permutations=self.config.n_permutations,
            fdr_alpha=self.config.fdr_alpha,
            rounding_rule=self.config.rounding_rule,
            strict_exceed=self.config.strict_block_exceed,
            kmeans_n_init=self.config.kmeans_n_init,
            p_value_correction=self.config.p_value_correction,
            random_state=self.config.random_state,
        )

        # 4. Graph Construction and Dual Reassignment [PAPER §2.4 vs Alg. 1]
        unique_cahc_clusters = sorted(np.unique(initial_labels))
        G_sim = build_similarity_graph(
            pairwise_df=pairwise_df,
            all_clusters=unique_cahc_clusters,
            fdr_alpha=self.config.fdr_alpha,
        )

        # 4a. Algorithm 1 Step 4(b): Connected Components
        labels_cc, cluster_to_cc = reassign_by_connected_components(
            initial_labels=initial_labels,
            G_sim=G_sim,
        )

        # 4b. Section 2.4 & Section 3.2: Maximal Cliques
        labels_clique, cluster_to_clique, clique_audit = reassign_by_cliques(
            initial_labels=initial_labels,
            G_sim=G_sim,
            min_size=self.config.clique_min_size,
        )

        primary_labels = labels_clique if self.config.reassignment_primary == "clique" else labels_cc

        self.results_ = {
            "m_star": int(m_star),
            "G_star": int(G_star),
            "scores_df": scores_df,
            "initial_labels": initial_labels,
            "pairwise_df": pairwise_df,
            "similarity_graph": G_sim,
            "labels_cc": labels_cc,
            "labels_clique": labels_clique,
            "labels_primary": primary_labels,
            "cluster_to_cc": cluster_to_cc,
            "cluster_to_clique": cluster_to_clique,
            "clique_audit": clique_audit,
            "graph_descriptors": clique_audit.get("graph_descriptors", {}),
            "config": asdict(self.config),
            "n_samples": n_samples,
            "n_initial_clusters": len(unique_cahc_clusters),
            "n_cc_clusters": len(np.unique(labels_cc)),
            "n_clique_clusters": len(np.unique(labels_clique)),
        }
        return self


class TemporalRepSpatECG:
    """Minimal-change adaptation of repSpat to multi-lead ECG / 3D VCG.

    Mapping:
    - s_t = (tau_t, 0) in R^2 (chronological domain embedded into R^2)
    - X(s_t) = v(t) in R^3 (continuous 3D electrical dipole via Kors transform)

    Preserves exact mathematical functions:
    - Contiguity = temporal neighborhood in elapsed time
    - Attributes = continuous 3D VCG voltages
    - Initial clusters = contiguous temporal intervals
    - Blocks = attribute-based k-means on VCG vectors within each temporal interval
    - Reassignment = detects temporally separated intervals sharing recurrent VCG distributions.
    """

    def __init__(self, config: RepSpatConfig | None = None) -> None:
        self.config = config or RepSpatConfig()
        self.model = ExactRepSpat(config=self.config)
        self.results_: dict[str, Any] = {}

    def fit(
        self,
        ecg_or_vcg: np.ndarray,
        sampling_rate: float = 500.0,
        is_vcg: bool = False,
        ground_truth_waves: np.ndarray | None = None,
    ) -> TemporalRepSpatECG:
        """Fits Temporal repSpat on an ECG or VCG recording.

        Parameters
        ----------
        ecg_or_vcg : np.ndarray
            Either 12-lead ECG [n_samples, 12] or 3D VCG [n_samples, 3].
        sampling_rate : float
            Sampling frequency in Hz (default: 500.0).
        is_vcg : bool
            If True, ecg_or_vcg is already 3D VCG; if False, applies Kors transform.
        ground_truth_waves : np.ndarray, optional
            Optional wave delineations (0=iso, 1=P, 2=QRS, 3=T) for concordance metrics.

        Returns
        -------
        self : TemporalRepSpatECG
        """
        arr = np.asarray(ecg_or_vcg, dtype=float)
        if not is_vcg:
            vcg = ecg_to_vcg_kors(arr)
        else:
            vcg = arr

        # [ECG-ADAPTATION]: Invariant neighborhood scale in milliseconds
        if self.config.m_ms_grid is not None:
            m_samples = [
                max(2, int(round(ms * float(sampling_rate) / 1000.0)))
                for ms in self.config.m_ms_grid
            ]
            self.model.config.m_grid = sorted(list(set(m_samples)))

        n_samples = len(vcg)
        time_tau = np.arange(n_samples, dtype=float) / float(sampling_rate)

        # [ECG-ADAPTATION]: Embed time line into R^2 as (tau_t, 0)
        # Keeps formal 2D domain of repSpat paper while enforcing chronological contiguity.
        S = np.column_stack([time_tau, np.zeros_like(time_tau)])
        X = vcg  # [n_samples, 3]

        # Execute exact repSpat
        self.model.fit(S, X)
        self.results_ = dict(self.model.results_)
        self.results_["time"] = time_tau
        self.results_["vcg"] = vcg
        self.results_["sampling_rate"] = sampling_rate

        # If ground-truth annotations provided, compute wave concordance (ARI, NMI)
        if ground_truth_waves is not None:
            gt = np.asarray(ground_truth_waves, dtype=int)
            # Remove background / non-wave if desired, or test over whole signal
            valid_mask = ~np.isnan(gt) & (gt >= 0)
            gt_valid = gt[valid_mask]

            ari_initial = float(adjusted_rand_score(gt_valid, self.results_["initial_labels"][valid_mask]))
            nmi_initial = float(normalized_mutual_info_score(gt_valid, self.results_["initial_labels"][valid_mask]))

            ari_cc = float(adjusted_rand_score(gt_valid, self.results_["labels_cc"][valid_mask]))
            nmi_cc = float(normalized_mutual_info_score(gt_valid, self.results_["labels_cc"][valid_mask]))

            ari_clique = float(adjusted_rand_score(gt_valid, self.results_["labels_clique"][valid_mask]))
            nmi_clique = float(normalized_mutual_info_score(gt_valid, self.results_["labels_clique"][valid_mask]))

            self.results_["concordance"] = {
                "ari_initial": ari_initial,
                "nmi_initial": nmi_initial,
                "ari_cc": ari_cc,
                "nmi_cc": nmi_cc,
                "ari_clique": ari_clique,
                "nmi_clique": nmi_clique,
            }

        return self


class VCGStateRepSpatResidual:
    """Secondary state-space adaptation: s_t = v(t) in R^3, X(s_t) = r(t) in R^p.

    In this formulation:
    - Domain s_t is the 3D VCG cardiac electrical state space.
    - Contiguity links L connect nearby electrical states, NOT nearby times.
    - Attribute X(s_t) is the non-dipolar residual r(t) = E(t) - A v(t).
    - Clusters C_g are connected patches in VCG state space.
    - MMD^2 compares the residual distribution across separated regions of state space.

    Note: This is scientifically interesting but is a SUBSTANTIAL adaptation,
    not the reference minimal-change translation.
    """

    def __init__(self, config: RepSpatConfig | None = None) -> None:
        self.config = config or RepSpatConfig()
        self.model = ExactRepSpat(config=self.config)
        self.results_: dict[str, Any] = {}

    def fit(self, ecg: np.ndarray, sampling_rate: float = 500.0) -> VCGStateRepSpatResidual:
        """Fits VCG state-space repSpat on 12-lead ECG.

        Parameters
        ----------
        ecg : np.ndarray of shape [n_samples, 12]
        sampling_rate : float
        """
        ecg = np.asarray(ecg, dtype=float)
        vcg = ecg_to_vcg_kors(ecg)

        # Compute pseudo-inverse dipolar forward model A: E_dip = vcg @ A^T
        # Least-squares fit of 12-lead ECG from 3-lead VCG:
        # A, residuals, rank, s = np.linalg.lstsq(vcg, ecg, rcond=None)
        A, _, _, _ = np.linalg.lstsq(vcg, ecg, rcond=None)
        ecg_dipolar = vcg @ A
        residual = ecg - ecg_dipolar  # [n_samples, 12]

        # Domain S = vcg [n_samples, 3] (state space)
        # Attributes X = residual [n_samples, 12]
        self.model.fit(S=vcg, X=residual)
        self.results_ = dict(self.model.results_)
        self.results_["domain_vcg"] = vcg
        self.results_["residual_attributes"] = residual
        self.results_["forward_matrix_A"] = A
        return self
