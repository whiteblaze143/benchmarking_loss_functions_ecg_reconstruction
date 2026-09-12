"""KirkVCG repSpat ECG Encoder.

Translates sample-level Repeated Electrophysiological Pattern (REP) labelings
into rich, interpretable, vector-space embeddings for patient phenotyping,
arrhythmia detection, and longitudinal cardiac monitoring.

Features extracted per recording (Section 3.2 of specification):
1. Occupancy: Fractional time spent in each REP state.
2. Transition Dynamics: Empirical Markov transition probability matrix between states.
3. Within-REP Variability: Feature dispersion and beat-to-beat variability within each REP.
4. REP Centroids: Morphological dipole centers in 3D VCG space and feature space R^p.
5. Clique Structure & Rhythm Regularity: Number/sizes of cliques, anomalous fraction.
"""
from __future__ import annotations

import numpy as np


class KirkVCGRepSpatEncoder:
    """Encodes sample-level REP labels and electrophysiological features into recording-level biomarkers."""

    def __init__(self, max_states: int = 10):
        self.max_states = max_states

    def encode(
        self,
        rep_labels: np.ndarray,
        X: np.ndarray,
        vcg: np.ndarray | None = None,
        clique_metadata: list[dict] | None = None,
    ) -> dict:
        """Extracts complete electrophysiological fingerprint from fitted REP labels.

        Parameters
        ----------
        rep_labels : np.ndarray
            Sample-level pattern labels of length n.
        X : np.ndarray
            Electrophysiological feature matrix [n, p].
        vcg : np.ndarray, optional
            3D VCG coordinates [n, 3].
        clique_metadata : list of dict, optional
            Metadata from clique reassignment indicating which labels are REPs.

        Returns
        -------
        encoding : dict
            {
                'occupancy': dict[int, float],
                'occupancy_vector': np.ndarray [max_states],
                'transition_matrix': np.ndarray [max_states, max_states],
                'within_rep_variability': dict[int, np.ndarray],
                'mean_variability_per_rep': dict[int, float],
                'rep_centroids_feature': dict[int, np.ndarray],
                'rep_centroids_vcg': dict[int, np.ndarray],
                'n_cliques': int,
                'n_rep_states': int,
                'n_unique_states': int,
                'anomaly_fraction': float,
                'flat_feature_vector': np.ndarray,
            }
        """
        labels = np.asarray(rep_labels, dtype=int)
        X_arr = np.asarray(X, dtype=np.float64)
        n = len(labels)
        p = X_arr.shape[1]

        unique_states = sorted(np.unique(labels))
        n_unique_states = len(unique_states)

        # Identify which states are REPs vs unique/anomalous
        rep_state_ids = set()
        if clique_metadata is not None:
            for item in clique_metadata:
                if item.get("is_rep", False):
                    rep_state_ids.add(item["rep_id"])
        else:
            rep_state_ids = set(unique_states)

        # 1. State Occupancy
        occupancy: dict[int, float] = {}
        for state in unique_states:
            occupancy[state] = float(np.sum(labels == state) / float(n))

        # Vectorized occupancy bounded by max_states
        occupancy_vec = np.zeros(self.max_states, dtype=np.float64)
        for i, state in enumerate(unique_states[: self.max_states]):
            occupancy_vec[i] = occupancy[state]

        # 2. Transition Dynamics Matrix (Markov transition probabilities)
        # Count transitions: y_{t} -> y_{t+1}
        trans_counts = np.zeros((self.max_states, self.max_states), dtype=np.float64)
        state_to_idx = {state: idx for idx, state in enumerate(unique_states[: self.max_states])}

        for t in range(n - 1):
            s_from = labels[t]
            s_to = labels[t + 1]
            if s_from in state_to_idx and s_to in state_to_idx:
                idx_from = state_to_idx[s_from]
                idx_to = state_to_idx[s_to]
                trans_counts[idx_from, idx_to] += 1.0

        # Row-normalize to obtain probabilities
        row_sums = trans_counts.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            trans_counts,
            row_sums,
            out=np.zeros_like(trans_counts),
            where=row_sums > 0,
        )

        # 3. Within-REP Variability and Centroids
        within_rep_var: dict[int, np.ndarray] = {}
        mean_var_per_rep: dict[int, float] = {}
        rep_centroids_feature: dict[int, np.ndarray] = {}
        rep_centroids_vcg: dict[int, np.ndarray] = {}

        for state in unique_states:
            mask = (labels == state)
            X_state = X_arr[mask]
            centroid = np.mean(X_state, axis=0)
            rep_centroids_feature[state] = centroid

            var_vec = np.var(X_state, axis=0) if len(X_state) > 1 else np.zeros(p)
            within_rep_var[state] = var_vec
            mean_var_per_rep[state] = float(np.mean(var_vec))

            if vcg is not None:
                vcg_state = vcg[mask]
                rep_centroids_vcg[state] = np.mean(vcg_state, axis=0)

        # 4. Clique Structure & Rhythm Regularity
        n_reps = len([s for s in unique_states if s in rep_state_ids])
        n_singletons = len([s for s in unique_states if s not in rep_state_ids])

        # Anomaly fraction = fraction of time spent in singleton/non-REP states
        anomaly_samples = np.sum([np.sum(labels == s) for s in unique_states if s not in rep_state_ids])
        anomaly_fraction = float(anomaly_samples / float(n)) if n > 0 else 0.0

        # 5. VCG Geometric Biomarkers: Loop Planarity and Spatial Angle
        loop_planarity = 1.0
        spatial_qrst_angle = 0.0
        if vcg is not None and len(vcg) > 3:
            from ..vcg.kinematics import compute_vcg_loop_planarity, compute_spatial_qrst_angle
            loop_planarity = compute_vcg_loop_planarity(vcg)
            sorted_by_occ = sorted(unique_states, key=lambda s: occupancy.get(s, 0.0), reverse=True)
            if len(sorted_by_occ) >= 2 and sorted_by_occ[0] in rep_centroids_vcg and sorted_by_occ[1] in rep_centroids_vcg:
                c1 = rep_centroids_vcg[sorted_by_occ[0]]
                c2 = rep_centroids_vcg[sorted_by_occ[1]]
                spatial_qrst_angle = compute_spatial_qrst_angle(c1, c2, degrees=True)

        # 6. Fixed-dimension summary embedding vector (for ML classification)
        # Combines occupancy, top transition probabilities, anomaly fraction, and mean variability
        diag_trans = np.diag(transition_matrix)  # State persistence
        offdiag_trans = transition_matrix[~np.eye(self.max_states, dtype=bool)][: 10]  # First 10 transitions
        avg_vars = np.array([mean_var_per_rep.get(s, 0.0) for s in unique_states[: self.max_states]])
        if len(avg_vars) < self.max_states:
            avg_vars = np.pad(avg_vars, (0, self.max_states - len(avg_vars)))

        flat_vector = np.concatenate([
            occupancy_vec,                     # [max_states]
            diag_trans,                        # [max_states]
            offdiag_trans,                     # [10]
            avg_vars,                          # [max_states]
            np.array([anomaly_fraction, float(n_reps), float(n_cliques), loop_planarity, spatial_qrst_angle]),
        ])

        return {
            "occupancy": occupancy,
            "occupancy_vector": occupancy_vec,
            "transition_matrix": transition_matrix,
            "within_rep_variability": within_rep_var,
            "mean_variability_per_rep": mean_var_per_rep,
            "rep_centroids_feature": rep_centroids_feature,
            "rep_centroids_vcg": rep_centroids_vcg,
            "n_cliques": n_cliques,
            "n_rep_states": n_reps,
            "n_unique_states": n_singletons,
            "anomaly_fraction": anomaly_fraction,
            "loop_planarity": loop_planarity,
            "spatial_qrst_angle": spatial_qrst_angle,
            "flat_feature_vector": flat_vector,
        }
