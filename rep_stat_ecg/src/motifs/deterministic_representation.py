"""Canonical Deterministic ECG Reference Representation Extractor (Milestone M5).

Implements the authoritative feature extraction for:
- R0_basic (23D): Standard physical VCG dipole field statistics
- R0_rich (101D): Capacity-matched physical VCG control
- R1 (67D): Spatial domain categorical occupancy & dwell statistics
- R2_H (68D): Pure Hilbert functional trajectory statistics
- R3 (101D): Coupled physical + Hilbert functional representation
- R4 (122D): Full canonical reference representation Z_{12}^*
"""
from __future__ import annotations

import numpy as np
from scipy import stats


class CanonicalReferenceRepresentationExtractor:
    """Extracts R0_basic, R0_rich, R1, R2_H, R3, and R4 from 12-lead ECG signals."""

    def __init__(
        self,
        centroids: np.ndarray,
        median: np.ndarray,
        mad: np.ndarray,
        Z_H: np.ndarray,
        lead_directions: np.ndarray,
    ):
        self.centroids = centroids.astype(np.float64)  # (64, 3)
        self.median = median.astype(np.float64)        # (3,)
        self.mad = mad.astype(np.float64)              # (3,)
        self.Z_H = Z_H.astype(np.float64)              # (64, 9)
        self.A = lead_directions.astype(np.float64)    # (8, 3)

        self.c_std = (self.centroids - self.median) / (self.mad + 1e-12)  # (64, 3)
        self.c_norm_sq = np.sum(self.c_std ** 2, axis=1, keepdims=True).T  # (1, 64)
        self.A_pinv = np.linalg.pinv(self.A)           # (3, 8)

    def extract_from_ecg(self, E_12L: np.ndarray, fs: float = 500.0) -> dict[str, np.ndarray]:
        """Extract all baseline representation feature vectors from 12-lead ECG.
        
        Args:
            E_12L: Array of shape (12, T) in physical mV units.
            fs: Sampling frequency in Hz.
        Returns:
            dict with keys 'R0_basic', 'R0_rich', 'R1', 'R2_H', 'R3', 'R4'.
        """
        # Independent leads: I, II, V1-V6 (indices 0, 1, 6, 7, 8, 9, 10, 11)
        ind_idx = [0, 1, 6, 7, 8, 9, 10, 11]
        E_ind = E_12L[ind_idx, :].astype(np.float64)  # (8, T)
        T = E_ind.shape[1]
        dt = 1.0 / fs

        # 1. Deterministic Physical Trajectory: v(t) = A^+ E_ind(t)
        v = np.dot(self.A_pinv, E_ind)  # (3, T)
        E_dipole = np.dot(self.A, v)    # (8, T)
        r = E_ind - E_dipole            # (8, T)

        # 2. Hilbert Domain Assignments: h_H(t) = z_{G(v(t))}^H
        v_std = (v.T - self.median) / (self.mad + 1e-12)  # (T, 3)
        v_norm_sq = np.sum(v_std ** 2, axis=1, keepdims=True)  # (T, 1)
        dists_sq = v_norm_sq - 2.0 * np.dot(v_std, self.c_std.T) + self.c_norm_sq  # (T, 64)
        domain_assignments = np.argmin(dists_sq, axis=1)  # (T,)
        h_H = self.Z_H[domain_assignments, :]  # (T, 9)

        # ======================================================================
        # BLOCK 1: Physical VCG Base Features (23D)
        # ======================================================================
        mean_v = np.mean(v, axis=1)  # (3,)
        cov_v = np.cov(v)             # (3, 3)
        cov_v_triu = cov_v[np.triu_indices(3)]  # (6,)
        range_v = np.ptp(v, axis=1)   # (3,)

        norm_v = np.linalg.norm(v, axis=0)  # (T,)
        norm_stats = np.array([
            np.mean(norm_v),
            np.max(norm_v),
            np.std(norm_v),
            np.percentile(norm_v, 95),
        ])  # (4,)

        dv = np.diff(v, axis=1)
        path_length_v = np.sum(np.linalg.norm(dv, axis=0)) / float(T)  # (1,)

        vdot = dv / dt  # (3, T-1)
        vel_v = np.linalg.norm(vdot, axis=0)  # (T-1,)
        vel_stats = np.array([np.mean(vel_v), np.percentile(vel_v, 95)])  # (2,)

        d2v = np.diff(dv, axis=1) / (dt ** 2)  # (3, T-2)
        vddot = d2v
        acc_stats = np.array([np.mean(np.linalg.norm(d2v, axis=0))])  # (1,)

        eigvals_v, _ = np.linalg.eigh(cov_v)
        inertia_eigs = np.sort(eigvals_v)[::-1]  # (3,)

        r0_basic = np.concatenate([
            mean_v, cov_v_triu, range_v, norm_stats, [path_length_v],
            vel_stats, acc_stats, inertia_eigs
        ])  # 23D

        # ======================================================================
        # BLOCK 2: Rich Physical Controls (78D -> total R0_rich = 23 + 78 = 101D)
        # ======================================================================
        # 1. Extended coordinate quantiles (15)
        quantiles_v = np.percentile(v, [10, 25, 50, 75, 90], axis=1).T.flatten()  # (15,)

        # 2. Velocity Covariance & Stats (12)
        cov_vdot = np.cov(vdot)
        cov_vdot_triu = cov_vdot[np.triu_indices(3)]  # (6,)
        range_vdot = np.ptp(vdot, axis=1)            # (3,)
        vel_max = np.max(vel_v)                      # (1,)
        vel_std = np.std(vel_v)                      # (1,)
        speed_iqr = np.percentile(vel_v, 75) - np.percentile(vel_v, 25)  # (1,)
        vel_rich_stats = np.concatenate([cov_vdot_triu, range_vdot, [vel_max, vel_std, speed_iqr]])  # (12,)

        # 3. Angular Momentum / Loop Dynamics (12)
        # L(t) = v(t) x vdot(t) for t = 0..T-2
        v_sub = v[:, :-1]
        L = np.cross(v_sub.T, vdot.T).T  # (3, T-1)
        mean_L = np.mean(L, axis=1)      # (3,)
        cov_L = np.cov(L)
        cov_L_triu = cov_L[np.triu_indices(3)]  # (6,)
        mag_L = np.linalg.norm(L, axis=0)
        mag_L_mean = np.mean(mag_L)      # (1,)
        mag_L_p95 = np.percentile(mag_L, 95)  # (1,)
        eig_L, _ = np.linalg.eigh(cov_L)
        sum_eig_L = np.sum(eig_L)
        loop_planarity = (np.min(eig_L) / (sum_eig_L + 1e-12)) if sum_eig_L > 0 else 0.0  # (1,)
        ang_mom_stats = np.concatenate([mean_L, cov_L_triu, [mag_L_mean, mag_L_p95, loop_planarity]])  # (12,)

        # 4. Acceleration Covariance & Multi-scale (12)
        cov_vddot = np.cov(vddot)
        cov_vddot_triu = cov_vddot[np.triu_indices(3)]  # (6,)
        # Multi-scale rolling variance
        w50 = int(0.050 * fs)   # 25 samples
        w200 = int(0.200 * fs)  # 100 samples
        short_vars = []
        long_vars = []
        for ax in range(3):
            vx = v[ax, :]
            # Sub-segment variances
            n_seg50 = T // w50
            if n_seg50 > 1:
                v_reshaped = vx[:n_seg50 * w50].reshape(n_seg50, w50)
                short_vars.append(np.mean(np.var(v_reshaped, axis=1)))
            else:
                short_vars.append(np.var(vx))

            n_seg200 = T // w200
            if n_seg200 > 1:
                v_reshaped = vx[:n_seg200 * w200].reshape(n_seg200, w200)
                long_vars.append(np.mean(np.var(v_reshaped, axis=1)))
            else:
                long_vars.append(np.var(vx))
        acc_multiscale_stats = np.concatenate([cov_vddot_triu, short_vars, long_vars])  # (12,)

        # 5. Spatial Octant Occupancy & Radial Shells (12)
        signs = (v_std >= 0).astype(int)
        octant_ids = signs[:, 0] * 4 + signs[:, 1] * 2 + signs[:, 2]
        octant_counts = np.bincount(octant_ids, minlength=8)
        octant_occupancy = octant_counts[:8] / float(T)  # (8,)

        r_std = np.linalg.norm(v_std, axis=1)
        radial_shells = np.array([
            np.mean(r_std < 0.5),
            np.mean((r_std >= 0.5) & (r_std < 1.0)),
            np.mean((r_std >= 1.0) & (r_std < 2.0)),
            np.mean(r_std >= 2.0),
        ])  # (4,)
        spatial_density_stats = np.concatenate([octant_occupancy, radial_shells])  # (12,)

        # 6. Phase-Plane Bivariate Statistics (9)
        phase_corrs = []
        phase_areas = []
        vel_skews = []
        for ax in range(3):
            v_ax = v[ax, :-1]
            vd_ax = vdot[ax, :]
            c = np.corrcoef(v_ax, vd_ax)[0, 1]
            phase_corrs.append(0.0 if np.isnan(c) else c)
            phase_areas.append(np.ptp(v_ax) * np.ptp(vd_ax))
            vel_skews.append(float(stats.skew(vd_ax)))
        phase_stats = np.concatenate([phase_corrs, phase_areas, vel_skews])  # (9,)

        # 7. Temporal Dynamics (6)
        half_T = T // 2
        drift_mean = np.mean(v[:, half_T:], axis=1) - np.mean(v[:, :half_T], axis=1)  # (3,)
        drift_energy = np.mean(norm_v[half_T:] ** 2) - np.mean(norm_v[:half_T] ** 2)   # (1,)
        # Autocorrelation of norm_v at lag 50 (100ms) and 100 (200ms)
        norm_v_centered = norm_v - np.mean(norm_v)
        var_norm_v = np.var(norm_v) + 1e-12
        autocorr_1 = np.mean(norm_v_centered[:-50] * norm_v_centered[50:]) / var_norm_v
        autocorr_2 = np.mean(norm_v_centered[:-100] * norm_v_centered[100:]) / var_norm_v
        temporal_stats = np.concatenate([drift_mean, [drift_energy, autocorr_1, autocorr_2]])  # (6,)

        rich_physical_features = np.concatenate([
            quantiles_v, vel_rich_stats, ang_mom_stats, acc_multiscale_stats,
            spatial_density_stats, phase_stats, temporal_stats
        ])  # 78D

        r0_rich = np.concatenate([r0_basic, rich_physical_features])  # 23 + 78 = 101D

        # ======================================================================
        # BLOCK 3: Spatial Domain Occupancy Histogram -> R1 (67D)
        # ======================================================================
        domain_counts = np.bincount(domain_assignments, minlength=64)
        domain_hist = domain_counts / float(T)  # (64,)
        occ_entropy = -np.sum(domain_hist * np.log(domain_hist + 1e-12))  # (1,)
        n_unique_domains = np.sum(domain_counts > 0)                      # (1,)
        
        diff_domains = np.diff(domain_assignments) != 0
        n_transitions = np.sum(diff_domains)
        mean_dwell = float(T) / float(max(n_transitions, 1))             # (1,)

        r1 = np.concatenate([domain_hist, [occ_entropy, n_unique_domains, mean_dwell]])  # 67D

        # ======================================================================
        # BLOCK 4: Hilbert Functional Trajectory -> R2_H (68D)
        # ======================================================================
        mean_h = np.mean(h_H, axis=0)  # (9,)
        cov_h = np.cov(h_H.T)          # (9, 9)
        cov_h_triu = cov_h[np.triu_indices(9)]  # (45,)
        range_h = np.ptp(h_H, axis=0)  # (9,)

        dh = np.diff(h_H, axis=0)
        path_length_h = np.sum(np.linalg.norm(dh, axis=1)) / float(T)  # (1,)

        if n_transitions > 0:
            h_transitions = np.linalg.norm(dh[diff_domains, :], axis=1)
            mean_trans_dist = float(np.mean(h_transitions))            # (1,)
        else:
            mean_trans_dist = 0.0

        mean_h_norm = np.mean(np.linalg.norm(h_H, axis=1))             # (1,)

        r2_h = np.concatenate([
            mean_h, cov_h_triu, range_h,
            [path_length_h, mean_trans_dist, mean_dwell, occ_entropy, mean_h_norm]
        ])  # 68D

        # ======================================================================
        # BLOCK 5: Physical-Functional Coupling (10D)
        # ======================================================================
        v_centered = v - np.mean(v, axis=1, keepdims=True)
        h_centered = h_H[:, :3].T - np.mean(h_H[:, :3].T, axis=1, keepdims=True)
        cross_cov = np.dot(v_centered, h_centered.T) / float(T)  # (3, 3)
        cross_cov_flat = cross_cov.flatten()                      # (9,)
        norm_vh_corr = float(np.corrcoef(norm_v, np.linalg.norm(h_H, axis=1))[0, 1])
        if np.isnan(norm_vh_corr):
            norm_vh_corr = 0.0
        coupling_features = np.concatenate([cross_cov_flat, [norm_vh_corr]])  # 10D

        # R3 = Coupled Physical + Hilbert
        r3 = np.concatenate([r0_basic, r2_h, coupling_features])  # 23 + 68 + 10 = 101D

        # ======================================================================
        # BLOCK 6: Residual ECG Statistics -> R4 (122D)
        # ======================================================================
        lead_rms = np.sqrt(np.mean(r ** 2, axis=1))  # (8,)
        limb_rms_sq = np.sum(lead_rms[:2] ** 2)
        precord_rms_sq = np.sum(lead_rms[2:] ** 2)
        regional_ratio = precord_rms_sq / (limb_rms_sq + 1e-12)  # (1,)

        lead_signal_rms = np.sqrt(np.mean(E_ind ** 2, axis=1)) + 1e-12
        residual_frac = lead_rms / lead_signal_rms  # (8,)

        dr = np.diff(r, axis=1) / dt
        hf_power = np.mean(dr ** 2, axis=1)[[0, 1, 3, 6]]  # Leads I, II, V2, V5 (4,)

        residual_features = np.concatenate([lead_rms, [regional_ratio], residual_frac, hf_power])  # 21D

        r4 = np.concatenate([r3, residual_features])  # 101 + 21 = 122D

        return {
            "R0_basic": r0_basic,
            "R0_rich": r0_rich,
            "R1": r1,
            "R2_H": r2_h,
            "R3": r3,
            "R4": r4,
        }
