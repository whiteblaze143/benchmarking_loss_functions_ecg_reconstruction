"""Dependence calibration functions for repSpat MMD inference on QVCG microstates.

Stage M3H-CAL: Validates Type-I error calibration and power of frozen repSpat
under patient / beat / time dependence hierarchy.

Contains only pure functions:
- generate_deterministic_seed
- build_patient_bundles
- split_patients_disjoint
- split_beats_shared_patient
- apply_mean_shift
- apply_covariance_shift
- generate_vector_ar1_beats
- compute_lag1_autocorrelation
- compute_temporal_dependence_summary
- compute_wilson_interval
- compute_bootstrap_fdr_ci
- summarize_pairwise_calibration
- compute_fdp
- evaluate_pair_test
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from rep_stat_ecg.src.motifs.mmd import (
    build_attribute_blocks,
    compute_n_blocks,
    precompute_block_kernel_sums,
    run_block_permutation_test,
)


# ==============================================================================
# 0. Deterministic Seed Generator
# ==============================================================================

def generate_deterministic_seed(
    run_id: str,
    scenario: str,
    domain: int,
    effect: float | str,
    replicate: int,
) -> int:
    """Generate process- and worker-independent 32-bit seed via SHA-256."""
    s = f"{run_id}_{scenario}_{domain}_{effect}_{replicate}"
    digest = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % (2**32)


# ==============================================================================
# 1. Patient & Beat Resampling Hierarchy
# ==============================================================================

def build_patient_bundles(domain_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group domain microstates by patient into atomic patient bundles."""
    bundles = {}
    for pid, group in domain_df.groupby("patient_id"):
        bundles[str(pid)] = group.copy().reset_index(drop=True)
    return bundles


def split_patients_disjoint(
    bundles: dict[str, pd.DataFrame],
    rng: np.random.RandomState,
    ratio: tuple[int, int] = (1, 1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split patient bundles into two disjoint pseudo-domains (CAL0-D).
    
    Supports balanced (1:1) and imbalanced (e.g. 1:3) splits.
    Guarantees exact set invariants: A and B are strictly disjoint and
    their union covers the sampled bundle set.
    """
    pids = sorted(list(bundles.keys()))
    if len(pids) < 2:
        raise ValueError(f"Need at least 2 patients for disjoint split, got {len(pids)}")
    
    r_A, r_B = ratio
    total_parts = r_A + r_B
    n_A = max(1, int(round(len(pids) * r_A / total_parts)))
    n_A = min(n_A, len(pids) - 1)  # Ensure at least 1 in B
    
    permuted_pids = list(rng.permutation(pids))
    pids_A = permuted_pids[:n_A]
    pids_B = permuted_pids[n_A:]
    
    df_A = pd.concat([bundles[p] for p in pids_A], ignore_index=True)
    df_B = pd.concat([bundles[p] for p in pids_B], ignore_index=True)
    return df_A, df_B


def split_beats_shared_patient(
    bundles: dict[str, pd.DataFrame],
    rng: np.random.RandomState,
    ratio: tuple[int, int] = (1, 1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole beats so both pseudo-domains share patients (CAL0-S).
    
    Requires patients to have >= 2 complete beats.
    Guarantees:
    1. Every sampled patient is present in both A and B.
    2. Complete beats remain intact; individual beat microstates are never split.
    3. Remaining beats are allocated to approach the target ratio (1:1 or 1:3).
    """
    pids = sorted(list(bundles.keys()))
    
    # Step 1: Guarantee every patient has 1 beat in A and 1 beat in B
    beats_A: list[tuple[str, Any]] = []
    beats_B: list[tuple[str, Any]] = []
    remaining_beats: list[tuple[str, Any]] = []
    
    for pid in pids:
        b_df = bundles[pid]
        unique_beats = sorted(list(b_df["beat_id"].unique()))
        if len(unique_beats) == 1:
            # Fallback if a patient has only 1 beat
            if rng.rand() < 0.5:
                beats_A.append((pid, unique_beats[0]))
            else:
                beats_B.append((pid, unique_beats[0]))
        else:
            perm = list(rng.permutation(unique_beats))
            beats_A.append((pid, perm[0]))
            beats_B.append((pid, perm[1]))
            for b in perm[2:]:
                remaining_beats.append((pid, b))
                
    # Step 2: Allocate remaining beats toward target ratio
    r_A, r_B = ratio
    total_beats = len(beats_A) + len(beats_B) + len(remaining_beats)
    target_A = max(len(beats_A), int(round(total_beats * r_A / (r_A + r_B))))
    needed_A = max(0, target_A - len(beats_A))
    
    if remaining_beats:
        perm_rem = [remaining_beats[i] for i in rng.permutation(len(remaining_beats))]
        beats_A.extend(perm_rem[:needed_A])
        beats_B.extend(perm_rem[needed_A:])
        
    # Assemble DataFrames
    dfs_A = []
    dfs_B = []
    set_A = set(beats_A)
    set_B = set(beats_B)
    
    for pid in pids:
        b_df = bundles[pid]
        p_beats_A = [b for (p, b) in set_A if p == pid]
        p_beats_B = [b for (p, b) in set_B if p == pid]
        
        if p_beats_A:
            dfs_A.append(b_df[b_df["beat_id"].isin(p_beats_A)])
        if p_beats_B:
            dfs_B.append(b_df[b_df["beat_id"].isin(p_beats_B)])
            
    df_A = pd.concat(dfs_A, ignore_index=True) if dfs_A else pd.DataFrame()
    df_B = pd.concat(dfs_B, ignore_index=True) if dfs_B else pd.DataFrame()
    return df_A, df_B


# ==============================================================================
# 2. Controlled Transformations (CAL1, CAL2, CAL3)
# ==============================================================================

def apply_mean_shift(
    data: np.ndarray,
    delta: float,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Apply standardized mean shift along unit vector u = (1,1,1)/sqrt(3).
    
    Parameters mean and std are frozen at the source-domain level before splitting:
        x* = (x - mu) / sigma
        x*' = x* + delta * u
        x' = x + delta * u * sigma
    Preserves covariance identically.
    """
    data = np.asarray(data, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    std_safe = np.where(std < 1e-12, 1.0, std)
    
    u = np.ones(3, dtype=np.float64) / math.sqrt(3.0)
    shift_vector = delta * u * std_safe
    return data + shift_vector


def apply_covariance_shift(
    data: np.ndarray,
    gamma: float,
    mean: np.ndarray,
    std: np.ndarray,
    eigvecs_std: np.ndarray,
) -> np.ndarray:
    """Scale variance along leading PC in standardized descriptor space by gamma.
    
    Parameters mean, std, and eigvecs_std (V_g*) are estimated once from the full
    source domain before any replicate split:
        x* = (x - mu_g) / sigma_g
        y = x* @ V_g*
        y'_1 = gamma * y_1
        x*' = y' @ (V_g*)^T
        x' = x*' * sigma_g + mu_g
    Preserves sample mean and scales variance along leading standardized PC by gamma^2.
    """
    data = np.asarray(data, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    std_safe = np.where(std < 1e-12, 1.0, std)
    V_star = np.asarray(eigvecs_std, dtype=np.float64)
    
    # 1. Standardize using frozen domain parameters
    x_star = (data - mean) / std_safe
    
    # 2. Project onto frozen standardized eigenvectors
    Y = x_star @ V_star
    
    # 3. Scale leading PC (last column in eigh)
    Y[:, -1] *= gamma
    
    # 4. Reconstruct in standardized coordinates
    x_star_prime = Y @ V_star.T
    
    # 5. Re-scale to original physical units
    transformed = x_star_prime * std_safe + mean
    return transformed


def generate_vector_ar1_beats(
    mean: np.ndarray,
    cov: np.ndarray,
    phi: float,
    beat_lengths: list[int],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Generate synthetic beats via stationary multivariate vector AR(1).
    
    Model:
        X_t - mu = phi * (X_{t-1} - mu) + epsilon_t,
        epsilon_t ~ N(0, (1 - phi^2) * Cov)
    Stationary Initial Condition:
        X_{b, 1} ~ N(mu, Cov)  [Starts directly from stationary distribution; no transients]
        
    Guarantees that marginally X_t ~ N(mu, Cov) for all t and any phi in (-1, 1).
    Only temporal autocorrelation changes with phi.
    """
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    # Ensure positive semi-definite
    cov = 0.5 * (cov + cov.T)
    min_eig = np.min(np.linalg.eigvalsh(cov))
    if min_eig < 1e-8:
        cov += (1e-8 - min_eig) * np.eye(3)
        
    noise_cov = max(1e-12, 1.0 - phi**2) * cov
    noise_cov = 0.5 * (noise_cov + noise_cov.T)
    
    rows = []
    for b_idx, length in enumerate(beat_lengths):
        if length <= 0:
            continue
        X_beat = np.zeros((length, 3), dtype=np.float64)
        # Stationary initialization: X_{b, 1} ~ N(mu, Cov)
        X_beat[0] = rng.multivariate_normal(mean, cov)
        noise = rng.multivariate_normal(np.zeros(3), noise_cov, size=length)
        
        for t in range(1, length):
            X_beat[t] = mean + phi * (X_beat[t - 1] - mean) + noise[t]
            
        for t in range(length):
            rows.append({
                "patient_id": f"sim_pat_{b_idx // 4:03d}",
                "beat_id": b_idx,
                "original_time": float(b_idx * 0.8 + t * 0.02),
                "normalized_phase": float(t / max(length, 1)),
                "s": float(X_beat[t, 0]),
                "rho": float(X_beat[t, 1]),
                "kappa": float(X_beat[t, 2]),
                "domain_id": 0,
            })
            
    return pd.DataFrame(rows)


def compute_lag1_autocorrelation(
    df: pd.DataFrame,
    col: str = "s",
) -> float:
    """Compute pooled within-beat lag-1 autocorrelation (panel ACF1).
    
    Centers each beat by its mean to measure within-beat temporal dependence
    without cross-sectional beat-baseline inflation.
    """
    x_prev = []
    x_curr = []
    
    sort_cols = [c for c in ["original_time", "normalized_phase"] if c in df.columns]
    
    for _, beat_group in df.groupby(["patient_id", "beat_id"]):
        if sort_cols:
            beat_sorted = beat_group.sort_values(by=sort_cols)
        else:
            beat_sorted = beat_group
        vals = beat_sorted[col].to_numpy(dtype=np.float64)
        if len(vals) >= 2:
            vals_centered = vals - np.mean(vals)
            x_prev.extend(vals_centered[:-1])
            x_curr.extend(vals_centered[1:])
            
    if len(x_prev) < 2:
        return 0.0
        
    v1 = np.asarray(x_prev, dtype=np.float64)
    v2 = np.asarray(x_curr, dtype=np.float64)
    
    std1 = np.std(v1)
    std2 = np.std(v2)
    if std1 < 1e-12 or std2 < 1e-12:
        return 0.0
        
    corr = np.corrcoef(v1, v2)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def compute_temporal_dependence_summary(df: pd.DataFrame) -> dict[str, float]:
    """Compute lag-1 autocorrelation for s, rho, kappa."""
    return {
        "acf1_s": compute_lag1_autocorrelation(df, "s"),
        "acf1_rho": compute_lag1_autocorrelation(df, "rho"),
        "acf1_kappa": compute_lag1_autocorrelation(df, "kappa"),
    }


# ==============================================================================
# 3. Statistical Summaries & Verification Gates
# ==============================================================================

def compute_wilson_interval(
    k: int,
    n: int,
    conf: float = 0.95,
) -> tuple[float, float]:
    """Calculate Wilson score interval for binomial proportion."""
    if n <= 0:
        return 0.0, 1.0
    p_hat = k / n
    z = scipy.stats.norm.ppf(1.0 - (1.0 - conf) / 2.0)
    z2 = z ** 2
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n ** 2))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return float(low), float(high)


def compute_bootstrap_fdr_ci(
    fdp_values: np.ndarray,
    n_boot: int = 2000,
    conf: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean FDR and percentile bootstrap confidence interval across family replicates.
    
    Returns:
        mean_fdr: Empirical mean of replicate FDPs.
        low_ci: Lower bound of bootstrap interval.
        high_ci: Upper bound of bootstrap interval (one-sided gate comparison: high_ci <= 0.075).
    """
    fdp_arr = np.asarray(fdp_values, dtype=np.float64)
    R = len(fdp_arr)
    if R == 0:
        return 0.0, 0.0, 1.0
    mean_fdr = float(np.mean(fdp_arr))
    
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(fdp_arr, size=R, replace=True)
        boot_means[b] = np.mean(sample)
        
    alpha_tail = (1.0 - conf) / 2.0
    low_ci = float(np.percentile(boot_means, 100.0 * alpha_tail))
    high_ci = float(np.percentile(boot_means, 100.0 * (1.0 - alpha_tail)))
    return mean_fdr, low_ci, high_ci


def summarize_pairwise_calibration(
    p_values: np.ndarray,
    alpha_thresholds: tuple[float, ...] = (0.01, 0.025, 0.05, 0.10),
) -> dict[str, Any]:
    """Summarize empirical rejection rates and evaluate calibration gate."""
    p_values = np.asarray(p_values, dtype=np.float64)
    n = len(p_values)
    if n == 0:
        return {"n_tests": 0, "verdict": "NO_TESTS"}
        
    rates = {}
    for a in alpha_thresholds:
        k = int(np.sum(p_values <= a))
        rates[f"rejection_rate_{str(a).replace('.', '')}"] = float(k / n)
        
    k_050 = int(np.sum(p_values <= 0.05))
    low_050, high_050 = compute_wilson_interval(k_050, n, conf=0.95)
    rate_050 = rates.get("rejection_rate_005", float(k_050 / n))
    
    # Calibration Gate: Upper 95% CI <= 0.075
    if high_050 <= 0.075:
        if rate_050 < 0.02:
            verdict = "PASS_WITH_CONSERVATISM"
        else:
            verdict = "PASS"
    else:
        verdict = "FAIL"
        
    return {
        "n_tests": n,
        **rates,
        "wilson95_050": [low_050, high_050],
        "verdict": verdict,
    }


def compute_fdp(
    p_values: np.ndarray,
    ground_truth_null: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, int, int]:
    """Compute false discovery proportion: FDP = V / max(R, 1)."""
    p_values = np.asarray(p_values, dtype=np.float64)
    ground_truth_null = np.asarray(ground_truth_null, dtype=bool)
    
    rejections = p_values <= alpha
    R = int(np.sum(rejections))
    V = int(np.sum(rejections & ground_truth_null))
    
    fdp = float(V / max(R, 1)) if R > 0 else 0.0
    return fdp, R, V


# ==============================================================================
# 4. Pair Test Evaluation Runner
# ==============================================================================

def evaluate_pair_test(
    df_A: pd.DataFrame,
    df_B: pd.DataFrame,
    n_perm: int = 999,
    seed: int = 42,
    cahc_neighborhood_size: int = 10,
    kernel: str = "IMQ",
    kernel_param: float = 1.0,
) -> tuple[float, float, int, int, int]:
    """Run frozen repSpat block-permutation test between two pseudo-domains.
    
    Returns:
        obs_mmd2: Observed MMD^2 statistic.
        p_val: Permutation p-value.
        n_exceed: Count of null statistics exceeding observed.
        n_blocks_A: Number of K-means blocks in domain A.
        n_blocks_B: Number of K-means blocks in domain B.
    """
    xi_A = df_A[["s", "rho", "kappa"]].to_numpy(dtype=np.float64)
    xi_B = df_B[["s", "rho", "kappa"]].to_numpy(dtype=np.float64)
    
    n_blocks_A = compute_n_blocks(len(xi_A), cahc_neighborhood_size=cahc_neighborhood_size)
    n_blocks_B = compute_n_blocks(len(xi_B), cahc_neighborhood_size=cahc_neighborhood_size)
    
    blocks_A = build_attribute_blocks(xi_A, n_blocks=n_blocks_A, random_state=seed)
    blocks_B = build_attribute_blocks(xi_B, n_blocks=n_blocks_B, random_state=seed + 1)
    
    S, block_sizes, self_diag = precompute_block_kernel_sums(
        blocks_A, blocks_B, kernel=kernel, kernel_param=kernel_param
    )
    
    rng = np.random.RandomState(seed + 2)
    obs_mmd2, p_val, _, _ = run_block_permutation_test(
        S, block_sizes, len(blocks_A), len(blocks_B), B_max=n_perm, rng=rng, self_diagonal=self_diag
    )
    n_exceed = int(round(p_val * (n_perm + 1) - 1))
    return float(obs_mmd2), float(p_val), n_exceed, len(blocks_A), len(blocks_B)
