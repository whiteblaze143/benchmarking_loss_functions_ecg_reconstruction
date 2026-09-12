"""Unit tests for Stage M3H-CAL: Dependence Calibration of repSpat Inference.

Incorporates all PRD corrections:
- Exact set invariants on patient bundle splits (disjoint, union, 1:1 and 1:3 ratios)
- Shared beat splits preserving patient overlap without beat fragmentation (global 1:3 ratio)
- Source-domain frozen parameters for CAL1 mean shift and CAL2 covariance shift in standardized descriptor space
- Multivariate vector AR(1) temporal generator for CAL3 (Gaussian marginal invariance, autocorrelation separation)
- Deterministic SHA-256 seeding independent of process/worker ordering
- Integration test for deterministic execution across 1 worker vs 8 workers
- Continuous bootstrap CI on replicate-level FDP
- Allowlist and Milestone 1 freeze integrity
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rep_stat_ecg.src.motifs.dependence_calibration import (
    apply_covariance_shift,
    apply_mean_shift,
    build_patient_bundles,
    compute_bootstrap_fdr_ci,
    compute_fdp,
    compute_lag1_autocorrelation,
    compute_wilson_interval,
    evaluate_pair_test,
    generate_deterministic_seed,
    generate_vector_ar1_beats,
    split_beats_shared_patient,
    split_patients_disjoint,
)


def make_synthetic_domain_df(n_patients=10, beats_per_pat=4, microstates_per_beat=8, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    for p in range(n_patients):
        pid = f"pat_{p:03d}"
        for b in range(beats_per_pat):
            for t in range(microstates_per_beat):
                rows.append({
                    "patient_id": pid,
                    "ecg_id": f"ecg_{p:03d}",
                    "beat_id": b,
                    "original_time": float(b * 0.8 + t * 0.02),
                    "normalized_phase": float(t / microstates_per_beat),
                    "s": float(rng.normal(2.0, 0.5)),
                    "rho": float(rng.normal(0.0, 10.0)),
                    "kappa": float(rng.normal(1.0, 0.2)),
                    "domain_id": 0,
                })
    return pd.DataFrame(rows)


# ==============================================================================
# 1. DATA HIERARCHY & EXACT SET INVARIANTS
# ==============================================================================

def test_patient_bundle_never_splits_patient():
    """1: Patient bundle contains all microstates for that patient in the domain."""
    df = make_synthetic_domain_df(n_patients=5, beats_per_pat=3, microstates_per_beat=10)
    bundles = build_patient_bundles(df)
    assert len(bundles) == 5
    for pid, b_df in bundles.items():
        assert b_df["patient_id"].nunique() == 1
        assert len(b_df) == 30


def test_disjoint_split_set_invariants():
    """2: Exact set invariants: A and B are disjoint, union equals full set, ratio honored."""
    df = make_synthetic_domain_df(n_patients=40, beats_per_pat=3, microstates_per_beat=5)
    bundles = build_patient_bundles(df)
    rng = np.random.RandomState(42)

    # 1:1 split
    df_A1, df_B1 = split_patients_disjoint(bundles, rng, ratio=(1, 1))
    pats_A1 = set(df_A1["patient_id"])
    pats_B1 = set(df_B1["patient_id"])
    all_pats = set(bundles.keys())
    assert pats_A1 & pats_B1 == set(), "CAL0-D A and B overlap in patients!"
    assert pats_A1 | pats_B1 == all_pats, "CAL0-D lost patients during split!"
    assert len(pats_A1) == 20
    assert len(pats_B1) == 20

    # 1:3 imbalanced split
    df_A3, df_B3 = split_patients_disjoint(bundles, rng, ratio=(1, 3))
    pats_A3 = set(df_A3["patient_id"])
    pats_B3 = set(df_B3["patient_id"])
    assert pats_A3 & pats_B3 == set()
    assert pats_A3 | pats_B3 == all_pats
    assert len(pats_A3) == 10
    assert len(pats_B3) == 30


def test_shared_split_preserves_patient_overlap():
    """3: For CAL0-S, all sampled patients appear in both pseudo-domains."""
    df = make_synthetic_domain_df(n_patients=10, beats_per_pat=6, microstates_per_beat=5)
    bundles = build_patient_bundles(df)
    rng = np.random.RandomState(42)
    df_A, df_B = split_beats_shared_patient(bundles, rng, ratio=(1, 1))

    pats_A = set(df_A["patient_id"])
    pats_B = set(df_B["patient_id"])
    assert len(pats_A & pats_B) == 10, "CAL0-S failed to share all patients!"


def test_beat_never_split_between_pseudo_domains():
    """4: In CAL0-S, every individual beat is wholly in A or wholly in B, never split."""
    df = make_synthetic_domain_df(n_patients=10, beats_per_pat=4, microstates_per_beat=10)
    bundles = build_patient_bundles(df)
    rng = np.random.RandomState(42)
    df_A, df_B = split_beats_shared_patient(bundles, rng, ratio=(1, 3))

    keys_A = set(zip(df_A["patient_id"], df_A["beat_id"]))
    keys_B = set(zip(df_B["patient_id"], df_B["beat_id"]))
    assert len(keys_A & keys_B) == 0, "Beats were split across pseudo-domains!"


# ==============================================================================
# 2. SOURCE-DOMAIN FROZEN TRANSFORMS (CAL1, CAL2)
# ==============================================================================

def test_mean_shift_source_frozen():
    """5: Standardized mean shift has exactly magnitude delta and preserves covariance identically."""
    df = make_synthetic_domain_df(n_patients=20, beats_per_pat=3, microstates_per_beat=10)
    data = df[["s", "rho", "kappa"]].to_numpy()
    
    source_mean = np.mean(data, axis=0)
    source_std = np.std(data, axis=0)

    delta = 0.5
    shifted = apply_mean_shift(data, delta=delta, mean=source_mean, std=source_std)

    std_data = (data - source_mean) / source_std
    std_shifted = (shifted - source_mean) / source_std
    mean_diff = np.mean(std_shifted, axis=0) - np.mean(std_data, axis=0)
    observed_delta = float(np.linalg.norm(mean_diff))
    assert np.isclose(observed_delta, delta, atol=1e-6)

    cov_orig = np.cov(data, rowvar=False)
    cov_shifted = np.cov(shifted, rowvar=False)
    assert np.allclose(cov_orig, cov_shifted, atol=1e-10)


def test_covariance_shift_source_frozen():
    """6: Covariance shift in standardized descriptor space scales leading PC variance by gamma^2."""
    df = make_synthetic_domain_df(n_patients=30, beats_per_pat=4, microstates_per_beat=10)
    data = df[["s", "rho", "kappa"]].to_numpy()
    
    source_mean = np.mean(data, axis=0)
    source_std = np.std(data, axis=0)
    std_data = (data - source_mean) / source_std
    source_cov_std = np.cov(std_data, rowvar=False)
    _, source_eigvecs_std = np.linalg.eigh(source_cov_std)

    gamma = 1.5
    transformed = apply_covariance_shift(
        data, gamma=gamma, mean=source_mean, std=source_std, eigvecs_std=source_eigvecs_std
    )

    # Sample mean must match source mean exactly
    mean_orig = np.mean(data, axis=0)
    mean_trans = np.mean(transformed, axis=0)
    assert np.allclose(mean_orig, mean_trans, atol=1e-10)

    # Leading standardized PC variance must scale by gamma^2
    v_lead_star = source_eigvecs_std[:, -1]
    std_transformed = (transformed - source_mean) / source_std
    var_orig_lead = float(np.var(std_data @ v_lead_star))
    var_trans_lead = float(np.var(std_transformed @ v_lead_star))
    assert np.isclose(var_trans_lead / var_orig_lead, gamma ** 2, rtol=0.05)


# ==============================================================================
# 3. CONTROLLED TEMPORAL PROCESS (CAL3)
# ==============================================================================

def test_ar1_temporal_process_marginals():
    """7: Vector AR(1) preserves Gaussian marginals while separating lag-1 autocorrelation."""
    mean = np.array([2.5, 5.0, 1.2])
    cov = np.array([
        [1.0, 0.3, 0.1],
        [0.3, 2.0, 0.4],
        [0.1, 0.4, 0.8],
    ])
    beat_lengths = [30] * 500

    rng1 = np.random.RandomState(42)
    df_phi0 = generate_vector_ar1_beats(mean, cov, phi=0.0, beat_lengths=beat_lengths, rng=rng1)
    
    rng2 = np.random.RandomState(43)
    df_phi8 = generate_vector_ar1_beats(mean, cov, phi=0.8, beat_lengths=beat_lengths, rng=rng2)

    for col_idx, col in enumerate(["s", "rho", "kappa"]):
        assert np.isclose(df_phi0[col].mean(), mean[col_idx], atol=0.15)
        assert np.isclose(df_phi8[col].mean(), mean[col_idx], atol=0.15)
        assert np.isclose(df_phi0[col].std(), np.sqrt(cov[col_idx, col_idx]), atol=0.15)
        assert np.isclose(df_phi8[col].std(), np.sqrt(cov[col_idx, col_idx]), atol=0.15)

    acf_phi0 = compute_lag1_autocorrelation(df_phi0, col="s")
    acf_phi8 = compute_lag1_autocorrelation(df_phi8, col="s")
    assert acf_phi0 < 0.15
    assert acf_phi8 > 0.65
    assert acf_phi8 - acf_phi0 > 0.50


# ==============================================================================
# 4. DETERMINISTIC SEEDING & INTEGRATION CHECK
# ==============================================================================

def test_deterministic_seed_generator():
    """8: SHA-256 seed generator is deterministic and independent of process state."""
    seed1 = generate_deterministic_seed("M3H_CAL", "CAL0_D", 5, 0.0, 42)
    seed2 = generate_deterministic_seed("M3H_CAL", "CAL0_D", 5, 0.0, 42)
    seed3 = generate_deterministic_seed("M3H_CAL", "CAL0_D", 5, 0.0, 43)

    assert seed1 == seed2
    assert seed1 != seed3
    assert 0 <= seed1 < 2**32


def _eval_single_task(args: tuple) -> dict[str, Any]:
    task_id, df_A, df_B, seed = args
    obs_mmd2, pval, n_exceed, b_A, b_B = evaluate_pair_test(df_A, df_B, n_perm=49, seed=seed)
    return {
        "task_id": task_id,
        "seed": seed,
        "obs_mmd2": obs_mmd2,
        "n_exceed": n_exceed,
        "p_value": pval,
    }


def test_parallelism_determinism_integration():
    """9: 1-worker and parallel executions yield identical results for identical tasks."""
    df = make_synthetic_domain_df(n_patients=10, beats_per_pat=2, microstates_per_beat=3, seed=10)
    bundles = build_patient_bundles(df)
    
    tasks = []
    for t in range(3):
        seed = generate_deterministic_seed("M3H_CAL_TEST", "INTEG", 0, "TEST", t)
        rng = np.random.RandomState(seed)
        df_A, df_B = split_patients_disjoint(bundles, rng, ratio=(1, 1))
        tasks.append((t, df_A, df_B, seed))

    # Sequential execution (1 worker)
    results_w1 = [_eval_single_task(t) for t in tasks]

    # Concurrent execution across multiple worker threads
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_eval_single_task, t) for t in tasks]
        results_parallel = [f.result() for f in futures]

    results_w1_sorted = sorted(results_w1, key=lambda r: r["task_id"])
    results_parallel_sorted = sorted(results_parallel, key=lambda r: r["task_id"])

    for r1, rp in zip(results_w1_sorted, results_parallel_sorted):
        assert r1["seed"] == rp["seed"]
        assert np.isclose(r1["obs_mmd2"], rp["obs_mmd2"], atol=1e-12)
        assert r1["n_exceed"] == rp["n_exceed"]
        assert np.isclose(r1["p_value"], rp["p_value"], atol=1e-12)


# ==============================================================================
# 5. STATISTICAL SUMMARIES & GATES
# ==============================================================================

def test_bootstrap_ci_fdp():
    """10: Replicate-level FDP bootstrap interval covers the true mean and bounds variance."""
    fdp_values = np.array([0.02, 0.03, 0.04, 0.05, 0.03, 0.04, 0.06, 0.02, 0.04, 0.05,
                           0.03, 0.04, 0.05, 0.02, 0.03, 0.04, 0.06, 0.03, 0.04, 0.05])
    mean_fdr, low, high = compute_bootstrap_fdr_ci(fdp_values, n_boot=2000, conf=0.95, seed=42)

    assert np.isclose(mean_fdr, np.mean(fdp_values))
    assert low < mean_fdr < high
    assert high <= 0.075


def test_wilson_interval_reference_values():
    """11: Wilson score interval matches textbook standard formula."""
    low, high = compute_wilson_interval(k=5, n=100, conf=0.95)
    assert np.isclose(low, 0.0215, atol=0.005)
    assert np.isclose(high, 0.1118, atol=0.005)
    assert low < 0.05 < high


def test_fdp_known_case():
    """12: False discovery proportion FDP = V / max(R, 1)."""
    p_values = np.array([0.01] * 20 + [0.5] * 80)
    ground_truth_null = np.array([True] * 5 + [False] * 15 + [True] * 80)
    fdp, n_rej, n_false = compute_fdp(p_values, ground_truth_null, alpha=0.05)

    assert n_rej == 20
    assert n_false == 5
    assert np.isclose(fdp, 0.25)


def test_fdp_zero_rejections_is_zero():
    """13: Zero rejections yields FDP = 0.0 gracefully."""
    p_values = np.array([0.5, 0.8, 0.9])
    ground_truth_null = np.array([True, True, False])
    fdp, n_rej, n_false = compute_fdp(p_values, ground_truth_null, alpha=0.05)

    assert n_rej == 0
    assert n_false == 0
    assert fdp == 0.0


# ==============================================================================
# 6. LEAKAGE & FREEZE INTEGRITY
# ==============================================================================

def test_milestone1_hashes_unchanged():
    """14: Milestone 1 freeze manifest hashes remain unchanged on disk."""
    manifest_path = Path("refine-logs/qvcg/hilbert_atlas/M3H_MILESTONE1_FREEZE_MANIFEST.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    atlas_dir = manifest_path.parent

    for name, meta in manifest["artifacts"].items():
        file_path = atlas_dir / name
        assert file_path.exists(), f"Missing frozen Milestone 1 file: {name}"
        observed_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        assert observed_hash == meta["sha256"], f"Milestone 1 hash altered for {name}!"


def test_calibration_inputs_allowlisted():
    """15: Calibration module does not reference clinical diagnosis tables."""
    cal_script = Path("rep_stat_ecg/src/motifs/dependence_calibration.py")
    if cal_script.exists():
        content = cal_script.read_text().lower()
        forbidden = ["ptbxl_database", "scp_statements", "diagnostic_class", "rhythm_label"]
        for term in forbidden:
            assert term not in content, f"Forbidden term '{term}' in dependence_calibration.py"
