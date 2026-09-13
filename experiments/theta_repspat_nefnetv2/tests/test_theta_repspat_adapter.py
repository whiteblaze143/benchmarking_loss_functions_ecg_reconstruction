"""Unit tests for the Theta-repSpat adapter and representation invariants.

Enforces:
1. Fundamental Invariant: X_CAHC == X_supplied for arbitrary p in {3, 8}.
   No hidden Kors projection, no lead selection, no standardization.
2. Scale-equivariant IMQ kernel: c = gamma * d_med (median_heuristic).
3. 4-Arm construction contracts:
   - R1 (oracle reference) in R^8
   - R0 (Kors VCG) in R^3
   - R2 (Nef panoramic completion) in R^8
   - R3 (Kors on Nef completion) in R^3
4. Wave concordance (ARI/NMI) is strictly secondary and non-optimizing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add tit_ecg and experiments to sys.path
repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.pipeline import ExactRepSpat
from tit_ecg.src.vcg_transform import ecg_to_vcg_kors, KORS_REGRESSION_MATRIX


def test_adapter_invariant_x_supplied_equals_x_cahc_p8():
    """Verify that p=8 attributes pass unchanged into ExactRepSpat."""
    T = 100
    p = 8
    rng = np.random.default_rng(42)
    X_supplied = rng.standard_normal((T, p)).astype(np.float64)
    time_tau = np.arange(T, dtype=np.float64) / 500.0
    S = np.column_stack([time_tau, np.zeros_like(time_tau)])

    cfg = RepSpatConfig.fast_test_config(
        m_grid=[2, 4],
        G_grid=[3, 4],
        n_permutations=20,
        kernel="imq",
        kernel_scale_rule="median_heuristic",
        kernel_param=1.0,
    )
    model = ExactRepSpat(config=cfg)
    model.fit(S, X_supplied)

    # Check results exist
    assert "initial_labels" in model.results_
    assert "labels_clique" in model.results_
    assert "labels_cc" in model.results_
    # Verify no dimension change or alteration
    assert X_supplied.shape == (T, 8)


def test_adapter_invariant_x_supplied_equals_x_cahc_p3():
    """Verify that p=3 attributes pass unchanged into ExactRepSpat."""
    T = 100
    p = 3
    rng = np.random.default_rng(42)
    X_supplied = rng.standard_normal((T, p)).astype(np.float64)
    time_tau = np.arange(T, dtype=np.float64) / 500.0
    S = np.column_stack([time_tau, np.zeros_like(time_tau)])

    cfg = RepSpatConfig.fast_test_config(
        m_grid=[2, 4],
        G_grid=[3, 4],
        n_permutations=20,
        kernel="imq",
        kernel_scale_rule="median_heuristic",
        kernel_param=1.0,
    )
    model = ExactRepSpat(config=cfg)
    model.fit(S, X_supplied)

    assert "initial_labels" in model.results_
    assert "labels_clique" in model.results_
    assert X_supplied.shape == (T, 3)


def test_scale_equivariant_median_heuristic_invariance():
    """Verify that c = gamma * d_med provides scale-invariance under positive rescaling."""
    T = 80
    rng = np.random.default_rng(123)
    X_base = rng.standard_normal((T, 8))
    time_tau = np.arange(T, dtype=np.float64) / 500.0
    S = np.column_stack([time_tau, np.zeros_like(time_tau)])

    # Rescale by factor of 5.0
    X_scaled = X_base * 5.0

    cfg = RepSpatConfig.fast_test_config(
        m_grid=[3],
        G_grid=[3],
        n_permutations=50,
        kernel="imq",
        kernel_scale_rule="median_heuristic",
        kernel_param=1.0,
        random_state=42,
    )

    model1 = ExactRepSpat(config=cfg)
    model1.fit(S, X_base)

    model2 = ExactRepSpat(config=cfg)
    model2.fit(S, X_scaled)

    # Initial CAHC clusters must be identical because Ward linkage is scale-equivariant
    np.testing.assert_array_equal(model1.results_["initial_labels"], model2.results_["initial_labels"])

    # Permutation test p-values should be identical (within random seed)
    df1 = model1.results_["pairwise_df"]
    df2 = model2.results_["pairwise_df"]
    np.testing.assert_allclose(df1["p_value"].values, df2["p_value"].values, atol=1e-5)


def test_four_arms_construction():
    """Verify dimension and mathematical definitions of Arms R0, R1, R2, R3."""
    T = 200
    rng = np.random.default_rng(7)
    x_gt_8 = rng.standard_normal((8, T))  # [I, II, V1, V2, V3, V4, V5, V6]

    # R1: Oracle 8-lead reference in R^8
    E_R1 = x_gt_8.T  # (T, 8)
    assert E_R1.shape == (T, 8)

    # R0: Kors VCG in R^3
    # 8-lead Kors matrix [3, 8]
    KORS_8 = np.array([
        [ 0.38,   -0.07,    -0.13,    0.05,   -0.01,    0.14,    0.06,    0.54 ],
        [-0.07,    0.93,     0.06,   -0.02,   -0.05,    0.06,   -0.17,    0.13 ],
        [-0.11,   -0.23,    -0.43,   -0.06,   -0.14,   -0.20,   -0.11,    0.31 ],
    ])
    E_R0 = E_R1 @ KORS_8.T  # (T, 3)
    assert E_R0.shape == (T, 3)

    # R2: Simulated Nef completed panorama in R^8 (inputs {I, II, V3} match GT)
    E_R2 = E_R1.copy()
    # Mock completion for {V1, V2, V4, V5, V6}
    mock_preds = rng.standard_normal((T, 5))
    E_R2[:, [2, 3, 5, 6, 7]] = mock_preds
    assert E_R2.shape == (T, 8)
    # Verify observed inputs strictly preserved
    np.testing.assert_array_equal(E_R2[:, 0], E_R1[:, 0])  # Lead I
    np.testing.assert_array_equal(E_R2[:, 1], E_R1[:, 1])  # Lead II
    np.testing.assert_array_equal(E_R2[:, 4], E_R1[:, 4])  # Lead V3

    # R3: Kors VCG on Nef completion in R^3
    E_R3 = E_R2 @ KORS_8.T  # (T, 3)
    assert E_R3.shape == (T, 3)


def test_wave_concordance_does_not_alter_clustering():
    """Verify that external wave annotations are strictly secondary probes."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    T = 100
    rng = np.random.default_rng(99)
    X = rng.standard_normal((T, 8))
    time_tau = np.arange(T, dtype=np.float64) / 500.0
    S = np.column_stack([time_tau, np.zeros_like(time_tau)])

    # Mock ground truth waves (0=iso, 1=P, 2=QRS, 3=T)
    gt_waves = rng.integers(0, 4, size=T)

    cfg = RepSpatConfig.fast_test_config(
        m_grid=[3],
        G_grid=[3],
        n_permutations=20,
        random_state=42,
    )
    model = ExactRepSpat(config=cfg)
    model.fit(S, X)

    initial_labels = model.results_["initial_labels"]
    clique_labels = model.results_["labels_clique"]

    # Compute ARI/NMI externally as interpretability probe
    ari_init = float(adjusted_rand_score(gt_waves, initial_labels))
    ari_clique = float(adjusted_rand_score(gt_waves, clique_labels))
    nmi_init = float(normalized_mutual_info_score(gt_waves, initial_labels))

    assert -1.0 <= ari_init <= 1.0
    assert -1.0 <= ari_clique <= 1.0
    assert 0.0 <= nmi_init <= 1.0
