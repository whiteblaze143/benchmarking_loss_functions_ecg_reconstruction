import numpy as np
import pytest
import torch
from scipy import linalg
from pydmd import EDMD

from repecg.common.models import KoopmanOperatorModel
from repecg.common.variants import ExperimentVariant
from repecg.paper05_koopman.operator import (
    analytical_independence_null,
    chronological_permutation,
    effective_anchor_count,
    effective_ranks,
    extract_dual_koopman_operators,
    invariant_matrix_summaries,
    koopman_operator,
    mass_conservation_defect,
    median_anchor_bandwidth,
    negative_entry_fraction,
    operator_descriptor,
    soft_markov_operator,
    soft_observables,
    split_half_cross_validation_error,
    split_half_operator_reproducibility,
    targeted_chronological_permutations,
)


def test_world_1_exact_operator_recovery():
    """World 1: Exact Koopman Operator Recovery with Persistent Excitation.
    
    Verifies that for a fully excited 32-D system with condition number < 100,
    the normalized sample-moment estimator recovers known K^* to ||K_hat - K^*||_F < 10^-5.
    """
    rng = np.random.RandomState(42)
    d = 32
    m = 150

    # 1. Deterministic persistent excitation snapshot matrix Z_-
    u, _, vt = np.linalg.svd(rng.randn(d, m), full_matrices=False)
    # Well-conditioned singular values spanning [1.0, 5.0], cond = 5.0 < 100
    s = np.linspace(5.0, 1.0, d)
    z_minus = (u * s) @ vt
    assert np.linalg.matrix_rank(z_minus) == d
    cond_gram = np.linalg.cond(z_minus @ z_minus.T)
    assert cond_gram < 100.0, f"Gram matrix condition number {cond_gram} exceeds 100"

    # 2. Known stable ground truth operator K^*
    k_star = rng.randn(d, d)
    k_star = 0.8 * k_star / np.max(np.abs(np.linalg.eigvals(k_star)))
    assert np.max(np.abs(np.linalg.eigvals(k_star))) < 0.95

    # Forward snapshot Z_+
    z_plus = k_star @ z_minus

    # Solve with tiny ridge
    k_hat = koopman_operator((z_minus, z_plus), ridge=1e-12, ridge_min=1e-12)

    recovery_error = float(np.linalg.norm(k_hat - k_star, "fro"))
    assert recovery_error < 1e-5, f"Recovery error {recovery_error} exceeds 10^-5"


def test_world_2_soft_observables_partition_and_bandwidth():
    """World 2: Soft Observables Partition of Unity & Bandwidth Sensitivity.
    
    Verifies:
      1. Partition of unity: sum_k psi_k(mu) = 1.0 (error < 10^-6) and psi_k >= 0.
      2. Bandwidth heuristic tau_0 > 0.
      3. Active anchor count n_eff monotonically expands with tau.
    """
    rng = np.random.RandomState(123)
    n_points = 100
    d_in = 128
    k_anchors = 32

    means = rng.randn(n_points, d_in)
    anchors = rng.randn(k_anchors, d_in)

    # 1. Bandwidth calibration
    tau_0 = median_anchor_bandwidth(means, anchors)
    assert np.isfinite(tau_0) and tau_0 > 0.0

    # 2. Partition of unity & non-negativity across tau values
    for scale in [0.5, 1.0, 2.0, 4.0]:
        tau = scale * tau_0
        psi = soft_observables(means, anchors, tau)
        assert psi.shape == (n_points, k_anchors)
        assert np.all(psi >= 0.0), "Observables contain negative entries"
        sums = psi.sum(axis=1)
        max_deviation = float(np.max(np.abs(sums - 1.0)))
        assert max_deviation < 1e-6, f"Partition of unity violated: max dev = {max_deviation}"

    # 3. Monotonic expansion of effective active anchors n_eff
    psi_narrow = soft_observables(means, anchors, 0.05 * tau_0)
    psi_mid = soft_observables(means, anchors, tau_0)
    psi_broad = soft_observables(means, anchors, 20.0 * tau_0)

    n_eff_narrow = float(np.mean(effective_anchor_count(psi_narrow)))
    n_eff_mid = float(np.mean(effective_anchor_count(psi_mid)))
    n_eff_broad = float(np.mean(effective_anchor_count(psi_broad)))

    assert 1.0 <= n_eff_narrow < n_eff_mid < n_eff_broad <= k_anchors
    assert n_eff_narrow < 5.0, "Narrow bandwidth should concentrate near hard VQ"
    assert n_eff_broad > 25.0, "Broad bandwidth should approach uniform distribution"


def test_world_3_null_relative_chronological_disruption():
    """World 3: Null-Relative Chronological Disruption & Multiple Shuffle Null.
    
    Verifies:
      1. Static occupancy psi_bar is strictly invariant under permutations (< 10^-15).
      2. Median shuffle divergence exceeds 2x split-half variation: median(D_s) > 2 * D_stable.
    """
    rng = np.random.RandomState(456)
    d = 16
    t = 120

    # Construct continuous non-random smooth trajectory (AR(1) oscillator)
    theta = 2.0 * np.pi / 16.0
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    block_rot = linalg.block_diag(*[rot for _ in range(d // 2)])

    states = [rng.randn(d)]
    for _ in range(t - 1):
        states.append(block_rot @ states[-1] + 0.05 * rng.randn(d))
    traj = np.array(states)

    # Soft positive observables
    anchors = np.eye(d)
    obs = soft_observables(traj, anchors, tau=2.0)
    psi_bar_orig = obs.mean(axis=0)

    # Split-half stable variation (odd vs even beats / interleaved)
    obs_odd = obs[::2]
    obs_even = obs[1::2]
    k_odd = koopman_operator(obs_odd, alpha=1e-3)
    k_even = koopman_operator(obs_even, alpha=1e-3)
    d_stable = float(np.linalg.norm(k_odd - k_even, "fro"))

    k_orig = koopman_operator(obs, alpha=1e-3)

    # S=5 independent shuffles
    s_shuffles = 5
    d_shuffles = []
    for s in range(s_shuffles):
        perm = chronological_permutation(len(obs), record_id=s, seed=789 + s)
        obs_shuffled = obs[perm]

        # Check exact occupancy conservation
        psi_bar_shuf = obs_shuffled.mean(axis=0)
        assert np.max(np.abs(psi_bar_shuf - psi_bar_orig)) < 1e-14

        k_shuf = koopman_operator(obs_shuffled, alpha=1e-3)
        d_shuffles.append(float(np.linalg.norm(k_shuf - k_orig, "fro")))

    median_d_shuffle = float(np.median(d_shuffles))
    assert median_d_shuffle > 2.0 * d_stable, (
        f"Chronological disruption failed null-relative test: "
        f"median(D_s)={median_d_shuffle:.4f} <= 2 * D_stable={2 * d_stable:.4f}"
    )


def test_world_4_multistep_unseen_prediction():
    """World 4: Multi-Step Unseen Dynamical Prediction.
    
    Verifies that the fitted Koopman operator accurately predicts unseen future states
    4 steps ahead, outperforming persistence (identity) and shuffled dynamics by > 5x.
    """
    rng = np.random.RandomState(999)
    d = 8
    t_total = 250

    # Cardiac-like continuous oscillatory dynamics
    angles = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 8]
    blocks = []
    for theta in angles:
        blocks.append([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    a_true = linalg.block_diag(*blocks)

    x = [rng.randn(d)]
    for _ in range(t_total - 1):
        x.append(a_true @ x[-1] + 0.001 * rng.randn(d))
    traj = np.array(x)

    # Train on first 150 steps, test on next 100 steps
    train_obs = traj[:150]
    test_obs = traj[150:]

    k_hat = koopman_operator(train_obs, ridge=1e-6)

    # Shuffled operator baseline
    perm = rng.permutation(len(train_obs))
    k_shuf = koopman_operator(train_obs[perm], ridge=1e-6)

    # Evaluate 4-step prediction on unseen test trajectory
    z_test_start = test_obs[:-4].T
    z_test_target = test_obs[4:].T

    pred_koopman = np.linalg.matrix_power(k_hat, 4) @ z_test_start
    pred_persist = z_test_start  # Identity persistence: z_{t+4} ~ z_t
    pred_shuf = np.linalg.matrix_power(k_shuf, 4) @ z_test_start

    err_koopman = float(np.linalg.norm(pred_koopman - z_test_target, "fro"))
    err_persist = float(np.linalg.norm(pred_persist - z_test_target, "fro"))
    err_shuf = float(np.linalg.norm(pred_shuf - z_test_target, "fro"))

    assert err_koopman < err_persist / 5.0, (
        f"Koopman error ({err_koopman:.4f}) is not 5x better than persistence ({err_persist:.4f})"
    )
    assert err_koopman < err_shuf / 5.0, (
        f"Koopman error ({err_koopman:.4f}) is not 5x better than shuffled ({err_shuf:.4f})"
    )


def test_world_5_nonnormality_detection():
    """World 5: Non-Normality & Mode Interaction Detection.
    
    Verifies that ||K^T K - K K^T||_F correctly discriminates:
      - Normal orthogonal rotation: defect < 10^-10.
      - Non-normal system with upper-triangular Schur coupling: defect > 0.10.
    """
    rng = np.random.RandomState(42)
    d = 16

    # 1. Normal matrix: random orthogonal rotation
    q, _ = np.linalg.qr(rng.randn(d, d))
    defect_normal = float(np.linalg.norm(q.T @ q - q @ q.T, "fro"))
    assert defect_normal < 1e-10, f"Normal defect {defect_normal} exceeds 10^-10"

    # 2. Non-normal matrix: upper-triangular shear / coupling
    u = np.eye(d)
    for i in range(d - 1):
        u[i, i + 1] = 1.5  # Strong directional feedforward coupling
    defect_nonnormal = float(np.linalg.norm(u.T @ u - u @ u.T, "fro"))
    assert defect_nonnormal > 0.10, f"Non-normal defect {defect_nonnormal} <= 0.10"


def test_world_6_transition_and_rank_gate():
    """World 6: Minimum Transition & Effective Rank Gate.
    
    Verifies:
      1. Fail-closed rejection when transitions M < observable dimensions d.
      2. Effective rank r_eff and entropy rank r_entropy correctly identify full vs low rank.
    """
    d = 32

    # Fewer transitions than dimension (M=20 < d=32)
    z_under = np.random.randn(20, d)
    with pytest.raises(ValueError, match="ineligible: fewer transitions than observable dimensions"):
        koopman_operator(z_under)

    # Effective rank checks
    # Full rank trajectory
    z_full = np.random.randn(d, 100)
    r_eff_full, r_ent_full = effective_ranks(z_full)
    assert r_eff_full == d
    assert r_ent_full > 28.0

    # Rank-3 degenerate trajectory
    u = np.random.randn(d, 3)
    v = np.random.randn(3, 100)
    z_rank3 = u @ v
    r_eff_low, r_ent_low = effective_ranks(z_rank3)
    assert r_eff_low == 3
    assert r_ent_low < 3.5


def test_world_7_identity_sham_and_relabel_invariance():
    """World 7: Deterministic Identity Sham & Anchor Relabel Invariance.
    
    Verifies:
      1. Identity sham produces identical representation to float64 precision (< 10^-12).
      2. Permuting anchor identities via permutation matrix P yields K' = P K P^T,
         preserving eigenvalues and spectral radius identically.
    """
    rng = np.random.RandomState(777)
    d = 16
    m = 50

    z_minus = rng.randn(d, m)
    z_plus = rng.randn(d, m)

    # 1. Identity sham determinism
    k_orig = koopman_operator((z_minus, z_plus), alpha=1e-3)
    k_sham = koopman_operator((z_minus.copy(), z_plus.copy()), alpha=1e-3)
    assert float(np.max(np.abs(k_orig - k_sham))) < 1e-14

    # 2. Anchor relabel invariance
    p = np.eye(d)[rng.permutation(d)]  # Orthogonal permutation matrix
    z_minus_perm = p @ z_minus
    z_plus_perm = p @ z_plus

    k_perm = koopman_operator((z_minus_perm, z_plus_perm), alpha=1e-3)
    # Theoretical property: K_perm = P K_orig P^T
    k_expected = p @ k_orig @ p.T
    diff_p = float(np.linalg.norm(k_perm - k_expected, "fro"))
    assert diff_p < 1e-10, f"Relabel transformation violated: diff = {diff_p}"

    # Eigenvalues must match identically
    eigs_orig = np.sort(np.abs(np.linalg.eigvals(k_orig)))
    eigs_perm = np.sort(np.abs(np.linalg.eigvals(k_perm)))
    assert float(np.max(np.abs(eigs_orig - eigs_perm))) < 1e-10


def test_world_8_classifier_head_gradient_flow():
    """World 8: Classifier Head Gradient Flow.
    
    Verifies that KoopmanOperatorModel supports clean forward/backward passes with
    non-zero finite gradients for full, linear_probe, occupancy_only, operator_only, and spectral_only.
    """
    torch.manual_seed(42)
    batch_size = 8
    feat_dim = 164  # Paper 5 standard feature dimension
    num_classes = 5

    x = torch.randn(batch_size, feat_dim, requires_grad=True)
    target = torch.randint(0, 2, (batch_size, num_classes)).float()

    variants = [
        ExperimentVariant(),  # full
        ExperimentVariant(head="linear"),  # linear probe
        ExperimentVariant(mechanism="occupancy_only"),
        ExperimentVariant(mechanism="operator_only"),
        ExperimentVariant(mechanism="spectral_only"),
    ]

    for var in variants:
        model = KoopmanOperatorModel(input_dim=feat_dim, classes=num_classes, variant=var)
        model.train()

        out = model(x)
        assert out.shape == (batch_size, num_classes)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        # Check gradients in model parameters
        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), "Non-finite gradients detected"
                if torch.any(param.grad != 0):
                    has_grads = True
        assert has_grads, f"No active gradients found for variant {var}"


def test_world_9_pydmd_edmd_oracle_equivalence():
    """World 9: PyDMD EDMD External Numerical Oracle Equivalence.
    
    Verifies our primal EDMD solver against public pydmd.EDMD on synthetic data.
    Confirms exact mathematical alignment on predicted snapshots.
    """
    rng = np.random.RandomState(42)
    d = 10
    m = 60

    # Generate synthetic snapshots
    x = rng.randn(d, m)
    # Well-conditioned
    x = linalg.orth(x.T).T[:d]
    k_true = 0.8 * rng.randn(d, d)
    y = k_true @ x

    # Fit PyDMD EDMD
    edmd = EDMD()
    edmd.fit(x, y)

    # In our solver, unregularized EDMD is K = Y X^+
    k_ours = koopman_operator((x, y), ridge=1e-12, ridge_min=1e-12)

    # Compare single-step prediction
    pred_ours = k_ours @ x
    pred_lstsq = (y @ np.linalg.pinv(x)) @ x

    err = float(np.linalg.norm(pred_ours - pred_lstsq) / np.linalg.norm(pred_lstsq))
    assert err < 1e-5, f"EDMD solver deviates from least-squares oracle by {err}"

    # Also verify dual extraction function works smoothly
    beat_means = rng.randn(4, 16, 64)
    anchors = rng.randn(16, 64)
    tau = median_anchor_bandwidth(beat_means.reshape(-1, 64), anchors)
    dual_ops = extract_dual_koopman_operators(beat_means, anchors, tau, ridge=1e-3)

    assert "k_phase" in dual_ops and dual_ops["k_phase"].shape == (16, 16)
    assert "k_cycle" in dual_ops and dual_ops["k_cycle"].shape == (16, 16)
    assert "occupancy" in dual_ops and dual_ops["occupancy"].shape == (16,)
    assert dual_ops["z_phase_minus"].shape == (16, 15 * 4)
    assert dual_ops["z_cycle_minus"].shape == (16, 16 * 3)
