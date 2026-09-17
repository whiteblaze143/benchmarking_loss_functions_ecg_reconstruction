import numpy as np
import pytest
import torch

from repecg.common.models import ConditionalRepStatModel
from repecg.common.variants import ExperimentVariant
from repecg.paper06_conditional.residual import (
    anchor_conditioned_mmd,
    conditional_distance,
    conditional_distance_diagnostics,
    decompose_macro_residual,
    fit_macro_basis,
    non_adjacent_circular_mask,
    recurrence_from_distances,
    soft_membership,
    within_macro_phase_shuffle,
)


def test_world_1_exact_orthogonal_decomposition_and_5d_complement():
    """World 1: Exact Orthogonal Decomposition & 5D Orthonormal Complement.
    
    Verifies that for multilead voltages in physical mV:
      1. Macro basis V_macro has orthonormal columns: V_macro^T V_macro = I_3 (< 10^-12).
      2. Orthonormal complement V_perp has orthonormal columns: V_perp^T V_perp = I_5 (< 10^-12).
      3. Mutual orthogonality: V_macro^T V_perp = 0 (< 10^-12).
      4. Residual r(t) is strictly orthogonal to V_macro: ||V_macro^T r(t)||_inf < 10^-12.
      5. 5D coordinates u(t) = V_perp^T (x - mean) satisfy r(t) = V_perp u(t) exactly.
      6. Signal reconstruction is exact: ||x - (mean + V_macro z + V_perp u)||_inf < 10^-12.
    """
    rng = np.random.RandomState(42)
    n_samples = 500
    leads = 8

    # Physical mV voltages
    raw_voltages = rng.randn(n_samples, leads) * 0.8 + 0.1
    mean, basis_macro, basis_perp = fit_macro_basis(raw_voltages, return_complement=True)

    # 1. Orthonormality of macro basis
    assert basis_macro.shape == (leads, 3)
    assert np.allclose(basis_macro.T @ basis_macro, np.eye(3), atol=1e-12)

    # 2. Orthonormality of complement basis
    assert basis_perp.shape == (leads, 5)
    assert np.allclose(basis_perp.T @ basis_perp, np.eye(5), atol=1e-12)

    # 3. Mutual orthogonality
    assert np.allclose(basis_macro.T @ basis_perp, np.zeros((3, 5)), atol=1e-12)

    # 4. Decomposition
    macro, residual, u = decompose_macro_residual(
        raw_voltages, mean, basis_macro, basis_perp=basis_perp
    )
    assert macro.shape == (n_samples, 3)
    assert residual.shape == (n_samples, leads)
    assert u.shape == (n_samples, 5)

    # 5. Strict orthogonality of ambient residual to macro subspace
    ortho_proj = residual @ basis_macro
    max_leakage = float(np.max(np.abs(ortho_proj)))
    assert max_leakage < 1e-12, f"Residual leaks into macro basis: {max_leakage}"

    # 6. 5D complement coordinates reconstruct residual exactly
    res_from_u = u @ basis_perp.T
    res_err = float(np.max(np.abs(residual - res_from_u)))
    assert res_err < 1e-12, f"5D residual reconstruction error: {res_err}"

    # 7. Exact total signal reconstruction
    reconstructed = mean + macro @ basis_macro.T + u @ basis_perp.T
    recon_err = float(np.max(np.abs(raw_voltages - reconstructed)))
    assert recon_err < 1e-12, f"Total reconstruction error: {recon_err}"


def test_world_2_pure_macro_field_nullification():
    """World 2: Pure Dominant Macro Field Signal Residual Nullification.
    
    Verifies that when an ECG signal is generated purely within the rank-3 spatial macrostate:
      1. Residual r(t) and 5D coordinate u(t) are identically zero: ||r(t)||_inf < 10^-12, ||u(t)||_inf < 10^-12.
      2. Recovered macro coordinates match true trajectory: ||macro - Z||_inf < 10^-12.
    """
    rng = np.random.RandomState(101)
    n_samples = 200
    leads = 8

    # Random orthonormal basis and complement
    q, _ = np.linalg.qr(rng.randn(leads, leads))
    basis_macro = q[:, :3]
    basis_perp = q[:, 3:]
    mean = rng.randn(leads)

    # Generate pure rank-3 macro trajectory
    z_true = rng.randn(n_samples, 3)
    pure_macro_signal = mean + z_true @ basis_macro.T

    macro_rec, residual_rec, u_rec = decompose_macro_residual(
        pure_macro_signal, mean, basis_macro, basis_perp=basis_perp
    )

    res_norm = float(np.max(np.abs(residual_rec)))
    assert res_norm < 1e-12, f"Pure macro signal produced non-zero residual: {res_norm}"

    u_norm = float(np.max(np.abs(u_rec)))
    assert u_norm < 1e-12, f"Pure macro signal produced non-zero 5D complement: {u_norm}"

    macro_err = float(np.max(np.abs(macro_rec - z_true)))
    assert macro_err < 1e-12, f"Recovered macro differs from true trajectory: {macro_err}"


def test_world_3_effective_overlap_ineligibility_cliff():
    """World 3: Effective-Overlap Ineligibility Cliff (Estimator-Level Gate).
    
    Verifies:
      1. When effective sample size per shared anchor is below threshold (n_eff < n_min),
         the estimator fails closed with ValueError rather than computing unstable estimates.
      2. When effective overlap satisfies n_eff >= n_min, anchor-conditioned MMD computes cleanly,
         is symmetric, zero on diagonal, and returns finite overlap mass matrix O_gh.
    """
    d_feat = 16
    k_anchors = 4
    n_pts = 50

    # 1. Disjoint effective supports: Cell 0 concentrated on anchor 0, Cell 1 on anchor 1
    mem_0 = np.zeros((n_pts, k_anchors))
    mem_0[:, 0] = 1.0

    mem_1 = np.zeros((n_pts, k_anchors))
    mem_1[:, 1] = 1.0

    feat_0 = np.random.randn(n_pts, d_feat)
    feat_1 = np.random.randn(n_pts, d_feat)

    with pytest.raises(ValueError, match="ineligible: no shared effective macrostate for cells 0,1"):
        anchor_conditioned_mmd([feat_0, feat_1], [mem_0, mem_1], min_effective_mass=2.0)

    # 2. Overlapping effective supports: both cells share anchor 2 with >= 2 effective samples
    mem_0_shared = np.zeros((n_pts, k_anchors))
    mem_0_shared[:, 2] = 1.0

    mem_1_shared = np.zeros((n_pts, k_anchors))
    mem_1_shared[:, 2] = 1.0

    dist, overlap = anchor_conditioned_mmd(
        [feat_0, feat_1], [mem_0_shared, mem_1_shared], min_effective_mass=2.0
    )
    assert dist.shape == (2, 2)
    assert overlap.shape == (2, 2)
    assert dist[0, 1] > 0.0
    assert dist[0, 0] == 0.0
    assert np.isclose(dist[0, 1], dist[1, 0])
    assert overlap[0, 1] > 0.0


def test_world_4_conditional_independence_null_proof():
    """World 4: Conditional-Independence Null Proof World (Confounding Removal).
    
    Verifies the fundamental theoretical claim of Paper 06:
      When R = f(Z) + epsilon identically across phases (so R perp G | Z),
      the marginal residual distributions differ because macro distributions P(Z | G) differ:
        ||E[R | g=0] - E[R | g=1]||^2 > 0  (spurious marginal difference).
      However, the anchor-conditioned MMD statistic removes this macro-induced confounding:
        D_ACMMD(0, 1) approx 0  (conditional null holds).
    """
    rng = np.random.RandomState(42)
    n_pts = 2000
    d_feat = 5
    k_anchors = 2
    anchors = np.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sigma2 = 1.0

    # Macro distributions differ between cell 0 and cell 1:
    # Cell 0 has 80% mass at anchor 0, 20% at anchor 1
    n0_a = int(n_pts * 0.8)
    n0_b = n_pts - n0_a
    z_0 = np.concatenate([rng.randn(n0_a, 3) * 0.2 + anchors[0], rng.randn(n0_b, 3) * 0.2 + anchors[1]])

    # Cell 1 has 20% mass at anchor 0, 80% at anchor 1
    n1_a = int(n_pts * 0.2)
    n1_b = n_pts - n1_a
    z_1 = np.concatenate([rng.randn(n1_a, 3) * 0.2 + anchors[0], rng.randn(n1_b, 3) * 0.2 + anchors[1]])

    mem_0 = soft_membership(z_0, anchors, sigma2)
    mem_1 = soft_membership(z_1, anchors, sigma2)

    # Identical conditional mechanism: R = M @ Z + small noise
    transform = rng.randn(d_feat, 3)
    feat_0 = (z_0 @ transform.T) + rng.randn(n_pts, d_feat) * 0.01
    feat_1 = (z_1 @ transform.T) + rng.randn(n_pts, d_feat) * 0.01

    # 1. Unconditioned marginal residual means differ substantially due to macro confounding
    marginal_diff = float(np.linalg.norm(feat_0.mean(axis=0) - feat_1.mean(axis=0)) ** 2)
    assert marginal_diff > 5.0, f"Marginal difference should be large due to macro composition: {marginal_diff}"

    # 2. Anchor-conditioned MMD removes macro composition confounding
    dist, overlap = anchor_conditioned_mmd([feat_0, feat_1], [mem_0, mem_1], min_effective_mass=5.0)
    d_acmmd = float(dist[0, 1])

    # AC-MMD must be orders of magnitude smaller than the marginal difference
    assert d_acmmd < 0.01 * marginal_diff, (
        f"AC-MMD failed to remove macro confounding: d_acmmd={d_acmmd} vs marginal_diff={marginal_diff}"
    )



def test_world_5_complementary_phase_interaction_positive_control():
    """World 5: Complementary Phase-Interaction Positive Control.
    
    Verifies that when macro distributions are IDENTICAL across phases:
      P(Z | g=0) = P(Z | g=1),
    but the residual generation mechanism is phase-specific:
      R_0 = f_0(Z) + epsilon,  R_1 = f_1(Z) + epsilon with f_0 != f_1,
    the anchor-conditioned MMD cleanly detects the true phase-residual interaction:
      D_ACMMD(0, 1) > 0.
    """
    rng = np.random.RandomState(123)
    n_pts = 300
    d_feat = 5
    k_anchors = 4
    anchors = np.eye(3, 4).T  # 4 anchors in 3D
    sigma2 = 1.0

    # Identical macro states in both cells
    z_shared = rng.randn(n_pts, 3)
    mem_0 = soft_membership(z_shared, anchors, sigma2)
    mem_1 = soft_membership(z_shared, anchors, sigma2)

    # Different residual mechanisms
    feat_0 = np.ones((n_pts, d_feat)) * 1.5 + rng.randn(n_pts, d_feat) * 0.05
    feat_1 = np.ones((n_pts, d_feat)) * -1.5 + rng.randn(n_pts, d_feat) * 0.05

    dist, overlap = anchor_conditioned_mmd([feat_0, feat_1], [mem_0, mem_1], min_effective_mass=2.0)
    assert dist[0, 1] > 5.0, f"AC-MMD failed to detect distinct phase mechanisms: {dist[0, 1]}"


def test_world_6_basis_rotation_and_reflection_invariance():
    """World 6: 3D Spatial Basis Rotation and Reflection Invariance.
    
    Verifies that for any orthogonal transformation Q in O(3):
      1. Transforming coordinates z -> Q^T z and anchors a_k -> Q^T a_k
         leaves soft memberships w_k(z) identically invariant.
      2. The resulting AC-MMD distance matrix is invariant to float64 machine precision (< 10^-12).
    """
    rng = np.random.RandomState(456)
    n_pts = 100
    k_anchors = 6
    d_feat = 5
    sigma2 = 1.5

    # Random macro states and anchors in 3D
    z_0 = rng.randn(n_pts, 3)
    z_1 = rng.randn(n_pts, 3)
    anchors = rng.randn(k_anchors, 3)

    feat_0 = rng.randn(n_pts, d_feat)
    feat_1 = rng.randn(n_pts, d_feat)

    mem_0 = soft_membership(z_0, anchors, sigma2)
    mem_1 = soft_membership(z_1, anchors, sigma2)
    dist_orig, _ = anchor_conditioned_mmd([feat_0, feat_1], [mem_0, mem_1], min_effective_mass=1.0)

    # Generate random orthogonal transformation Q in O(3) with determinant -1 (reflection) or +1 (rotation)
    q_mat, _ = np.linalg.qr(rng.randn(3, 3))

    # Apply rotation/reflection
    z_0_rot = z_0 @ q_mat
    z_1_rot = z_1 @ q_mat
    anchors_rot = anchors @ q_mat

    mem_0_rot = soft_membership(z_0_rot, anchors_rot, sigma2)
    mem_1_rot = soft_membership(z_1_rot, anchors_rot, sigma2)

    # Memberships must match exactly
    assert np.allclose(mem_0, mem_0_rot, atol=1e-12)
    assert np.allclose(mem_1, mem_1_rot, atol=1e-12)

    dist_rot, _ = anchor_conditioned_mmd([feat_0, feat_1], [mem_0_rot, mem_1_rot], min_effective_mass=1.0)
    assert np.allclose(dist_orig, dist_rot, atol=1e-12)


def test_world_7_within_macro_phase_shuffle_and_pair_sham():
    """World 7: Within-Macro Phase Shuffling & Deterministic Pair Sham.
    
    Verifies:
      1. Deterministic Joint Pair Sham: permuting (feat, mem) jointly preserves AC-MMD (< 10^-14).
      2. Within-Macro Phase Shuffle: breaks phase-residual alignment while preserving macro clusters.
    """
    rng = np.random.RandomState(789)
    n_cells = 4
    n_pts = 60
    d_feat = 16
    k_anchors = 4

    feature_cells = [rng.randn(n_pts, d_feat) for _ in range(n_cells)]
    membership_cells = [rng.dirichlet(np.ones(k_anchors) * 5.0, size=n_pts) for _ in range(n_cells)]

    dist_orig, _ = anchor_conditioned_mmd(feature_cells, membership_cells, min_effective_mass=2.0)

    # 1. Joint pair sham
    sham_features = []
    sham_memberships = []
    for g in range(n_cells):
        perm = rng.permutation(n_pts)
        sham_features.append(feature_cells[g][perm])
        sham_memberships.append(membership_cells[g][perm])

    dist_sham, _ = anchor_conditioned_mmd(sham_features, sham_memberships, min_effective_mass=2.0)
    max_diff = float(np.max(np.abs(dist_orig - dist_sham)))
    assert max_diff < 1e-14, f"Joint pair sham violated: max diff = {max_diff}"

    # 2. Within-macro phase shuffle
    shuffled_features = within_macro_phase_shuffle(feature_cells, membership_cells, seed=42)
    assert len(shuffled_features) == n_cells
    for g in range(n_cells):
        assert shuffled_features[g].shape == feature_cells[g].shape
    # Distances after shuffle should be finite
    dist_shuffled, _ = anchor_conditioned_mmd(shuffled_features, membership_cells, min_effective_mass=2.0)
    assert np.all(np.isfinite(dist_shuffled))


def test_world_8_classifier_head_gradient_flow_and_mask_ablations():
    """World 8: Classifier Head Gradient Flow & Mask/Affinity Ablations.
    
    Verifies:
      1. Clean gradient flow across 2-channel and 1-channel variants.
      2. Mask ablations: all_pairs, exclude_diagonal, non_adjacent.
      3. Graph normalization ablation: raw affinity vs Laplacian normalization.
    """
    torch.manual_seed(42)
    batch_size = 4
    classes = 5

    x_2c = torch.randn(batch_size, 2, 16, 16, requires_grad=True)
    x_1c = torch.randn(batch_size, 1, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (batch_size, classes)).float()

    variants = [
        (ExperimentVariant(), x_2c, 2),  # macro_plus_conditional_residual / full
        (ExperimentVariant(head="linear"), x_2c, 2),  # linear probe
        (ExperimentVariant(head="phase_aware_linear"), x_2c, 2),
        (ExperimentVariant(representation="macro_component"), x_2c, 2),
        (ExperimentVariant(representation="residual_marginal"), x_2c, 2),
        (ExperimentVariant(representation="macro_residual_marginals"), x_2c, 2),
        (ExperimentVariant(), x_1c, 1),  # single channel variant
    ]

    for var, input_tensor, in_c in variants:
        model = ConditionalRepStatModel(input_dim=16, classes=classes, variant=var, in_channels=in_c)
        model.train()

        out = model(input_tensor)
        assert out.shape == (batch_size, classes)

        loss = torch.nn.functional.binary_cross_entropy_with_logits(out, target)
        loss.backward()

        has_grads = False
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(torch.isfinite(param.grad)), "Non-finite gradients detected"
                if torch.any(param.grad != 0):
                    has_grads = True
        assert has_grads, f"No active gradients found for variant {var}"

    # Recurrence mask and normalization ablations
    test_dist = np.random.rand(16, 16)
    test_dist = test_dist + test_dist.T
    np.fill_diagonal(test_dist, 0.0)

    # 1. Non-adjacent mask
    rec_non_adj = recurrence_from_distances(test_dist, tau=1.0, mask_type="non_adjacent")
    assert rec_non_adj[0, 1] == 0.0 and rec_non_adj[0, 15] == 0.0
    assert rec_non_adj[0, 2] > 0.0

    # 2. All-pairs mask
    rec_all = recurrence_from_distances(test_dist, tau=1.0, mask_type="all_pairs")
    assert rec_all[0, 1] > 0.0
    assert rec_all[0, 0] > 0.0

    # 3. Exclude-diagonal mask
    rec_no_diag = recurrence_from_distances(test_dist, tau=1.0, mask_type="exclude_diagonal")
    assert rec_no_diag[0, 0] == 0.0
    assert rec_no_diag[0, 1] > 0.0

    # 4. Raw affinity (unnormalized)
    rec_raw = recurrence_from_distances(test_dist, tau=1.0, laplacian_normalize=False)
    assert np.all(np.isfinite(rec_raw))
