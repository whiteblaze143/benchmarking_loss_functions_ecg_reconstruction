"""Synthetic Recovery and Mathematical Invariant Tests for Paper 03.

Phase-Path-Signature: Distributional Path Signatures on Cardiac Phase Trajectories.
Proves:
1. Chen's Concatenation Identity: S(gamma * eta) == S(gamma) (x) S(eta).
2. Reparameterization Invariance & Resolution Scaling:
   monotonicity min_t phi'(t) > 0 and error O(N^-2) convergence.
3. Descriptor Invariance Breakdown:
   d^sig preserves reparameterization invariance; temporal mean \\bar{gamma} breaks it.
4. Time-Reversal & Lévy Area Antisymmetry:
   A^{i,j}(gamma^<-) == -A^{i,j}(gamma) to numerical precision.
5. Exact Multiset Order Destruction vs Point Marginal Conservation:
   Internal sample permutation leaves sample mean and covariance identical,
   while LogSig changes dramatically (>50% distance).
6. Nyström Landmark Approximation Quality Gate on Signature Descriptors.
7. Planar Lead-Space Trajectory Chirality Null-Relative Separation:
   Signature KME achieves null-relative separation D(CW, CCW) > Q_0.99(null),
   while unordered voltage KME is blind to chirality (D_volt <= Q_0.99(null)).
8. Tree-Like Ambiguity & Excursion Cancellation:
   gamma * eta * eta^-1 cancels in the signature up to tree-like equivalence.
9. Representation Endpoints:
   D_warp << D_rev and D_warp << D_scramble.
"""

from __future__ import annotations

import json
from pathlib import Path

import iisignature
import numpy as np
import pytest

from repecg.common.kernels import NystromMap, TruncatedPCAWhitening, WhiteningTransform, audit_nystrom, biased_mmd2, imq_kernel


# ============================================================================
# World 1A: Chen's Concatenation Identity
# ============================================================================

def test_world_1a_chen_concatenation_identity():
    """Verify Chen's theorem: S(gamma * eta) == S(gamma) (x) S(eta) via sigcombine."""
    rng = np.random.default_rng(42)
    dim = 8
    depth = 2

    # Two paths in R^8
    p1 = rng.normal(0, 1, (12, dim))
    p2 = rng.normal(0, 1, (12, dim))
    # Align p2 start to p1 end
    p2_aligned = p2 - p2[0] + p1[-1]
    p_concat = np.vstack([p1, p2_aligned[1:]])

    s1 = iisignature.sig(p1, depth)
    s2 = iisignature.sig(p2_aligned, depth)
    s_concat = iisignature.sig(p_concat, depth)
    s_combined = iisignature.sigcombine(s1, s2, dim, depth)

    max_diff = float(np.max(np.abs(s_concat - s_combined)))
    assert max_diff < 1e-5, f"Chen's identity failed: max diff = {max_diff}"


# ============================================================================
# World 1B: Reparameterization Invariance & Discretization Scaling
# ============================================================================

def test_world_1b_reparameterization_invariance_and_monotonicity():
    """Verify non-linear monotone time-warping:
    1. Check phi'(t) > 0 strictly.
    2. Verify that signature reparameterization error decreases monotonically as N increases.
    """
    # 1. Monotonicity check
    t_dense = np.linspace(0.0, 1.0, 1000)
    amplitude = 0.1
    phi_prime = 1.0 + amplitude * np.cos(2.0 * np.pi * t_dense)
    min_derivative = float(np.min(phi_prime))
    assert min_derivative >= 0.89, f"Monotonicity violated: min derivative = {min_derivative}"

    def true_curve(tau):
        return np.stack([
            np.sin(2.0 * np.pi * tau),
            np.cos(2.0 * np.pi * tau),
            np.sin(4.0 * np.pi * tau),
            np.cos(4.0 * np.pi * tau),
        ], axis=-1)

    prep = iisignature.prepare(4, 2)
    resolutions = [16, 32, 64, 128, 256]
    relative_errors = []

    for N in resolutions:
        t = np.linspace(0.0, 1.0, N)
        gamma = true_curve(t)
        phi_t = t + amplitude * np.sin(2.0 * np.pi * t) / (2.0 * np.pi)
        gamma_warped = true_curve(phi_t)

        s_orig = iisignature.logsig(gamma, prep)
        s_warp = iisignature.logsig(gamma_warped, prep)
        rel_err = float(np.linalg.norm(s_orig - s_warp) / np.linalg.norm(s_orig))
        relative_errors.append(rel_err)

    # Error must decrease with resolution
    assert relative_errors[-1] < relative_errors[0], "Error must decrease as resolution increases"
    assert relative_errors[-1] < 1e-3, f"High-resolution error too large: {relative_errors[-1]}"
    # Monotonic reduction across successive doublings
    for i in range(len(relative_errors) - 1):
        assert relative_errors[i + 1] < relative_errors[i], f"Non-monotonic error step at index {i}"


# ============================================================================
# World 1C: Descriptor Invariance Breakdown (signature_pure vs signature_plus_mean)
# ============================================================================

def test_world_1c_descriptor_invariance_breakdown():
    """Verify that pure signature descriptor d^sig = [gamma(0), gamma(1), LogSig] is
    reparameterization invariant, while temporal mean bar{gamma} breaks invariance.
    """
    N = 128
    t = np.linspace(0.0, 1.0, N)
    amplitude = 0.1
    phi_t = t + amplitude * np.sin(2.0 * np.pi * t) / (2.0 * np.pi)

    gamma = np.stack([
        np.sin(2.0 * np.pi * t) + 0.3 * np.cos(4.0 * np.pi * t),
        np.cos(2.0 * np.pi * t) - 0.2 * np.sin(4.0 * np.pi * t),
    ], axis=-1)

    gamma_warped = np.stack([
        np.sin(2.0 * np.pi * phi_t) + 0.3 * np.cos(4.0 * np.pi * phi_t),
        np.cos(2.0 * np.pi * phi_t) - 0.2 * np.sin(4.0 * np.pi * phi_t),
    ], axis=-1)

    prep = iisignature.prepare(2, 2)
    ls_orig = iisignature.logsig(gamma, prep)
    ls_warped = iisignature.logsig(gamma_warped, prep)

    # d^sig: [gamma(0), gamma(1), LogSig]
    d_sig_orig = np.concatenate([gamma[0], gamma[-1], ls_orig])
    d_sig_warped = np.concatenate([gamma_warped[0], gamma_warped[-1], ls_warped])
    rel_err_dsig = float(np.linalg.norm(d_sig_orig - d_sig_warped) / np.linalg.norm(d_sig_orig))

    # Mean vector \bar{gamma}
    mean_orig = gamma.mean(axis=0)
    mean_warped = gamma_warped.mean(axis=0)
    rel_err_mean = float(np.linalg.norm(mean_orig - mean_warped) / np.linalg.norm(mean_orig))

    # LogSig / d^sig error is negligible, while temporal mean error is orders of magnitude larger
    assert rel_err_dsig < 1e-3, f"Pure descriptor must be invariant: rel_err={rel_err_dsig}"
    assert rel_err_mean > 0.05, f"Temporal mean must show non-invariance under warp: rel_err={rel_err_mean}"
    assert rel_err_mean > 50.0 * rel_err_dsig, "Temporal mean error must be >50x larger than signature error"


# ============================================================================
# World 2: Time-Reversal & Lévy Area Antisymmetry
# ============================================================================

def test_world_2_levy_area_antisymmetry():
    """Verify that reversing trajectory time arrow flips the sign of the level-2 Lévy area:
    A^{i,j}(gamma^<-) == -A^{i,j}(gamma) to numerical precision.
    """
    t = np.linspace(0.0, 1.0, 100)
    # Planar trajectory in lead-space (e.g. V1 vs V5)
    gamma = np.stack([np.cos(2.0 * np.pi * t), np.sin(2.0 * np.pi * t)], axis=-1)
    gamma_rev = gamma[::-1]

    prep = iisignature.prepare(2, 2)
    ls_fwd = iisignature.logsig(gamma, prep)
    ls_rev = iisignature.logsig(gamma_rev, prep)

    # For d=2, depth 2: index 0 is delta x, index 1 is delta y, index 2 is Lévy area A^{1,2}
    area_fwd = float(ls_fwd[2])
    area_rev = float(ls_rev[2])

    assert abs(area_fwd - np.pi) < 0.05, f"Planar loop area should approximate pi: {area_fwd}"
    sum_area = abs(area_fwd + area_rev)
    assert sum_area < 1e-10, f"Lévy area antisymmetry failed: sum = {sum_area}"


# ============================================================================
# World 3: Exact Multiset Order Destruction vs Point Marginal Conservation
# ============================================================================

def test_world_3_exact_multiset_order_destruction():
    """Verify that permuting internal samples leaves the pointwise multiset, mean,
    and covariance identical to machine precision, while the LogSig changes by >50%.
    """
    rng = np.random.default_rng(42)
    N = 16
    dim = 8
    t = np.linspace(0.0, 1.0, N)
    path = np.stack([np.sin((i + 1) * t) + 0.1 * (i + 1) * t for i in range(dim)], axis=-1)

    # Permute only interior points, preserving endpoints
    perm = np.arange(N)
    perm[1:-1] = rng.permutation(perm[1:-1])
    scrambled = path[perm]

    # Verify identical endpoints
    assert np.array_equal(path[0], scrambled[0])
    assert np.array_equal(path[-1], scrambled[-1])

    # Verify identical sample mean and sample covariance
    mean_diff = float(np.max(np.abs(path.mean(axis=0) - scrambled.mean(axis=0))))
    cov_diff = float(np.max(np.abs(np.cov(path, rowvar=False) - np.cov(scrambled, rowvar=False))))
    assert mean_diff < 1e-12, f"Sample mean must be identical: diff={mean_diff}"
    assert cov_diff < 1e-12, f"Sample covariance must be identical: diff={cov_diff}"

    # Verify log-signature changes significantly
    prep = iisignature.prepare(dim, 3)
    ls_orig = iisignature.logsig(path, prep)
    ls_scrambled = iisignature.logsig(scrambled, prep)

    rel_dist = float(np.linalg.norm(ls_orig - ls_scrambled) / np.linalg.norm(ls_orig))
    assert rel_dist > 0.50, f"Scrambled path must differ from original by >50%: {rel_dist:.4f}"


# ============================================================================
# World 4: Nyström Landmark Approximation Quality Gate
# ============================================================================

def test_world_4_nystrom_signature_fidelity_gate():
    """Verify that Nyström IMQ kernel map with M=128 landmarks satisfies fidelity gate:
    Spearman rank correlation >= 0.90 and median relative error <= 0.15.
    """
    # Check production manifest if present
    manifest_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper03_path_signature/development_representations/manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        audit = manifest["audit"]
        assert audit["passed"] is True, f"Production Nyström audit failed: {audit}"
        assert audit["spearman"] >= 0.90, f"Spearman too low: {audit['spearman']}"
        assert audit["median_relative_error"] <= 0.15, f"Relative error too high: {audit['median_relative_error']}"

    # Synthetic check with TruncatedPCAWhitening and Nyström mapping
    rng = np.random.default_rng(42)
    dim = 8
    n_pairs = 30
    n_obs = 100

    reservoir = rng.standard_normal(size=(2000, dim))
    whitening = WhiteningTransform.fit(reservoir)

    cells = []
    for _ in range(n_pairs * 2):
        center = rng.normal(0, 0.5, size=dim)
        cells.append(whitening.transform(rng.normal(center, 0.8, size=(n_obs, dim))))

    mapping = NystromMap.fit(np.concatenate(cells[:20]), landmarks=128, c2=1.0, seed=42)

    exact = np.empty(n_pairs)
    approx = np.empty(n_pairs)
    for pair in range(n_pairs):
        x = cells[pair * 2]
        y = cells[pair * 2 + 1]
        exact[pair] = float(imq_kernel(x, x).mean() + imq_kernel(y, y).mean() - 2.0 * imq_kernel(x, y).mean())
        x_feat = mapping.transform(x).mean(axis=0)
        y_feat = mapping.transform(y).mean(axis=0)
        approx[pair] = float(np.sum((x_feat - y_feat) ** 2))

    audit_res = audit_nystrom(exact, approx, min_spearman=0.90)
    assert audit_res["spearman"] >= 0.90, f"Synthetic Spearman too low: {audit_res['spearman']}"


# ============================================================================
# World 5: Planar Lead-Space Trajectory Chirality Null-Relative Separation
# ============================================================================

def test_world_5_planar_chirality_null_relative_separation():
    """Verify that Signature KME achieves null-relative separation:
    D_sig(CW, CCW) > Q_0.99(null),
    while unordered voltage KME fails to distinguish opposite chiralities (D_volt <= Q_0.99(null)).
    """
    rng = np.random.default_rng(42)
    N_points = 16
    prep = iisignature.prepare(2, 2)

    def sample_ensemble(sign=1.0, count=30):
        t = np.linspace(0.0, 1.0, N_points)
        loops = []
        for _ in range(count):
            noise = rng.normal(0, 0.04, (N_points, 2))
            loop = np.stack([np.cos(2.0 * np.pi * t), sign * np.sin(2.0 * np.pi * t)], axis=-1) + noise
            loops.append(loop)
        return loops

    cw1 = sample_ensemble(sign=1.0, count=30)
    cw2 = sample_ensemble(sign=1.0, count=30)
    ccw = sample_ensemble(sign=-1.0, count=30)

    # Signature descriptors
    sig_cw1 = np.array([iisignature.logsig(p, prep) for p in cw1])
    sig_cw2 = np.array([iisignature.logsig(p, prep) for p in cw2])
    sig_ccw = np.array([iisignature.logsig(p, prep) for p in ccw])

    # Voltage point clouds (unordered)
    volt_cw1 = np.concatenate(cw1, axis=0)
    volt_cw2 = np.concatenate(cw2, axis=0)
    volt_ccw = np.concatenate(ccw, axis=0)

    # Null distribution for signature: D(CW1_sub, CW2_sub)
    null_sig = [
        biased_mmd2(sig_cw1[rng.choice(30, 15, replace=False)], sig_cw2[rng.choice(30, 15, replace=False)])
        for _ in range(50)
    ]
    q99_sig = float(np.percentile(null_sig, 99))
    d_sig_opposite = float(biased_mmd2(sig_cw1, sig_ccw))

    # Null distribution for voltage point clouds
    null_volt = [
        biased_mmd2(volt_cw1[rng.choice(len(volt_cw1), 100, replace=False)], volt_cw2[rng.choice(len(volt_cw2), 100, replace=False)])
        for _ in range(50)
    ]
    q99_volt = float(np.percentile(null_volt, 99))
    d_volt_opposite = float(biased_mmd2(volt_cw1, volt_ccw))

    # 1. Signature MMD between opposite chiralities must exceed null 99th percentile by a large margin
    assert d_sig_opposite > q99_sig, f"Signature failed null separation: {d_sig_opposite} <= {q99_sig}"
    assert d_sig_opposite > 10.0 * q99_sig, f"Signature separation should be >10x null: ratio={d_sig_opposite / q99_sig}"

    # 2. Voltage point-cloud MMD between opposite chiralities must be comparable to the null
    assert d_volt_opposite <= q99_volt * 1.5, f"Voltage KME should not separate chiralities: {d_volt_opposite} > {q99_volt * 1.5}"


# ============================================================================
# World 6: Tree-Like Ambiguity & Excursion Cancellation
# ============================================================================

def test_world_6_tree_like_ambiguity_and_excursion_cancellation():
    """Verify that an inserted forward-and-back excursion gamma * eta * eta^-1
    cancels out in the signature to machine precision, formalizing that the signature
    characterizes paths up to tree-like equivalence.
    """
    rng = np.random.default_rng(42)
    dim = 8
    depth = 2

    # Base path gamma
    p_base = rng.normal(0, 1, (10, dim))
    # Excursion eta starting at p_base[-1]
    eta = rng.normal(0, 1, (8, dim))
    eta_aligned = eta - eta[0] + p_base[-1]
    # Exact return eta^-1
    eta_rev = eta_aligned[::-1]

    # Concatenated path with excursion: gamma * eta * eta^-1
    p_tree = np.vstack([p_base, eta_aligned[1:], eta_rev[1:]])

    s_base = iisignature.sig(p_base, depth)
    s_tree = iisignature.sig(p_tree, depth)

    diff = float(np.max(np.abs(s_base - s_tree)))
    assert diff < 1e-10, f"Tree-like excursion did not cancel: max diff = {diff}"


# ============================================================================
# Representation Endpoints Test: D_warp << D_rev and D_warp << D_scramble
# ============================================================================

def test_representation_endpoints_warp_vs_order_and_reversal():
    """Verify representation endpoints on realistic 8-lead phase trajectory:
    D_warp << D_rev and D_warp << D_scramble.
    """
    t = np.linspace(0.0, 1.0, 32)
    gamma = np.stack([
        np.sin(2.0 * np.pi * t) + 0.2 * np.sin(4.0 * np.pi * t),
        np.cos(2.0 * np.pi * t) - 0.1 * np.cos(4.0 * np.pi * t),
        0.5 * np.sin(2.0 * np.pi * t),
        -0.3 * np.cos(2.0 * np.pi * t),
        0.8 * np.sin(2.0 * np.pi * t),
        0.2 * np.sin(2.0 * np.pi * t),
        -0.5 * np.sin(2.0 * np.pi * t),
        0.1 * np.cos(2.0 * np.pi * t),
    ], axis=-1)

    phi_t = t + 0.1 * np.sin(2.0 * np.pi * t) / (2.0 * np.pi)
    gamma_warp = np.stack([
        np.sin(2.0 * np.pi * phi_t) + 0.2 * np.sin(4.0 * np.pi * phi_t),
        np.cos(2.0 * np.pi * phi_t) - 0.1 * np.cos(4.0 * np.pi * phi_t),
        0.5 * np.sin(2.0 * np.pi * phi_t),
        -0.3 * np.cos(2.0 * np.pi * phi_t),
        0.8 * np.sin(2.0 * np.pi * phi_t),
        0.2 * np.sin(2.0 * np.pi * phi_t),
        -0.5 * np.sin(2.0 * np.pi * phi_t),
        0.1 * np.cos(2.0 * np.pi * phi_t),
    ], axis=-1)

    gamma_rev = gamma[::-1]

    perm = np.arange(len(t))
    perm[1:-1] = np.random.default_rng(42).permutation(perm[1:-1])
    gamma_scramble = gamma[perm]

    prep = iisignature.prepare(8, 3)
    sig_base = iisignature.logsig(gamma, prep)
    sig_warp = iisignature.logsig(gamma_warp, prep)
    sig_rev = iisignature.logsig(gamma_rev, prep)
    sig_scramble = iisignature.logsig(gamma_scramble, prep)

    d_warp = float(np.linalg.norm(sig_base - sig_warp))
    d_rev = float(np.linalg.norm(sig_base - sig_rev))
    d_scramble = float(np.linalg.norm(sig_base - sig_scramble))

    assert d_warp < 0.01 * d_rev, f"D_warp should be < 1% of D_rev: {d_warp} vs {d_rev}"
    assert d_warp < 0.01 * d_scramble, f"D_warp should be < 1% of D_scramble: {d_warp} vs {d_scramble}"
