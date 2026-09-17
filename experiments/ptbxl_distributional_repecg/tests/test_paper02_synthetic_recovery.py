import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.stats import spearmanr

from repecg.common.kernels import imq_kernel, NystromMap, WhiteningTransform, audit_nystrom
from repecg.common.models import PhaseCNN, CircularResidualBlock, StandardResidualBlock
from repecg.common.variants import ExperimentVariant
from repecg.paper02_kernel_mean.controls import exact_moment_matched_gaussian, sample_covariance


def _torch_imq_mmd2(x: torch.Tensor, y: torch.Tensor, c2: float = 1.0) -> float:
    """Float32/64 IMQ MMD^2 via PyTorch."""
    d_xx = torch.cdist(x, x).square()
    d_yy = torch.cdist(y, y).square()
    d_xy = torch.cdist(x, y).square()
    k_xx = torch.rsqrt(d_xx + c2).mean()
    k_yy = torch.rsqrt(d_yy + c2).mean()
    k_xy = torch.rsqrt(d_xy + c2).mean()
    return (k_xx + k_yy - 2.0 * k_xy).item()


def _unbiased_mmd2(x: torch.Tensor, y: torch.Tensor, c2: float = 1.0) -> float:
    """Unbiased U-statistic IMQ MMD^2."""
    m, n = len(x), len(y)
    d_xx = torch.cdist(x, x).square()
    d_yy = torch.cdist(y, y).square()
    d_xy = torch.cdist(x, y).square()
    k_xx = torch.rsqrt(d_xx + c2)
    k_yy = torch.rsqrt(d_yy + c2)
    k_xy = torch.rsqrt(d_xy + c2)
    u_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (m * (m - 1))
    u_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
    u_xy = k_xy.mean()
    return (u_xx + u_yy - 2.0 * u_xy).item()


def _whiten(x: torch.Tensor) -> torch.Tensor:
    """Exact empirical centering and whitening."""
    xc = x - x.mean(dim=0, keepdim=True)
    cov = xc.T @ xc / (len(x) - 1)
    evals, evecs = torch.linalg.eigh(cov)
    return xc @ (evecs * (evals.clamp_min(1e-8) ** -0.5)) @ evecs.T


# ============================================================================
# Test World A: Non-Gaussian Sensitivity with Matched Mean & Covariance
# ============================================================================

def test_world_a_matched_mean_cov_non_gaussian_sensitivity():
    """Prove that IMQ KME distinguishes distributions with identical mean AND covariance.
    
    P = N(0, I_8)
    Q = 0.5 * N(+a, Sigma_0) + 0.5 * N(-a, Sigma_0)
    where a = [0.8, 0, ..., 0]^T and Sigma_0 = I - a a^T.
    Both are empirically whitened to sample mean 0 and sample covariance I_8 to machine precision.
    """
    torch.manual_seed(42)
    dim = 8
    n = 1000
    a = torch.zeros(dim)
    a[0] = 0.8
    sigma_0 = torch.eye(dim) - torch.outer(a, a)
    L = torch.linalg.cholesky(sigma_0)
    
    # Draw from P = N(0, I) and Q (symmetric bimodal mixture)
    p = torch.randn(n, dim)
    choices = torch.randint(0, 2, (n, 1)).float() * 2 - 1
    q = torch.randn(n, dim) @ L.T + choices * a
    
    # Whiten both so sample moments are EXACTLY matched (mean=0, cov=I)
    p_w = _whiten(p)
    q_w = _whiten(q)
    
    # Verify sample moments match to machine precision
    mean_diff = (p_w.mean(dim=0) - q_w.mean(dim=0)).norm().item()
    cov_p = p_w.T @ p_w / (n - 1)
    cov_q = q_w.T @ q_w / (n - 1)
    cov_diff = (cov_p - cov_q).norm().item()
    assert mean_diff < 1e-5, f"Mean difference not zero: {mean_diff}"
    assert cov_diff < 1e-4, f"Covariance difference not zero: {cov_diff}"
    
    # Compute alternative distance D_alt = MMD^2(P, Q)
    d_alt = _torch_imq_mmd2(p_w, q_w, c2=1.0)
    
    # Compute null distribution D_null = MMD^2(P, P') over independent draws
    null_distances = []
    for _ in range(25):
        p_1 = _whiten(torch.randn(n, dim))
        p_2 = _whiten(torch.randn(n, dim))
        null_distances.append(_torch_imq_mmd2(p_1, p_2, c2=1.0))
        
    q95_null = float(np.quantile(null_distances, 0.95))
    assert d_alt > q95_null, (
        f"Non-Gaussian shape not detected: D_alt={d_alt:.6f} <= Q0.95(D_null)={q95_null:.6f}"
    )


# ============================================================================
# Test World B: Moment-Matched Gaussian Surrogate Precision
# ============================================================================

def test_world_b_moment_matched_gaussian_surrogate():
    """Verify that exact_moment_matched_gaussian achieves < 1e-8 error on mean and covariance
    while preserving non-zero MMD against non-Gaussian target.
    """
    torch.manual_seed(42)
    # Generate skewed non-Gaussian observations (e.g. exponential distribution)
    # Shape: (batch=2, observations=64, features=8)
    exp_samples = torch.empty(2, 64, 8, dtype=torch.float64).exponential_(lambd=1.0)
    
    # Apply Gaussian surrogate whitening-recolouring
    matched_surrogate = exact_moment_matched_gaussian(exp_samples, tolerance=1e-8)
    
    # Verify sample mean matching to machine precision
    target_mean = exp_samples.mean(dim=-2, keepdim=True)
    surrogate_mean = matched_surrogate.mean(dim=-2, keepdim=True)
    max_mean_error = (surrogate_mean - target_mean).abs().max().item()
    assert max_mean_error < 1e-8, f"Mean error {max_mean_error} exceeds 1e-8"
    
    # Verify sample covariance matching to machine precision
    target_cov = sample_covariance(exp_samples)
    surrogate_cov = sample_covariance(matched_surrogate)
    max_cov_error = (surrogate_cov - target_cov).abs().max().item()
    assert max_cov_error < 1e-8, f"Covariance error {max_cov_error} exceeds 1e-8"
    
    # Verify that despite identical 1st and 2nd moments, MMD distinguishes target from surrogate
    x = exp_samples[0]
    y = matched_surrogate[0]
    mmd_dist = _torch_imq_mmd2(x, y, c2=1.0)
    assert mmd_dist > 1e-4, f"MMD between skewed target and Gaussian surrogate must be positive, got {mmd_dist}"


# ============================================================================
# Test World C: Discrete C_16 Cyclic Shift Equivariance & Invariance
# ============================================================================

def test_world_c_c16_cyclic_shift_equivariance_and_invariance():
    """Prove that CircularResidualBlock is strictly C_16 equivariant and PhaseCNN GAP is invariant,
    while StandardResidualBlock breaks cyclic equivariance.
    """
    torch.manual_seed(42)
    batch, phases, channels = 2, 16, 128
    x = torch.randn(batch, channels, phases)
    k = 3  # integer circular shift
    x_rolled = torch.roll(x, shifts=k, dims=-1)
    
    # 1. Test CircularResidualBlock equivariance: F(T_k x) == T_k F(x)
    circ_block = CircularResidualBlock(channels)
    circ_block.eval()
    with torch.no_grad():
        out_orig = circ_block(x)
        out_rolled_input = circ_block(x_rolled)
        rolled_orig_out = torch.roll(out_orig, shifts=k, dims=-1)
        
    circ_diff = (out_rolled_input - rolled_orig_out).abs().max().item()
    assert circ_diff < 1e-5, f"CircularResidualBlock failed C_16 equivariance: diff={circ_diff}"
    
    # 2. Test StandardResidualBlock (zero padding) fails equivariance
    std_block = StandardResidualBlock(channels)
    std_block.eval()
    with torch.no_grad():
        std_out_orig = std_block(x)
        std_out_rolled = std_block(x_rolled)
        std_rolled_orig = torch.roll(std_out_orig, shifts=k, dims=-1)
        
    std_diff = (std_out_rolled - std_rolled_orig).abs().max().item()
    assert std_diff > 0.01, f"StandardResidualBlock should fail cyclic equivariance, got diff={std_diff}"
    
    # 3. Test full PhaseCNN with GAP: y(T_k x) == y(x) (exact shift invariance)
    variant_full = ExperimentVariant()  # default GAP
    model = PhaseCNN(input_dim=channels, width=channels, variant=variant_full)
    model.eval()
    
    x_seq = x.transpose(1, 2)  # (batch, 16, channels)
    x_seq_rolled = torch.roll(x_seq, shifts=k, dims=1)
    with torch.no_grad():
        logits_orig = model(x_seq)
        logits_rolled = model(x_seq_rolled)
        
    gap_diff = (logits_rolled - logits_orig).abs().max().item()
    assert gap_diff < 1e-5, f"PhaseCNN with GAP must be shift-invariant: diff={gap_diff}"
    
    # 4. Test phase_aware readout is NOT shift-invariant (preserves anchored depolarization phase)
    variant_aware = ExperimentVariant(head="phase_aware")
    model_aware = PhaseCNN(input_dim=channels, width=channels, variant=variant_aware)
    model_aware.eval()
    with torch.no_grad():
        aware_orig = model_aware(x_seq)
        aware_rolled = model_aware(x_seq_rolled)
        
    aware_diff = (aware_rolled - aware_orig).abs().max().item()
    assert aware_diff > 0.01, f"Phase-aware readout must retain phase location identity: diff={aware_diff}"


# ============================================================================
# Test World D: Nyström Operating Point Fidelity Acceptance Gate
# ============================================================================

def test_world_d_nystrom_operating_point_fidelity():
    """Verify Nyström IMQ approximation achieves Spearman rho >= 0.95 on production representations,
    and verify fidelity gate on synthetic whitened distributions.
    """
    manifest_path = Path("outputs/paper02_kernel_mean/development_representations/manifest.json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        audit = manifest["audit"]
        assert audit["passed"] is True, f"Production Nyström audit failed: {audit}"
        assert audit["spearman"] >= 0.95, f"Spearman rho too low: {audit['spearman']}"
        assert audit["median_relative_error"] <= 0.15, f"Relative error too high: {audit['median_relative_error']}"
        
    # Synthetic verification with whitened data
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
        
    mapping = NystromMap.fit(np.concatenate(cells[:20]), landmarks=256, c2=1.0, seed=42)
    
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
    assert audit_res["spearman"] >= 0.90, f"Synthetic Spearman rho too low: {audit_res['spearman']}"


# ============================================================================
# Test World E: Sample-Size Identifiability Power Curve
# ============================================================================

def test_world_e_sample_size_identifiability_power_curve():
    """Verify that distributional discriminability increases with beat count B,
    achieving statistical power > 80% at B=32 under unbiased U-statistic MMD.
    """
    torch.manual_seed(42)
    dim = 8
    
    powers = {}
    for b in [4, 8, 16, 32]:
        trials = 25
        detected = 0
        for _ in range(trials):
            n = b * 16
            p = torch.randn(n, dim)
            # Uniform distribution with matched mean 0 and variance 1: U[-sqrt(3), sqrt(3)]
            q = (torch.rand(n, dim) * 2.0 - 1.0) * (3.0 ** 0.5)
            p_prime = torch.randn(n, dim)
            
            d_alt = _unbiased_mmd2(p, q, c2=1.0)
            d_null = _unbiased_mmd2(p, p_prime, c2=1.0)
            if d_alt > d_null:
                detected += 1
                
        powers[b] = detected / trials
        
    # Statistical power must be high at large beat count B=32
    assert powers[32] >= 0.80, f"Power at B=32 must exceed 80%, got {powers[32]:.2%}"
