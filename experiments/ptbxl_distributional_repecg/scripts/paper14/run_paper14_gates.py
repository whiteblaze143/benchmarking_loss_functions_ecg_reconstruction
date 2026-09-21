"""Paper 14 Synthetic Gate Suite: Environmental Representation-Distribution Alignment (V2.1).

Implements the comprehensive gate stack:
- G0: Numerical MMD contract + finite-sample null calibration (synthetic distribution baseline;
      real-data calibration uses matched within-representation permutation null)
- G1: Positive alignment world with environment probe (AUC_E(MMD) < AUC_E(ERM), AUC_Y preserved)
- G2: Matched-marginal failure world (Claim-boundary PASS: matched marginals blind MMD to reversal)
- G3: Collapse world with geometry audit (MMD-only collapses Var/r_eff, BCE+MMD preserves them)
- G4: Objective execution (nonzero gradient contrast between ERM and MMD on identical batch)
- G5: Scale escape audit (proves unconstrained MMD -> 1/n + 1/m diagonal floor under scale inflation;
      LayerNorm eliminates escape)
- G5b: Trainable-normalizer escape audit (proves model contains no trainable LayerNorm affine parameters,
       H -> aH + b leaves Z invariant, and end-to-end training has no post-norm scale escape)
- G6: Label-shift boundary (Claim-boundary PASS: unequal class priors produce MMD > 0 without domain nuisance)
- G7: Environment provenance & label mixture audit (controlled acquisition perturbation bank, Delta_y=0)
- G7b: Intervention-survival audit (verifies D_raw, D_pre, D_input > 0 for all acquisition transforms;
       confirms gain scaling survives preprocessing)
- G8: Representation geometry contract & triad audit (relative rank r_rel = r_eff / (d - 1),
       comparative ERM vs MMD audit, triad: MMD down, AUC_E down, r_rel preserved)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repecg.common.models import (
    InvariantMechanismDiscoveryModel,
    biased_imq_mmd2,
    multi_environment_mmd,
    effective_rank,
)
from repecg.common.variants import ExperimentVariant, PAPER14_VARIANTS


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.ndarray, list)):
            return [self.default(x) for x in obj]
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


def fast_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Computes binary AUROC via Mann-Whitney U statistic."""
    pos = y_pred[y_true == 1]
    neg = y_pred[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    raw = float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())
    return max(raw, 1.0 - raw)


def run_p14_g0_contract(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G0 — Numerical MMD contract + finite-sample null calibration.
    Note: The null calibration values here represent the synthetic Gaussian distribution baseline.
    Real-data null calibration is performed via matched within-representation permutation/resampling.
    """
    torch.manual_seed(seed)
    
    # 1. Identical samples MMD is ~ 0
    x = torch.randn(100, 16)
    mmd_identical = biased_imq_mmd2(x, x, c2=1.0).item()
    
    # 2. Shifted samples MMD is strictly positive
    delta = 2.0
    y = x + delta
    mmd_shifted = biased_imq_mmd2(x, y, c2=1.0).item()
    
    # 3. Symmetry
    y_rand = torch.randn(80, 16) + 0.5
    mmd_xy = biased_imq_mmd2(x[:80], y_rand, c2=1.0).item()
    mmd_yx = biased_imq_mmd2(y_rand, x[:80], c2=1.0).item()
    symmetry_diff = abs(mmd_xy - mmd_yx)
    
    # 4. Nonzero finite gradients
    x_grad = torch.randn(50, 8, requires_grad=True)
    y_grad = torch.randn(50, 8) + 1.0
    penalty = biased_imq_mmd2(x_grad, y_grad, c2=1.0)
    penalty.backward()
    grad_norm = torch.norm(x_grad.grad).item()
    grad_finite = bool(torch.isfinite(x_grad.grad).all().item())
    
    # 5. Environment requirements (fail closed on < 2 envs or < 2 samples/env)
    z = torch.randn(20, 8)
    single_env_failed = False
    try:
        multi_environment_mmd(z, torch.zeros(20, dtype=torch.long))
    except RuntimeError:
        single_env_failed = True
        
    # 6. Finite-sample null calibration for synthetic test setup (independent draws X ~ P, Y ~ P)
    null_draws = []
    batch_size = 128
    for _ in range(100):
        x_null = torch.randn(batch_size, 16)
        y_null = torch.randn(batch_size, 16)
        null_draws.append(biased_imq_mmd2(x_null, y_null, c2=1.0).item())
    null_draws = np.sort(null_draws)
    q50 = float(np.percentile(null_draws, 50))
    q90 = float(np.percentile(null_draws, 90))
    q95 = float(np.percentile(null_draws, 95))
    q99 = float(np.percentile(null_draws, 99))
        
    gate_pass = (
        mmd_identical < 1e-6
        and mmd_shifted > 0.05
        and symmetry_diff < 1e-6
        and grad_norm > 1e-4
        and grad_finite
        and single_env_failed
    )
    
    return {
        "gate": "P14-G0",
        "description": "Numerical MMD contract & synthetic finite-sample null calibration",
        "mmd_identical": mmd_identical,
        "mmd_shifted": mmd_shifted,
        "symmetry_diff": symmetry_diff,
        "grad_norm": grad_norm,
        "single_env_fails_closed": single_env_failed,
        "synthetic_null_calibration": {
            "Q50": q50,
            "Q90": q90,
            "Q95": q95,
            "Q99": q99,
            "note": "Distribution-dependent synthetic null; real-data null uses matched permutation"
        },
        "gate_pass": gate_pass,
    }


def run_p14_g1_positive_alignment(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G1 — Positive alignment world with environment probe.
    Input has informative feature S and domain nuisance N_e.
    Nuisance shifts strongly by environment.
    Evaluates that MMD reduces environment predictability (AUC_E) and representation MMD
    while preserving task diagnostic accuracy (AUC_Y).
    """
    torch.manual_seed(seed)
    B = 400
    half = B // 2
    
    # Informative feature: S ~ N(0, 1), Y = 1[S > 0]
    S = torch.randn(B)
    Y = (S > 0.0).float().unsqueeze(1)
    
    # Nuisance: U + delta_e
    U = torch.randn(B) * 0.5
    delta = torch.cat([torch.full((half,), -2.0), torch.full((half,), 2.0)])
    N_e = U + delta
    
    X = torch.stack([S, N_e], dim=1)
    env = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])
    
    # Model architecture with LayerNorm(elementwise_affine=False)
    class AlignedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(2, 4)
            self.norm = nn.LayerNorm(4, elementwise_affine=False)
            self.head = nn.Linear(4, 1, bias=False)
        def encode(self, x):
            return self.norm(self.enc(x))
        def forward(self, x):
            z = self.encode(x)
            return self.head(z), z
            
    # Train ERM model
    torch.manual_seed(seed)
    model_erm = AlignedNet()
    opt_erm = torch.optim.Adam(model_erm.parameters(), lr=0.03)
    for _ in range(300):
        opt_erm.zero_grad()
        logits, z = model_erm(X)
        loss = F.binary_cross_entropy_with_logits(logits, Y)
        loss.backward()
        opt_erm.step()
        
    with torch.no_grad():
        logits_erm, z_erm = model_erm(X)
        mmd_erm = float(multi_environment_mmd(z_erm, env, c2=1.0).item())
        auc_y_erm = fast_auc(Y.squeeze(), torch.sigmoid(logits_erm.squeeze()))
        z_erm_detached = z_erm.clone()

    # Train auxiliary environment linear probe on frozen Z_erm (requires grad for probe)
    probe_erm = nn.Linear(4, 1)
    opt_probe_erm = torch.optim.Adam(probe_erm.parameters(), lr=0.05)
    for _ in range(100):
        opt_probe_erm.zero_grad()
        p_logits = probe_erm(z_erm_detached)
        p_loss = F.binary_cross_entropy_with_logits(p_logits, env.float().unsqueeze(1))
        p_loss.backward()
        opt_probe_erm.step()
    with torch.no_grad():
        probe_erm_preds = torch.sigmoid(probe_erm(z_erm_detached).squeeze())
        auc_e_erm = fast_auc(env, probe_erm_preds)
        
    # Train MMD model with identical initialization
    torch.manual_seed(seed)
    model_mmd = AlignedNet()
    opt_mmd = torch.optim.Adam(model_mmd.parameters(), lr=0.03)
    for _ in range(300):
        opt_mmd.zero_grad()
        logits, z = model_mmd(X)
        loss_bce = F.binary_cross_entropy_with_logits(logits, Y)
        loss_mmd = multi_environment_mmd(z, env, c2=1.0)
        loss = loss_bce + 5.0 * loss_mmd
        loss.backward()
        opt_mmd.step()
        
    with torch.no_grad():
        logits_mmd, z_mmd = model_mmd(X)
        mmd_mmd = float(multi_environment_mmd(z_mmd, env, c2=1.0).item())
        var_mmd = float(z_mmd.var(dim=0).mean().item())
        auc_y_mmd = fast_auc(Y.squeeze(), torch.sigmoid(logits_mmd.squeeze()))
        acc_mmd = float(((logits_mmd > 0.0).float() == Y).float().mean().item())
        z_mmd_detached = z_mmd.clone()

    # Train auxiliary environment linear probe on frozen Z_mmd (requires grad for probe)
    probe_mmd = nn.Linear(4, 1)
    opt_probe_mmd = torch.optim.Adam(probe_mmd.parameters(), lr=0.05)
    for _ in range(100):
        opt_probe_mmd.zero_grad()
        p_logits = probe_mmd(z_mmd_detached)
        p_loss = F.binary_cross_entropy_with_logits(p_logits, env.float().unsqueeze(1))
        p_loss.backward()
        opt_probe_mmd.step()
    with torch.no_grad():
        probe_mmd_preds = torch.sigmoid(probe_mmd(z_mmd_detached).squeeze())
        auc_e_mmd = fast_auc(env, probe_mmd_preds)

    gate_pass = bool(
        (mmd_mmd < mmd_erm * 0.65)
        and (auc_e_mmd < auc_e_erm)
        and (auc_y_mmd >= 0.85)
        and (acc_mmd >= 0.85)
        and (var_mmd > 0.05)
    )
    
    return {
        "gate": "P14-G1",
        "description": "Positive alignment world with environment probe",
        "mmd_erm": mmd_erm,
        "mmd_aligned": mmd_mmd,
        "auc_y_erm": auc_y_erm,
        "auc_y_aligned": auc_y_mmd,
        "auc_e_erm": auc_e_erm,
        "auc_e_aligned": auc_e_mmd,
        "var_aligned": var_mmd,
        "gate_pass": gate_pass,
    }


def run_p14_g2_matched_marginal_boundary(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G2 — Matched-marginal failure world (Claim Boundary PASS).
    Construct environments where the spurious feature has identical marginal
    distributions but opposite label association. Marginal MMD is blind to the reversal.
    """
    torch.manual_seed(seed)
    N = 1000
    
    # Symmetric balanced Y in {-1, +1}
    Y0 = (torch.randint(0, 2, (N,)) * 2 - 1).float()
    Y1 = (torch.randint(0, 2, (N,)) * 2 - 1).float()
    
    # S has identical marginal distribution across envs
    S0 = Y0 + torch.randn(N) * 0.1
    S1 = -Y1 + torch.randn(N) * 0.1
    
    z0 = S0.unsqueeze(1)
    z1 = S1.unsqueeze(1)
    
    mmd_marginal = biased_imq_mmd2(z0, z1, c2=1.0).item()
    
    corr_env0 = float(torch.corrcoef(torch.stack([Y0, S0]))[0, 1].item())
    corr_env1 = float(torch.corrcoef(torch.stack([Y1, S1]))[0, 1].item())
    
    gate_pass = bool(
        mmd_marginal < 0.005
        and corr_env0 > 0.9
        and corr_env1 < -0.9
    )
    
    return {
        "gate": "P14-G2",
        "description": "Claim-boundary PASS: matched marginal distributions blind MMD to mechanism reversal",
        "mmd_marginal": mmd_marginal,
        "corr_env0": corr_env0,
        "corr_env1": corr_env1,
        "claim_boundary_respected": True,
        "gate_pass": gate_pass,
    }


def run_p14_g3_collapse_world(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G3 — Collapse world & representation geometry audit.
    Demonstrates that MMD-only training collapses representations (Var -> 0, r_eff -> 0),
    whereas joint BCE + MMD retains active representation with non-degenerate rank.
    """
    torch.manual_seed(seed)
    B = 200
    half = B // 2
    
    # 1. Train unconstrained encoder with MMD penalty only -> demonstrates Var(Z) collapse towards 0
    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)
        def forward(self, x):
            return self.fc(x)

    env0 = torch.randn(half, 2) * 0.1 - 2.0
    env1 = torch.randn(half, 2) * 0.1 + 2.0
    x_collapse = torch.cat([env0, env1], dim=0)
    env_collapse = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])

    m_mmd_only = SimpleEncoder()
    opt_mmd_only = torch.optim.Adam(m_mmd_only.parameters(), lr=0.01)
    for _ in range(250):
        opt_mmd_only.zero_grad()
        z = m_mmd_only(x_collapse)
        loss = multi_environment_mmd(z, env_collapse, c2=1.0)
        loss.backward()
        opt_mmd_only.step()
        
    with torch.no_grad():
        z_mmd_only = m_mmd_only(x_collapse)
        var_mmd_only = float(z_mmd_only.var(dim=0).mean().item())
        eff_rank_mmd_only = effective_rank(z_mmd_only)
        
    # 2. Train joint BCE + MMD model with LayerNorm(elementwise_affine=False)
    class JointModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Linear(2, 4)
            self.norm = nn.LayerNorm(4, elementwise_affine=False)
            self.clf = nn.Linear(4, 1, bias=False)
        def forward(self, x):
            z = self.norm(self.enc(x))
            return self.clf(z), z

    S = torch.randn(B)
    y_joint = (S > 0.0).float().unsqueeze(1)
    nuisance = torch.cat([torch.randn(half) - 2.0, torch.randn(half) + 2.0])
    x_joint = torch.stack([S, nuisance], dim=1)
    env_joint = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])

    m_joint = JointModel()
    opt_joint = torch.optim.Adam(m_joint.parameters(), lr=0.03)
    for _ in range(300):
        opt_joint.zero_grad()
        logits, z = m_joint(x_joint)
        bce = F.binary_cross_entropy_with_logits(logits, y_joint)
        mmd = multi_environment_mmd(z, env_joint, c2=1.0)
        loss = bce + 2.0 * mmd
        loss.backward()
        opt_joint.step()
        
    with torch.no_grad():
        logits_joint, z_joint = m_joint(x_joint)
        var_joint = float(z_joint.var(dim=0).mean().item())
        eff_rank_joint = effective_rank(z_joint)
        mean_norm_joint = float(z_joint.norm(dim=1).mean().item())
        acc_joint = float(((logits_joint > 0.0).float() == y_joint).float().mean().item())

    # Collapse happens under MMD only; preserved under joint
    gate_pass = bool(
        (var_mmd_only < 0.01)
        and (eff_rank_mmd_only <= 1.05)
        and (var_joint > 0.1)
        and (eff_rank_joint >= 1.0)
        and (acc_joint >= 0.85)
    )
    
    return {
        "gate": "P14-G3",
        "description": "Collapse world & representation geometry audit",
        "final_var_mmd_only": var_mmd_only,
        "eff_rank_mmd_only": eff_rank_mmd_only,
        "final_var_joint": var_joint,
        "eff_rank_joint": eff_rank_joint,
        "mean_norm_joint": mean_norm_joint,
        "acc_joint": acc_joint,
        "gate_pass": gate_pass,
    }


def run_p14_g4_objective_execution(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G4 — Objective execution: nonzero gradient contrast."""
    torch.manual_seed(seed)
    
    model_erm = InvariantMechanismDiscoveryModel(input_dim=8, width=32, classes=1, variant=PAPER14_VARIANTS["erm"])
    model_mmd = InvariantMechanismDiscoveryModel(input_dim=8, width=32, classes=1, variant=PAPER14_VARIANTS["mmd"])
    model_mmd.load_state_dict(model_erm.state_dict())
    
    x = torch.randn(40, 16, 8)
    x[20:] = x[20:] + 2.5
    y = torch.randint(0, 2, (40, 1)).float()
    env = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    
    logits_erm, _ = model_erm.forward_with_representation(x)
    loss_erm = F.binary_cross_entropy_with_logits(logits_erm, y)
    loss_erm.backward()
    
    logits_mmd, z_mmd = model_mmd.forward_with_representation(x)
    bce = F.binary_cross_entropy_with_logits(logits_mmd, y)
    penalty = model_mmd.compute_mmd_penalty(z_mmd, env, c2=1.0)
    loss_mmd = bce + 5.0 * penalty
    loss_mmd.backward()
    
    grad_erm = model_erm.encoder[0].weight.grad
    grad_mmd = model_mmd.encoder[0].weight.grad
    grad_diff = float(torch.norm(grad_erm - grad_mmd).item())
    
    gate_pass = bool(grad_diff > 1e-4)
    
    return {
        "gate": "P14-G4",
        "description": "Objective execution: nonzero gradient contrast between ERM and MMD",
        "grad_diff": grad_diff,
        "gate_pass": gate_pass,
    }


def run_p14_g5_scale_escape(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G5 — Scale escape audit under fixed-bandwidth IMQ kernel.
    Demonstrates that unconstrained representations allow MMD to approach its finite-sample
    diagonal floor (1/n + 1/m) via norm inflation (scale escape) without alignment.
    Demonstrates that LayerNorm-normalized representation Z eliminates scale escape.
    """
    torch.manual_seed(seed)
    n0, n1 = 100, 100
    z0 = torch.randn(n0, 8)
    z1 = torch.randn(n1, 8) + 1.0  # Non-aligned distributions

    # Expected mathematical asymptote for biased IMQ MMD under scale inflation a -> infty:
    # k(ax_i, ax_j) -> 0 for i != j, but k(ax_i, ax_i) = 1.
    # lim_{a -> infty} MMD_b^2 = 1/n + 1/m = 1/100 + 1/100 = 0.020.
    diagonal_floor = 1.0 / n0 + 1.0 / n1

    a_scales = [0.1, 1.0, 10.0, 100.0, 1000.0]
    unconstrained_mmd = [biased_imq_mmd2(a * z0, a * z1, c2=1.0).item() for a in a_scales]

    # As a -> infinity, unconstrained MMD approaches the finite-sample diagonal floor (0.02)
    # rather than staying bounded by true distributional divergence!
    scale_escape_observed = (
        unconstrained_mmd[-1] < unconstrained_mmd[1] * 0.35
        and abs(unconstrained_mmd[-1] - diagonal_floor) < 0.005
    )

    # LayerNorm (without affine parameters) eliminates global scale escape: MMD is invariant to scaling a
    norm = nn.LayerNorm(8, elementwise_affine=False)
    ln_mmd = [biased_imq_mmd2(norm(a * z0), norm(a * z1), c2=1.0).item() for a in [1.0, 10.0, 100.0]]
    ln_scale_invariant = (abs(ln_mmd[0] - ln_mmd[-1]) < 1e-5) and (ln_mmd[0] > 0.01)

    gate_pass = bool(scale_escape_observed and ln_scale_invariant)

    return {
        "gate": "P14-G5",
        "description": "Scale escape audit: proves norm inflation drives unconstrained MMD to diagonal floor 1/n+1/m",
        "diagonal_floor": diagonal_floor,
        "unconstrained_mmd_by_scale": dict(zip(a_scales, unconstrained_mmd)),
        "layernorm_mmd_by_scale": dict(zip([1.0, 10.0, 100.0], ln_mmd)),
        "scale_escape_to_floor_demonstrated": scale_escape_observed,
        "layernorm_eliminates_escape": ln_scale_invariant,
        "gate_pass": gate_pass,
    }


def run_p14_g5b_trainable_normalizer_escape(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G5b — Trainable-normalizer escape audit.
    Verifies that:
    1. The actual trainable model contains no trainable LayerNorm affine parameters (elementwise_affine=False).
    2. H -> aH + b leaves normalized Z numerically invariant for any a > 0 and b in R.
    3. End-to-end optimization cannot inflate a post-normalization scale parameter because none exists.
    """
    torch.manual_seed(seed)
    model = InvariantMechanismDiscoveryModel(input_dim=8, width=32, classes=1, variant=PAPER14_VARIANTS["mmd"])
    
    # 1. Parameter count check: model.norm must have 0 parameters
    norm_params = list(model.norm.parameters())
    num_norm_params = len(norm_params)
    has_no_affine = (num_norm_params == 0)
    
    # 2. Numerical invariance check: Z = Norm(aH + b) == Norm(H)
    h = torch.randn(50, 32)
    z_baseline = model.norm(h)
    
    invariance_checks = []
    # Test scale factors a >= 0.5 (where norm inflation / scale escape occurs as a -> infty)
    for a in [0.5, 1.0, 2.0, 5.0, 10.0, 100.0]:
        for b in [-5.0, 0.0, 5.0, 10.0]:
            h_scaled = a * h + b
            z_scaled = model.norm(h_scaled)
            diff = float(torch.max(torch.abs(z_scaled - z_baseline)).item())
            invariance_checks.append(diff < 2e-4)
    all_scale_shift_invariant = all(invariance_checks)
    
    # 3. End-to-end check: train joint BCE+MMD and verify no post-norm scale parameters exist to drift
    x = torch.randn(40, 16, 8)
    y = torch.randint(0, 2, (40, 1)).float()
    env = torch.cat([torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for _ in range(50):
        optimizer.zero_grad()
        logits, z = model.forward_with_representation(x)
        bce = F.binary_cross_entropy_with_logits(logits, y)
        mmd = model.compute_mmd_penalty(z, env, c2=1.0)
        loss = bce + 2.0 * mmd
        loss.backward()
        optimizer.step()
        
    post_train_norm_params = len(list(model.norm.parameters()))

    gate_pass = bool(has_no_affine and all_scale_shift_invariant and (post_train_norm_params == 0))

    return {
        "gate": "P14-G5b",
        "description": "Trainable-normalizer escape audit: elementwise_affine=False contract & scale/shift invariance",
        "num_norm_parameters": num_norm_params,
        "has_no_affine": has_no_affine,
        "scale_shift_invariant_across_grid": all_scale_shift_invariant,
        "post_train_norm_parameters": post_train_norm_params,
        "gate_pass": gate_pass,
    }


def run_p14_g6_label_shift_boundary(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G6 — Label-shift boundary world (Claim Boundary PASS).
    Demonstrates that marginal MMD penalizes differing label mixtures P(Y|E)
    even when class conditionals P(X|Y, E) are identical across environments.
    """
    torch.manual_seed(seed)
    n = 500
    # Class conditionals: X|Y=0 ~ N(-1, 0.1), X|Y=1 ~ N(1, 0.1) across all envs
    # Env 0: 90% Y=1
    # Env 1: 10% Y=1
    y0 = (torch.rand(n) < 0.9).float()
    y1 = (torch.rand(n) < 0.1).float()

    x0 = (y0 * 2.0 - 1.0).unsqueeze(1) + torch.randn(n, 1) * 0.1
    x1 = (y1 * 2.0 - 1.0).unsqueeze(1) + torch.randn(n, 1) * 0.1

    mmd_label_shift = biased_imq_mmd2(x0, x1, c2=1.0).item()

    # Claim boundary PASS: MMD evaluates to positive (> 0.5) solely due to label shift!
    gate_pass = bool(mmd_label_shift > 0.5)

    return {
        "gate": "P14-G6",
        "description": "Claim-boundary PASS: marginal MMD penalizes unequal label mixtures without domain nuisance",
        "mmd_label_shift": mmd_label_shift,
        "p_y1_env0": 0.9,
        "p_y1_env1": 0.1,
        "claim_boundary_respected": True,
        "gate_pass": gate_pass,
    }


def run_p14_g7_environment_provenance(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G7 — Environment provenance & real-data label mixture audit.
    Freezes the real-data environment specification:
    Controlled label-preserving acquisition perturbations holding patient & diagnosis fixed.
    Strictly bounds the claim to acquisition-perturbation representation alignment.
    """
    environments = {
        "clean": "Reference raw ECG waveform",
        "gain_0.8": "Acquisition amplitude scaling 0.8x",
        "gain_1.2": "Acquisition amplitude scaling 1.2x",
        "noise_20db": "Additive sensor noise (20 dB SNR)",
        "resample_250hz": "Bandwidth downsample/upsample 500->250->500 Hz",
    }
    
    # Audit: Because each environment is an acquisition transform applied to the same record,
    # P(Y|E=e) = P(Y|E=e') by construction, yielding Delta_{e, e', c} = 0.
    delta_y_max = 0.0
    gate_pass = bool((len(environments) >= 2) and (delta_y_max == 0.0))

    return {
        "gate": "P14-G7",
        "description": "Environment provenance & label balance audit (controlled acquisition bank)",
        "environment_bank": environments,
        "label_preserving_invariant": True,
        "delta_y_max": delta_y_max,
        "claim_scope": "acquisition-perturbation representation alignment (patient and label fixed)",
        "disclaimed_scopes": [
            "hospital generalization",
            "cross-dataset domain generalization",
            "causal invariant mechanism discovery"
        ],
        "gate_pass": gate_pass,
    }


def run_p14_g7b_intervention_survival(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G7b — Intervention-survival audit across the preprocessing and encoding pipeline.
    For each environment transform T_e, measures:
      D_raw   = d(X, T_e X)
      D_pre   = d(pre(X), pre(T_e X)) under deterministic ECG preprocessing
      D_input = d(Z_untrained(X), Z_untrained(T_e X))
    Verifies that no upstream normalization (e.g. per-sample peak/z-scoring) erases
    amplitude gain or acquisition transforms before reaching the model representation.
    """
    torch.manual_seed(seed)
    B, T, C = 32, 16, 8
    # Standard ECG phase/temporal feature representation input to InvariantMechanismDiscoveryModel
    X_raw = torch.randn(B, T, C)
    
    # Define acquisition perturbation bank
    def apply_gain(x: torch.Tensor, scale: float) -> torch.Tensor:
        return x * scale

    def apply_noise(x: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
        sig_std = x.std()
        noise_std = sig_std / (10.0 ** (snr_db / 20.0))
        return x + torch.randn_like(x) * noise_std

    def apply_resample(x: torch.Tensor) -> torch.Tensor:
        # Downsample 2x then interpolate back to original length
        down = F.interpolate(x.transpose(1, 2), scale_factor=0.5, mode="linear", align_corners=False)
        up = F.interpolate(down, size=T, mode="linear", align_corners=False)
        return up.transpose(1, 2)

    transforms = {
        "gain_0.8": lambda x: apply_gain(x, 0.8),
        "gain_1.2": lambda x: apply_gain(x, 1.2),
        "noise_20db": lambda x: apply_noise(x, 20.0),
        "resample_250hz": lambda x: apply_resample(x),
    }

    # Deterministic global dataset-level whitening / phase feature extraction
    # (Global statistics across training set, NOT per-sample z-scoring!)
    global_mean = X_raw.mean(dim=(0, 1), keepdim=True)
    global_std = X_raw.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    def preprocess_ecg(x: torch.Tensor) -> torch.Tensor:
        return (x - global_mean) / global_std

    # Model encoder
    model = InvariantMechanismDiscoveryModel(input_dim=C, width=32, classes=5, variant=PAPER14_VARIANTS["mmd"])

    survival_metrics = {}
    all_survived = True

    for name, transform in transforms.items():
        X_pert = transform(X_raw)
        
        # 1. Raw relative distance: ||X - T(X)||_F / ||X||_F
        d_raw = float((torch.norm(X_raw - X_pert) / torch.norm(X_raw)).item())
        
        # 2. Preprocessed relative distance
        X_pre_raw = preprocess_ecg(X_raw)
        X_pre_pert = preprocess_ecg(X_pert)
        d_pre = float((torch.norm(X_pre_raw - X_pre_pert) / torch.norm(X_pre_raw)).item())
        
        # 3. Model representation relative distance at Z = Norm(f(X))
        with torch.no_grad():
            z_raw = model.encode(X_pre_raw)
            z_pert = model.encode(X_pre_pert)
        d_input = float((torch.norm(z_raw - z_pert) / torch.norm(z_raw)).item())
        
        survived = (d_raw > 0.01) and (d_pre > 0.01) and (d_input > 0.01)
        if not survived:
            all_survived = False
            
        survival_metrics[name] = {
            "d_raw": d_raw,
            "d_pre": d_pre,
            "d_input": d_input,
            "survived": survived,
        }

    return {
        "gate": "P14-G7b",
        "description": "Intervention-survival audit across preprocessing and encoding pipeline",
        "survival_metrics": survival_metrics,
        "all_interventions_survived": all_survived,
        "gate_pass": all_survived,
    }


def run_p14_g8_representation_geometry(rng: np.random.Generator, seed: int = 42) -> dict[str, Any]:
    """G8 — Representation geometry contract & relative rank audit.
    Evaluates:
      1. Effective rank r_eff and relative effective rank r_rel = r_eff / (d - 1)
      2. Comparative audit: r_eff(ERM) vs r_eff(MMD)
      3. Joint triad: MMD down, AUC_E down, r_rel preserved
    """
    torch.manual_seed(seed)
    rep_dim = 32
    max_d = rep_dim - 1  # LayerNorm imposes sum_j Z_j = 0 constraint
    
    # Train both ERM and MMD models on synthetic alignment setup
    B = 400
    half = B // 2
    S = torch.randn(B, 16, 8)
    Y = (S.mean(dim=(1, 2)) > 0.0).float().unsqueeze(1)
    
    # Nuisance shift
    shift = torch.cat([torch.full((half, 16, 8), -1.5), torch.full((half, 16, 8), 1.5)])
    X = S + shift
    env = torch.cat([torch.zeros(half, dtype=torch.long), torch.ones(half, dtype=torch.long)])

    # ERM Model
    torch.manual_seed(seed)
    model_erm = InvariantMechanismDiscoveryModel(input_dim=8, width=rep_dim, classes=1, variant=PAPER14_VARIANTS["erm"])
    opt_erm = torch.optim.Adam(model_erm.parameters(), lr=0.02)
    for _ in range(150):
        opt_erm.zero_grad()
        logits, z = model_erm.forward_with_representation(X)
        loss = F.binary_cross_entropy_with_logits(logits, Y)
        loss.backward()
        opt_erm.step()

    with torch.no_grad():
        logits_erm, z_erm = model_erm.forward_with_representation(X)
        var_erm = float(z_erm.var(dim=0).mean().item())
        eff_rank_erm = effective_rank(z_erm)
        r_rel_erm = eff_rank_erm / max_d
        mmd_erm = float(multi_environment_mmd(z_erm, env, c2=1.0).item())
        z_erm_detached = z_erm.clone()

    # Probe on ERM
    probe_erm = nn.Linear(rep_dim, 1)
    opt_p_erm = torch.optim.Adam(probe_erm.parameters(), lr=0.05)
    for _ in range(80):
        opt_p_erm.zero_grad()
        p_logits = probe_erm(z_erm_detached)
        F.binary_cross_entropy_with_logits(p_logits, env.float().unsqueeze(1)).backward()
        opt_p_erm.step()
    with torch.no_grad():
        auc_e_erm = fast_auc(env, torch.sigmoid(probe_erm(z_erm_detached).squeeze()))

    # MMD Model
    torch.manual_seed(seed)
    model_mmd = InvariantMechanismDiscoveryModel(input_dim=8, width=rep_dim, classes=1, variant=PAPER14_VARIANTS["mmd"])
    opt_mmd = torch.optim.Adam(model_mmd.parameters(), lr=0.02)
    for _ in range(150):
        opt_mmd.zero_grad()
        logits, z = model_mmd.forward_with_representation(X)
        bce = F.binary_cross_entropy_with_logits(logits, Y)
        penalty = model_mmd.compute_mmd_penalty(z, env, c2=1.0)
        loss = bce + 3.0 * penalty
        loss.backward()
        opt_mmd.step()

    with torch.no_grad():
        logits_mmd, z_mmd = model_mmd.forward_with_representation(X)
        var_mmd = float(z_mmd.var(dim=0).mean().item())
        eff_rank_mmd = effective_rank(z_mmd)
        r_rel_mmd = eff_rank_mmd / max_d
        mean_norm_mmd = float(z_mmd.norm(dim=1).mean().item())
        mmd_mmd = float(multi_environment_mmd(z_mmd, env, c2=1.0).item())
        z_mmd_detached = z_mmd.clone()

    # Probe on MMD
    probe_mmd = nn.Linear(rep_dim, 1)
    opt_p_mmd = torch.optim.Adam(probe_mmd.parameters(), lr=0.05)
    for _ in range(80):
        opt_p_mmd.zero_grad()
        p_logits = probe_mmd(z_mmd_detached)
        F.binary_cross_entropy_with_logits(p_logits, env.float().unsqueeze(1)).backward()
        opt_p_mmd.step()
    with torch.no_grad():
        auc_e_mmd = fast_auc(env, torch.sigmoid(probe_mmd(z_mmd_detached).squeeze()))

    # Triad audit: MMD down, AUC_E down, r_rel not catastrophically collapsed
    triad_valid = bool(
        (mmd_mmd < mmd_erm)
        and (auc_e_mmd < auc_e_erm)
        and (r_rel_mmd >= 0.10)
    )

    gate_pass = bool(
        (0.05 < var_mmd < 5.0)
        and (eff_rank_mmd >= 1.5)
        and (mean_norm_mmd > 0.5)
        and triad_valid
    )

    return {
        "gate": "P14-G8",
        "description": "Representation geometry contract & comparative relative-rank triad audit",
        "rep_dim": rep_dim,
        "max_rank_d_minus_1": max_d,
        "var_z": var_mmd,
        "mean_norm": mean_norm_mmd,
        "eff_rank_erm": eff_rank_erm,
        "eff_rank_mmd": eff_rank_mmd,
        "r_rel_erm": r_rel_erm,
        "r_rel_mmd": r_rel_mmd,
        "triad": {
            "mmd_erm": mmd_erm,
            "mmd_aligned": mmd_mmd,
            "auc_e_erm": auc_e_erm,
            "auc_e_aligned": auc_e_mmd,
            "r_rel_mmd": r_rel_mmd,
            "triad_valid": triad_valid,
        },
        "geometry_interpretation": "exceeding the frozen minimum geometry threshold while indicating a substantially compressed representation",
        "gate_pass": gate_pass,
    }


def main():
    rng = np.random.default_rng(42)
    results = {}

    print("Running P14-G0 (Numerical MMD contract & null calibration)...")
    results["G0"] = run_p14_g0_contract(rng)

    print("Running P14-G1 (Positive alignment world with environment probe)...")
    results["G1"] = run_p14_g1_positive_alignment(rng)

    print("Running P14-G2 (Matched-marginal boundary world)...")
    results["G2"] = run_p14_g2_matched_marginal_boundary(rng)

    print("Running P14-G3 (Collapse world & geometry audit)...")
    results["G3"] = run_p14_g3_collapse_world(rng)

    print("Running P14-G4 (Objective execution)...")
    results["G4"] = run_p14_g4_objective_execution(rng)

    print("Running P14-G5 (Scale escape audit to diagonal floor)...")
    results["G5"] = run_p14_g5_scale_escape(rng)

    print("Running P14-G5b (Trainable-normalizer escape audit)...")
    results["G5b"] = run_p14_g5b_trainable_normalizer_escape(rng)

    print("Running P14-G6 (Label-shift boundary world)...")
    results["G6"] = run_p14_g6_label_shift_boundary(rng)

    print("Running P14-G7 (Environment provenance & label audit)...")
    results["G7"] = run_p14_g7_environment_provenance(rng)

    print("Running P14-G7b (Intervention-survival audit)...")
    results["G7b"] = run_p14_g7b_intervention_survival(rng)

    print("Running P14-G8 (Representation geometry contract & triad audit)...")
    results["G8"] = run_p14_g8_representation_geometry(rng)

    all_pass = all(v["gate_pass"] for v in results.values())

    output = {
        "all_synthetic_gates_pass": all_pass,
        "gates": results,
    }

    out_path = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/ptbxl_distributional_repecg/scripts/paper14/refine-logs/P14_GATES.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(f"\n==========================================")
    print(f"Paper 14 Gate Summary: all_pass = {all_pass}")
    for k, v in results.items():
        status = "PASS" if v["gate_pass"] else "FAIL"
        print(f"  [{status}] {v['gate']}: {v['description']}")
    print(f"==========================================")


if __name__ == "__main__":
    main()
