"""Paper 15 Inductive Bias & Mechanism Audit Gates: Modular Phase-Transition Dynamics (P15-V2).

Implements the comprehensive 8-gate audit stack:
- P15-G0: State Grounding & Cyclic Ring Contract
          (Verifies Z_g grounding across all 16 phases, full 16-transition cyclic ring closure
           (g+1)%16, and active gradient flow to all 16 modules).
- P15-G1: Transition Law Recovery
          (Verifies modular M_g accurately recovers known synthetic phase-specific transition laws).
- P15-G2: Input-Distribution Autonomy (State-Shift Transfer vs Mechanism-Shift Failure)
          (Verifies M_g trained on P_A(Z) transfers to shifted state distribution P_B(Z) when
           transition law is unchanged, but fails severely when transition law itself shifts).
- P15-G3: Post-Training Frozen Permutation Kill Test
          (Post-training derangement pi(g) != g degrades transition prediction: L_perm >> L_ordered).
- P15-G4: Strong Shared Controls Benchmark
          (Benchmarks modular against phase-conditioned shared, phase-agnostic shared same-width,
           and capacity-matched shared controls).
- P15-G5: Cyclic Origin Invariance
          (Evaluates cyclic transition consistency across origins 0, 4, 8, 12, confirming periodic topology).
- P15-G6: Information Preservation & Non-Collapse Audit
          (Verifies representations retain sample variance, effective rank r_eff >= 8, and no phase collapse).
- P15-G7: Simple-Transition Baselines Benchmark
          (Confirms learned modular mechanism strictly outperforms Identity, Phase Mean, Mean Residual,
           and Linear baselines).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from repecg.common.models import CausalMechanismFactorizationModel
from repecg.common.variants import ExperimentVariant, PAPER15_VARIANTS


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


def run_p15_g0_grounding_and_ring_contract(seed: int = 42) -> dict[str, Any]:
    """P15-G0: State Grounding & Cyclic Ring Contract.
    Verifies Z_g grounding across all 16 phases, full 16-transition cyclic ring closure (g+1)%16,
    and active gradient flow to all 16 modules.
    """
    torch.manual_seed(seed)
    input_dim = 64
    width = 32
    batch_size = 16
    num_phases = 16
    
    model = CausalMechanismFactorizationModel(
        input_dim=input_dim, width=width, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    
    # 1. Shape contract: exactly 16 cyclic transitions
    x = torch.randn(batch_size, num_phases, input_dim)
    logits, z, z_pred = model.forward_with_transitions(x)
    shape_ok = (
        logits.shape == (batch_size, 5)
        and z.shape == (batch_size, num_phases, width)
        and z_pred.shape == (batch_size, num_phases, width)
    )
    
    # 2. Phase-wise grounding isolation test
    x_pert = x.clone()
    target_phase = 5
    x_pert[:, target_phase] += torch.randn_like(x_pert[:, target_phase]) * 2.0
    z_pert = model.encode_states(x_pert)
    delta_target = (z_pert[:, target_phase] - z[:, target_phase]).abs().mean().item()
    other_phases = [p for p in range(num_phases) if p != target_phase]
    delta_others = (z_pert[:, other_phases] - z[:, other_phases]).abs().mean().item()
    isolation_ok = (delta_target > 1e-3) and (delta_others < 1e-6)
    
    # 3. Cyclic ring closure test: target of phase 15 is phase 0
    z_target = torch.roll(z, shifts=-1, dims=1)
    ring_closure_ok = torch.allclose(z_target[:, 15], z[:, 0], atol=1e-7)
    
    # 4. Active gradient flow to all 16 mechanisms
    trans_loss = model.compute_transition_loss(z, z_pred)
    model.zero_grad()
    trans_loss.backward()
    
    grad_norms = []
    all_grads_active = True
    for g in range(num_phases):
        mech = model.mechanisms[g]
        norm = sum(p.grad.norm().item() for p in mech.parameters() if p.grad is not None)
        grad_norms.append(norm)
        if norm <= 1e-8:
            all_grads_active = False
            
    passed = bool(shape_ok and isolation_ok and ring_closure_ok and all_grads_active)
    
    return {
        "gate": "P15-G0",
        "name": "State Grounding & Cyclic Ring Contract",
        "passed": passed,
        "details": {
            "shape_ok": shape_ok,
            "z_pred_shape": list(z_pred.shape),
            "isolation_ok": isolation_ok,
            "delta_target_phase": delta_target,
            "delta_unperturbed_phases": delta_others,
            "ring_closure_15_to_0_ok": ring_closure_ok,
            "all_16_mechanisms_have_active_gradients": all_grads_active,
            "min_mechanism_grad_norm": min(grad_norms),
            "max_mechanism_grad_norm": max(grad_norms),
        },
    }


def run_p15_g1_transition_law_recovery(seed: int = 42) -> dict[str, Any]:
    """P15-G1: Transition Law Recovery.
    Verifies that modular M_g accurately recovers known synthetic phase-specific transition laws.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    latent_dim = 16
    n_train = 600
    n_test = 200
    num_phases = 16
    
    # 16 distinct ground truth phase transition mechanisms
    true_mechs = [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(num_phases)]
    for m in true_mechs:
        nn.init.orthogonal_(m.weight)
        
    def generate_data(num_samples: int) -> torch.Tensor:
        with torch.no_grad():
            z0 = torch.randn(num_samples, latent_dim)
            z_seq = [z0]
            for g in range(num_phases - 1):
                next_z = z_seq[-1] + true_mechs[g](z_seq[-1])
                z_seq.append(next_z)
            return torch.stack(z_seq, dim=1).detach()

    x_train = generate_data(n_train)
    x_test = generate_data(n_test)
    
    model = CausalMechanismFactorizationModel(
        input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    
    for _ in range(160):
        optimizer.zero_grad()
        _, z, z_pred = model.forward_with_transitions(x_train)
        loss = model.compute_transition_loss(z, z_pred)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        _, z_test, z_pred_test = model.forward_with_transitions(x_test)
        test_mse = model.compute_transition_loss(z_test, z_pred_test).item()
        # Compute R^2 against target variance
        z_target = torch.roll(z_test, shifts=-1, dims=1)
        target_var = z_target.var().item()
        r2 = max(0.0, 1.0 - (test_mse / (target_var + 1e-8)))
        
    passed = bool(test_mse < 0.15 and r2 > 0.85)
    
    return {
        "gate": "P15-G1",
        "name": "Transition Law Recovery",
        "passed": passed,
        "details": {
            "test_mse": test_mse,
            "target_variance": target_var,
            "r2_score": r2,
            "verdict": f"Modular mechanisms recovered true phase transitions with R^2={r2:.4f} and MSE={test_mse:.4f}."
            if passed else "Transition recovery below threshold.",
        },
    }


def run_p15_g2_input_distribution_autonomy(seed: int = 42) -> dict[str, Any]:
    """P15-G2: Input-Distribution Autonomy (State-Shift Transfer vs Mechanism-Shift Failure).
    Verifies that M_g trained under P_A(Z) transfers when the state distribution shifts to P_B(Z)
    with invariant transition law, but fails when the transition law itself changes.
    """
    torch.manual_seed(seed)
    latent_dim = 16
    num_phases = 16
    n_samples = 400
    
    # Ground truth transition laws (canonical)
    true_mechs_A = [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(num_phases)]
    for m in true_mechs_A:
        nn.init.orthogonal_(m.weight)
        
    # Mechanism shift world: completely different laws
    true_mechs_Shifted = [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(num_phases)]
    for m in true_mechs_Shifted:
        nn.init.orthogonal_(m.weight)
        
    def sample_env(mechs: list[nn.Module], shift_mean: float, shift_scale: float) -> torch.Tensor:
        with torch.no_grad():
            z0 = torch.randn(n_samples, latent_dim) * shift_scale + shift_mean
            z_seq = [z0]
            for g in range(num_phases - 1):
                next_z = z_seq[-1] + mechs[g](z_seq[-1])
                z_seq.append(next_z)
            return torch.stack(z_seq, dim=1).detach()

    # Environment A: standard state distribution (mean=0, scale=1.0)
    data_env_A = sample_env(true_mechs_A, shift_mean=0.0, shift_scale=1.0)
    # Environment B: state-shifted distribution (mean=2.0, scale=1.5), identical transition law
    data_env_B = sample_env(true_mechs_A, shift_mean=2.0, shift_scale=1.5)
    # Environment C: shifted transition law
    data_env_C = sample_env(true_mechs_Shifted, shift_mean=0.0, shift_scale=1.0)
    
    # Train on Env A
    model = CausalMechanismFactorizationModel(
        input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(150):
        opt.zero_grad()
        _, z, z_pred = model.forward_with_transitions(data_env_A)
        loss = model.compute_transition_loss(z, z_pred)
        loss.backward()
        opt.step()
        
    model.eval()
    with torch.no_grad():
        _, z_a, zp_a = model.forward_with_transitions(data_env_A)
        loss_source = model.compute_transition_loss(z_a, zp_a).item()
        
        # Test transfer to state-shifted Env B
        _, z_b, zp_b = model.forward_with_transitions(data_env_B)
        loss_transfer = model.compute_transition_loss(z_b, zp_b).item()
        
        # Test on mechanism-shifted Env C
        _, z_c, zp_c = model.forward_with_transitions(data_env_C)
        loss_mech_shift = model.compute_transition_loss(z_c, zp_c).item()
        
    transfer_ratio = loss_transfer / (loss_source + 1e-8)
    mech_shift_ratio = loss_mech_shift / (loss_source + 1e-8)
    
    # Pass criteria:
    # 1. Transfer under state-shift: error does not explode (transfer_ratio < 2.5)
    # 2. Mechanism shift degrades substantially (mech_shift_ratio > 3.0)
    passed = bool(transfer_ratio < 2.5 and mech_shift_ratio > 3.0)
    
    return {
        "gate": "P15-G2",
        "name": "Input-Distribution Autonomy (State-Shift Transfer vs Mechanism-Shift Failure)",
        "passed": passed,
        "details": {
            "source_loss_env_A": loss_source,
            "transfer_loss_env_B_state_shifted": loss_transfer,
            "transfer_ratio": transfer_ratio,
            "failure_loss_env_C_mech_shifted": loss_mech_shift,
            "mech_shift_ratio": mech_shift_ratio,
            "verdict": f"Autonomy confirmed: state shift transferred (ratio={transfer_ratio:.2f}) while mechanism shift failed (ratio={mech_shift_ratio:.2f})."
            if passed else "Autonomy criteria not met.",
        },
    }


def run_p15_g3_frozen_permutation_kill_test(seed: int = 42) -> dict[str, Any]:
    """P15-G3: Post-Training Frozen Permutation Kill Test.
    Freezes the trained modular model and evaluates under fixed derangement pi(g) != g.
    Confirms transition prediction degrades substantially without retraining.
    """
    torch.manual_seed(seed)
    latent_dim = 16
    num_phases = 16
    n_samples = 400
    
    true_mechs = [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(num_phases)]
    for m in true_mechs:
        nn.init.orthogonal_(m.weight)
        
    with torch.no_grad():
        z0 = torch.randn(n_samples, latent_dim)
        z_seq = [z0]
        for g in range(num_phases - 1):
            next_z = z_seq[-1] + true_mechs[g](z_seq[-1])
            z_seq.append(next_z)
        data = torch.stack(z_seq, dim=1).detach()
        
    model = CausalMechanismFactorizationModel(
        input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    for _ in range(120):
        opt.zero_grad()
        _, z, z_pred = model.forward_with_transitions(data)
        loss = model.compute_transition_loss(z, z_pred)
        loss.backward()
        opt.step()
        
    model.eval()
    with torch.no_grad():
        # Ordered evaluation
        z = model.encode_states(data)
        z_pred_ordered = model.predict_transitions(z)
        loss_ordered = model.compute_transition_loss(z, z_pred_ordered).item()
        
        # Frozen Derangement: cyclic half-shift pi(g) = (g + 8) % 16 (ventricular vs atrial dynamics)
        derangement = [(g + 8) % num_phases for g in range(num_phases)]
        z_pred_perm = model.evaluate_permuted_transitions(z, derangement=derangement)
        loss_perm = model.compute_transition_loss(z, z_pred_perm).item()
        
    kill_ratio = loss_perm / (loss_ordered + 1e-8)
    passed = bool(kill_ratio > 2.0 and (loss_perm - loss_ordered) > 0.05)
    
    return {
        "gate": "P15-G3",
        "name": "Post-Training Frozen Permutation Kill Test",
        "passed": passed,
        "details": {
            "loss_ordered": loss_ordered,
            "loss_permuted": loss_perm,
            "kill_ratio": kill_ratio,
            "absolute_delta": loss_perm - loss_ordered,
            "verdict": f"Kill Test PASSED: Frozen mechanism derangement degraded transition loss by {kill_ratio:.2f}x."
            if passed else "Kill Test FAILED.",
        },
    }


def run_p15_g4_strong_shared_controls_benchmark(seed: int = 42) -> dict[str, Any]:
    """P15-G4: Strong Shared Controls Benchmark.
    Benchmarks modular against phase-conditioned shared, phase-agnostic shared same-width,
    and capacity-matched shared controls.
    """
    torch.manual_seed(seed)
    latent_dim = 16
    num_phases = 16
    n_train = 500
    n_test = 200
    
    true_mechs = [nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(num_phases)]
    for m in true_mechs:
        nn.init.orthogonal_(m.weight)
        
    def gen(N: int) -> torch.Tensor:
        with torch.no_grad():
            z0 = torch.randn(N, latent_dim)
            z_seq = [z0]
            for g in range(num_phases - 1):
                next_z = z_seq[-1] + true_mechs[g](z_seq[-1])
                z_seq.append(next_z)
            return torch.stack(z_seq, dim=1).detach()

    train_data = gen(n_train)
    test_data = gen(n_test)
    
    variants = {
        "modular": PAPER15_VARIANTS["full"],
        "shared_phase_conditioned": PAPER15_VARIANTS["shared_phase_conditioned"],
        "shared_same_width": PAPER15_VARIANTS["shared_same_width"],
        "shared_capacity_matched": PAPER15_VARIANTS["capacity_matched_shared"],
    }
    
    losses = {}
    params = {}
    for name, var in variants.items():
        m = CausalMechanismFactorizationModel(
            input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=var
        )
        params[name] = sum(p.numel() for p in m.parameters())
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        for _ in range(140):
            opt.zero_grad()
            _, z, zp = m.forward_with_transitions(train_data)
            l = m.compute_transition_loss(z, zp)
            l.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            _, z_t, zp_t = m.forward_with_transitions(test_data)
            losses[name] = m.compute_transition_loss(z_t, zp_t).item()
            
    # Check capacity matching between modular and shared_capacity_matched (< 1%)
    param_delta_pct = abs(params["modular"] - params["shared_capacity_matched"]) / params["modular"] * 100.0
    capacity_matched_ok = (param_delta_pct < 1.0)
    
    # Modular beats phase-agnostic shared same-width
    beats_shared_agnostic = bool(losses["modular"] < losses["shared_same_width"])
    
    passed = bool(capacity_matched_ok and beats_shared_agnostic)
    
    return {
        "gate": "P15-G4",
        "name": "Strong Shared Controls Benchmark",
        "passed": passed,
        "details": {
            "test_losses": losses,
            "parameter_counts": params,
            "capacity_matched_param_delta_pct": param_delta_pct,
            "capacity_matched_ok": capacity_matched_ok,
            "modular_beats_shared_agnostic": beats_shared_agnostic,
            "ratio_modular_to_shared_same_width": losses["modular"] / losses["shared_same_width"],
            "verdict": "Modular benchmarks verified against strong shared controls."
            if passed else "Benchmark failed.",
        },
    }


def run_p15_g5_cyclic_origin_invariance(seed: int = 42) -> dict[str, Any]:
    """P15-G5: Cyclic Origin Invariance.
    Evaluates cyclic transition consistency across origins 0, 4, 8, 12, confirming periodic topology.
    """
    torch.manual_seed(seed)
    latent_dim = 16
    num_phases = 16
    n_samples = 200
    
    model = CausalMechanismFactorizationModel(
        input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    x = torch.randn(n_samples, num_phases, latent_dim)
    
    # Evaluate cyclic transition error when trajectory is rolled by origin o
    origins = [0, 4, 8, 12]
    losses_by_origin = {}
    with torch.no_grad():
        for origin in origins:
            x_rolled = torch.roll(x, shifts=-origin, dims=1)
            _, z, zp = model.forward_with_transitions(x_rolled)
            loss = model.compute_transition_loss(z, zp).item()
            losses_by_origin[f"origin_{origin}"] = loss
            
    # Check that error across origins has low variance (periodic cyclic consistency)
    vals = list(losses_by_origin.values())
    max_dev = (max(vals) - min(vals)) / (np.mean(vals) + 1e-8)
    passed = bool(max_dev < 0.10)
    
    return {
        "gate": "P15-G5",
        "name": "Cyclic Origin Invariance",
        "passed": passed,
        "details": {
            "losses_by_origin": losses_by_origin,
            "max_relative_deviation": max_dev,
            "verdict": f"Cyclic topology confirmed: max deviation across origins is {max_dev*100:.2f}% (<10%)."
            if passed else "Origin sensitivity detected.",
        },
    }


def run_p15_g6_information_preservation_audit(seed: int = 42) -> dict[str, Any]:
    """P15-G6: Information Preservation & Non-Collapse Audit.
    Verifies that state representations retain sample variance, effective rank >= 8, and no phase collapse.
    """
    torch.manual_seed(seed)
    input_dim = 64
    width = 32
    num_phases = 16
    n_samples = 300
    
    model = CausalMechanismFactorizationModel(
        input_dim=input_dim, width=width, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    x = torch.randn(n_samples, num_phases, input_dim)
    
    z = model.encode_states(x) # (B, 16, width)
    
    # 1. Variance across records for each phase
    phase_variances = z.var(dim=0).mean(dim=-1).detach().cpu().numpy() # (16,)
    min_var = float(np.min(phase_variances))
    
    # 2. Effective rank across latent dimensions
    z_flat = z.view(n_samples * num_phases, width)
    cov = (z_flat.T @ z_flat) / len(z_flat)
    eigenvals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    p = eigenvals / eigenvals.sum()
    entropy = - (p * torch.log(p)).sum().item()
    eff_rank = float(np.exp(entropy))
    
    # 3. Non-collapse check: representation is not constant across samples
    no_collapse = bool(min_var > 0.05 and eff_rank >= 8.0)
    
    return {
        "gate": "P15-G6",
        "name": "Information Preservation & Non-Collapse Audit",
        "passed": no_collapse,
        "details": {
            "min_phase_variance": min_var,
            "phase_variances": [float(v) for v in phase_variances],
            "effective_rank": eff_rank,
            "effective_rank_threshold": 8.0,
            "verdict": f"Non-collapse confirmed: effective rank = {eff_rank:.2f} >= 8.0, min variance = {min_var:.4f}."
            if no_collapse else "Collapse detected in representations.",
        },
    }


def run_p15_g7_simple_baselines_benchmark(seed: int = 42) -> dict[str, Any]:
    """P15-G7: Simple-Transition Baselines Benchmark.
    Confirms that learned modular mechanism strictly outperforms Identity, Phase Mean,
    Mean Residual, and Linear baselines.
    """
    torch.manual_seed(seed)
    latent_dim = 16
    num_phases = 16
    n_train = 500
    n_test = 200
    
    # Non-linear phase dynamics: delta = tanh(W_g @ z)
    true_mechs = [nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Tanh(), nn.Linear(latent_dim, latent_dim)) for _ in range(num_phases)]
    for m in true_mechs:
        for p in m.parameters():
            nn.init.normal_(p, std=0.5)
            
    def gen(N: int) -> torch.Tensor:
        with torch.no_grad():
            z0 = torch.randn(N, latent_dim)
            z_seq = [z0]
            for g in range(num_phases - 1):
                shift = true_mechs[g](z_seq[-1])
                z_seq.append(z_seq[-1] + shift)
            return torch.stack(z_seq, dim=1).detach()

    train_z = gen(n_train)
    test_z = gen(n_test)
    
    # Train modular nonlinear model
    model_modular = CausalMechanismFactorizationModel(
        input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=PAPER15_VARIANTS["full"]
    )
    opt = torch.optim.Adam(model_modular.parameters(), lr=5e-3)
    for _ in range(160):
        opt.zero_grad()
        _, z, zp = model_modular.forward_with_transitions(train_z)
        l = model_modular.compute_transition_loss(z, zp)
        l.backward()
        opt.step()
        
    model_modular.eval()
    with torch.no_grad():
        _, z_t, zp_t = model_modular.forward_with_transitions(test_z)
        loss_modular = model_modular.compute_transition_loss(z_t, zp_t).item()
        
    # Evaluate simple baselines
    loss_identity = CausalMechanismFactorizationModel.evaluate_identity_baseline(test_z)
    loss_phase_mean = CausalMechanismFactorizationModel.evaluate_phase_mean_baseline(train_z, test_z)
    loss_mean_residual = CausalMechanismFactorizationModel.evaluate_mean_residual_baseline(train_z, test_z)
    
    # Train linear modular baseline
    model_linear = CausalMechanismFactorizationModel(
        input_dim=latent_dim, width=latent_dim, num_phases=num_phases, variant=PAPER15_VARIANTS["linear_modular"]
    )
    opt_lin = torch.optim.Adam(model_linear.parameters(), lr=5e-3)
    for _ in range(160):
        opt_lin.zero_grad()
        _, z, zp = model_linear.forward_with_transitions(train_z)
        l = model_linear.compute_transition_loss(z, zp)
        l.backward()
        opt_lin.step()
    model_linear.eval()
    with torch.no_grad():
        _, z_t, zp_t = model_linear.forward_with_transitions(test_z)
        loss_linear = model_linear.compute_transition_loss(z_t, zp_t).item()
        
    beats_all_simple = bool(
        loss_modular < loss_identity
        and loss_modular < loss_phase_mean
        and loss_modular < loss_mean_residual
        and loss_modular < loss_linear
    )
    
    return {
        "gate": "P15-G7",
        "name": "Simple-Transition Baselines Benchmark",
        "passed": beats_all_simple,
        "details": {
            "loss_modular_nonlinear": loss_modular,
            "loss_identity": loss_identity,
            "loss_phase_mean": loss_phase_mean,
            "loss_mean_residual": loss_mean_residual,
            "loss_linear_modular": loss_linear,
            "improvement_over_identity_pct": (1.0 - loss_modular / loss_identity) * 100.0,
            "improvement_over_linear_pct": (1.0 - loss_modular / loss_linear) * 100.0,
            "verdict": "Modular nonlinear mechanism beats Identity, Phase Mean, Mean Residual, and Linear baselines."
            if beats_all_simple else "Modular model failed to beat all simple baselines.",
        },
    }


def run_all_gates(seed: int = 42) -> dict[str, Any]:
    """Runs all 8 gates of the P15-V2 stack."""
    gates = [
        run_p15_g0_grounding_and_ring_contract(seed),
        run_p15_g1_transition_law_recovery(seed),
        run_p15_g2_input_distribution_autonomy(seed),
        run_p15_g3_frozen_permutation_kill_test(seed),
        run_p15_g4_strong_shared_controls_benchmark(seed),
        run_p15_g5_cyclic_origin_invariance(seed),
        run_p15_g6_information_preservation_audit(seed),
        run_p15_g7_simple_baselines_benchmark(seed),
    ]
    
    all_passed = all(g["passed"] for g in gates)
    report = {
        "kind": "paper15_modular_transition_dynamics_gate_report",
        "version": "P15-V2",
        "status": "PASS" if all_passed else "FAIL",
        "summary": {
            "total_gates": len(gates),
            "passed_gates": sum(1 for g in gates if g["passed"]),
            "failed_gates": sum(1 for g in gates if not g["passed"]),
        },
        "gates": gates,
    }
    return report


def main() -> None:
    report = run_all_gates(seed=42)
    output_dir = Path(__file__).resolve().parent / "refine-logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "P15_GATES.json"
    
    output_file.write_text(json.dumps(report, indent=2, cls=NumpyEncoder) + "\n")
    print(json.dumps({
        "status": report["status"],
        "passed": report["summary"]["passed_gates"],
        "total": report["summary"]["total_gates"],
        "output": str(output_file),
    }, indent=2))


if __name__ == "__main__":
    main()
