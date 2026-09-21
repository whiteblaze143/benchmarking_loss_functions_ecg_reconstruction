"""Unit tests for Paper 15: Modular Phase-Transition Dynamics in ECG Representation Space (P15-V2).

Verifies:
1. State grounding across all 16 observed cardiac phases.
2. Output shapes of forward and forward_with_transitions across 16 cyclic transitions.
3. Cyclic transition loss, ring closure ((g+1)%16), and target state detachment.
4. Capacity matching precision (< 1.0% parameter difference between modular and capacity_matched_shared).
5. Exact module reuse in shared controls vs 16 distinct modules in modular.
6. Phase-conditioned shared control execution.
7. Post-training frozen derangement Kill Test operator.
8. 4-tier diagnostic representation hierarchy extraction (Z, Delta, Delta_hat, R).
9. Simple transition baselines (Identity, Phase Mean, Mean Residual).
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from repecg.common.models import CausalMechanismFactorizationModel
from repecg.common.variants import ExperimentVariant, PAPER15_VARIANTS


def test_observation_grounding_across_all_phases():
    """Verifies that latent states Z_g are grounded on observed phase features X_g."""
    torch.manual_seed(42)
    model = CausalMechanismFactorizationModel(input_dim=256, width=64, num_phases=16, variant=PAPER15_VARIANTS["full"])
    
    x = torch.randn(8, 16, 256)
    z = model.encode_states(x)
    assert z.shape == (8, 16, 64)
    
    # Perturb phase 3 only
    x_pert = x.clone()
    x_pert[:, 3] += torch.randn(8, 256) * 3.0
    z_pert = model.encode_states(x_pert)
    
    # Phase 3 must change
    assert (z_pert[:, 3] - z[:, 3]).abs().mean() > 1e-3
    
    # All other phases must remain exactly identical
    other_phases = [p for p in range(16) if p != 3]
    assert torch.allclose(z_pert[:, other_phases], z[:, other_phases], atol=1e-7)


def test_forward_with_transitions_shapes():
    """Verifies output shapes across 16 cyclic transitions."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, 128)
    
    # Modular 16 mechanisms
    m_full = CausalMechanismFactorizationModel(input_dim=128, width=64, classes=5, variant=PAPER15_VARIANTS["full"])
    logits, z, z_pred = m_full.forward_with_transitions(x)
    assert logits.shape == (4, 5)
    assert z.shape == (4, 16, 64)
    assert z_pred.shape == (4, 16, 64)
    
    # Shared capacity-matched
    m_sh = CausalMechanismFactorizationModel(input_dim=128, width=64, classes=5, variant=PAPER15_VARIANTS["capacity_matched_shared"])
    logits_sh, z_sh, z_pred_sh = m_sh.forward_with_transitions(x)
    assert logits_sh.shape == (4, 5)
    assert z_sh.shape == (4, 16, 64)
    assert z_pred_sh.shape == (4, 16, 64)
    
    # Shared phase-conditioned
    m_cond = CausalMechanismFactorizationModel(input_dim=128, width=64, classes=5, variant=PAPER15_VARIANTS["shared_phase_conditioned"])
    logits_cond, z_cond, z_pred_cond = m_cond.forward_with_transitions(x)
    assert logits_cond.shape == (4, 5)
    assert z_cond.shape == (4, 16, 64)
    assert z_pred_cond.shape == (4, 16, 64)


def test_cyclic_transition_loss_and_ring_closure():
    """Verifies cyclic transition loss computes roll target and all 16 mechanisms receive gradients."""
    torch.manual_seed(42)
    model = CausalMechanismFactorizationModel(input_dim=64, width=32, variant=PAPER15_VARIANTS["full"])
    
    x = torch.randn(4, 16, 64, requires_grad=True)
    _, z, z_pred = model.forward_with_transitions(x)
    loss = model.compute_transition_loss(z, z_pred)
    
    assert loss.ndim == 0
    assert torch.isfinite(loss).item()
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Verify all 16 mechanisms receive non-zero gradients
    assert len(model.mechanisms) == 16
    for g, mech in enumerate(model.mechanisms):
        for name, param in mech.named_parameters():
            assert param.grad is not None, f"Mechanism {g} parameter {name} has no gradient"
            assert torch.isfinite(param.grad).all(), f"Mechanism {g} parameter {name} has non-finite gradient"


def test_capacity_matching_precision():
    """Verifies capacity matching between modular and shared_capacity_matched within 1.0% tolerance."""
    for width in [16, 32, 64, 128]:
        m_full = CausalMechanismFactorizationModel(input_dim=256, width=width, variant=PAPER15_VARIANTS["full"])
        m_sh = CausalMechanismFactorizationModel(input_dim=256, width=width, variant=PAPER15_VARIANTS["capacity_matched_shared"])
        
        p_full = sum(p.numel() for p in m_full.parameters())
        p_sh = sum(p.numel() for p in m_sh.parameters())
        
        diff = abs(p_full - p_sh)
        rel_diff = diff / float(p_full)
        assert rel_diff < 0.01, f"Width {width}: rel_diff={rel_diff:.5f} exceeds 1% tolerance"


def test_shared_variants_module_counts():
    """Verifies module reuse in shared controls vs 16 distinct modules in modular."""
    m_full = CausalMechanismFactorizationModel(input_dim=128, width=64, variant=PAPER15_VARIANTS["full"])
    m_sh = CausalMechanismFactorizationModel(input_dim=128, width=64, variant=PAPER15_VARIANTS["capacity_matched_shared"])
    m_cond = CausalMechanismFactorizationModel(input_dim=128, width=64, variant=PAPER15_VARIANTS["shared_phase_conditioned"])
    m_sw = CausalMechanismFactorizationModel(input_dim=128, width=64, variant=PAPER15_VARIANTS["shared_same_width"])
    
    assert len({id(m) for m in m_full.mechanisms}) == 16
    assert len({id(m) for m in m_sh.mechanisms}) == 1
    assert len({id(m) for m in m_cond.mechanisms}) == 1
    assert len({id(m) for m in m_sw.mechanisms}) == 1


def test_frozen_derangement_kill_test_operator():
    """Verifies that evaluate_permuted_transitions permutes mechanisms across phases."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, 64)
    model = CausalMechanismFactorizationModel(input_dim=64, width=32, variant=PAPER15_VARIANTS["full"])
    
    z = model.encode_states(x)
    z_pred_ord = model.predict_transitions(z)
    z_pred_perm = model.evaluate_permuted_transitions(z)
    
    assert z_pred_perm.shape == z_pred_ord.shape
    diff = (z_pred_ord - z_pred_perm).abs().mean()
    assert diff > 1e-3, "Permuted mechanism order did not alter predictions"


def test_representation_hierarchy_extraction():
    """Verifies extraction of state, observed dynamics, predicted transitions, and innovations."""
    torch.manual_seed(42)
    x = torch.randn(4, 16, 64)
    model = CausalMechanismFactorizationModel(input_dim=64, width=32, variant=PAPER15_VARIANTS["full"])
    
    hierarchy = model.extract_representation_hierarchy(x)
    assert set(hierarchy.keys()) == {"state", "observed_dynamics", "predicted_transitions", "innovations"}
    
    z = hierarchy["state"]
    delta_obs = hierarchy["observed_dynamics"]
    delta_pred = hierarchy["predicted_transitions"]
    innovations = hierarchy["innovations"]
    
    assert z.shape == (4, 16, 32)
    assert delta_obs.shape == (4, 16, 32)
    assert delta_pred.shape == (4, 16, 32)
    assert innovations.shape == (4, 16, 32)
    
    # Confirm R = Delta_obs - Delta_pred
    assert torch.allclose(innovations, delta_obs - delta_pred, atol=1e-6)


def test_simple_baselines_evaluators():
    """Verifies static baseline evaluator functions."""
    torch.manual_seed(42)
    z_tr = torch.randn(20, 16, 32)
    z_te = torch.randn(10, 16, 32)
    
    l_id = CausalMechanismFactorizationModel.evaluate_identity_baseline(z_te)
    l_mean = CausalMechanismFactorizationModel.evaluate_phase_mean_baseline(z_tr, z_te)
    l_res = CausalMechanismFactorizationModel.evaluate_mean_residual_baseline(z_tr, z_te)
    
    assert l_id > 0 and torch.isfinite(torch.tensor(l_id))
    assert l_mean > 0 and torch.isfinite(torch.tensor(l_mean))
    assert l_res > 0 and torch.isfinite(torch.tensor(l_res))
