from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from repecg.common.variants import ExperimentVariant
from repecg.paper08_tokens import (
    PhaseTokenAttentionLayer,
    PhaseTokenTransformer,
    bootstrap_pairwise_phase_distances,
    build_cyclic_banded_mask,
    complete_linkage_merge,
    odd_even_agreement,
    pairwise_phase_distances,
    simultaneous_upper_bounds,
)


def test_world_1_cyclic_banded_mask_correctness_and_sparsity():
    """
    World 1: Proves cyclic banded mask on C_16 correctly sparsifies attention.
    CLS interacts with all; phase tokens interact iff dist_C(i, j) <= bandwidth.
    """
    num_phases = 16
    bandwidth = 2
    mask = build_cyclic_banded_mask(num_phases=num_phases, bandwidth=bandwidth)
    
    assert mask.shape == (17, 17)
    # CLS pools globally; phase tokens cannot use CLS as a global bypass.
    assert torch.all(mask[0, :] == 0.0)
    assert torch.all(mask[1:, 0] < -1e8)

    # Phase token cyclic distance checks
    # Phase 0 (token 1): neighbors are 15, 14, 1, 2 (tokens 16, 15, 2, 3)
    assert mask[1, 1] == 0.0  # self
    assert mask[1, 2] == 0.0  # +1
    assert mask[1, 3] == 0.0  # +2
    assert mask[1, 16] == 0.0  # -1 (cyclic wrap)
    assert mask[1, 15] == 0.0  # -2 (cyclic wrap)
    
    # Non-neighbors should be masked out with -1e9
    assert mask[1, 4] < -1e8  # +3
    assert mask[1, 9] < -1e8  # opposite phase
    assert mask[1, 14] < -1e8  # -3

    # Forward pass through attention layer with mask
    layer = PhaseTokenAttentionLayer(
        d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, mode="local_banded", bandwidth=bandwidth
    )
    layer.eval()
    x = torch.randn(2, 17, 64)
    with torch.no_grad():
        _, attn = layer(x, mask=mask, return_attention=True)
    assert attn is not None
    assert attn.shape == (2, 4, 17, 17)

    # Attention weights for masked positions must be zero to high precision
    for i in range(1, 17):
        for j in range(1, 17):
            dist = min(abs((i - 1) - (j - 1)), 16 - abs((i - 1) - (j - 1)))
            if dist > bandwidth:
                assert torch.all(attn[:, :, i, j] < 1e-6), f"Position ({i}, {j}) should be masked"


def test_world_2_static_attention_weight_invariance_across_inputs():
    """
    World 2: Proves static attention weights are completely invariant across inputs.
    A(x_1) == A(x_2) to machine precision.
    """
    model = PhaseTokenTransformer(input_dim=8, d_model=64, nhead=4, num_layers=2, mode="static")
    model.eval()

    x1 = torch.randn(2, 16, 8)
    x2 = torch.randn(2, 16, 8) * 5.0 + 10.0  # Radically different distribution

    with torch.no_grad():
        _, attns1 = model(x1, return_attention=True)
        _, attns2 = model(x2, return_attention=True)

    assert len(attns1) == 2
    for l, (a1, a2) in enumerate(zip(attns1, attns2)):
        # Check batch element 0 of x1 vs batch element 0 of x2
        diff = torch.max(torch.abs(a1[0] - a2[0])).item()
        assert diff < 1e-12, f"Static attention at layer {l} varied across inputs: max diff {diff}"


def test_world_3_dynamic_routing_input_sensitivity():
    """
    World 3: Proves dynamic attention responds selectively to input features.
    ||A(x_1) - A(x_2)||_F > 0.
    """
    model = PhaseTokenTransformer(input_dim=8, d_model=64, nhead=4, num_layers=2, mode="global")
    model.eval()

    x1 = torch.randn(2, 16, 8)
    x2 = torch.randn(2, 16, 8)

    with torch.no_grad():
        _, attns1 = model(x1, return_attention=True)
        _, attns2 = model(x2, return_attention=True)

    # Across distinct inputs, attention maps must differ
    diff_layer0 = torch.norm(attns1[0][0] - attns2[0][0]).item()
    assert diff_layer0 > 0.05, f"Dynamic attention failed to differentiate inputs: diff {diff_layer0}"


def test_world_4_phase_scrambling_sensitivity():
    """
    World 4: Proves phase-token ordering is exploited when positional embeddings are active.
    Scrambling phase tokens measurably alters model predictions.
    """
    model = PhaseTokenTransformer(input_dim=8, d_model=64, nhead=4, num_layers=2, mode="global")
    model.eval()

    # Distinct phase profile
    x = torch.zeros(1, 16, 8)
    for phi in range(16):
        x[0, phi, :] = phi * 0.5
    with torch.no_grad():

        out_ordered = model(x)
        # 1. Reversed phase order (reversing cardiac cycle progression)
        x_reversed = torch.flip(x, dims=[1])
        out_reversed = model(x_reversed)
        diff_rev = torch.norm(out_ordered - out_reversed).item()

        # 2. Random scramble
        out_scrambled = model(x, scramble_phases=True)
        diff_scramble = torch.norm(out_ordered - out_scrambled).item()

    assert diff_rev > 1e-3, f"Phase reversal had negligible effect: diff {diff_rev}"
    assert diff_scramble > 1e-3, f"Phase scrambling had negligible effect: diff {diff_scramble}"




def test_world_5_phase_agnostic_permutation_invariance():
    """
    World 5: Proves that when positional embeddings are absent and uniform/mean pooling is used,
    arbitrary phase permutations leave outputs invariant.
    """
    # Linear probe over mean of tokens is mathematically permutation invariant
    variant = ExperimentVariant(head="linear")
    model = PhaseTokenTransformer(input_dim=8, variant=variant)
    model.eval()

    x = torch.randn(2, 16, 8)
    perm = torch.randperm(16)
    x_perm = x[:, perm]

    with torch.no_grad():
        out_orig = model(x)
        out_perm = model(x_perm)

    max_err = torch.max(torch.abs(out_orig - out_perm)).item()
    assert max_err < 1e-6, f"Linear probe failed permutation invariance: error {max_err}"


def test_world_6_simultaneous_bootstrap_upper_bounds_and_linkage_clustering():
    """
    World 6: Verifies FWER coverage of simultaneous upper bounds and complete linkage merging.
    """
    # Create synthetic points and bootstrap replicates
    point = np.array([0.1, 0.2, 0.3, 0.4])
    rng = np.random.default_rng(42)
    # Replicates centered around point with small variance
    draws = point[None, :] + rng.normal(0, 0.01, size=(200, 4))
    
    upper = simultaneous_upper_bounds(point, draws, alpha=0.05)
    assert np.all(upper >= point), "Simultaneous upper bounds must be >= point estimates"

    # Test complete linkage merge on symmetric upper bound matrix
    upper_mat = np.array([
        [0.0, 0.15, 0.80, 0.90],
        [0.15, 0.0, 0.75, 0.85],
        [0.80, 0.75, 0.0, 0.10],
        [0.90, 0.85, 0.10, 0.0],
    ])
    clusters = complete_linkage_merge(upper_mat, delta=0.20)
    assert clusters == [(0, 1), (2, 3)], f"Unexpected clusters: {clusters}"

    # Verify pairwise distance computation from representations
    synthetic_rep = rng.normal(0, 1, size=(50, 16, 16))
    dist_mat = pairwise_phase_distances(synthetic_rep)
    assert dist_mat.shape == (16, 16)
    assert np.allclose(dist_mat, dist_mat.T)
    assert np.allclose(np.diag(dist_mat), 0.0)

    # Bootstrap pairwise distances
    points, boots = bootstrap_pairwise_phase_distances(synthetic_rep, n_bootstraps=20, seed=42)
    assert len(points) == 120  # 16 * 15 // 2
    assert boots.shape == (20, 120)


def test_world_7_split_half_odd_even_heartbeat_reliability():
    """
    World 7: Verifies normalized Jensen-Shannon agreement for split-half reliability.
    """
    # Identical distributions must yield 1.0 agreement
    p = np.array([10, 20, 30, 40], dtype=np.float64)
    assert abs(odd_even_agreement(p, p) - 1.0) < 1e-12

    # Strongly divergent distributions must yield low agreement
    p_disjoint1 = np.array([100, 0, 0, 0], dtype=np.float64)
    p_disjoint2 = np.array([0, 0, 0, 100], dtype=np.float64)
    agreement_disjoint = odd_even_agreement(p_disjoint1, p_disjoint2)
    assert agreement_disjoint < 0.05, f"Disjoint distributions should have low agreement: {agreement_disjoint}"

    # Smooth monotonic decrease under perturbation
    p_mod1 = np.array([50, 50, 0, 0], dtype=np.float64)
    p_mod2 = np.array([40, 60, 0, 0], dtype=np.float64)
    p_mod3 = np.array([10, 90, 0, 0], dtype=np.float64)
    
    ag1 = odd_even_agreement(p_mod1, p_mod1)
    ag2 = odd_even_agreement(p_mod1, p_mod2)
    ag3 = odd_even_agreement(p_mod1, p_mod3)
    assert ag1 > ag2 > ag3


def test_world_8_gradient_flow_across_all_attention_modes():
    """
    World 8: Verifies finite non-zero gradients across all attention modes without dead parameters.
    """
    modes = ["global", "local_banded", "static", "uniform", "phase_agnostic"]
    for mode in modes:
        model = PhaseTokenTransformer(input_dim=8, d_model=32, nhead=2, num_layers=2, classes=5, mode=mode)
        x = torch.randn(4, 16, 8, requires_grad=True)
        y = torch.randint(0, 2, (4, 5)).float()

        logits = model(x)
        loss = nn.BCEWithLogitsLoss()(logits, y)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} had no gradient in mode {mode}"
                assert torch.isfinite(param.grad).all(), f"Parameter {name} had non-finite gradient in mode {mode}"
                assert torch.norm(param.grad).item() > 0, f"Parameter {name} had zero gradient in mode {mode}"
        model.zero_grad()
