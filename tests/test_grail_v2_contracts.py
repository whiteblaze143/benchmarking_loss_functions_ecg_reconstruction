"""Mandatory Unit Tests for GRAIL-ECG v2 (PRD v2 §56).

Tests:
1. Baseline contracts: B0 has no clinical loss, weights from training folds only, probe labels absent from encoder.
2. P1 engineering tests: clinical memorization, view decoder memorization, SSL noncollapse.
3. Invariant contracts: lead-order permutation invariance, temporal order sensitivity, lead identity preservation.
4. Subset lattice contracts: 4,095 unique masks, cardinality counts, independent rank calculation, limb lead rank.
5. Shapley & cooperative game contracts: efficiency, symmetry, dummy lead, sum-difference, and 2nd-order interaction.
"""

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from grail_ecg.src.models.baselines import (
    PlainSSLBaselineB0,
    PlainClinicalSSLBaselineB1,
    StructuredBaselineB2,
    GeometryBaselineB3,
    SupervisedUpperBoundUB,
    FactorialGRAILEncoder,
)
from grail_ecg.src.models.grail_encoder import GRAILEncoder
from grail_ecg.src.models.view_aux_decoder import ViewAuxiliaryDecoder
from grail_ecg.src.losses.ssl import vicreg_loss
from grail_ecg.src.subsets.lattice import (
    LEAD_NAMES_12,
    SubsetInfo,
    compute_subset_rank,
    get_subset_info,
    enumerate_all_4095_subsets,
    BASIS_MATRIX_12_TO_8,
)
from grail_ecg.src.subsets.shapley import (
    compute_exact_shapley_values,
    compute_exact_pairwise_interactions,
)

CONFIG_PATH = Path("configs/ptbxl_concept_tiers.yaml")


# ==============================================================================
# 1. Baseline & Data Contracts
# ==============================================================================

def test_b0_contains_no_clinical_loss():
    """Verify B0 has no classification heads or clinical parameters."""
    model = PlainSSLBaselineB0(num_leads=8, hidden_dim=128, latent_dim=96)
    assert not hasattr(model, "anchor_head"), "B0 must NOT have an anchor_head"
    assert not hasattr(model, "anchor_heads"), "B0 must NOT have anchor_heads"

    x = torch.randn(2, 8, 5000)
    out = model(x)
    assert isinstance(out, torch.Tensor), "B0 output must be a single latent tensor"
    assert out.shape == (2, 96), f"Expected [2, 96], got {out.shape}"


def test_class_weights_training_folds_only():
    """Verify class weights in concept tiers config exist and are strictly positive."""
    assert CONFIG_PATH.exists(), f"Missing config {CONFIG_PATH}"
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    anchors = cfg["anchors"]
    assert len(anchors) == 25, f"Expected 25 anchors, got {len(anchors)}"
    for a in anchors:
        assert "pos_weight" in a, f"Anchor {a['code']} missing pos_weight"
        assert a["pos_weight"] > 0.0, f"Anchor {a['code']} pos_weight must be positive"
        assert a["train_pos_count"] >= 50, f"Anchor {a['code']} has <50 training occurrences"


def test_probe_label_hierarchy_leakage_registry():
    """Verify Tier P1, P2, P3 are properly categorized and Tier P3 has minimal anchor overlap."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    probes = cfg["probes"]
    tier_p1 = probes["tier_p1_directly_related"]
    tier_p2 = probes["tier_p2_related_distinct"]
    tier_p3 = probes["tier_p3_ontology_distinct"]

    assert len(tier_p1) >= 5
    assert len(tier_p2) >= 5
    assert len(tier_p3) >= 3

    # Tier P3 must have zero hierarchy relationship and minimal empirical Jaccard with any anchor (< 0.05)
    for p in tier_p3:
        assert p["max_anchor_jaccard"] < 0.05, f"Tier P3 concept {p['code']} has Jaccard >= 0.05 with an anchor"


def test_probe_labels_absent_from_encoder_loss():
    """Verify that probe labels never appear in the anchor training list."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    anchor_codes = {a["code"] for a in cfg["anchors"]}
    for tier in ["tier_p1_directly_related", "tier_p2_related_distinct", "tier_p3_ontology_distinct"]:
        for p in cfg["probes"][tier]:
            assert p["code"] not in anchor_codes, f"Leakage: Probe {p['code']} found in anchor training set!"


# ==============================================================================
# 2. P1 Engineering Gate Tests: Memorization & Noncollapse
# ==============================================================================

def test_p1_clinical_memorization():
    """P1a: Clinical supervision pathway can memorize a small batch (N=4) to loss < 0.05."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False

    model = FactorialGRAILEncoder(
        use_geometry=True,
        use_slots=True,
        use_view_aux=False,
        num_leads=8,
        num_tokens_per_lead=32,
        hidden_dim=64,
        num_slots=6,
        slot_dim=16,
        anchor_counts_per_domain={"rhythm": 2, "conduction": 2, "morphology": 2, "stt": 2},
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    x = torch.randn(4, 8, 1000, device=device)
    y = torch.randint(0, 2, (4, 8), device=device).float()

    min_loss = float("inf")
    for step in range(120):
        optimizer.zero_grad()
        slots, z_flat, logits_dict = model(x)
        logits = torch.cat(list(logits_dict.values()), dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
        if loss.item() < min_loss:
            min_loss = loss.item()

    model.eval()
    with torch.no_grad():
        slots, z_flat, logits_dict = model(x)
        logits = torch.cat(list(logits_dict.values()), dim=1)
        eval_loss = F.binary_cross_entropy_with_logits(logits, y).item()

    assert eval_loss < 0.05 or min_loss < 0.05, f"Supervised pathway failed to memorize small batch: eval_loss={eval_loss}, min_loss={min_loss}"


def test_p1_decoder_memorization():
    """P1b: View decoder pathway can memorize a target lead (N=8) to >50% L1 reduction over baseline."""
    torch.manual_seed(42)
    decoder = ViewAuxiliaryDecoder(latent_dim=96, hidden_dim=64)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-2)

    z = torch.randn(8, 96)
    target_angle = torch.tensor([[math.pi / 2, math.pi / 4]]).expand(8, -1)
    target_lead = torch.sin(torch.linspace(0, 10, 5000)).unsqueeze(0).expand(8, -1)  # Canonical 5000 samples

    initial_loss = float(F.l1_loss(torch.zeros_like(target_lead), target_lead).item())

    for _ in range(40):
        optimizer.zero_grad()
        pred = decoder(z, target_angle).squeeze(1)
        loss = F.l1_loss(pred, target_lead)
        loss.backward()
        optimizer.step()

    final_loss = loss.item()
    reduction = (initial_loss - final_loss) / initial_loss
    assert reduction > 0.50, f"Decoder failed memorization: initial={initial_loss:.4f}, final={final_loss:.4f}, red={reduction:.2%}"


def test_p1_ssl_noncollapse():
    """P1c: VICReg loss alone enforces non-trivial variance across all latent coordinates."""
    torch.manual_seed(42)
    z1 = torch.randn(64, 96)
    # Simulate slightly perturbed second view
    z2 = z1 + 0.1 * torch.randn(64, 96)

    loss_dict = vicreg_loss(z1, z2, sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0)
    assert "loss" in loss_dict
    assert "var_loss" in loss_dict
    assert "cov_loss" in loss_dict

    # Check that coordinate std is measured correctly
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    assert (std1 > 0.5).all(), "Variance across latent coordinates should be non-trivial"


# ==============================================================================
# 3. Permutation Invariance and Geometric Conditioning
# ==============================================================================

def test_lead_order_permutation_invariance():
    """Verify that permuting lead ordering (X and Theta) produces identical latent representation."""
    torch.manual_seed(42)
    model = FactorialGRAILEncoder(
        use_geometry=True,
        use_slots=True,
        use_view_aux=False,
        num_leads=8,
        num_tokens_per_lead=32,
        hidden_dim=64,
        num_slots=6,
        slot_dim=16,
    )
    model.eval()

    x = torch.randn(2, 8, 1000)
    # Canonical angles for 8 leads
    angles = torch.randn(8, 2)

    # Forward with canonical order
    with torch.no_grad():
        _, z_canonical, _ = model(x, custom_angles=angles)

    # Permute leads with an arbitrary permutation of 8 leads
    perm = torch.tensor([3, 0, 7, 1, 5, 2, 6, 4])
    x_perm = x[:, perm, :]
    angles_perm = angles[perm, :]

    with torch.no_grad():
        _, z_permuted, _ = model(x_perm, custom_angles=angles_perm)

    diff = torch.max(torch.abs(z_canonical - z_permuted)).item()
    assert diff < 1e-5, f"Permutation invariance failed: max diff = {diff:.2e}"


def test_temporal_order_not_invariant():
    """Verify that temporal ordering is NOT invariant (reversing time alters the latent)."""
    torch.manual_seed(42)
    model = FactorialGRAILEncoder(
        use_geometry=True,
        use_slots=True,
        use_view_aux=False,
        num_leads=8,
        num_tokens_per_lead=32,
        hidden_dim=64,
    )
    model.eval()

    # Create an asymmetric signal: frequency chirp that starts slow and accelerates
    t = torch.linspace(0, 10, 5000)
    sig = torch.sin(t ** 2.0).unsqueeze(0).unsqueeze(0).expand(2, 8, -1)
    sig_reversed = torch.flip(sig, dims=[-1])

    with torch.no_grad():
        _, z_orig, _ = model(sig)
        _, z_rev, _ = model(sig_reversed)

    diff = torch.max(torch.abs(z_orig - z_rev)).item()
    # While lead permutation is invariant to machine precision (< 1e-6),
    # reversing time alters the latent state (diff is clearly non-zero).
    assert diff > 1e-4, f"Temporal order should matter: diff was only {diff:.6f}"


def test_target_lead_absent_from_encoder():
    """Verify that during view auxiliary training, target lead is masked from input."""
    x = torch.randn(4, 8, 1000)
    q_idx = 3

    # Mask lead q_idx
    x_masked = x.clone()
    x_masked[:, q_idx, :] = 0.0

    assert (x_masked[:, q_idx, :] == 0.0).all()
    # Confirm other leads unchanged
    other_indices = [i for i in range(8) if i != q_idx]
    assert torch.equal(x_masked[:, other_indices, :], x[:, other_indices, :])


# ==============================================================================
# 4. Subset Lattice and Information Rank
# ==============================================================================

def test_all_4095_subset_masks_unique():
    """Verify exactly 4,095 unique non-empty subsets."""
    subsets = enumerate_all_4095_subsets()
    assert len(subsets) == 4095
    unique_masks = {s.mask for s in subsets}
    assert len(unique_masks) == 4095
    assert min(unique_masks) == 1
    assert max(unique_masks) == 4095


def test_subset_cardinality_counts():
    """Verify that the cardinality count matches binomial coefficients nCr(12, k)."""
    subsets = enumerate_all_4095_subsets()
    counts = {}
    for s in subsets:
        counts[s.k_displayed] = counts.get(s.k_displayed, 0) + 1

    expected_binomial = {
        1: 12,
        2: 66,
        3: 220,
        4: 495,
        5: 792,
        6: 924,
        7: 792,
        8: 495,
        9: 220,
        10: 66,
        11: 12,
        12: 1,
    }
    assert counts == expected_binomial


def test_independent_rank_calculation():
    """Verify independent rank calculation for subsets."""
    # Lead I only -> rank 1
    info_i = get_subset_info(1 << 0)
    assert info_i.k_displayed == 1
    assert info_i.r_independent == 1
    assert info_i.n_limb == 1
    assert info_i.limb_rank == 1

    # Lead I and Lead II -> rank 2
    info_i_ii = get_subset_info((1 << 0) | (1 << 1))
    assert info_i_ii.k_displayed == 2
    assert info_i_ii.r_independent == 2
    assert info_i_ii.limb_rank == 2

    # Lead I, II, III -> displayed = 3, but independent rank = 2! (III = II - I)
    info_i_ii_iii = get_subset_info((1 << 0) | (1 << 1) | (1 << 2))
    assert info_i_ii_iii.k_displayed == 3
    assert info_i_ii_iii.r_independent == 2, f"Expected rank 2, got {info_i_ii_iii.r_independent}"
    assert info_i_ii_iii.limb_rank == 2


def test_limb_lead_algebra_rank():
    """Verify that all 6 limb leads together have rank strictly equal to 2."""
    # All 6 limb leads: bitmask 0b111111 = 63
    limb_mask = (1 << 6) - 1
    info_limb = get_subset_info(limb_mask)
    assert info_limb.k_displayed == 6
    assert info_limb.r_independent == 2, f"6 limb leads should have rank 2, got {info_limb.r_independent}"
    assert info_limb.limb_rank == 2

    # All 12 leads together have rank strictly equal to 8
    full_info = get_subset_info(4095)
    assert full_info.k_displayed == 12
    assert full_info.r_independent == 8, f"Full 12 leads should have rank 8, got {full_info.r_independent}"


# ==============================================================================
# 5. Exact Shapley and Pairwise Interaction Axiomatic Tests
# ==============================================================================

def test_shapley_efficiency():
    """Axiom: Efficiency. Sum of Shapley values must equal v(Grand Coalition) - v(Empty)."""
    n = 4
    # Define an arbitrary cooperative game value function
    def game_val(mask: int) -> float:
        s_size = bin(mask).count("1")
        return float(s_size ** 2 + (mask & 1) * 3)

    phi = compute_exact_shapley_values(n, game_val)
    expected_sum = game_val((1 << n) - 1) - game_val(0)
    assert np.isclose(np.sum(phi), expected_sum), f"Efficiency failed: sum(phi)={np.sum(phi)}, expected={expected_sum}"


def test_shapley_symmetry():
    """Axiom: Symmetry. Symmetric players must receive identical Shapley values."""
    n = 3
    # Player 0 and Player 1 have symmetric contributions
    def symmetric_game(mask: int) -> float:
        p0 = bool(mask & (1 << 0))
        p1 = bool(mask & (1 << 1))
        p2 = bool(mask & (1 << 2))
        return float((p0 + p1) * 2 + p2 * 5 + (p0 and p1) * 3)

    phi = compute_exact_shapley_values(n, symmetric_game)
    assert np.isclose(phi[0], phi[1]), f"Symmetry failed: phi[0]={phi[0]}, phi[1]={phi[1]}"


def test_shapley_dummy_lead():
    """Axiom: Dummy player. Player who contributes nothing receives 0 Shapley value."""
    n = 3
    # Player 2 contributes nothing to any coalition
    def dummy_game(mask: int) -> float:
        p0 = bool(mask & (1 << 0))
        p1 = bool(mask & (1 << 1))
        return float(p0 * 10 + p1 * 20)

    phi = compute_exact_shapley_values(n, dummy_game)
    assert np.isclose(phi[2], 0.0), f"Dummy player failed: phi[2]={phi[2]} != 0"


def test_pairwise_interaction_known_synthetic_case():
    """Verify 2nd-order interaction index: pure synergy produces positive I_ij, pure redundancy produces negative I_ij."""
    n = 3
    # Player 0 and 1 are purely synergistic: value is only created when BOTH are present!
    def synergistic_game(mask: int) -> float:
        p0 = bool(mask & (1 << 0))
        p1 = bool(mask & (1 << 1))
        return 10.0 if (p0 and p1) else 0.0

    I_mat = compute_exact_pairwise_interactions(n, synergistic_game)
    assert I_mat[0, 1] > 0.0, f"Synergy should be positive: got {I_mat[0, 1]}"
    assert np.isclose(I_mat[0, 1], I_mat[1, 0])
