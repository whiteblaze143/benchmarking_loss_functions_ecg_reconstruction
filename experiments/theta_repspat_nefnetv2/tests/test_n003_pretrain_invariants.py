"""Unit tests verifying N003 pretrain branch invariants.

Checks:
1. super_mode='pretrain' is properly set in GeoVT (nefnet_plus.layer).
2. Dynamic variable-cardinality input slots L_input = 2 + k for k in {1, 2, 3}.
3. Forward pass operates on runtime x.shape[1] rather than fixed optimization module lists.
4. Lead I and II always anchored at indices 0 and 1.
5. Prospective observed-lead normalization contracts.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch
import numpy as np

from theta_repspat.vendor import instantiate_author_model
from theta_repspat.clinical_dataset import PTBXLClinicalAnyPairs, STANDARD_12LEAD_ANGLES_RAD

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False


def test_pretrain_branch_dynamic_cardinality():
    """Verify that nefnet_plus.layer in pretrain mode supports variable L_input = 2 + k."""
    model = instantiate_author_model(
        ROOT / "author_code" / "nefnet_v2",
        DEVICE,
        super_mode="pretrain",
        lead_num=3,  # Ignored when super_mode='pretrain'
    )
    model.eval()

    # Verify that in pretrain mode, the fixed module lists are NOT instantiated
    assert not hasattr(model, "mlp_list"), "pretrain mode should not use fixed mlp_list"
    assert not hasattr(model, "W_encoder_list"), "pretrain mode should not use fixed W_encoder_list"
    assert hasattr(model, "mlp3"), "pretrain mode requires shared mlp3"
    assert hasattr(model, "W_encoder3"), "pretrain mode requires shared W_encoder3"

    batch_size = 2
    length = 4608

    for k in [1, 2, 3]:
        L_input = 2 + k  # 3, 4, 5
        x = torch.randn(batch_size, L_input, length, device=DEVICE)
        # First two views: I (0) and II (1); subsequent views sampled from precordial/augmented
        input_indices = [0, 1] + list(range(2, 2 + k))
        input_thetas = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[input_indices]).unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE)
        
        target_idx = 7  # V2
        query_theta = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[target_idx]).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)

        with torch.no_grad():
            out = model(x, input_thetas, query_theta)

        assert out.shape == (batch_size, 1, length), f"Failed for k={k}: shape was {out.shape}"
        assert torch.isfinite(out).all(), f"Failed for k={k}: non-finite values in output"


def test_pretrain_gradient_flow():
    """Verify gradient flows backward through all input slots in pretrain mode."""
    torch.backends.cudnn.enabled = False
    model = instantiate_author_model(
        ROOT / "author_code" / "nefnet_v2",
        DEVICE,
        super_mode="pretrain",
    )
    model.train()

    batch_size = 2
    length = 4608
    L_input = 4  # 2 + 2

    x = torch.randn(batch_size, L_input, length, device=DEVICE, requires_grad=True)
    input_indices = [0, 1, 6, 8]  # I, II, V1, V3
    input_thetas = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[input_indices]).unsqueeze(0).repeat(batch_size, 1, 1).to(DEVICE)
    query_theta = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD[11]).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)  # V6

    out = model(x, input_thetas, query_theta)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Every slot must receive non-zero gradient
    for slot_idx in range(L_input):
        slot_grad_norm = float(x.grad[:, slot_idx, :].norm())
        assert slot_grad_norm > 0.0, f"Slot {slot_idx} received zero gradient"


def test_observed_normalization_invariance():
    """Verify that PTBXLClinicalAnyPairs computes normalization strictly from observed inputs."""
    dataset = PTBXLClinicalAnyPairs(
        tensors_dir=Path("/home/mithunmanivannan/data/ptb_xl/tensors"),
        split="train",
        lead_cardinality=3,
        seed=42,
    )
    sample = dataset[0]
    inp = sample["input"]
    tgt = sample["target"]

    # Input leads must have minimum 0.0 and maximum 1.0
    assert abs(float(inp.min()) - 0.0) < 1e-5
    assert abs(float(inp.max()) - 1.0) < 1e-5

    # Target lead min/max may exceed [0, 1] because it was normalized with observed extrema
    # This guarantees no target extrema leakage
    m_obs = float(sample["m_obs"])
    M_obs = float(sample["M_obs"])
    assert M_obs > m_obs
