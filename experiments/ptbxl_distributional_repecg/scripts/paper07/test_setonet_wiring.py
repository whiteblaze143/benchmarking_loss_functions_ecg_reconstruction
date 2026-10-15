#!/usr/bin/env python3
"""Fast GPU wiring test for SetONet on PTB-XL."""

import sys
from pathlib import Path
import torch

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts/paper07"))

from train_setonet_ptbxl import PTBXLSetONetAdapter

def main():
    print("Testing SetONet instantiation and forward/backward on GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"

    for variant in ["standard", "key"]:
        print(f"Testing SetONet variant: {variant}")
        model = PTBXLSetONetAdapter(
            operator_dim=8,
            response_dim=128,
            classes=5,
            p=64,
            variant=variant,
        ).to(device)

        batch_size = 16
        m = 5
        ops = torch.randn(batch_size, m, 8, device=device)
        res = torch.randn(batch_size, m, 16, 128, device=device)
        targets = (torch.rand(batch_size, 5, device=device) > 0.7).float()

        logits = model(ops, res)
        assert logits.shape == (batch_size, 5), f"Shape mismatch: {logits.shape}"

        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(grad_norm), "Non-finite gradient"
        print(f"  Passed {variant}: loss={loss.item():.4f}, grad_norm={grad_norm.item():.4f}")

    print("ALL SETONET WIRING TESTS PASSED!")

if __name__ == "__main__":
    main()
