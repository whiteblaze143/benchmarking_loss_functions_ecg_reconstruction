"""Unit tests for EchoNext model adaptations: SetOperator and SetONet.

Verifies:
1. Valid output tensor shapes for binary classification (batch, 1).
2. Permutation invariance of set processing over the sensor/operator axis.
3. Finite loss and finite gradients under backward pass.
4. Support for all tabular conditions (ECG-only, native/trunk fusion, late concat).
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

REPO_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
SCRIPTS_DIR = REPO_ROOT / "experiments/ptbxl_distributional_repecg/scripts/paper07/echonext"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from models.setoperator_echonext import build_setoperator_echonext
from models.setonet_echonext import build_setonet_echonext


def test_setoperator():
    print("Testing SetOperatorEchoNext...")
    batch_size = 4
    n_sensors = 8
    operator_dim = 8

    # Fake input tensors
    operators = torch.randn(batch_size, n_sensors, operator_dim)
    responses = torch.randn(batch_size, n_sensors, 16, 128)
    tabular = torch.randn(batch_size, 7)

    for cond in ["SO-ECG", "SO-ECG+Tab-Native", "SO-ECG+Tab-Concat"]:
        model = build_setoperator_echonext(cond)
        tab_in = None if cond == "SO-ECG" else tabular
        out = model(operators, responses, tabular=tab_in)

        logits = out["logits"]
        assert logits.shape == (batch_size, 1), f"Unexpected shape {logits.shape} for {cond}"
        assert torch.isfinite(logits).all(), f"Non-finite logits for {cond}"

        # Backward test
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name} for {cond}"

        print(f"  Passed {cond}: Shape {logits.shape}, finite gradients verified.")

    # Permutation invariance test on SO-ECG
    model = build_setoperator_echonext("SO-ECG")
    model.eval()
    with torch.no_grad():
        perm = torch.randperm(n_sensors)
        ops_perm = operators[:, perm]
        resp_perm = responses[:, perm]

        out_orig = model(operators, responses)["logits"]
        out_perm = model(ops_perm, resp_perm)["logits"]
        diff = (out_orig - out_perm).abs().max().item()
        assert diff < 1e-4, f"Permutation difference too large: {diff}"
        print(f"  Passed Permutation Invariance: max diff = {diff:.2e}")


def test_setonet():
    print("Testing SetONetEchoNext...")
    batch_size = 4
    n_sensors = 8
    operator_dim = 8

    operators = torch.randn(batch_size, n_sensors, operator_dim)
    responses = torch.randn(batch_size, n_sensors, 16, 128)
    tabular = torch.randn(batch_size, 7)

    for cond in ["SO-ECG", "SO-ECG+Tab-Trunk", "SO-ECG+Tab-Concat"]:
        model = build_setonet_echonext(cond)
        tab_in = None if cond == "SO-ECG" else tabular
        out = model(operators, responses, tabular=tab_in)

        logits = out["logits"]
        assert logits.shape == (batch_size, 1), f"Unexpected shape {logits.shape} for {cond}"
        assert torch.isfinite(logits).all(), f"Non-finite logits for {cond}"

        # Backward test
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name} for {cond}"

        print(f"  Passed {cond}: Shape {logits.shape}, finite gradients verified.")

    # Permutation invariance test
    model = build_setonet_echonext("SO-ECG")
    model.eval()
    with torch.no_grad():
        perm = torch.randperm(n_sensors)
        ops_perm = operators[:, perm]
        resp_perm = responses[:, perm]

        out_orig = model(operators, responses)["logits"]
        out_perm = model(ops_perm, resp_perm)["logits"]
        diff = (out_orig - out_perm).abs().max().item()
        assert diff < 1e-4, f"Permutation difference too large: {diff}"
        print(f"  Passed Permutation Invariance: max diff = {diff:.2e}")


if __name__ == "__main__":
    test_setoperator()
    test_setonet()
    print("\nALL ARCHITECTURE TESTS PASSED SUCCESSFULLY!")
