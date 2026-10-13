#!/usr/bin/env python3
"""Unit tests for Faithful Operator Embedding in OperatorSetModel."""

import numpy as np
import torch
import torch.nn as nn
from repecg.paper07_operator import OperatorSetModel


def test_faithful_operator():
    print("=== Testing Faithful Operator Embedding in OperatorSetModel ===")
    torch.manual_seed(42)

    # 1. Instantiate OperatorSetModel in faithful mode
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="faithful")
    model.eval()

    # 2. Canonical basis operators: 8 independent leads
    # [I, II, V1, V2, V3, V4, V5, V6]
    q_I = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_II = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    q_V2 = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    q_V3 = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    # Derived leads via exact linear algebra
    q_III = (q_II - q_I) / np.sqrt(2.0)
    q_aVR = (-q_I - q_II) / np.sqrt(2.0)
    q_aVL = (q_I - 0.5 * q_II) / np.sqrt(1.25)
    q_aVF = (q_II - 0.5 * q_I) / np.sqrt(1.25)
    q_V3_minus_V2 = (q_V3 - q_V2) / np.sqrt(2.0)

    operators = torch.stack([q_I, q_II, q_III, q_aVR, q_aVL, q_aVF, q_V2, q_V3, q_V3_minus_V2]).unsqueeze(0) # (1, 9, 8)

    # 3. Test encode_operator
    e_q = model.encode_operator(operators, None) # (1, 9, 64)
    assert e_q.shape == (1, 9, 64), f"Expected shape (1, 9, 64), got {e_q.shape}"
    print("[PASS] e_q output shape is strictly (1, 9, 64)")

    # 4. Test exact projection property P e_q == q
    P_e_q = e_q[..., :8] # Projection selecting first 8 coordinates
    max_diff = torch.max(torch.abs(P_e_q - operators)).item()
    assert max_diff < 1e-7, f"P e_q does not match q! Max diff: {max_diff}"
    print(f"[PASS] Exact operator recovery: P e_q == q (max abs diff = {max_diff:.2e})")

    # 5. Test limb lead algebra preservation
    # P e_{III} == (P e_{II} - P e_I) / sqrt(2)
    P_e_I = e_q[0, 0, :8]
    P_e_II = e_q[0, 1, :8]
    P_e_III = e_q[0, 2, :8]
    diff_III = torch.max(torch.abs(P_e_III - (P_e_II - P_e_I) / np.sqrt(2.0))).item()
    assert diff_III < 1e-7, f"Lead III algebra violated: diff = {diff_III}"
    print(f"[PASS] Lead III algebra exact: P e_III == (P e_II - P e_I)/sqrt(2) (diff = {diff_III:.2e})")

    # aVR algebra
    P_e_aVR = e_q[0, 3, :8]
    diff_aVR = torch.max(torch.abs(P_e_aVR - (-P_e_I - P_e_II) / np.sqrt(2.0))).item()
    assert diff_aVR < 1e-7, f"Lead aVR algebra violated: diff = {diff_aVR}"
    print(f"[PASS] Lead aVR algebra exact: P e_aVR == -(P e_I + P e_II)/sqrt(2) (diff = {diff_aVR:.2e})")

    # aVL algebra
    P_e_aVL = e_q[0, 4, :8]
    diff_aVL = torch.max(torch.abs(P_e_aVL - (P_e_I - 0.5 * P_e_II) / np.sqrt(1.25))).item()
    assert diff_aVL < 1e-7, f"Lead aVL algebra violated: diff = {diff_aVL}"
    print(f"[PASS] Lead aVL algebra exact: P e_aVL == (P e_I - 0.5 P e_II)/sqrt(1.25) (diff = {diff_aVL:.2e})")

    # aVF algebra
    P_e_aVF = e_q[0, 5, :8]
    diff_aVF = torch.max(torch.abs(P_e_aVF - (P_e_II - 0.5 * P_e_I) / np.sqrt(1.25))).item()
    assert diff_aVF < 1e-7, f"Lead aVF algebra violated: diff = {diff_aVF}"
    print(f"[PASS] Lead aVF algebra exact: P e_aVF == (P e_II - 0.5 P e_I)/sqrt(1.25) (diff = {diff_aVF:.2e})")

    # Oblique V3 - V2 algebra
    P_e_V2 = e_q[0, 6, :8]
    P_e_V3 = e_q[0, 7, :8]
    P_e_V3_V2 = e_q[0, 8, :8]
    diff_V3_V2 = torch.max(torch.abs(P_e_V3_V2 - (P_e_V3 - P_e_V2) / np.sqrt(2.0))).item()
    assert diff_V3_V2 < 1e-7, f"V3 - V2 algebra violated: diff = {diff_V3_V2}"
    print(f"[PASS] Oblique V3-V2 algebra exact: P e_V3-V2 == (P e_V3 - P e_V2)/sqrt(2) (diff = {diff_V3_V2:.2e})")

    # 6. Test full forward pass with synthetic responses
    batch_size = 4
    num_channels = 6
    ops = torch.randn(batch_size, num_channels, 8)
    ops = ops / torch.norm(ops, dim=-1, keepdim=True)
    resps = torch.randn(batch_size, num_channels, 16, 128)

    model.train()
    logits = model(ops, resps)
    assert logits.shape == (batch_size, 5), f"Expected logits shape ({batch_size}, 5), got {logits.shape}"
    print(f"[PASS] Full forward pass runs cleanly: logits shape is {logits.shape}")

    # 7. Test backward pass and gradient flow
    target = torch.randint(0, 2, (batch_size, 5)).float()
    loss = nn.BCEWithLogitsLoss()(logits, target)
    loss.backward()

    op_grad_norm = sum(p.grad.norm().item() for p in model.operator.parameters() if p.grad is not None)
    assert op_grad_norm > 0, "No gradients flowed to residual operator parameters!"
    print(f"[PASS] Gradient flow verified: residual operator grad norm = {op_grad_norm:.4f}")

    print("\n>>> ALL 7 FAITHFUL OPERATOR CHECKS PASSED SUCCESSFULLY! <<<\n")


if __name__ == "__main__":
    test_faithful_operator()
