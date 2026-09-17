from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from pydmd import DMD

from repecg.common.models import HankelDynamicsModel
from repecg.common.variants import ExperimentVariant


def test_world_1_linear_system_recovery() -> None:
    """World 1: Linear Dynamical System Exact Operator Recovery & Low-Data Prediction."""
    np.random.seed(42)
    # Part (a): Sufficiently observed system (T >= 100, d=4)
    # Generate stable block-orthogonal discrete-time linear system x_{t+1} = A^* x_t
    theta1, theta2 = 0.3, 0.5
    R1 = np.array([[np.cos(theta1), -np.sin(theta1)], [np.sin(theta1), np.cos(theta1)]])
    R2 = np.array([[np.cos(theta2), -np.sin(theta2)], [np.sin(theta2), np.cos(theta2)]])
    A_star = np.block([[R1, np.zeros((2, 2))], [np.zeros((2, 2)), R2]]) * 0.98

    x0 = np.array([1.0, 0.5, -0.5, 0.8], dtype=np.float64)
    trajectory = [x0]
    for _ in range(120):
        trajectory.append(A_star @ trajectory[-1])
    X_traj = np.array(trajectory).T  # (4, 121)
    X1 = X_traj[:, :-1]  # (4, 120)
    X2 = X_traj[:, 1:]   # (4, 120)

    # Solve least-squares with minimal machine-scale ridge
    C11 = X1 @ X1.T + 1e-10 * np.eye(4)
    C21 = X2 @ X1.T
    A_hat = C21 @ np.linalg.inv(C11)
    frob_err = np.linalg.norm(A_hat - A_star, "fro")
    assert frob_err < 1e-5, f"World 1a failed: frob_err={frob_err:.2e}"

    # Under additive Gaussian noise (SNR = 20 dB)
    noise = np.random.randn(*X_traj.shape) * 0.02
    X_noisy = X_traj + noise
    X1_n, X2_n = X_noisy[:, :-1], X_noisy[:, 1:]
    A_hat_noisy = (X2_n @ X1_n.T) @ np.linalg.inv(X1_n @ X1_n.T + 1e-4 * np.eye(4))
    rel_err = np.linalg.norm(A_hat_noisy - A_star, "fro") / np.linalg.norm(A_star, "fro")
    assert rel_err < 0.08, f"World 1a noisy failed: rel_err={rel_err:.4f}"

    # Part (b): Low-data Paper-04 geometry (T=16, d=4, L=4 => DH=16)
    # Evaluate one-step prediction error on held-out continuation
    X_train = X_traj[:, :16]
    X_test = X_traj[:, 16:32]
    H_blocks_train = [X_train[:, i:i+13] for i in range(4)]
    H_train = np.concatenate(H_blocks_train, axis=0) # (16, 13)
    H1_tr = H_train[:, :-1]
    H2_tr = H_train[:, 1:]
    A_dh16 = (H2_tr @ H1_tr.T) @ np.linalg.inv(H1_tr @ H1_tr.T + 1e-4 * np.eye(16))

    H_blocks_test = [X_test[:, i:i+13] for i in range(4)]
    H_test = np.concatenate(H_blocks_test, axis=0)
    H1_te = H_test[:, :-1]
    H2_te = H_test[:, 1:]
    pred_err = np.linalg.norm(H2_te - A_dh16 @ H1_te, "fro") / np.linalg.norm(H2_te, "fro")
    assert pred_err < 0.05, f"World 1b prediction error too high: {pred_err:.4f}"


def test_world_2_finite_memory_delay_augmentation() -> None:
    """World 2: Finite-Memory Recovery Through Delay Augmentation (AR(2) Oscillator)."""
    r = 0.90
    omega = np.pi / 6.0
    s = [1.0, r * np.cos(omega)]
    for _ in range(250):
        s.append(2 * r * np.cos(omega) * s[-1] - (r**2) * s[-2])
    s = np.array(s, dtype=np.float64)

    # L = 1 (Markovian assumption on scalar s_t):
    X_L1 = s[:-1][None, :]
    Y_L1 = s[1:][None, :]
    a_L1 = (Y_L1 @ X_L1.T) / (X_L1 @ X_L1.T)
    res_L1 = np.linalg.norm(Y_L1 - a_L1 @ X_L1) / np.linalg.norm(Y_L1)

    # L = 2 (Delay augmentation captures [s_t, s_{t-1}] state):
    H_L2 = np.stack([s[1:-1], s[:-2]], axis=0)
    H_L2_next = np.stack([s[2:], s[1:-1]], axis=0)
    A_L2 = (H_L2_next @ H_L2.T) @ np.linalg.inv(H_L2 @ H_L2.T)
    res_L2 = np.linalg.norm(H_L2_next - A_L2 @ H_L2) / np.linalg.norm(H_L2_next)

    # L = 4 (Overspecified delay depth):
    H_L4 = np.stack([s[3:-1], s[2:-2], s[1:-3], s[:-4]], axis=0)
    H_L4_next = np.stack([s[4:], s[3:-1], s[2:-2], s[1:-3]], axis=0)
    A_L4 = (H_L4_next @ H_L4.T) @ np.linalg.inv(H_L4 @ H_L4.T + 1e-8 * np.eye(4))
    res_L4 = np.linalg.norm(H_L4_next - A_L4 @ H_L4) / np.linalg.norm(H_L4_next)

    assert res_L1 > 0.25, f"L=1 residual unexpectedly low: {res_L1:.4f}"
    assert res_L2 < 0.01, f"L=2 residual too high: {res_L2:.6f}"
    assert res_L2 < res_L1 / 20.0
    assert abs(res_L4 - res_L2) < 0.02


def test_world_3_ridge_floor_and_degeneracy() -> None:
    """World 3: Scale-Relative Ridge with Floor & Degenerate Input Stability."""
    torch.manual_seed(42)
    # Test 1: All-zero input (rank 0)
    model = HankelDynamicsModel(input_dim=128, proj_dim=4, lag=4)
    zero_input = torch.zeros(4, 16, 128, requires_grad=True)
    out_zero = model(zero_input)
    assert out_zero.shape == (4, 5)
    assert torch.isfinite(out_zero).all(), "All-zero input produced non-finite output"

    # Backward gradient check
    loss = out_zero.sum()
    loss.backward()
    assert zero_input.grad is not None
    assert torch.isfinite(zero_input.grad).all(), "All-zero input backward grad non-finite"

    # Test 2: Constant signal x_t = c (rank 1)
    const_input = torch.ones(4, 16, 128, requires_grad=True) * 3.14
    out_const = model(const_input)
    assert torch.isfinite(out_const).all()

    # Test 3: Condition number bounded by lambda_min floor
    DH = 16
    X_zero = torch.zeros(1, DH, 16)
    C11 = torch.bmm(X_zero, X_zero.transpose(-1, -2))
    scale = C11.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    reg_lambda = torch.clamp(1e-4 * scale, min=1e-6)
    assert abs(reg_lambda.item() - 1e-6) < 1e-9
    C11_reg = C11 + reg_lambda[:, None, None] * torch.eye(DH).unsqueeze(0)
    cond = torch.linalg.cond(C11_reg)
    assert cond.item() == 1.0


def test_world_4_out_of_sample_order_sensitivity() -> None:
    """World 4: Out-of-Sample Temporal Order Sensitivity."""
    np.random.seed(42)
    T_total = 500
    t = np.arange(T_total)
    signal = np.stack([
        np.cos(2 * np.pi * t / 16.0),
        np.sin(2 * np.pi * t / 16.0),
        np.cos(4 * np.pi * t / 16.0),
        np.sin(4 * np.pi * t / 16.0),
    ], axis=0)  # (4, 500)

    train_sig = signal[:, :300]
    test_sig = signal[:, 300:]

    H_train = np.concatenate([train_sig[:, i:i+290] for i in range(4)], axis=0)
    X_tr = H_train[:, :-1]
    Y_tr = H_train[:, 1:]
    A_ordered = (Y_tr @ X_tr.T) @ np.linalg.inv(X_tr @ X_tr.T + 1e-4 * np.eye(16))

    # Permuting the physical temporal timeline destroys true transitions
    perm_t = np.random.permutation(train_sig.shape[1])
    train_sig_shuffled = train_sig[:, perm_t]
    H_shuff = np.concatenate([train_sig_shuffled[:, i:i+290] for i in range(4)], axis=0)
    X_sh = H_shuff[:, :-1]
    Y_sh = H_shuff[:, 1:]
    A_shuffled = (Y_sh @ X_sh.T) @ np.linalg.inv(X_sh @ X_sh.T + 1e-4 * np.eye(16))

    H_test = np.concatenate([test_sig[:, i:i+190] for i in range(4)], axis=0)
    X_te = H_test[:, :-1]
    Y_te = H_test[:, 1:]
    e_ordered = np.linalg.norm(Y_te - A_ordered @ X_te, "fro") / np.linalg.norm(Y_te, "fro")
    e_shuffled = np.linalg.norm(Y_te - A_shuffled @ X_te, "fro") / np.linalg.norm(Y_te, "fro")

    assert e_ordered < 0.05, f"Ordered test error too high: {e_ordered:.4f}"
    assert e_shuffled > 2.0 * e_ordered, f"Shuffled did not degrade sufficiently: ordered={e_ordered:.4f}, shuffled={e_shuffled:.4f}"


def test_world_5_historical_ineligibility_preservation() -> None:
    """World 5: Historical Spectral-Descriptor Ineligibility Preservation (Negative Finding)."""
    manifest_path = Path(
        "/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/"
        "paper04_hankel_dynamics/development_representations/manifest.json"
    )
    assert manifest_path.exists(), "Historical manifest must exist"
    payload = json.loads(manifest_path.read_text())

    assert payload["status"] == "ineligible_nystrom_fidelity"
    audits = payload["audits"]
    assert "128" in audits and "256" in audits

    assert not audits["128"]["passed"]
    assert not audits["256"]["passed"]
    assert audits["128"]["spearman"] < 0.70
    assert audits["256"]["spearman"] < 0.70
    assert audits["128"]["median_relative_error"] > 0.80
    assert audits["256"]["median_relative_error"] > 0.80


def test_world_6_cyclic_closure_advantage() -> None:
    """World 6: Cyclic Closure Advantage on Periodic Orbits."""
    # Synthetic cardiac activation waveform with asymmetric multi-harmonic structure
    phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    wave1 = np.sin(phi) + 0.5 * np.sin(2 * phi) + 0.3 * np.cos(3 * phi)
    wave2 = np.cos(phi) - 0.4 * np.sin(2 * phi) + 0.2 * np.sin(4 * phi)
    X_ecg = np.stack([wave1, wave2], axis=0)  # (2, 16)

    # (a) Open-chain Hankel: 14 transitions, discarding wrap-around 15 -> 0
    H_open = np.concatenate([X_ecg[:, :15], X_ecg[:, 1:]], axis=0)  # (4, 15)
    X_open = H_open[:, :-1]  # (4, 14)
    Y_open = H_open[:, 1:]   # (4, 14)
    A_open = (Y_open @ X_open.T) @ np.linalg.inv(X_open @ X_open.T + 1e-4 * np.eye(4))

    # (b) Cyclic Hankel: all 16 transitions with wrap-around
    H_cyclic = np.concatenate([X_ecg, np.roll(X_ecg, -1, axis=1)], axis=0)  # (4, 16)
    X_cyclic = H_cyclic
    Y_cyclic = np.roll(H_cyclic, -1, axis=1)
    A_cyclic = (Y_cyclic @ X_cyclic.T) @ np.linalg.inv(X_cyclic @ X_cyclic.T + 1e-4 * np.eye(4))

    # Boundary transition test: predicting state 0 from state 15
    h_boundary = np.concatenate([X_ecg[:, 15], X_ecg[:, 0]], axis=0)
    h_target = np.concatenate([X_ecg[:, 0], X_ecg[:, 1]], axis=0)

    err_open = np.linalg.norm(h_target - A_open @ h_boundary) / np.linalg.norm(h_target)
    err_cyclic = np.linalg.norm(h_target - A_cyclic @ h_boundary) / np.linalg.norm(h_target)

    assert err_cyclic < 0.10, f"Cyclic boundary error unexpectedly high: {err_cyclic:.4f}"
    assert err_cyclic < err_open / 2.0, f"Cyclic did not outperform open chain: cyclic={err_cyclic:.4f}, open={err_open:.4f}"


def test_world_7_regularization_path_tradeoff() -> None:
    """World 7: Regularization Path Bias-Variance Tradeoff."""
    np.random.seed(42)
    D = 8
    t_steps = 25
    X_clean = np.random.randn(D, t_steps)
    U_mat, s_vals, Vt_mat = np.linalg.svd(X_clean, full_matrices=False)
    s_decay = np.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.01, 0.005, 0.001])
    X_ill = U_mat @ np.diag(s_decay) @ Vt_mat
    A_true = np.eye(D) * 0.7
    Y_clean = A_true @ X_ill
    Y_noisy = Y_clean + np.random.randn(*Y_clean.shape) * 0.05

    X_te_ill = np.random.randn(D, 30)
    Y_te_clean = A_true @ X_te_ill

    lambdas = [1e-7, 1e-5, 1e-3, 0.1, 1.0, 10.0]
    test_errors = []
    norm_A = []

    for lam in lambdas:
        C11 = X_ill @ X_ill.T + lam * np.eye(D)
        C21 = Y_noisy @ X_ill.T
        A_est = C21 @ np.linalg.inv(C11)
        test_err = np.linalg.norm(Y_te_clean - A_est @ X_te_ill, "fro") / np.linalg.norm(Y_te_clean, "fro")
        test_errors.append(test_err)
        norm_A.append(np.linalg.norm(A_est, "fro"))

    # Verify:
    # 1. Norm of A strictly decreases as regularization grows
    for i in range(len(norm_A) - 1):
        assert norm_A[i] >= norm_A[i+1] - 1e-6
    # 2. Optimal test error occurs at intermediate lambda (bias-variance trade-off)
    min_idx = int(np.argmin(test_errors))
    assert 0 < min_idx < len(lambdas) - 1, f"Expected internal minimum, got index {min_idx} for lambdas {lambdas}"


def test_world_8_solve_orientation_and_gradients() -> None:
    """World 8: Differentiable Operator Solve & Float64 Unit Check."""
    torch.manual_seed(42)
    B, DH, M = 2, 16, 16
    X = torch.randn(B, DH, M, dtype=torch.float64)
    Y = torch.randn(B, DH, M, dtype=torch.float64)
    C11 = torch.bmm(X, X.transpose(-1, -2)) + 1e-4 * torch.eye(DH, dtype=torch.float64).unsqueeze(0)
    C21 = torch.bmm(Y, X.transpose(-1, -2))

    A_inv = torch.bmm(C21, torch.linalg.inv(C11))
    A_solve = torch.linalg.solve(C11, C21.transpose(-1, -2)).transpose(-1, -2)
    max_err = torch.max(torch.abs(A_inv - A_solve)).item()
    assert max_err < 1e-10, f"Solve orientation error: {max_err:.2e}"

    model = HankelDynamicsModel(input_dim=128, proj_dim=4, lag=4, classes=5)
    features = torch.randn(4, 16, 128, requires_grad=True)
    out = model(features)
    assert out.shape == (4, 5)
    loss = nn.functional.binary_cross_entropy_with_logits(out, torch.ones(4, 5))
    loss.backward()

    assert model.proj.weight.grad is not None
    grad_norm = model.proj.weight.grad.norm().item()
    assert grad_norm > 1e-6, f"Gradient vanished: norm={grad_norm:.2e}"
    assert torch.isfinite(model.proj.weight.grad).all()


def test_world_9_pydmd_oracle_equivalence() -> None:
    """World 9: PyDMD Independent Numerical Oracle Equivalence."""
    A_star = np.array([[0.8, -0.2], [0.2, 0.8]])
    x0 = np.array([1.0, 0.5])
    trajectory = [x0]
    for _ in range(50):
        trajectory.append(A_star @ trajectory[-1])
    X_traj = np.array(trajectory).T  # (2, 51)

    # 1. Public PyDMD
    dmd = DMD(svd_rank=2)
    dmd.fit(X_traj)
    eigs_pydmd = np.sort(np.abs(dmd.eigs))

    # 2. Our least-squares formulation
    X1 = X_traj[:, :-1]
    X2 = X_traj[:, 1:]
    A_ours = (X2 @ X1.T) @ np.linalg.inv(X1 @ X1.T)
    eigs_ours = np.sort(np.abs(np.linalg.eigvals(A_ours)))

    assert np.allclose(eigs_pydmd, eigs_ours, atol=1e-6), (
        f"PyDMD eigs {eigs_pydmd} != our eigs {eigs_ours}"
    )
