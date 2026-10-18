#!/usr/bin/env python3
"""Unit tests for Observable-Concept SetOperator (OCSetOperator)."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest
import torch

from experiments.echonext_operator_benchmark.oc_setoperator import OCSetOperator


@pytest.fixture
def sample_ecg_batch():
    torch.manual_seed(42)
    B, m, d, T = 2, 4, 8, 2048
    X = torch.randn(B, d, T)
    # Random measurement operators
    Q = torch.randn(B, m, d)
    # Orthogonalize rows
    for b in range(B):
        q_b, _ = torch.linalg.qr(Q[b].T)
        Q[b] = q_b.T[:m]
    Y = torch.bmm(Q, X)
    tabular = torch.randn(B, 7)
    return {"X": X, "Q": Q, "Y": Y, "tabular": tabular, "B": B, "m": m, "d": d, "T": T}


def test_oc_setoperator_transparency_sum(sample_ecg_batch):
    """Verifies that the output logit exactly equals b + sum f_j(c_j) + f_tab(x_tab)."""
    model = OCSetOperator(mode="strict", in_leads=8, num_tabular=7)
    model.eval()

    Q = sample_ecg_batch["Q"]
    Y = sample_ecg_batch["Y"]
    tab = sample_ecg_batch["tabular"]

    with torch.no_grad():
        out = model(Q, Y, tabular=tab)

    logits = out["logits"].squeeze(-1)  # (B,)
    expected_sum = model.bias.expand(sample_ecg_batch["B"]).clone()
    for key in model.concept_keys:
        expected_sum = expected_sum + out["concept_contributions"][key]
    expected_sum = expected_sum + out["tab_contribution"]

    torch.testing.assert_close(logits, expected_sum, atol=1e-5, rtol=1e-5)


def test_oc_setoperator_basis_rotation_invariance(sample_ecg_batch):
    """Verifies that rotating the measurement basis R in O(m) leaves outputs identical."""
    model = OCSetOperator(mode="strict", in_leads=8, num_tabular=7)
    model.eval()

    Q = sample_ecg_batch["Q"]
    Y = sample_ecg_batch["Y"]
    tab = sample_ecg_batch["tabular"]

    # Generate random orthogonal rotation R in O(m)
    m = sample_ecg_batch["m"]
    R, _ = torch.linalg.qr(torch.randn(m, m))
    R_batch = R.unsqueeze(0).expand(sample_ecg_batch["B"], -1, -1)

    Q_rot = torch.bmm(R_batch, Q)
    Y_rot = torch.bmm(R_batch, Y)

    with torch.no_grad():
        out_orig = model(Q, Y, tabular=tab)
        out_rot = model(Q_rot, Y_rot, tabular=tab)

    # Logits and probs must be identical
    torch.testing.assert_close(out_rot["logits"], out_orig["logits"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_rot["probs"], out_orig["probs"], atol=1e-5, rtol=1e-5)

    # Concept estimates and contributions must be identical
    for key in model.concept_keys:
        torch.testing.assert_close(out_rot["concept_means"][key], out_orig["concept_means"][key], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_rot["concept_contributions"][key], out_orig["concept_contributions"][key], atol=1e-5, rtol=1e-5)


def test_oc_setoperator_permutation_invariance(sample_ecg_batch):
    """Verifies that permuting measurement rows leaves outputs identical."""
    model = OCSetOperator(mode="strict", in_leads=8, num_tabular=7)
    model.eval()

    Q = sample_ecg_batch["Q"]
    Y = sample_ecg_batch["Y"]
    tab = sample_ecg_batch["tabular"]

    perm = [2, 0, 3, 1]
    Q_perm = Q[:, perm, :]
    Y_perm = Y[:, perm, :]

    with torch.no_grad():
        out_orig = model(Q, Y, tabular=tab)
        out_perm = model(Q_perm, Y_perm, tabular=tab)

    torch.testing.assert_close(out_perm["logits"], out_orig["logits"], atol=1e-5, rtol=1e-5)
    for key in model.concept_keys:
        torch.testing.assert_close(out_perm["concept_means"][key], out_orig["concept_means"][key], atol=1e-5, rtol=1e-5)


def test_oc_setoperator_patient_evidence_report(sample_ecg_batch):
    """Verifies generation of structured clinical evidence breakdown."""
    model = OCSetOperator(mode="strict", in_leads=8, num_tabular=7)
    model.eval()

    # Single patient
    Q_single = sample_ecg_batch["Q"][:1]
    Y_single = sample_ecg_batch["Y"][:1]
    tab_single = sample_ecg_batch["tabular"][:1]

    with torch.no_grad():
        out = model(Q_single, Y_single, tabular=tab_single, return_evidence_report=True)

    assert "patient_report" in out
    assert "Predicted SHD Risk" in out["patient_report"]
    assert "Clinical Concept Evidence Table" in out["patient_report"]
    assert "observability_audit" in out
    print("\n--- SAMPLE PATIENT EVIDENCE REPORT ---")
    print(out["patient_report"])
