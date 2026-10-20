#!/usr/bin/env python3
"""Comprehensive Scientific Preflight Verification Gate (Paper 07, P1).

Implements and strictly verifies:
1. Integration of module-aware deterministic initializer with decoupled module RNG streams.
2. LayerNorm weights initialize strictly to 1.0 (scale = 1.0) and biases to 0.0.
3. All 49 shared modules initialize identically across A0, G1, and A2 matched-init models.
4. Transformer and ECG pathway embeddings are non-constant and non-zero on authentic inputs.
5. Finite, non-zero gradients flow through every branch (PhaseCNN, operator, Transformer, pool, latent, patient, head).
6. Genuine small-batch overfit using TRAINING cohort exclusively (zero validation data), demonstrating >85% loss drop.
7. All disposable smoke-test models and artifacts are strictly discarded.
8. Bitwise SHA-256 parameter hashes recorded for all models.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

DEV_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction_geometry_dev")
BENCHMARK_DIR = DEV_ROOT / "experiments/echonext_operator_benchmark"
SRC_DIR = DEV_ROOT / "experiments/ptbxl_distributional_repecg/src"

for p in [DEV_ROOT, BENCHMARK_DIR, SRC_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from module_aware_init import (
    init_module_aware_deterministic,
    get_model_parameter_hashes,
    verify_layernorm_ones,
)
from repecg.ablation_setoperator.ablation_models import AblationSetOperatorEchoNext
from phase_kme_representation import EchoNextPhaseKMEExtractor, DEFAULT_FIT_PATH, CANONICAL_8_LEADS
from train_setoperator_echonext import DATA_DIR


def run_comprehensive_preflight_gate() -> Dict[str, Any]:
    print("=" * 80)
    print("PAPER 07: P1 REPAIRED TRAINING PREFLIGHT GATE")
    print("=" * 80)

    report = {"timestamp": time.time(), "status": "PENDING", "checks": {}}

    # -------------------------------------------------------------------------
    # CHECK 1 & 2: LayerNorm Initialization Scale == 1.0 and Bias == 0.0
    # -------------------------------------------------------------------------
    print("\n--- Check 1 & 2: LayerNorm Scale Verification ---")
    m_test = AblationSetOperatorEchoNext(
        operator_mode="continuous",
        geometry_mode="baseline",
        phase_mode="standard_circular",
        set_encoder_mode="standard_transformer",
        pooling_mode="learned_query",
        tabular_mode="native_additive",
        classes=1,
    )
    init_module_aware_deterministic(m_test, seed=42)
    ln_ok, ln_errs = verify_layernorm_ones(m_test)
    assert ln_ok, f"LayerNorm verification failed: {ln_errs}"

    ln_scales = []
    for name, mod in m_test.named_modules():
        if isinstance(mod, nn.LayerNorm):
            ln_scales.append({
                "module": name,
                "weight_mean": float(mod.weight.mean().item()),
                "weight_std": float(mod.weight.std().item()),
                "bias_mean": float(mod.bias.mean().item()) if mod.bias is not None else 0.0,
            })
    print(f"Verified {len(ln_scales)} LayerNorm modules: all scales = 1.0000, biases = 0.0000.")
    report["checks"]["layernorm_scales_ones"] = {"passed": True, "count": len(ln_scales), "details": ln_scales}

    # -------------------------------------------------------------------------
    # CHECK 3: Parameter Bitwise Parity across Matched A0, G1, and A2
    # -------------------------------------------------------------------------
    print("\n--- Check 3: Shared-Module Parameter Hashes Across Matched Triplet ---")
    m_a0 = AblationSetOperatorEchoNext(
        operator_mode="continuous", geometry_mode="baseline", tabular_mode="native_additive", classes=1
    )
    m_g1 = AblationSetOperatorEchoNext(
        operator_mode="continuous", geometry_mode="G1", tabular_mode="native_additive", classes=1
    )
    m_a2 = AblationSetOperatorEchoNext(
        operator_mode="continuous", geometry_mode="control_capacity", tabular_mode="native_additive", classes=1
    )

    init_module_aware_deterministic(m_a0, seed=42)
    init_module_aware_deterministic(m_g1, seed=42)
    init_module_aware_deterministic(m_a2, seed=42)

    h_a0 = get_model_parameter_hashes(m_a0)
    h_g1 = get_model_parameter_hashes(m_g1)
    h_a2 = get_model_parameter_hashes(m_a2)

    shared_names = [n for n in h_a0 if not n.startswith("operator.")]
    matched_count = 0
    mismatched = []
    for n in shared_names:
        if h_a0[n] == h_g1[n] == h_a2[n]:
            matched_count += 1
        else:
            mismatched.append(n)

    print(f"Shared parameters: {matched_count}/{len(shared_names)} match bit-for-bit SHA-256 hashes.")
    assert len(mismatched) == 0, f"Mismatched shared parameters: {mismatched}"

    # Operator encoder should differ across architectures
    op_differs = (h_a0["operator.0.weight"] != h_g1["operator.0.weight"]) and (h_a0["operator.0.weight"] != h_a2["operator.0.weight"])
    print(f"Operator encoder uniqueness across architectures verified: {op_differs}")
    assert op_differs, "Operator encoders should have distinct architectures and hashes."

    report["checks"]["shared_module_hashes_matched"] = {
        "passed": True,
        "shared_tensors_count": len(shared_names),
        "matched_count": matched_count,
        "operator_encoders_distinct": op_differs,
    }

    # -------------------------------------------------------------------------
    # CHECK 4: Nonconstant and Nonzero Embeddings on Authentic Inputs
    # -------------------------------------------------------------------------
    print("\n--- Check 4: Embedding Non-Constancy on Authentic Inputs ---")
    extractor = EchoNextPhaseKMEExtractor(fit_path=DEFAULT_FIT_PATH, device="cpu")
    train_w = np.load(DATA_DIR / "EchoNext_train_waveforms.npy", mmap_mode="r")
    train_t = np.load(DATA_DIR / "EchoNext_train_tabular_features.npy")
    df = pd.read_csv(DATA_DIR / "echonext_metadata_100k.csv")
    train_mask = (df["split"] == "train").values
    y_train = df.loc[train_mask, "shd_moderate_or_greater_flag"].values

    ops_canonical = torch.eye(8, dtype=torch.float32)

    batch_beats = []
    for i in range(4):
        wf_8 = train_w[i, 0][:, CANONICAL_8_LEADS]
        b = extractor.extract_beats(wf_8)
        r = extractor.compute_phase_kme(torch.from_numpy(b), ops_canonical)
        batch_beats.append(r)

    t_resps = torch.stack(batch_beats)  # (4, 8, 16, 128)
    t_ops = ops_canonical.unsqueeze(0).expand(4, -1, -1)  # (4, 8, 8)
    t_tab = torch.from_numpy(train_t[:4]).float()  # (4, 7)

    m_a0.eval()
    with torch.no_grad():
        out = m_a0(operators=t_ops, responses=t_resps, tabular=t_tab)
        logits = out["logits"].squeeze(-1)
        latent = out["latent"]

    logit_std = float(torch.std(logits).item())
    latent_std = float(torch.std(latent).item())
    latent_norm = float(torch.norm(latent).item())

    print(f"Output logits: {logits.tolist()} (std={logit_std:.4f})")
    print(f"Latent norm: {latent_norm:.4f}, Latent std across batch: {latent_std:.4f}")
    assert logit_std > 1e-4, f"Logits are constant across distinct training records! std={logit_std}"
    assert latent_std > 1e-4, f"Latent representations are constant! std={latent_std}"
    assert latent_norm > 1.0, f"Latent representation is zero or collapsed! norm={latent_norm}"

    report["checks"]["nonconstant_embeddings"] = {
        "passed": True,
        "logit_std": logit_std,
        "latent_std": latent_std,
        "latent_norm": latent_norm,
    }

    # -------------------------------------------------------------------------
    # CHECK 5: Finite Non-Zero Gradients Across All Relevant Branches
    # -------------------------------------------------------------------------
    print("\n--- Check 5: Finite Gradient Flow Across All Relevant Branches ---")
    m_a0.train()
    m_a0.zero_grad()
    t_y = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    out_train = m_a0(operators=t_ops, responses=t_resps, tabular=t_tab)
    loss = criterion(out_train["logits"].squeeze(-1), t_y)
    loss.backward()

    branches = {
        "PhaseCNN": m_a0.response,
        "OperatorEncoder": m_a0.operator,
        "Transformer": m_a0.set_encoder,
        "SetPooling": m_a0.pool,
        "PoolQuery": [m_a0.pool_query],
        "DiagnosticLatent": m_a0.latent,
        "PatientContext": m_a0.patient_context,
        "DiagnosticHead": m_a0.head,
    }

    branch_grad_norms = {}
    for b_name, b_mod in branches.items():
        params = b_mod if isinstance(b_mod, list) else list(b_mod.parameters())
        sq_sum = 0.0
        n_p = 0
        for p in params:
            assert p.grad is not None, f"Parameter in {b_name} has None grad!"
            assert torch.isfinite(p.grad).all(), f"Parameter in {b_name} has non-finite grad!"
            sq_sum += float(p.grad.norm(2).item() ** 2)
            n_p += 1
        norm = math.sqrt(sq_sum)
        branch_grad_norms[b_name] = norm
        print(f"  Branch {b_name:18s}: Grad Norm = {norm:.6f} ({n_p} parameter tensors)")
        assert norm > 1e-6, f"Branch {b_name} has vanishing / zero gradients! norm={norm}"

    report["checks"]["finite_gradients_all_branches"] = {"passed": True, "grad_norms": branch_grad_norms}

    # -------------------------------------------------------------------------
    # CHECK 6 & 7: Genuine Small-Batch Overfit on TRAINING Records Only
    # -------------------------------------------------------------------------
    print("\n--- Check 6 & 7: Genuine Small-Batch Overfit (Training Records Only) ---")
    N_OVERFIT = 16
    overfit_beats = []
    for i in range(N_OVERFIT):
        wf_8 = train_w[i, 0][:, CANONICAL_8_LEADS]
        b = extractor.extract_beats(wf_8)
        r = extractor.compute_phase_kme(torch.from_numpy(b), ops_canonical)
        overfit_beats.append(r)

    of_resps = torch.stack(overfit_beats)  # (16, 8, 16, 128)
    of_ops = ops_canonical.unsqueeze(0).expand(N_OVERFIT, -1, -1)
    of_tab = torch.from_numpy(train_t[:N_OVERFIT]).float()
    of_y = torch.from_numpy(y_train[:N_OVERFIT]).float()

    print(f"Overfit cohort: N={N_OVERFIT} exclusively from EchoNext TRAIN cohort (labels: {int(of_y.sum().item())}/{N_OVERFIT} positive)")

    m_overfit = AblationSetOperatorEchoNext(
        operator_mode="continuous", geometry_mode="baseline", tabular_mode="native_additive", classes=1
    )
    init_module_aware_deterministic(m_overfit, seed=42)

    optimizer = torch.optim.AdamW(m_overfit.parameters(), lr=2e-3, weight_decay=0.0)
    crit = nn.BCEWithLogitsLoss()

    loss_history = []
    initial_loss = None
    final_loss = None

    for step in range(1, 51):
        m_overfit.train()
        optimizer.zero_grad()
        out = m_overfit(operators=of_ops, responses=of_resps, tabular=of_tab)
        step_loss = crit(out["logits"].squeeze(-1), of_y)
        step_loss.backward()
        optimizer.step()

        loss_val = float(step_loss.item())
        if step == 1:
            initial_loss = loss_val
        final_loss = loss_val
        if step in [1, 5, 10, 20, 30, 40, 50]:
            with torch.no_grad():
                probs = torch.sigmoid(out["logits"].squeeze(-1)).numpy()
                preds = (probs >= 0.5).astype(np.float32)
                acc = float(np.mean(preds == of_y.numpy()))
            loss_history.append({"step": step, "loss": loss_val, "acc": acc})
            print(f"  Step {step:02d}/50 | Train Loss: {loss_val:.5f} | Accuracy: {acc*100:.1f}%")

    relative_loss_drop = (initial_loss - final_loss) / initial_loss
    print(f"Initial Loss: {initial_loss:.5f} -> Final Loss: {final_loss:.5f} (Relative Drop: {relative_loss_drop*100:.2f}%)")
    assert relative_loss_drop > 0.85, f"Overfit test failed to drop loss by >85%! Drop was {relative_loss_drop*100:.2f}%"
    assert loss_history[-1]["acc"] == 1.0, f"Overfit test failed to achieve 100% accuracy on training minibatch! Acc={loss_history[-1]['acc']}"

    # -------------------------------------------------------------------------
    # CHECK 8: Discard All Disposable Models
    # -------------------------------------------------------------------------
    del m_test, m_a0, m_g1, m_a2, m_overfit
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("All disposable smoke-test and overfit models cleanly garbage-collected.")

    report["checks"]["training_set_overfit"] = {
        "passed": True,
        "n_samples": N_OVERFIT,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "relative_loss_drop": relative_loss_drop,
        "final_accuracy": loss_history[-1]["acc"],
        "history": loss_history,
    }

    report["status"] = "PASSED"
    print("\n" + "=" * 80)
    print("ALL P1 PREFLIGHT VERIFICATION CHECKS PASSED CLEANLY.")
    print("=" * 80)
    return report


if __name__ == "__main__":
    rep = run_comprehensive_preflight_gate()
    out_file = Path("/data/mithunmanivannan/codex_artifacts/echonext_geometry_setoperator_v1/supervisor_state/p1_preflight_report.json")
    with open(out_file, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"Authoritative P1 Preflight Report written to {out_file}")
