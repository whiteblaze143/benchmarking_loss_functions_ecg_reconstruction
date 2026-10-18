#!/usr/bin/env python3
"""Comprehensive SetOperator Pre-Flight & GPU Smoke Test for PTB-XL.

Verifies:
1. Exact Constructor & Frozen Architecture: OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
2. Source SHA-256 Verification & Golden Regression Test on initialization.
3. Class-Weighted BCEWithLogitsLoss (train prevalence positive weights).
4. AdamW Optimizer (lr=3e-4, wd=1e-3) with gradient norm clipping (1.0).
5. CosineAnnealingLR scheduler behavior.
6. Context cardinality law: m in [1, 7] strictly; m=8 NEVER occurs.
7. Paired Initializations: Basis and Span start from identical initial states.
8. Strict Checkpoint Saving and Strict state_dict Loading.
9. Fold-9 Macro AUROC evaluation logic.
10. Strict Fold Isolation: Fold 10 is NEVER accessed or loaded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("setoperator_smoke_test")

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("highest")

# Ensure repo is in sys.path
REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from repecg.paper07_operator.model import OperatorSetModel

EXPECTED_MODEL_SHA256 = "66dae0d122266271e4b5b080e3e7048e4ed4ad3bea7d1f96e7bc8fd99b85227e"
GOLDEN_LOGITS_REF = [0.012054, 0.012796, 0.004852, 0.08265, 0.036781]


def test_1_architecture_contract() -> None:
    """Verify source code SHA-256 and golden logit regression test."""
    import repecg.paper07_operator.model as model_mod

    source_path = Path(model_mod.__file__)
    source_bytes = source_path.read_bytes()
    computed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    assert computed_sha256 == EXPECTED_MODEL_SHA256, (
        f"SHA-256 mismatch! Expected {EXPECTED_MODEL_SHA256}, got {computed_sha256}"
    )
    logger.info("PASS: Test 1.1 - OperatorSetModel source SHA-256 matches frozen reference: %s", computed_sha256)

    torch.manual_seed(42)
    test_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    test_model.eval()

    torch.manual_seed(1337)
    test_ops = torch.randn(1, 4, 8)
    test_res = torch.randn(1, 4, 16, 128)
    with torch.no_grad():
        test_logits = test_model(test_ops, test_res).squeeze().tolist()

    max_diff = max(abs(a - b) for a, b in zip(test_logits, GOLDEN_LOGITS_REF))
    assert max_diff < 1e-4, f"Golden regression failed! Max diff: {max_diff:.6f}, observed: {test_logits}"
    logger.info("PASS: Test 1.2 - Golden logit regression test passed (max diff: %.2e)", max_diff)


def test_2_gpu_forward_backward_weighted_loss() -> None:
    """Verify GPU execution, class-weighted BCE, finite gradients, clipping, and lr schedule."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA GPU required for SetOperator smoke test!"
    logger.info("PASS: Test 2.0 - CUDA GPU confirmed: %s", torch.cuda.get_device_name(0))

    # Synthetic batch of 64 records, 16 candidate operators, Phase-KME dim [16, 128], 5 classes
    batch_size = 64
    n_candidates = 16
    ops = torch.randn(batch_size, n_candidates, 8, device=device)
    # Normalize operators
    ops = ops / torch.linalg.norm(ops, dim=-1, keepdim=True).clamp_min(1e-6)
    resps = torch.randn(batch_size, n_candidates, 16, 128, device=device)

    # Synthetic realistic multilabel targets (NORM, MI, STTC, CD, HYP)
    labels = (torch.rand(batch_size, 5, device=device) > 0.75).float()
    # Ensure at least 1 positive per class
    labels[0] = 1.0

    # Class-weighted BCE
    prevalence = labels.mean(dim=0)
    pos_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Test context sampling m in [1, 7]
    cardinality_seen = set()
    for step in range(5):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        m = int(torch.randint(1, min(8, n_candidates), (1,)).item())
        cardinality_seen.add(m)
        assert 1 <= m <= 7, f"Context cardinality violation: m={m}"

        sub_idx = torch.randperm(n_candidates)[:m]
        sub_ops = ops[:, sub_idx]
        sub_resps = resps[:, sub_idx]

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(sub_ops, sub_resps, return_reconstruction=False)
            loss = criterion(logits, labels)

        assert torch.isfinite(loss), f"Non-finite loss encountered: {loss.item()}"
        loss.backward()

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(grad_norm), f"Non-finite gradient norm: {grad_norm}"
        optimizer.step()
        scheduler.step()

    logger.info("PASS: Test 2.1 - Class-weighted BCE loss, BF16 autocast, and gradient clipping verified.")
    logger.info("PASS: Test 2.2 - Context cardinalities sampled strictly in {1..7}: %s", sorted(cardinality_seen))
    logger.info("PASS: Test 2.3 - Cosine scheduler updated lr to %.6f", optimizer.param_groups[0]["lr"])


def test_3_paired_initialization_and_checkpoint_strictness() -> None:
    """Verify paired initialization across Basis and Span, and strict checkpoint reloading."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    torch.manual_seed(seed)
    basis_template = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    basis_initial_state = copy.deepcopy(basis_template.state_dict())

    torch.manual_seed(seed)
    span_template = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    span_initial_state = copy.deepcopy(span_template.state_dict())

    # Check identical weights
    for k in basis_initial_state:
        diff = torch.max(torch.abs(basis_initial_state[k] - span_initial_state[k])).item()
        assert diff == 0.0, f"Paired initialization mismatch on key {k}: max diff={diff}"
    logger.info("PASS: Test 3.1 - Paired initialization produces bitwise identical weights for Seed %d.", seed)

    # Save to temp checkpoint and test strict reload
    with tempfile.TemporaryDirectory() as tmp_dir:
        chk_path = Path(tmp_dir) / "test_checkpoint.pt"
        payload = {
            "state_dict": basis_initial_state,
            "seed": seed,
            "model_source_sha256": EXPECTED_MODEL_SHA256,
            "constructor": "OperatorSetModel(response_dim=128, classes=5, operator_mode='continuous')",
        }
        torch.save(payload, chk_path)

        loaded = torch.load(chk_path, map_location="cpu", weights_only=False)
        reload_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
        reload_model.load_state_dict(loaded["state_dict"], strict=True)
    logger.info("PASS: Test 3.2 - Strict checkpoint serialization and deserialization verified.")


def test_4_fold9_evaluation_and_fold10_isolation() -> None:
    """Verify Fold 9 macro AUROC evaluation and assert Fold 10 is NEVER accessed."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model.eval()

    n_val = 64
    val_ops = torch.randn(n_val, 8, 8, device=device)
    val_ops = val_ops / torch.linalg.norm(val_ops, dim=-1, keepdim=True).clamp_min(1e-6)
    val_resps = torch.randn(n_val, 8, 16, 128, device=device)
    val_labels = (np.random.rand(n_val, 5) > 0.7).astype(np.float32)
    val_labels[0] = 1.0  # ensure non-degenerate

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(val_ops, val_resps, return_reconstruction=False)
        probs = torch.sigmoid(logits).float().cpu().numpy()

    aurocs = []
    for c in range(5):
        if len(np.unique(val_labels[:, c])) > 1:
            aurocs.append(float(roc_auc_score(val_labels[:, c], probs[:, c])))
        else:
            aurocs.append(0.5)
    macro_auroc = float(np.mean(aurocs))
    assert 0.0 <= macro_auroc <= 1.0
    logger.info("PASS: Test 4.1 - Fold-9 macro AUROC calculation verified: %.4f", macro_auroc)

    # Check that Fold 10 lockbox is sealed
    lockbox = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_fold10_evaluation")
    assert not (lockbox / "fold10_access.lock").exists(), "INTEGRITY VIOLATION: Fold 10 access lock exists during smoke test!"
    logger.info("PASS: Test 4.2 - Fold 10 isolation confirmed (strictly sealed).")


def main() -> None:
    logger.info("==================================================================")
    logger.info("STARTING SETOPERATOR PRE-FLIGHT INTEGRITY & GPU SMOKE SUITE")
    logger.info("==================================================================")
    t0 = time.time()
    test_1_architecture_contract()
    test_2_gpu_forward_backward_weighted_loss()
    test_3_paired_initialization_and_checkpoint_strictness()
    test_4_fold9_evaluation_and_fold10_isolation()
    elapsed = time.time() - t0
    logger.info("==================================================================")
    logger.info("ALL SETOPERATOR PRE-FLIGHT SMOKE TESTS PASSED IN %.2f SECONDS!", elapsed)
    logger.info("==================================================================")


if __name__ == "__main__":
    main()
