"""Authoritative Five-Seed SetOperator Training Script (Folds 1-8).

Restores the exact Paper 07 shared-grid and development training protocol:
1. Exact Constructor: OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
2. Source SHA-256 Verification & Golden Regression Test on initialization.
3. Class-Weighted BCEWithLogitsLoss:
     prevalence = train_labels.mean(axis=0)
     positive_weight = ((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
     loss_fn = BCEWithLogitsLoss(pos_weight=positive_weight)
4. AdamW Optimizer with gradient norm clipping (1.0).
5. CosineAnnealingLR scheduler with T_max=epochs.
6. Max epochs (default 80) with patience-based early stopping (default 10) on Fold 9 macro AUROC.
7. Frozen Context Law: m = rng.integers(1, min(8, K)) strictly in {1, ..., 7}; m=8 NEVER occurs.
8. Paired Initialization: For each seed, Basis and Span start from the IDENTICAL initial weights.
9. Strict Fold Isolation: Trained exclusively on Folds 1-8; checkpoint selection on Fold 9;
   Fold 10 strictly SEALED.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_final_5seeds")

# Disable cudnn 1D conv bug on this environment
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")

from repecg.paper07_operator.model import OperatorSetModel

EXPECTED_MODEL_SHA256 = "66dae0d122266271e4b5b080e3e7048e4ed4ad3bea7d1f96e7bc8fd99b85227e"
GOLDEN_LOGITS_REF = [0.012054, 0.012796, 0.004852, 0.08265, 0.036781]


def verify_architecture_contract() -> None:
    """Verify source code SHA-256 and golden logit regression test."""
    import repecg.paper07_operator.model as model_mod

    source_path = Path(model_mod.__file__)
    source_bytes = source_path.read_bytes()
    computed_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if computed_sha256 != EXPECTED_MODEL_SHA256:
        raise RuntimeError(
            f"ARCHITECTURE INTEGRITY FAILURE: model.py SHA-256 mismatch!\n"
            f"Expected: {EXPECTED_MODEL_SHA256}\n"
            f"Observed: {computed_sha256}"
        )
    logger.info("Verified OperatorSetModel source SHA-256: %s", computed_sha256)

    # Golden logit regression test
    torch.manual_seed(42)
    test_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    test_model.eval()

    torch.manual_seed(1337)
    test_ops = torch.randn(1, 4, 8)
    test_res = torch.randn(1, 4, 16, 128)
    with torch.no_grad():
        test_logits = test_model(test_ops, test_res).squeeze().tolist()

    max_diff = max(abs(a - b) for a, b in zip(test_logits, GOLDEN_LOGITS_REF))
    if max_diff > 1e-4:
        raise RuntimeError(
            f"GOLDEN REGRESSION FAILURE: OperatorSetModel logits deviate from frozen reference!\n"
            f"Max diff: {max_diff:.6f}\n"
            f"Observed: {test_logits}\n"
            f"Reference: {GOLDEN_LOGITS_REF}"
        )
    logger.info("Passed golden logit regression test (max deviation: %.2e)", max_diff)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute multilabel classification metrics."""
    class_aurocs = []
    class_auprcs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            class_aurocs.append(float(roc_auc_score(y_true[:, c], y_prob[:, c])))
            class_auprcs.append(float(average_precision_score(y_true[:, c], y_prob[:, c])))
        else:
            class_aurocs.append(0.5)
            class_auprcs.append(0.0)
    return {
        "macro_auroc": float(np.mean(class_aurocs)),
        "micro_auroc": float(roc_auc_score(y_true.ravel(), y_prob.ravel())),
        "macro_auprc": float(np.mean(class_auprcs)),
    }


def evaluate_on_fold9(
    model: OperatorSetModel,
    val_data: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate on Fold 9 standard 8 basis leads for checkpoint selection."""
    model.eval()
    ops = val_data["operators"][:, :8]
    resps = val_data["responses"][:, :8]
    labels = val_data["labels"]

    probs = []
    with torch.inference_mode():
        for i in range(0, len(labels), batch_size):
            o_b = torch.as_tensor(ops[i : i + batch_size], dtype=torch.float32, device=device)
            r_b = torch.as_tensor(resps[i : i + batch_size], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(o_b, r_b, return_reconstruction=False)
            probs.append(torch.sigmoid(logits).float().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    return compute_metrics(labels, probs)


def train_single_seed(
    seed: int,
    intervention: str,
    initial_state: Dict[str, torch.Tensor],
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    output_dir: Path,
    epochs: int = 80,
    patience: int = 10,
    batch_size: int = 64,
    lr: float = 3e-4,
    wd: float = 1e-3,
    device_str: str = "cuda",
) -> Path:
    """Train a single seed model using the restored Paper 07 shared-grid protocol."""
    logger.info("=== Starting Training: Seed %d | Intervention: %s ===", seed, intervention)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    train_ops = train_data["operators"]
    train_resps = train_data["responses"]
    train_labels = train_data["labels"]
    n_records, n_candidates, _ = train_ops.shape

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)

    # Instantiate model and load exact matched initial state
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
    model.load_state_dict(initial_state)

    # Restored class-weighted BCE loss
    prevalence = torch.as_tensor(train_labels.mean(axis=0), dtype=torch.float32, device=device)
    positive_weight = ((1.0 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    logger.info(
        "[%s_s%d] Positive class weights: %s",
        intervention, seed, [round(w, 4) for w in positive_weight.tolist()]
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history: List[Dict[str, Any]] = []
    cardinality_histogram = {m: 0 for m in range(1, 9)}

    best_checkpoint_path = output_dir / f"{intervention}_s{seed}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_records)
        epoch_losses: List[float] = []

        for start in range(0, n_records, batch_size):
            idx = perm[start : start + batch_size]
            b_len = len(idx)

            # Sample m strictly in [1, 7]
            m = int(rng.integers(1, min(8, n_candidates)))
            cardinality_histogram[m] += b_len
            if m == 8:
                raise RuntimeError("CRITICAL INTEGRITY FAILURE: Context cardinality m=8 encountered in training!")

            # Uniform random choice of m distinct candidates per sample
            sub_indices = np.stack([rng.choice(n_candidates, size=m, replace=False) for _ in range(b_len)])
            c_ops = torch.as_tensor(train_ops[idx[:, None], sub_indices], dtype=torch.float32, device=device)
            c_resps = torch.as_tensor(train_resps[idx[:, None], sub_indices], dtype=torch.float32, device=device)
            batch_y = torch.as_tensor(train_labels[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(c_ops, c_resps, return_reconstruction=False)
                loss = criterion(logits, batch_y)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("[%s_s%d] Non-finite loss at epoch %d batch %d", intervention, seed, epoch, start)
                continue

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)):
                optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        scheduler.step()
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Evaluate on Fold 9 standard 8 basis leads
        val_metrics = evaluate_on_fold9(model, val_data, device, batch_size=batch_size)
        val_auroc = val_metrics["macro_auroc"]

        history.append({
            "epoch": epoch,
            "loss": mean_loss,
            "val_macro_auroc": val_auroc,
            "val_micro_auroc": val_metrics["micro_auroc"],
            "val_macro_auprc": val_metrics["macro_auprc"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        })

        if val_auroc > best_val_auroc + 1e-5:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
            logger.info(
                "[%s_s%d] Epoch %2d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f (*) BEST",
                intervention, seed, epoch, epochs, mean_loss, val_auroc,
            )
        else:
            stale += 1
            if epoch % 5 == 0 or stale >= patience:
                logger.info(
                    "[%s_s%d] Epoch %2d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f (stale %d/%d)",
                    intervention, seed, epoch, epochs, mean_loss, val_auroc, stale, patience,
                )

        if stale >= patience:
            logger.info("[%s_s%d] Early stopping patience (%d) reached at epoch %d.", intervention, seed, patience, epoch)
            break

    # Cardinality audit check
    logger.info("[%s_s%d] Training context cardinality distribution: %s", intervention, seed, cardinality_histogram)
    if cardinality_histogram[8] > 0:
        raise RuntimeError("Integrity violation: m=8 was observed during training!")
    if epochs >= 10:
        for m in range(1, 8):
            if cardinality_histogram[m] == 0:
                raise RuntimeError(f"Integrity violation: Cardinality m={m} was never observed during training!")

    assert best_state is not None, "Training failed to produce a valid model state"
    checkpoint_payload = {
        "state_dict": best_state,
        "seed": seed,
        "intervention": intervention,
        "best_epoch": best_epoch,
        "best_val_auroc": float(best_val_auroc),
        "history": history,
        "hyperparameters": {
            "lr": lr,
            "wd": wd,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "seed": seed,
            "positive_weights": positive_weight.cpu().tolist(),
        },
        "model_source_sha256": EXPECTED_MODEL_SHA256,
        "constructor": "OperatorSetModel(response_dim=128, classes=5, operator_mode='continuous')",
    }
    torch.save(checkpoint_payload, best_checkpoint_path)

    # Verify strict checkpoint load
    chk = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
    verify_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    verify_model.load_state_dict(chk["state_dict"], strict=True)
    logger.info(
        "Saved and strictly verified checkpoint %s (Best Fold-9 AUROC: %.4f at epoch %d)",
        best_checkpoint_path,
        best_val_auroc,
        best_epoch,
    )

    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train 5-seed matched Basis vs Span models using shared-grid protocol.")
    parser.add_argument("--mode", choices=["basis", "span", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--train-basis-npz", type=Path, required=True)
    parser.add_argument("--train-span-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch smoke test on small subset")
    args = parser.parse_args()

    # Pre-flight architecture and golden test check
    verify_architecture_contract()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    val_limit = 64 if args.smoke_test else None
    train_limit = 128 if args.smoke_test else None
    if args.smoke_test:
        args.epochs = 2
        args.patience = 2
        logger.info("SMOKE TEST MODE: Loading small slices (128 train, 64 val records)")

    logger.info("Preloading Fold-9 validation representations from %s...", args.val_npz)
    t0 = time.time()
    with np.load(args.val_npz) as d_val:
        val_data = {
            "operators": np.asarray(d_val["operators"][:val_limit]),
            "responses": np.asarray(d_val["responses"][:val_limit]),
            "labels": np.asarray(d_val["labels"][:val_limit]),
        }
    logger.info("Loaded %d validation records in %.2fs", len(val_data["labels"]), time.time() - t0)

    # Preload training representations for requested modes
    modes = ["basis", "span"] if args.mode == "both" else [args.mode]
    train_datasets: Dict[str, Dict[str, np.ndarray]] = {}

    for mode in modes:
        train_path = args.train_basis_npz if mode == "basis" else args.train_span_npz
        logger.info("Preloading %s training representations from %s...", mode.upper(), train_path)
        t0 = time.time()
        with np.load(train_path) as d_train:
            train_datasets[mode] = {
                "operators": np.asarray(d_train["operators"][:train_limit]),
                "responses": np.asarray(d_train["responses"][:train_limit]),
                "labels": np.asarray(d_train["labels"][:train_limit]),
            }
        logger.info(
            "Loaded %d %s records (%d candidates) in %.2fs",
            len(train_datasets[mode]["labels"]),
            mode.upper(),
            train_datasets[mode]["operators"].shape[1],
            time.time() - t0,
        )

    # Matched seed training: generate paired initializations for each seed
    checkpoints: Dict[str, Any] = {}
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    for seed in args.seeds:
        # Generate paired initial state for this seed
        torch.manual_seed(seed)
        template_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)
        initial_state = copy.deepcopy(template_model.state_dict())
        del template_model

        for mode in modes:
            chk_path = train_single_seed(
                seed=seed,
                intervention=mode,
                initial_state=initial_state,
                train_data=train_datasets[mode],
                val_data=val_data,
                output_dir=args.output_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                lr=args.lr,
                wd=args.wd,
                device_str=args.device,
            )
            sha = hashlib.sha256(chk_path.read_bytes()).hexdigest()
            checkpoints[f"{mode}_s{seed}"] = {
                "path": str(chk_path),
                "sha256": sha,
            }

    ledger_path = args.output_dir / "checkpoints_selection_ledger.json"
    ledger_path.write_text(json.dumps(checkpoints, indent=2) + "\n")
    logger.info("All training complete. Checkpoint ledger written to %s", ledger_path)


if __name__ == "__main__":
    main()
