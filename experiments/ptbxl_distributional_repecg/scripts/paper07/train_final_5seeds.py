"""Final Five-Seed SetOperator Training Script (Folds 1-8).

Fail-Closed Architecture:
1. Exact Constructor: OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
2. Source SHA-256 Verification & Golden Regression Test on initialization.
3. Frozen Context Law: m = rng.integers(1, 8) strictly in {1, ..., 7}; m=8 NEVER occurs.
4. Frozen Exposure Law: 1/2 basis + 1/4 dense + 1/4 sparse for Span; pure basis for Basis.
5. Strict Fold Isolation: Trained exclusively on Folds 1-8; checkpoint selection on Fold 9;
   Fold 10 strictly SEALED.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_final_5seeds")

# Ensure cudnn disabled for deterministic reproduction if required
torch.backends.cudnn.enabled = False

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


def sample_context_m(rng: np.random.Generator) -> int:
    """Sample context size m in exactly {1, 2, 3, 4, 5, 6, 7}. Cardinality 8 NEVER occurs."""
    m = int(rng.integers(1, 8))
    if m < 1 or m > 7:
        raise ValueError(f"CRITICAL VIOLATION: Sampled m={m} outside [1, 7]")
    return m


class RepresentationDataset(Dataset):
    """Dataset serving precomputed Phase-KME response features and operator vectors."""

    def __init__(
        self,
        npz_path: Path | str,
        mode: str,
        seed: int,
    ):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing representation file: {npz_path}")
        logger.info("Loading representations from %s", npz_path)
        data = np.load(npz_path)
        self.operators = torch.from_numpy(data["operators"]).float()  # (N, K, 8)
        self.responses = torch.from_numpy(data["responses"]).float()  # (N, K, 16, 128)
        self.labels = torch.from_numpy(data["labels"]).float()        # (N, 5)
        self.record_ids = data["record_ids"] if "record_ids" in data else data.get("ecg_ids", data.get("patient_ids", None))
        self.mode = mode
        self.rng = np.random.default_rng(seed)
        self.k_available = self.operators.shape[1]
        logger.info(
            "Loaded dataset: %d records, %d available operators per record",
            len(self.labels),
            self.k_available,
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        m = sample_context_m(self.rng)
        # Randomly choose m indices out of k_available
        perm = self.rng.permutation(self.k_available)
        context_idx = perm[:m]

        ops = self.operators[idx, context_idx]  # (m, 8)
        res = self.responses[idx, context_idx]  # (m, 16, 128)
        label = self.labels[idx]                # (5,)

        return {
            "operators": ops,
            "responses": res,
            "label": label,
            "m": torch.tensor(m, dtype=torch.long),
        }


class DeterministicValidationDataset(Dataset):
    """Deterministic validation dataset with a frozen context bank on Fold 9.

    Guarantees that every epoch, every seed, and every model architecture
    evaluates against the exact same context cardinalities m_i and operator subsets.
    Eliminates stochastic validation sampling variance during checkpoint selection.
    """

    def __init__(self, npz_path: Path | str, seed: int = 42):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing representation file: {npz_path}")
        logger.info("Loading deterministic validation representations from %s", npz_path)
        data = np.load(npz_path)
        self.operators = torch.from_numpy(data["operators"]).float()  # (N, K, 8)
        self.responses = torch.from_numpy(data["responses"]).float()  # (N, K, 16, 128)
        self.labels = torch.from_numpy(data["labels"]).float()        # (N, 5)
        self.record_ids = data["record_ids"] if "record_ids" in data else data.get("ecg_ids", data.get("patient_ids", None))
        self.k_available = self.operators.shape[1]

        # Freeze the deterministic validation context bank
        val_rng = np.random.default_rng(seed)
        self.frozen_contexts: list[tuple[int, np.ndarray]] = []
        for _ in range(len(self.labels)):
            m = int(val_rng.integers(1, 8))
            perm = val_rng.permutation(self.k_available)
            context_idx = perm[:m]
            self.frozen_contexts.append((m, context_idx))

        logger.info(
            "Initialized frozen validation bank: %d records, deterministic across all epochs & seeds",
            len(self.labels),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        m, context_idx = self.frozen_contexts[idx]
        ops = self.operators[idx, context_idx]  # (m, 8)
        res = self.responses[idx, context_idx]  # (m, 16, 128)
        label = self.labels[idx]                # (5,)

        return {
            "operators": ops,
            "responses": res,
            "label": label,
            "m": torch.tensor(m, dtype=torch.long),
        }


def collate_variable_m(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate batch with variable context cardinalities m_i."""
    batch_size = len(batch)
    m_values = [b["m"].item() for b in batch]
    max_m = max(m_values)

    padded_ops = torch.zeros(batch_size, max_m, 8, dtype=torch.float32)
    padded_res = torch.zeros(batch_size, max_m, 16, 128, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_m, dtype=torch.bool)
    labels = torch.stack([b["label"] for b in batch])

    for i, b in enumerate(batch):
        m = b["m"].item()
        padded_ops[i, :m] = b["operators"]
        padded_res[i, :m] = b["responses"]
        mask[i, m:] = True  # PyTorch key_padding_mask: True indicates position should be ignored (padding)

    return {
        "operators": padded_ops,
        "responses": padded_res,
        "mask": mask,
        "labels": labels,
        "m_values": torch.tensor(m_values, dtype=torch.long),
    }


def compute_auroc_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro AUROC across classes."""
    from sklearn.metrics import roc_auc_score
    aurocs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) > 1:
            aurocs.append(roc_auc_score(y_true[:, c], y_pred[:, c]))
    return float(np.mean(aurocs)) if aurocs else 0.5


def train_single_seed(
    seed: int,
    intervention: str,
    train_npz: Path,
    val_npz: Path,
    output_dir: Path,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 3e-4,
    device_str: str = "cuda",
    train_ds: RepresentationDataset | None = None,
    val_ds: DeterministicValidationDataset | None = None,
) -> Path:
    """Train a single seed model and select best checkpoint on Fold 9."""
    logger.info("=== Starting Training: Seed %d | Intervention: %s ===", seed, intervention)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    # Instantiate model with exact verified constructor
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous").to(device)

    if train_ds is None:
        train_ds = RepresentationDataset(train_npz, mode=intervention, seed=seed)
    else:
        train_ds.rng = np.random.default_rng(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_variable_m,
        num_workers=0,
    )

    if val_ds is None:
        val_ds = DeterministicValidationDataset(val_npz, seed=42)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_variable_m,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auroc = -1.0
    best_checkpoint_path = output_dir / f"{intervention}_s{seed}.pt"
    best_metrics = {}

    cardinality_histogram = {m: 0 for m in range(1, 9)}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            ops = batch["operators"].to(device)
            res = batch["responses"].to(device)
            mask = batch["mask"].to(device)
            targets = batch["labels"].to(device)

            # Audit training context cardinalities
            for m_val in batch["m_values"].tolist():
                cardinality_histogram[m_val] = cardinality_histogram.get(m_val, 0) + 1
                if m_val == 8:
                    raise RuntimeError("CRITICAL INTEGRITY FAILURE: Context cardinality m=8 encountered in training!")

            optimizer.zero_grad()
            logits = model(ops, res, mask=mask)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        # Validation on Fold 9 (Exclusive selection)
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                ops = batch["operators"].to(device)
                res = batch["responses"].to(device)
                mask = batch["mask"].to(device)
                logits = model(ops, res, mask=mask)
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu().numpy())
                val_targets.append(batch["labels"].numpy())

        y_val_pred = np.concatenate(val_preds, axis=0)
        y_val_true = np.concatenate(val_targets, axis=0)
        val_auroc = compute_auroc_macro(y_val_true, y_val_pred)

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Fold-9 Val Macro AUROC: %.4f",
            epoch,
            epochs,
            train_loss / max(1, n_batches),
            val_auroc,
        )

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_metrics = {
                "epoch": epoch,
                "val_auroc": float(val_auroc),
                "train_loss": float(train_loss / max(1, n_batches)),
            }
            # Save strict state dict
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "intervention": intervention,
                    "epoch": epoch,
                    "val_auroc": float(val_auroc),
                    "model_source_sha256": EXPECTED_MODEL_SHA256,
                    "constructor": "OperatorSetModel(response_dim=128, classes=5, operator_mode='continuous')",
                },
                best_checkpoint_path,
            )

    # Post-training batch audit verification
    logger.info("Context Cardinality Distribution during Training: %s", cardinality_histogram)
    if cardinality_histogram[8] > 0:
        raise RuntimeError("Integrity violation: m=8 was observed during training!")
    for m in range(1, 8):
        if cardinality_histogram[m] == 0:
            raise RuntimeError(f"Integrity violation: Cardinality m={m} was never observed during training!")

    # Verify strict checkpoint load
    chk = torch.load(best_checkpoint_path, map_location="cpu")
    verify_model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
    verify_model.load_state_dict(chk["state_dict"], strict=True)
    logger.info(
        "Saved and strictly verified checkpoint %s (Best Fold-9 AUROC: %.4f)",
        best_checkpoint_path,
        best_val_auroc,
    )

    return best_checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Train 5-seed matched Basis vs Span models.")
    parser.add_argument("--mode", choices=["basis", "span", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--train-basis-npz", type=Path, required=True)
    parser.add_argument("--train-span-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Pre-flight architecture and golden test check
    verify_architecture_contract()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["basis", "span"] if args.mode == "both" else [args.mode]
    checkpoints = {}

    logger.info("Preloading deterministic validation representations from %s...", args.val_npz)
    val_ds = DeterministicValidationDataset(args.val_npz, seed=42)

    for mode in modes:
        train_npz = args.train_basis_npz if mode == "basis" else args.train_span_npz
        logger.info("Preloading %s training representations from %s...", mode.upper(), train_npz)
        train_ds = RepresentationDataset(train_npz, mode=mode, seed=args.seeds[0])

        for seed in args.seeds:
            chk_path = train_single_seed(
                seed=seed,
                intervention=mode,
                train_npz=train_npz,
                val_npz=args.val_npz,
                output_dir=args.output_dir,
                epochs=args.epochs,
                device_str=args.device,
                train_ds=train_ds,
                val_ds=val_ds,
            )
            sha = hashlib.sha256(chk_path.read_bytes()).hexdigest()
            checkpoints[f"{mode}_s{seed}"] = {
                "path": str(chk_path),
                "sha256": sha,
            }

    ledger_path = args.output_dir / "checkpoints_selection_ledger.json"
    ledger_path.write_text(json.dumps(checkpoints, indent=2) + "\n")
    logger.info("Training complete. Checkpoint ledger written to %s", ledger_path)


if __name__ == "__main__":
    main()
