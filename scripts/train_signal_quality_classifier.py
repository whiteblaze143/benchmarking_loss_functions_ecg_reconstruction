#!/usr/bin/env python3
"""Train a frozen ECG-FM head on PTB-XL's signal-quality annotations.

Only folds 1-8 train the head and fold 9 selects the checkpoint. Fold 10 is
inventoried for provenance but is never loaded during training or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bootstrap_paths import ECG_FM_ROOT, setup_import_paths
from scripts.experiment_provenance import code_provenance

setup_import_paths(include_fairseq=True)
sys.path.insert(0, str(ECG_FM_ROOT))
from src.ecg_fm_classifier import ECGFMClassifier

ARTIFACT_COLUMNS = [
    "baseline_drift",
    "static_noise",
    "burst_noise",
    "electrodes_problems",
]
CLASSES = ["any_artifact", *ARTIFACT_COLUMNS]
SELECTOR_CLASSES = ["any_artifact", "baseline_drift", "static_noise", "burst_noise"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path) -> dict[str, object]:
    files = sorted(path.glob("*.pt"), key=lambda item: int(item.stem))
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item.stem}:{item.stat().st_size}\n".encode())
    return {"count": len(files), "sha256": digest.hexdigest()}


def quality_labels(database: pd.DataFrame) -> dict[int, np.ndarray]:
    present = database[ARTIFACT_COLUMNS].notna().to_numpy(dtype=np.float32)
    values = np.concatenate(
        [(present.max(axis=1) > 0).astype(np.float32)[:, None], present], axis=1
    )
    return {
        int(ecg_id): values[index]
        for index, ecg_id in enumerate(database["ecg_id"].to_numpy())
    }


class ECGDataset(Dataset):
    def __init__(self, path: Path, labels: dict[int, np.ndarray]):
        self.files = sorted(path.glob("*.pt"), key=lambda item: int(item.stem))
        self.labels = labels

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        signal = torch.load(path, weights_only=True).float()
        signal = (signal - signal.mean(dim=-1, keepdim=True)) / signal.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-8)
        return signal, torch.from_numpy(self.labels[int(path.stem)]), int(path.stem)


@torch.inference_mode()
def embed(model: ECGFMClassifier, loader: DataLoader, device: torch.device):
    features, labels, ids = [], [], []
    model.eval()
    for signals, batch_labels, batch_ids in loader:
        features.append(model.extract_embeddings(signals.to(device), pool=True).cpu())
        labels.append(batch_labels.float())
        ids.append(batch_ids.long())
    return torch.cat(features), torch.cat(labels), torch.cat(ids)


@torch.inference_mode()
def validation_score(
    head: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[float, dict[str, float | None]]:
    head.eval()
    chunks = [
        torch.sigmoid(head(features[start:start + 1024].to(device))).cpu()
        for start in range(0, len(features), 1024)
    ]
    probabilities = torch.cat(chunks).numpy()
    truth = labels.numpy()
    per_class: dict[str, float | None] = {}
    for index, name in enumerate(CLASSES):
        per_class[name] = (
            float(roc_auc_score(truth[:, index], probabilities[:, index]))
            if np.unique(truth[:, index]).size == 2
            else None
        )
    selector = float(np.mean([per_class[name] for name in SELECTOR_CLASSES]))
    return selector, per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "checkpoints/factorial_v4/parity/ecgfm_signal_quality.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_path = ROOT / "ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt"
    database_path = ROOT / "data/ptb_xl/ptbxl_database.csv"
    tensor_root = ROOT / "data/ptb_xl/tensors"
    labels = quality_labels(pd.read_csv(database_path))
    model = ECGFMClassifier(
        str(backbone_path), num_classes=len(CLASSES), dropout=0.3, freeze_backbone=True
    ).to(device)
    loaders = {
        split: DataLoader(
            ECGDataset(tensor_root / split, labels),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        for split in ("train", "val")
    }
    train_features, train_labels, train_ids = embed(model, loaders["train"], device)
    val_features, val_labels, val_ids = embed(model, loaders["val"], device)

    positives = train_labels.sum(dim=0)
    negatives = len(train_labels) - positives
    pos_weight = (negatives / positives.clamp_min(1)).clamp(max=50).to(device)
    generator = torch.Generator().manual_seed(args.seed)
    feature_loader = DataLoader(
        TensorDataset(train_features, train_labels),
        batch_size=256,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_score = -float("inf")
    best_state = None
    best_epoch = None
    best_per_class = None
    for epoch in range(args.epochs):
        model.head.train()
        for features, batch_labels in feature_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.head(features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        score, per_class = validation_score(model.head, val_features, val_labels, device)
        print(
            f"epoch={epoch + 1} val_selector_macro_auroc={score:.6f} "
            f"per_class={per_class}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_per_class = per_class
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.head.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No signal-quality checkpoint selected")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state_dict": best_state,
        "classes": CLASSES,
        "annotation_columns": ARTIFACT_COLUMNS,
        "annotation_semantics": "non-null PTB-XL quality annotation field",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_selector_macro_auroc": best_score,
        "best_val_per_class_auroc": best_per_class,
        "selector_classes": SELECTOR_CLASSES,
        "electrodes_problems_caveat": "rare: 23 train, 4 validation, 3 test positives",
        "train_positive_counts": {
            name: int(positives[index]) for index, name in enumerate(CLASSES)
        },
        "val_positive_counts": {
            name: int(val_labels[:, index].sum()) for index, name in enumerate(CLASSES)
        },
        "backbone": str(backbone_path.relative_to(ROOT)),
        "backbone_sha256": sha256(backbone_path),
        "database": str(database_path.relative_to(ROOT)),
        "database_sha256": sha256(database_path),
        "split_inventory": {
            split: inventory(tensor_root / split) for split in ("train", "val", "test")
        },
        "train_ids_sha256": hashlib.sha256(train_ids.numpy().tobytes()).hexdigest(),
        "val_ids_sha256": hashlib.sha256(val_ids.numpy().tobytes()).hexdigest(),
        "test_used_for_training_or_selection": False,
        "preprocessing": "per-lead record z-score",
        "optimizer": "AdamW(lr=1e-4,weight_decay=1e-4)",
        "loss": "BCEWithLogitsLoss with train-only capped inverse-prevalence weights",
        "selector": "highest fold-9 macro AUROC over any/baseline/static/burst",
        "architecture_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "architecture_provenance": code_provenance(
            ROOT,
            [
                "scripts/train_signal_quality_classifier.py",
                "scripts/signal_quality_classifier.py",
                "ecg_fm_integration/src/ecg_fm_classifier.py",
            ],
        ),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    output.with_suffix(".json").write_text(
        json.dumps(
            {key: value for key, value in payload.items() if key != "head_state_dict"},
            indent=2,
            allow_nan=False,
        )
    )
    print(f"Saved provenance-complete signal-quality classifier: {output}")


if __name__ == "__main__":
    main()
