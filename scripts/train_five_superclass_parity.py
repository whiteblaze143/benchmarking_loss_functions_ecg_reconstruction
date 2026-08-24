#!/usr/bin/env python3
"""Retrain the paper's frozen ECG-FM five-superclass head without test leakage."""

from __future__ import annotations

import argparse
import ast
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

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inventory(path: Path) -> dict[str, object]:
    files = sorted(path.glob("*.pt"), key=lambda item: int(item.stem))
    digest = hashlib.sha256()
    for item in files:
        digest.update(f"{item.stem}:{item.stat().st_size}\n".encode())
    return {"count": len(files), "sha256": digest.hexdigest()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def labels_by_id() -> dict[int, np.ndarray]:
    database = pd.read_csv(ROOT / "data/ptb_xl/ptbxl_database.csv")
    statements = pd.read_csv(ROOT / "data/ptb_xl/scp_statements.csv", index_col=0)
    output = {}
    for row in database.itertuples():
        labels = np.zeros(len(CLASSES), dtype=np.float32)
        for code, likelihood in ast.literal_eval(str(row.scp_codes)).items():
            if float(likelihood) >= 50 and code in statements.index:
                diagnostic_class = statements.loc[code].get("diagnostic_class")
                if diagnostic_class in CLASSES:
                    labels[CLASSES.index(diagnostic_class)] = 1.0
        output[int(row.ecg_id)] = labels
    return output


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
    embeddings, labels, ids = [], [], []
    model.eval()
    for signals, batch_labels, batch_ids in loader:
        values = model.extract_embeddings(signals.to(device), pool=True)
        embeddings.append(values.cpu())
        labels.append(batch_labels.float())
        ids.append(batch_ids.long())
    return torch.cat(embeddings), torch.cat(labels), torch.cat(ids)


@torch.inference_mode()
def macro_auroc(head: nn.Module, features: torch.Tensor, labels: torch.Tensor, device: torch.device) -> float:
    head.eval()
    probabilities = []
    for start in range(0, len(features), 1024):
        probabilities.append(torch.sigmoid(head(features[start:start + 1024].to(device))).cpu())
    return float(roc_auc_score(labels.numpy(), torch.cat(probabilities).numpy(), average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "checkpoints/factorial_v2/parity/ecgfm_five_superclass.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone_path = ROOT / "ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt"
    model = ECGFMClassifier(str(backbone_path), num_classes=5, dropout=0.3, freeze_backbone=True).to(device)
    labels = labels_by_id()
    tensor_root = ROOT / "data/ptb_xl/tensors"
    train_loader = DataLoader(ECGDataset(tensor_root / "train", labels), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(ECGDataset(tensor_root / "val", labels), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    train_features, train_labels, train_ids = embed(model, train_loader, device)
    val_features, val_labels, val_ids = embed(model, val_loader, device)

    generator = torch.Generator().manual_seed(args.seed)
    feature_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=256, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_auroc = -float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(args.epochs):
        model.head.train()
        for features, batch_labels in feature_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.head(features.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        score = macro_auroc(model.head, val_features, val_labels, device)
        print(f"epoch={epoch + 1} val_macro_auroc={score:.6f}")
        if score > best_auroc:
            best_auroc = score
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No parity classifier checkpoint selected")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "head_state_dict": best_state,
        "classes": CLASSES,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_auroc": best_auroc,
        "backbone": str(backbone_path.relative_to(ROOT)),
        "backbone_sha256": file_sha256(backbone_path),
        "split_inventory": {split: inventory(tensor_root / split) for split in ("train", "val", "test")},
        "train_ids_sha256": hashlib.sha256(train_ids.numpy().tobytes()).hexdigest(),
        "val_ids_sha256": hashlib.sha256(val_ids.numpy().tobytes()).hexdigest(),
        "test_used_for_training_or_selection": False,
        "preprocessing": "per-lead record z-score",
        "optimizer": "AdamW(lr=1e-4,weight_decay=1e-4)",
        "selector": "highest_validation_macro_auroc",
        "architecture_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip(),
        "architecture_provenance": code_provenance(ROOT, [
            "scripts/train_five_superclass_parity.py",
            "scripts/five_superclass_parity.py",
            "ecg_fm_integration/src/ecg_fm_classifier.py",
        ]),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    output.with_suffix(".json").write_text(json.dumps({key: value for key, value in payload.items() if key != "head_state_dict"}, indent=2))
    print(f"Saved provenance-complete parity classifier: {output}")


if __name__ == "__main__":
    main()
