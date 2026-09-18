"""Fold-7-only selection for the frozen Paper 12 V2 G4 linear probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from repecg.paper12_innovations import (
    PhaseCoordinateStandardizer,
    ProbeScaler,
    minibatch_order,
    probe_features,
)
from scripts.paper12.run_g3_residual_audit import fit_model


LRS = (1e-4, 3e-4, 1e-3)
WDS = (1e-5, 1e-4, 1e-3)
TUNING_SEEDS = (52, 53, 54)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def macro_auc(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(labels[:, c], probability[:, c]) for c in range(labels.shape[1])]))


def fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    pos_weight: torch.Tensor,
    *,
    lr: float,
    wd: float,
    seed: int,
) -> list[float]:
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], train_y.shape[1]).to(train_x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scores = []
    for epoch in range(1, 51):
        order = minibatch_order(len(train_x), seed, epoch, train_x.device)
        model.train()
        for start in range(0, len(order), 256):
            index = order[start : start + 256]
            loss = loss_fn(model(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            probability = torch.sigmoid(model(val_x)).cpu().numpy()
        scores.append(macro_auc(val_y, probability))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    with np.load(args.representations / "representation_train.npz") as item:
        train_raw = torch.from_numpy(np.asarray(item["kernel"])).to(device)
        train_y = torch.from_numpy(np.asarray(item["labels"])).to(device)
    with np.load(args.representations / "representation_val.npz") as item:
        val_raw = torch.from_numpy(np.asarray(item["kernel"])).to(device)
        val_y = np.asarray(item["labels"])
    standardizer = PhaseCoordinateStandardizer.fit(train_raw)
    train_z, val_z = standardizer.transform(train_raw), standardizer.transform(val_raw)
    density = fit_model(train_z, steps=1000, batch=128, seed=42, unit_scale_init=True)
    density.freeze_density()
    z_scaler = ProbeScaler.fit(probe_features(train_z))
    train_x = z_scaler.transform(probe_features(train_z))
    val_x = z_scaler.transform(probe_features(val_z))
    prevalence = train_y.mean(0)
    pos_weight = ((1 - prevalence) / prevalence.clamp_min(1e-8)).clamp_max(10)
    candidates = []
    for lr in LRS:
        for wd in WDS:
            seed_scores = [fit_probe(train_x, train_y, val_x, val_y, pos_weight, lr=lr, wd=wd, seed=seed) for seed in TUNING_SEEDS]
            mean_by_epoch = np.mean(seed_scores, axis=0)
            epoch = int(np.argmax(mean_by_epoch)) + 1
            candidates.append((float(mean_by_epoch[epoch - 1]), lr, wd, epoch))
            print(json.dumps({"lr": lr, "weight_decay": wd, "epoch": epoch, "score": candidates[-1][0]}), flush=True)
    candidates.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
    score, lr, wd, epoch = candidates[0]
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "phase_mean": standardizer.mean.cpu(),
            "phase_scale": standardizer.scale.cpu(),
            "density_state_dict": density.state_dict(),
            "z_probe_mean": z_scaler.mean.cpu(),
            "z_probe_scale": z_scaler.scale.cpu(),
        },
        args.output / "upstream.pt",
    )
    payload = {
        "kind": "paper12_v2_g4_probe_config",
        "learning_rate": lr,
        "weight_decay": wd,
        "epoch": epoch,
        "batch_size": 256,
        "positive_weights": [float(v) for v in pos_weight.cpu()],
        "tuning_seeds": list(TUNING_SEEDS),
        "selection_metric": "mean_fold7_macro_auroc",
        "selection_representation": "phase_standardized",
        "selection_score": score,
        "protocol_sha256": sha256(args.protocol),
        "kernel_fit_sha256": sha256(args.kernel_fit),
        "upstream_sha256": sha256(args.output / "upstream.pt"),
    }
    temporary = args.output / "probe_config.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output / "probe_config.json")
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
