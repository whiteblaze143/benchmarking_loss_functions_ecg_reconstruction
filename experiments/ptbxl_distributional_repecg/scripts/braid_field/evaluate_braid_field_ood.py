#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from repecg.braid_field import BraidFieldClassifier, BraidFieldConfig
from repecg.braid_field.features import FrozenResponseFeatureMap
from repecg.evaluation.novel_operators import get_ood_operator_bank


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as item:
        return {key: np.asarray(item[key]) for key in item.files}


def _load_model(path: Path, device: torch.device) -> BraidFieldClassifier:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = BraidFieldClassifier(BraidFieldConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _cache_index(cache: Path) -> dict[int, str]:
    frame = pd.read_csv(cache / "qc.csv").query("eligible")
    return {int(row.ecg_id): str(row.artifact) for _, row in frame.iterrows()}


def _load_physical(
    cache: Path,
    names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    values = []
    for name in names:
        with np.load(cache / name) as item:
            beats = np.asarray(item["beats"], dtype=np.float32)
        values.append(beats * std.astype(np.float32) + mean.astype(np.float32))
    return np.stack(values)


def _macro_auroc(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(labels, probability, average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--phase-cache", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--response-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operator-count", type=int, default=100)
    parser.add_argument("--operator-seed", type=int, default=2026)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    data = _load_npz(args.dataset / "val.npz")
    model = _load_model(args.checkpoint, device)
    fmap = FrozenResponseFeatureMap(args.response_fit, device=device)
    scaler = json.loads(args.scaler.read_text())
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    lookup = _cache_index(args.phase_cache)
    names = [lookup[int(ecg_id)] for ecg_id in data["ecg_ids"]]

    bank = get_ood_operator_bank(
        seed=args.operator_seed, dim=8, count=args.operator_count
    ).to(device=device, dtype=torch.float32)
    query_ops = torch.as_tensor(data["query_operators"], device=device)
    query_coords = torch.as_tensor(data["query_coords"], device=device)

    rows = []
    model.eval()
    for operator_index, q in enumerate(bank):
        probabilities = []
        for start in range(0, len(names), args.batch):
            stop = min(start + args.batch, len(names))
            physical = torch.as_tensor(
                _load_physical(args.phase_cache, names[start:stop], mean, std),
                device=device,
            )
            q_batch = q.view(1, 1, 8).expand(stop - start, -1, -1)
            with torch.inference_mode():
                response = fmap.project_basis(physical, q_batch)
                out = model(q_batch, response, query_ops, query_coords)
                probabilities.append(torch.sigmoid(out["logits"]).float().cpu())
        probability = torch.cat(probabilities).numpy()
        score = _macro_auroc(data["labels"], probability)
        row = {
            "operator_index": operator_index,
            "operator": q.detach().cpu().tolist(),
            "macro_auroc": score,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    payload = {
        "checkpoint": str(args.checkpoint),
        "operator_seed": args.operator_seed,
        "operator_count": args.operator_count,
        "rows": rows,
        "mean_macro_auroc": float(np.mean([row["macro_auroc"] for row in rows])),
        "std_macro_auroc": float(np.std([row["macro_auroc"] for row in rows])),
        "claim_scope": "unseen_in_span_measurement_operators_not_real_device_validation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
