#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from repecg.braid_field import BraidFieldClassifier, BraidFieldConfig, Q8_LEADS
from repecg.common.metrics import multilabel_metrics


NAMED_CONFIGS = {
    "Q8": tuple(range(8)),
    "6precordial": (2, 3, 4, 5, 6, 7),
    "2limb_independent": (0, 1),
    "3_I_II_V1": (0, 1, 2),
    "3_I_II_V5": (0, 1, 6),
    "1_I": (0,),
    "1_II": (1,),
}


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as item:
        return {key: np.asarray(item[key]) for key in item.files}


def _load_model(path: Path, device: torch.device) -> BraidFieldClassifier:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = BraidFieldConfig(**payload["config"])
    model = BraidFieldClassifier(config).to(device)
    model.load_state_dict(payload["state_dict"])
    return model.eval()


def _macro_auroc(labels: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(labels, probability, average="macro"))


def _predict(
    model: BraidFieldClassifier,
    data: dict[str, np.ndarray],
    subset: tuple[int, ...],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    result = []
    context_ops = torch.as_tensor(data["context_operators"][list(subset)], device=device)
    query_ops = torch.as_tensor(data["query_operators"], device=device)
    query_coords = torch.as_tensor(data["query_coords"], device=device)
    with torch.inference_mode():
        for start in range(0, len(data["labels"]), batch_size):
            stop = min(start + batch_size, len(data["labels"]))
            response = torch.as_tensor(
                data["context_responses"][start:stop, list(subset)], device=device
            )
            operators = context_ops.unsqueeze(0).expand(stop - start, -1, -1)
            out = model(operators, response, query_ops, query_coords)
            result.append(torch.sigmoid(out["logits"]).float().cpu())
    return torch.cat(result).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--all-subsets", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    data = _load(args.dataset / f"{args.split}.npz")
    model = _load_model(args.checkpoint, device)
    configs = dict(NAMED_CONFIGS)
    if args.all_subsets:
        for k in range(1, 9):
            for subset in itertools.combinations(range(8), k):
                configs["subset_" + "_".join(Q8_LEADS[i] for i in subset)] = subset

    rows = []
    q8_probability = _predict(model, data, tuple(range(8)), batch_size=args.batch, device=device)
    q8_score = _macro_auroc(data["labels"], q8_probability)
    for name, subset in configs.items():
        probability = _predict(model, data, subset, batch_size=args.batch, device=device)
        metrics = multilabel_metrics(data["labels"], probability)
        score = float(metrics["macro_auroc"])
        rows.append(
            {
                "configuration": name,
                "lead_indices": list(subset),
                "lead_names": [Q8_LEADS[i] for i in subset],
                "k": len(subset),
                "macro_auroc": score,
                "macro_auprc": float(metrics["macro_auprc"]),
                "delta_from_q8": q8_score - score,
                "retention": score / q8_score if q8_score else None,
            }
        )
    payload = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "q8_macro_auroc": q8_score,
        "rows": rows,
        "scope_warning": "standard-Q8 subset shift only; novel operator evaluation requires physical waveform synthesis",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
