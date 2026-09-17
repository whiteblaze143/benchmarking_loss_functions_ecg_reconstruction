from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for summary_path in sorted(args.cells.glob("*/summary.json")):
        all_summaries.append((summary_path.parent, json.loads(summary_path.read_text())))
    if not all_summaries:
        print("No completed cells found in", args.cells)
        return
    variants = sorted(list(set(summary["variant"] for _, summary in all_summaries)))
    for variant in variants:
        candidates = [(path, summary) for path, summary in all_summaries if summary.get("variant") == variant]
        if not candidates:
            continue
        best_path, best = max(candidates, key=lambda item: item[1].get("metrics", {}).get("macro_auroc", -1.0))
        ckpt = best_path / "checkpoint.pt"
        if ckpt.exists():
            shutil.copy2(ckpt, args.output / f"{variant}_best.pt")
        pred_npz = best_path / "predictions.npz"
        if pred_npz.exists():
            with np.load(pred_npz) as item:
                columns = {
                    "ecg_id": item["ecg_ids"],
                    "patient_id": item["patient_ids"],
                }
                for index, name in enumerate(CLASSES):
                    if "labels" in item and item["labels"].ndim == 2 and item["labels"].shape[1] > index:
                        columns[f"y_{name}"] = item["labels"][:, index]
                    if "probability" in item and item["probability"].ndim == 2 and item["probability"].shape[1] > index:
                        columns[f"p_{name}"] = item["probability"][:, index]
                pd.DataFrame(columns).to_csv(args.output / f"predictions_{variant}.csv", index=False)
        (args.output / f"metrics_{variant}.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    table = [{"cell": str(path), **summary} for path, summary in all_summaries]
    (args.output / "search_state.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    manifest = {
        "kind": "paper02_development_hyperparameter_search",
        "seed": args.seed,
        "cells": 36,
        "execution": "single_process_shared_tensor_cuda_streams",
        "folds": {"train": [1, 2, 3, 4, 5, 6, 7], "validation": [8]},
        "selection_metric": "macro_auroc",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
