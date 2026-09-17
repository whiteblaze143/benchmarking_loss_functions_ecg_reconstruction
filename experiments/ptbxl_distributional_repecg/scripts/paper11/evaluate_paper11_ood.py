from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.common.models import CausalStateECGModel

VARIANTS = ("kernel", "moments", "gaussian", "linear")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True, help="Path to training directory")
    parser.add_argument("--representations", type=Path, required=True, help="Path to OOD representations directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    best_models = {}
    for variant in VARIANTS:
        best_score = -np.inf
        best_path = None
        for cell_dir in (args.training / "cells").iterdir():
            if not cell_dir.is_dir() or not cell_dir.name.startswith(f"{variant}_lr"):
                continue
            summary_path = cell_dir / "summary.json"
            if not summary_path.exists():
                continue
            with open(summary_path) as f:
                summary = json.load(f)
            
            score = summary["metrics"]["macro_auroc"]
            if score > best_score:
                best_score = score
                best_path = cell_dir / "checkpoint.pt"
        
        if best_path is not None:
            best_models[variant] = best_path
            print(f"Best {variant} model: {best_path} (AUROC: {best_score:.4f})")

    if not best_models:
        print("No models found to evaluate.")
        return
    
    for file in sorted(args.representations.glob("representation_ood_*.npz")):
        dataset_name = file.stem.replace("representation_ood_", "")
        print(f"Evaluating on {dataset_name}...")
        
        with np.load(file, allow_pickle=True) as item:
            ood_data = {name: np.asarray(item[name]) for name in item.files}
        
        all_preds = {}
        for variant in VARIANTS:
            if variant not in best_models:
                continue
            
            checkpoint = torch.load(best_models[variant], map_location="cpu", weights_only=False)
            x_val = torch.from_numpy(ood_data[variant]).float().cuda()
            model = CausalStateECGModel(input_dim=x_val.shape[-1], classes=5).cuda()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                probs = torch.sigmoid(model(x_val)).float().cpu().numpy()
                
            all_preds[variant] = probs
            
        np.savez_compressed(
            args.output / f"ood_predictions_{dataset_name}.npz",
            ecg_ids=ood_data["ecg_ids"],
            **all_preds
        )
        print(f"Saved predictions for {dataset_name}.")


if __name__ == "__main__":
    main()
