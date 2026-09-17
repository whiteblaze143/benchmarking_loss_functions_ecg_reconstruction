from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from repecg.common.metrics import multilabel_metrics
from repecg.common.paper_models import create_paper_model
from repecg.common.variants import get_variants_for_paper


CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def aggregate_grid(paper_id: int, cells: Path, output: Path, seed: int) -> dict[str, object]:
    registry = get_variants_for_paper(paper_id)
    summaries = [
        (path.parent, json.loads(path.read_text()))
        for path in sorted(cells.glob("*/summary.json"))
    ]
    expected_total = 9 * len(registry)
    if len(summaries) != expected_total:
        raise RuntimeError(f"paper {paper_id:02d}: expected {expected_total} cells, found {len(summaries)}")
    observed = {summary.get("variant") for _, summary in summaries}
    if observed != set(registry):
        raise RuntimeError(
            f"paper {paper_id:02d}: expected variants {sorted(registry)}, found {sorted(observed)}"
        )
    output.mkdir(parents=True, exist_ok=True)
    for variant in registry:
        candidates = [(path, summary) for path, summary in summaries if summary["variant"] == variant]
        if len(candidates) != 9:
            raise RuntimeError(f"paper {paper_id:02d} variant {variant}: expected 9 cells, found {len(candidates)}")
        best_path, best = max(candidates, key=lambda item: item[1]["metrics"]["macro_auroc"])
        shutil.copy2(best_path / "checkpoint.pt", output / f"{variant}_best.pt")
        shutil.copy2(best_path / "predictions.npz", output / f"predictions_{variant}.npz")
        (output / f"metrics_{variant}.json").write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
    manifest = {
        "kind": f"paper{paper_id:02d}_development_hyperparameter_search",
        "paper_id": paper_id,
        "seed": seed,
        "cells": expected_total,
        "variants": list(registry),
        "cells_per_variant": 9,
        "execution": "single_process_shared_tensor_cuda_streams",
        "selection_metric": "macro_auroc",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def evaluate_development_variant(
    paper_id: int,
    variant_name: str,
    representations: Path,
    training: Path,
    output: Path,
    batch: int = 4096,
) -> dict[str, object]:
    registry = get_variants_for_paper(paper_id)
    if variant_name not in registry:
        raise ValueError(f"unknown paper {paper_id:02d} variant: {variant_name}")
    variant = registry[variant_name]
    representation_path = representations / "representation_val.npz"
    if not representation_path.is_file():
        raise FileNotFoundError(f"development representation missing: {representation_path}")
    with np.load(representation_path) as item:
        values = {name: np.asarray(item[name]) for name in item.files}
    key = variant.representation if variant.representation != "full" else "kernel"
    if key not in values:
        raise KeyError(f"variant {variant_name} requires representation {key!r}")
    checkpoint_path = training / f"{variant_name}_best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"best checkpoint missing: {checkpoint_path}")
    device = torch.device("cuda")
    features = torch.from_numpy(values[key]).to(device)
    target = values["labels"]
    model = create_paper_model(paper_id, features.shape[-1], target.shape[-1], variant).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    probability = []
    torch.backends.cudnn.enabled = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(features), batch):
            prediction = model(features[start : start + batch])
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            probability.append(torch.sigmoid(prediction).float().cpu())
    probability_array = torch.cat(probability).numpy()
    metrics = multilabel_metrics(target, probability_array)
    payload = {
        "kind": "development_selection_evaluation",
        "paper_id": paper_id,
        "variant": variant_name,
        "representation": key,
        "records": int(len(target)),
        "metrics": metrics,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"metrics_{variant_name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / f"predictions_{variant_name}.npz",
        probability=probability_array,
        labels=target,
        ecg_ids=values["ecg_ids"],
        patient_ids=values["patient_ids"],
    )
    return payload
