#!/usr/bin/env python3
"""Matched held-out dense PanoBench geometry assay for N002 and frozen N003a."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from theta_repspat.panobench import PANOBENCH_ANGLES_RAD, _upsample2x  # noqa: E402
from theta_repspat.vendor import instantiate_author_model, load_author_model  # noqa: E402
from eval_g002 import EXPECTED_LEN, compute_geometry_metrics  # noqa: E402
from eval_g002_c_kernel_alignment import compute_aligned_stress_and_cka  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRICS = ("spearman_rho_d", "knn_jaccard_k10", "knn_jaccard_k20", "cahc_ari",
           "eps_aligned", "cka")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_n003a(path: Path):
    model = instantiate_author_model(ROOT / "author_code/nefnet_v2", DEVICE, super_mode="pretrain")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def complete(values: np.ndarray, input_indices: np.ndarray, model, arm: str) -> np.ndarray:
    observed = values[input_indices]
    if arm == "N002":
        lo, hi = float(observed.min()), float(observed.max())
        span = hi - lo if hi > lo else 1.0
        normalized = (observed - lo) / span
    elif arm == "N003a":
        lo, span = -4.0, 8.0
        normalized = np.clip((observed - lo) / span, 1e-4, 1 - 1e-4)
    else:
        raise ValueError(arm)
    inputs = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(DEVICE)
    angles = torch.from_numpy(PANOBENCH_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)
    result = values.copy()
    for query_idx in range(44):
        if query_idx in input_indices:
            continue
        query = torch.from_numpy(PANOBENCH_ANGLES_RAD[query_idx]).unsqueeze(0).to(DEVICE)
        if arm == "N003a":
            query = query.unsqueeze(1)
        with torch.no_grad():
            prediction = model(inputs, angles, query).squeeze().cpu().numpy()
        result[query_idx] = prediction * span + lo
    return result.T


def paired_bootstrap(values: np.ndarray, seed: int, n_boot: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean()
                        for _ in range(n_boot)])
    return {"n_records": len(values), "mean": float(values.mean()),
            "ci95": [float(np.quantile(means, .025)), float(np.quantile(means, .975))]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n002", type=Path, default=ROOT / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt")
    parser.add_argument("--n003a", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/model_best.pt")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/frozen_artifact/freeze_manifest.json")
    parser.add_argument("--g002-records", type=Path, default=ROOT / "results/g002_eval/g002_panobench_records.json")
    parser.add_argument("--test-root", type=Path, default=Path("/data/mithunmanivannan/panobench/test"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/n003a_matched_dense_panobench")
    return parser.parse_args()


def main():
    args = parse_args()
    frozen = json.loads(args.freeze_manifest.read_text())
    if sha256(args.n003a) != frozen["selected_checkpoint_sha256"]:
        raise RuntimeError("N003a checkpoint differs from frozen artifact")
    old_rows = json.loads(args.g002_records.read_text())
    rng = np.random.default_rng(42)
    manifest = []
    for old in old_rows:
        crop_start = int(rng.integers(0, 5000 - EXPECTED_LEN + 1))
        third = int(rng.integers(2, 44))
        if third != int(old["third_view_idx"]):
            raise RuntimeError(f"failed to reconstruct G002 sampling for record {old['record']}")
        manifest.append({"record": str(old["record"]), "crop_start": crop_start,
                         "input_indices": [0, 1, third]})
    models = {"N002": load_author_model(ROOT / "author_code/nefnet_v2", args.n002, DEVICE),
              "N003a": load_n003a(args.n003a)}
    rows = []
    for index, item in enumerate(manifest):
        raw = scipy.io.loadmat(args.test_root / f"{item['record']}.mat")["Panobench"][:44].astype(np.float32)
        values = _upsample2x(raw)[:, item["crop_start"]:item["crop_start"] + EXPECTED_LEN]
        reference = values.T
        row = {**item, "arms": {}, "paired_delta_n003a_minus_n002": {}}
        for arm, model in models.items():
            representation = complete(values, np.asarray(item["input_indices"]), model, arm)
            row["arms"][arm] = {**compute_geometry_metrics(representation, reference),
                                **compute_aligned_stress_and_cka(representation, reference)}
        for metric in METRICS:
            row["paired_delta_n003a_minus_n002"][metric] = (
                row["arms"]["N003a"][metric] - row["arms"]["N002"][metric])
        rows.append(row)
        if (index + 1) % 10 == 0:
            print(f"evaluated={index + 1}/{len(manifest)}", flush=True)
    summary = {
        "protocol": {"record_source": str(args.g002_records),
                     "record_source_sha256": sha256(args.g002_records),
                     "n_records": len(rows), "sampling_reconstructed_from_g002_seed": 42,
                     "input_views": "I, II, and frozen third torso view", "queries": "remaining 41 views",
                     "bootstrap_unit": "PanoBench record"},
        "checkpoints": {"N002": sha256(args.n002), "N003a": sha256(args.n003a)},
        "arm_means": {arm: {metric: float(np.mean([r["arms"][arm][metric] for r in rows]))
                            for metric in METRICS} for arm in models},
        "paired_n003a_minus_n002": {
            metric: paired_bootstrap(np.asarray([r["paired_delta_n003a_minus_n002"][metric]
                                                 for r in rows]), 6200 + i)
            for i, metric in enumerate(METRICS)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "dense_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output / "dense_records.json").write_text(json.dumps(rows) + "\n")
    (args.output / "dense_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
