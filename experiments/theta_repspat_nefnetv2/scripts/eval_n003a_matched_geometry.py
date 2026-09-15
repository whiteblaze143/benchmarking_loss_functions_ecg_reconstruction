#!/usr/bin/env python3
"""Matched N002-vs-N003a G002-C/D geometry evaluation on frozen Emory records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import scipy.stats
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from theta_repspat.panorama import CANONICAL_ANGLES_RAD  # noqa: E402
from theta_repspat.vendor import instantiate_author_model, load_author_model  # noqa: E402
from eval_g002 import (INPUT_CANONICAL_INDICES, KORS_MATRIX, PREDICTED_CANONICAL_INDICES,
                       compute_geometry_metrics, load_clinical_record)  # noqa: E402
from eval_g002_c_kernel_alignment import compute_aligned_stress_and_cka  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_n003a(path: Path):
    model = instantiate_author_model(ROOT / "author_code/nefnet_v2", DEVICE, super_mode="pretrain")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def complete(x_gt: np.ndarray, model, arm: str) -> np.ndarray:
    observed = x_gt[INPUT_CANONICAL_INDICES]
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
    input_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[INPUT_CANONICAL_INDICES]).unsqueeze(0).to(DEVICE)
    completed = x_gt.copy()
    for query_idx in PREDICTED_CANONICAL_INDICES:
        query = torch.from_numpy(CANONICAL_ANGLES_RAD[query_idx]).unsqueeze(0).to(DEVICE)
        if arm == "N003a":
            query = query.unsqueeze(1)
        with torch.no_grad():
            prediction = model(inputs, input_angles, query).squeeze().cpu().numpy()
        completed[query_idx] = prediction * span + lo
    return completed.T


def nonlocal_metrics(x_arm: np.ndarray, x_ref: np.ndarray, delta_max: int = 50):
    step = x_ref.shape[0] // 500
    arm, ref = x_arm[::step][:500], x_ref[::step][:500]
    d_arm = dist.squareform(dist.pdist(arm)); d_ref = dist.squareform(dist.pdist(ref))
    def centered_imq(distance):
        iu = np.triu_indices(len(distance), 1)
        scale = float(np.median(distance[iu]))
        if scale < 1e-8:
            scale = 1.0
        kernel = 1.0 / np.sqrt((distance / scale) ** 2 + 1.0)
        center = np.eye(len(distance)) - np.ones_like(distance) / len(distance)
        return center @ kernel @ center
    k_arm, k_ref = centered_imq(d_arm), centered_imq(d_ref)
    i, j = np.triu_indices(len(ref), 1)
    mask = np.abs(i - j) * step > delta_max
    da, dr = d_arm[i[mask], j[mask]], d_ref[i[mask], j[mask]]
    ka, kr = k_arm[i[mask], j[mask]], k_ref[i[mask], j[mask]]
    return {"rho_nonlocal": float(scipy.stats.spearmanr(da, dr).statistic),
            "cka_nonlocal": float(np.dot(ka, kr) / (np.linalg.norm(ka) * np.linalg.norm(kr)))}


def patient_bootstrap(rows, key_path, patient_map, n_boot=10000, seed=42):
    grouped = defaultdict(list)
    for row in rows:
        patient = patient_map.get(row["record_id"])
        if patient is None or pd.isna(patient):
            continue
        value = row
        for key in key_path:
            value = value[key]
        grouped[str(patient)].append(float(value))
    values = np.asarray([np.mean(items) for items in grouped.values()])
    rng = np.random.default_rng(seed)
    boots = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)])
    return {"n_patients": len(values), "mean": float(values.mean()),
            "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n002", type=Path, default=ROOT / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt")
    parser.add_argument("--n003a", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/model_best.pt")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/frozen_artifact/freeze_manifest.json")
    parser.add_argument("--g002-records", type=Path, default=ROOT / "results/g002_eval/g002_clinical_records.json")
    parser.add_argument("--emory", type=Path, default=Path("/data/mithunmanivannan/heedb_emory"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/n003a_matched_geometry")
    parser.add_argument("--max-core", type=int, default=500)
    parser.add_argument("--max-nonlocal", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args(); torch.backends.cudnn.enabled = False
    frozen = json.loads(args.freeze_manifest.read_text())
    if sha256(args.n003a) != frozen["selected_checkpoint_sha256"]:
        raise RuntimeError("N003a checkpoint differs from frozen artifact")
    core_ids = [row["record_id"] for row in json.loads(args.g002_records.read_text())][:args.max_core]
    with os.scandir(args.emory / "WFDB/2010") as entries:
        nonlocal_ids = [entry.name[:-4] for entry in entries
                        if entry.name.endswith(".hea") and not entry.name.startswith("._")][:args.max_nonlocal]
    metadata = pd.read_csv(args.emory / "metadata/metadata.csv", usecols=["BDSPPatientID", "FileID"])
    patient_map = dict(zip(metadata.FileID, metadata.BDSPPatientID))
    models = {"N002": load_author_model(ROOT / "author_code/nefnet_v2", args.n002, DEVICE),
              "N003a": load_n003a(args.n003a)}
    all_ids = list(dict.fromkeys(core_ids + nonlocal_ids)); rows = []
    for index, record_id in enumerate(all_ids):
        x_gt = load_clinical_record(args.emory / "WFDB/2010" / f"{record_id}.hea")
        ref = x_gt.T; r0 = ref @ KORS_MATRIX.T
        completions = {arm: complete(x_gt, model, arm) for arm, model in models.items()}
        row = {"record_id": record_id, "in_core": record_id in core_ids,
               "in_nonlocal": record_id in nonlocal_ids, "arms": {}}
        for arm, representation in completions.items():
            metrics = {}
            if row["in_core"]:
                metrics.update(compute_geometry_metrics(representation, ref))
            if row["in_nonlocal"]:
                metrics.update(compute_aligned_stress_and_cka(representation, ref))
                metrics.update(nonlocal_metrics(representation, ref))
            row["arms"][arm] = metrics
        if row["in_core"]:
            row["R0_core"] = compute_geometry_metrics(r0, ref)
        if row["in_nonlocal"]:
            row["R0_nonlocal"] = {**compute_aligned_stress_and_cka(r0, ref), **nonlocal_metrics(r0, ref)}
        rows.append(row)
        if (index + 1) % 25 == 0 or index + 1 == len(all_ids):
            print(f"evaluated={index + 1}/{len(all_ids)}", flush=True)

    core = [row for row in rows if row["in_core"]]
    nonlocal_rows = [row for row in rows if row["in_nonlocal"]]
    for row in rows:
        for metric in ("spearman_rho_d", "knn_jaccard_k10", "cahc_ari"):
            if row["in_core"]:
                row.setdefault("paired_delta", {})[metric] = row["arms"]["N003a"][metric] - row["arms"]["N002"][metric]
        for metric in ("eps_aligned", "cka", "cka_nonlocal"):
            if row["in_nonlocal"]:
                row.setdefault("paired_delta", {})[metric] = row["arms"]["N003a"][metric] - row["arms"]["N002"][metric]
    summary = {
        "protocol": {"core_record_source": str(args.g002_records), "core_record_source_sha256": sha256(args.g002_records),
                     "n_core": len(core), "n_nonlocal": len(nonlocal_rows), "delta_max_samples": 50,
                     "sampling_hz": 500, "bootstrap_unit": "BDSPPatientID"},
        "checkpoints": {"N002": sha256(args.n002), "N003a": sha256(args.n003a)},
        "paired_n003a_minus_n002": {
            metric: patient_bootstrap(core if metric in ("spearman_rho_d", "knn_jaccard_k10", "cahc_ari") else nonlocal_rows,
                                      ["paired_delta", metric], patient_map, seed=5100 + i)
            for i, metric in enumerate(("spearman_rho_d", "knn_jaccard_k10", "cahc_ari", "eps_aligned", "cka", "cka_nonlocal"))
        },
    }
    for row in rows:
        if row["in_core"]:
            for metric in ("spearman_rho_d", "knn_jaccard_k10", "cahc_ari"):
                row.setdefault("delta_n003a_minus_r0", {})[metric] = row["arms"]["N003a"][metric] - row["R0_core"][metric]
        if row["in_nonlocal"]:
            for metric in ("eps_aligned", "cka", "cka_nonlocal"):
                row.setdefault("delta_n003a_minus_r0", {})[metric] = row["arms"]["N003a"][metric] - row["R0_nonlocal"][metric]
    summary["n003a_minus_r0"] = {
        metric: patient_bootstrap(core if metric in ("spearman_rho_d", "knn_jaccard_k10", "cahc_ari") else nonlocal_rows,
                                  ["delta_n003a_minus_r0", metric], patient_map, seed=5300 + i)
        for i, metric in enumerate(("spearman_rho_d", "knn_jaccard_k10", "cahc_ari", "eps_aligned", "cka", "cka_nonlocal"))
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "geometry_record_manifest.json").write_text(json.dumps({"core": core_ids, "nonlocal": nonlocal_ids}, indent=2) + "\n")
    (args.output / "matched_geometry_records.json").write_text(json.dumps(rows) + "\n")
    (args.output / "matched_geometry_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
