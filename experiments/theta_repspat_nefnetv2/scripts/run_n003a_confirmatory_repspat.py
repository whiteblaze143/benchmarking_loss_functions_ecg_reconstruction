#!/usr/bin/env python3
"""Confirmatory frozen Gate-3B repSpat comparison for N002, N003a, R0, and R1."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(REPO)]
from theta_repspat.panorama import CANONICAL_ANGLES_RAD  # noqa: E402
from theta_repspat.vendor import instantiate_author_model, load_author_model  # noqa: E402
from eval_g002 import KORS_MATRIX  # noqa: E402
from tit_ecg.src.dataset_adapters import ISPAdapter, LUDBAdapter  # noqa: E402
from tit_ecg.scripts.run_empirical_multi_dataset_benchmark import (  # noqa: E402
    beat_ids_from_segmentation, evaluate_empirical_null_and_power_record,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUTS, QUERIES = [0, 1, 4], [2, 3, 5, 6, 7]
ARMS = ["R0_Kors", "R1_Oracle", "N002", "N003a"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_n003a(path: Path):
    model = instantiate_author_model(ROOT / "author_code/nefnet_v2", DEVICE, super_mode="pretrain")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    return model


def completed(raw: np.ndarray, model, arm: str) -> np.ndarray:
    leads = raw[:, [0, 1, 6, 7, 8, 9, 10, 11]] if raw.shape[1] == 12 else raw
    n = len(leads)
    padded = np.pad(leads.T, ((0, 0), (0, max(0, 4608 - n))), mode="edge")[:, :4608]
    observed = padded[INPUTS]
    if arm == "N002":
        lo, hi = float(observed.min()), float(observed.max())
        span = hi - lo if hi > lo else 1.0
        normalized = (observed - lo) / span
    else:
        lo, span = -4.0, 8.0
        normalized = np.clip((observed - lo) / span, 1e-4, 1 - 1e-4)
    x = torch.from_numpy(normalized.astype(np.float32)).unsqueeze(0).to(DEVICE)
    angles = torch.from_numpy(CANONICAL_ANGLES_RAD[INPUTS]).unsqueeze(0).to(DEVICE)
    output = padded.copy()
    for query_idx in QUERIES:
        query = torch.from_numpy(CANONICAL_ANGLES_RAD[query_idx]).unsqueeze(0).to(DEVICE)
        if arm == "N003a":
            query = query.unsqueeze(1)
        with torch.no_grad():
            pred = model(x, angles, query).squeeze().cpu().numpy()
        output[query_idx] = pred * span + lo
    output[INPUTS] = padded[INPUTS]
    return output[:, :n].T


def run_task(task):
    dataset, record_id, arm, values, fs, segmentation, beat_ids, permutations = task
    result = evaluate_empirical_null_and_power_record(
        dataset, record_id, values, fs, segmentation, beat_ids,
        block_mode="attribute_kmeans", gamma=1.0,
        m_ms_grid=[32.0, 64.0, 128.0], G_grid=[4, 6, 8],
        n_permutations=permutations, random_state=42,
    )
    result["arm"] = arm
    for pair in result["diagnostic_pairs"]:
        pair["arm"] = arm
    return result


def atomic_json(payload, path: Path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    os.replace(temporary, path)


def paired_bootstrap(frame, arm_a: str, arm_b: str, seed: int):
    pivot = frame.pivot(index=["dataset", "record_id"], columns="arm", values="recurrence_rejection_rate")
    paired = pivot[[arm_a, arm_b]].dropna()
    delta = (paired[arm_a] - paired[arm_b]).to_numpy()
    if not len(delta):
        return {"n_paired_patients": 0, "mean_delta": None, "ci95": None}
    rng = np.random.default_rng(seed)
    boots = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(10000)])
    return {"n_paired_patients": len(delta), "mean_delta": float(delta.mean()),
            "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n002", type=Path, default=ROOT / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt")
    parser.add_argument("--n003a", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/model_best.pt")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/frozen_artifact/freeze_manifest.json")
    parser.add_argument("--exploratory-records", type=Path, default=ROOT / "results/theta_repspat_eval/theta_repspat_phase1_results.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/n003a_confirmatory_repspat_b1000")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args(); torch.backends.cudnn.enabled = False
    frozen = json.loads(args.freeze_manifest.read_text())
    if sha256(args.n003a) != frozen["selected_checkpoint_sha256"]:
        raise RuntimeError("N003a checkpoint differs from frozen artifact")
    prior = json.loads(args.exploratory_records.read_text())
    ids = {dataset: list(dict.fromkeys(str(row["record_id"]) for row in prior if row["dataset"] == dataset))
           for dataset in ("LUDB", "ISP")}
    if args.max_records:
        ids = {dataset: values[:args.max_records] for dataset, values in ids.items()}
    adapters = {"LUDB": LUDBAdapter(data_dir=str(REPO / "data/ludb")),
                "ISP": ISPAdapter(data_dir=str(REPO / "data/isp_delineation_dataset"))}
    models = {"N002": load_author_model(ROOT / "author_code/nefnet_v2", args.n002, DEVICE),
              "N003a": load_n003a(args.n003a)}
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "record_checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    tasks, results = [], []
    for dataset, record_ids in ids.items():
        for record_id in record_ids:
            n_samples = 1500 if dataset == "LUDB" else 3000
            rec = adapters[dataset].load_record(int(record_id), n_samples=n_samples)
            raw = rec["ecg"][:n_samples]; fs = float(rec["fs"]); segmentation = rec["segmentation"][:len(raw)]
            values = {"R1_Oracle": raw[:, [0, 1, 6, 7, 8, 9, 10, 11]] if raw.shape[1] == 12 else raw,
                      "R0_Kors": None}
            values["R0_Kors"] = values["R1_Oracle"] @ KORS_MATRIX.T
            values.update({arm: completed(raw, model, arm) for arm, model in models.items()})
            beat_ids = beat_ids_from_segmentation(segmentation)
            for arm in ARMS:
                path = checkpoint_dir / f"{dataset}_{record_id}_{arm}.json"
                if path.exists():
                    results.append(json.loads(path.read_text()))
                else:
                    tasks.append((dataset, record_id, arm, values[arm], fs, segmentation,
                                  beat_ids, args.permutations))
    print(f"records={sum(map(len, ids.values()))} tasks_missing={len(tasks)} B={args.permutations}", flush=True)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = {executor.submit(run_task, task): task[:3] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            path = checkpoint_dir / f"{result['dataset']}_{result['record_id']}_{result['arm']}.json"
            atomic_json(result, path); results.append(result)
            print(f"completed={len(results)}/{len(tasks) + len(results)} {futures[future]}", flush=True)
    expected = sum(map(len, ids.values())) * len(ARMS)
    if len(results) != expected:
        raise RuntimeError(f"Result reconciliation failed: {len(results)} != {expected}")
    pairs = [pair for result in results for pair in result.pop("diagnostic_pairs")]
    frame = pd.DataFrame(results)
    summary_rows = []
    for (dataset, arm), group in frame.groupby(["dataset", "arm"]):
        eligible = group[group.n_recurrence_control_pairs > 0]
        summary_rows.append({"dataset": dataset, "arm": arm, "n_records": len(group),
                             "n_eligible_patients": len(eligible),
                             "n_recurrence_pairs": int(group.n_recurrence_control_pairs.sum()),
                             "recurrence_rejections": int(group.recurrence_rejections.sum()),
                             "pooled_recurrence_rejection_rate": float(group.recurrence_rejections.sum() / group.n_recurrence_control_pairs.sum()) if group.n_recurrence_control_pairs.sum() else None,
                             "patient_mean_recurrence_rejection_rate": float(eligible.recurrence_rejection_rate.mean()) if len(eligible) else None,
                             "n_positive_pairs": int(group.n_alt_pairs.sum()),
                             "positive_rejections": int(group.alt_rejections.sum()),
                             "positive_control_power": float(group.alt_rejections.sum() / group.n_alt_pairs.sum()) if group.n_alt_pairs.sum() else None})
    summary = {"protocol": {"B": args.permutations, "block_mode": "attribute_kmeans", "gamma": 1.0,
                            "m_ms_grid": [32, 64, 128], "G_grid": [4, 6, 8], "global_BH_alpha": .05,
                            "purity_rule": "exact_phase_and_beat_membership", "record_ids": ids},
               "checkpoints": {"N002": sha256(args.n002), "N003a": sha256(args.n003a)},
               "results": summary_rows,
               "paired_recurrence_deltas": {"N003a_minus_N002": paired_bootstrap(frame, "N003a", "N002", 6101),
                                            "N003a_minus_R0": paired_bootstrap(frame, "N003a", "R0_Kors", 6102)}}
    frame.to_csv(args.output / "confirmatory_runs.csv", index=False)
    pd.DataFrame(pairs).to_csv(args.output / "confirmatory_pairs.csv", index=False)
    atomic_json(summary, args.output / "confirmatory_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
