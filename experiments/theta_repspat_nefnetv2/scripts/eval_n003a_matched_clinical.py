#!/usr/bin/env python3
"""Frozen, paired N002-vs-N003a reconstruction evaluation on identical Emory ECGs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from theta_repspat.panorama import CANONICAL_ANGLES_RAD  # noqa: E402
from theta_repspat.vendor import instantiate_author_model, load_author_model  # noqa: E402
from eval_g001_c import load_wfdb_record, pearson_r, ssim_1d  # noqa: E402

INPUT_INDICES = [0, 1, 4]
QUERY_INDICES = [2, 3, 5, 6, 7]
LEADS = ["V1", "V2", "V4", "V5", "V6"]
EXPECTED_LEN = 4608
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def center_crop(data: np.ndarray) -> np.ndarray:
    if data.shape[1] < EXPECTED_LEN:
        return np.pad(data, ((0, 0), (0, EXPECTED_LEN - data.shape[1])), mode="edge")
    start = (data.shape[1] - EXPECTED_LEN) // 2
    return data[:, start:start + EXPECTED_LEN]


def load_n003a(path: Path):
    model = instantiate_author_model(ROOT / "author_code/nefnet_v2", DEVICE, super_mode="pretrain")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_record(data: np.ndarray, model, arm: str) -> dict[str, np.ndarray]:
    observed = data[INPUT_INDICES]
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
    input_angles = torch.from_numpy(CANONICAL_ANGLES_RAD[INPUT_INDICES]).unsqueeze(0).to(DEVICE)
    predictions = {}
    for lead, query_idx in zip(LEADS, QUERY_INDICES):
        query = torch.from_numpy(CANONICAL_ANGLES_RAD[query_idx]).unsqueeze(0).to(DEVICE)
        if arm == "N003a":
            query = query.unsqueeze(1)
        with torch.no_grad():
            prediction = model(inputs, input_angles, query).squeeze().cpu().numpy()
        predictions[lead] = prediction * span + lo
    return predictions


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    data_range = float(target.max() - target.min())
    if data_range <= 0:
        data_range = 1.0
    return {
        "mae_mv": float(np.mean(np.abs(prediction - target))),
        "pearson_r": pearson_r(prediction, target),
        "ssim": ssim_1d(prediction, target, val_range=data_range),
    }


def patient_lead_means(rows: list[dict], arm: str, metric: str, lead: str | None = None):
    values = defaultdict(list)
    for row in rows:
        if row["patient_id"] is None:
            continue
        selected = LEADS if lead is None else [lead]
        values[row["patient_id"]].append(np.mean([row["arms"][arm][name][metric] for name in selected]))
    return {patient: float(np.mean(items)) for patient, items in values.items()}


def paired_summary(rows: list[dict], metric: str, lead: str | None, n_boot: int, seed: int):
    n002 = patient_lead_means(rows, "N002", metric, lead)
    n003a = patient_lead_means(rows, "N003a", metric, lead)
    patients = sorted(set(n002) & set(n003a))
    a = np.asarray([n002[p] for p in patients])
    b = np.asarray([n003a[p] for p in patients])
    delta = b - a
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for index in range(n_boot):
        boots[index] = delta[rng.integers(0, len(delta), len(delta))].mean()
    return {
        "n_patients": len(patients), "n002_mean": float(a.mean()), "n003a_mean": float(b.mean()),
        "delta_n003a_minus_n002": float(delta.mean()),
        "delta_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
    }


def record_summary(rows: list[dict], metric: str):
    values = {
        arm: np.asarray([np.mean([row["arms"][arm][lead][metric] for lead in LEADS])
                         for row in rows])
        for arm in ("N002", "N003a")
    }
    delta = values["N003a"] - values["N002"]
    return {"n_records": len(rows), "n002_mean": float(values["N002"].mean()),
            "n003a_mean": float(values["N003a"].mean()),
            "delta_n003a_minus_n002": float(delta.mean())}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n002", type=Path, default=ROOT / "results/panobench_3view_variablethird_geovt_seed123/model_final.pt")
    parser.add_argument("--n003a", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/model_best.pt")
    parser.add_argument("--freeze-manifest", type=Path, default=ROOT / "results/n003a_stage1_fixed_norm_seed123/frozen_artifact/freeze_manifest.json")
    parser.add_argument("--n002-records", type=Path, default=ROOT / "results/g001_c_eval/g001_c_results.json")
    parser.add_argument("--emory", type=Path, default=Path("/data/mithunmanivannan/heedb_emory"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/n003a_matched_clinical")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.backends.cudnn.enabled = False
    frozen = json.loads(args.freeze_manifest.read_text())
    if sha256(args.n003a) != frozen["selected_checkpoint_sha256"]:
        raise RuntimeError("N003a checkpoint differs from the pre-evaluation freeze")
    record_ids = [row["record_id"] for row in json.loads(args.n002_records.read_text())]
    if args.max_records:
        record_ids = record_ids[:args.max_records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("N002 evaluation record list contains duplicates")
    metadata = pd.read_csv(args.emory / "metadata/metadata.csv", usecols=["BDSPPatientID", "FileID"])
    patient_map = dict(zip(metadata.FileID, metadata.BDSPPatientID))
    missing = [record for record in record_ids if record not in patient_map or pd.isna(patient_map[record])]

    n002 = load_author_model(ROOT / "author_code/nefnet_v2", args.n002, DEVICE)
    n003a = load_n003a(args.n003a)
    rows = []
    for index, record_id in enumerate(record_ids):
        data, _ = load_wfdb_record(args.emory / "WFDB/2010" / f"{record_id}.hea")
        data = center_crop(data)
        predictions = {"N002": predict_record(data, n002, "N002"),
                       "N003a": predict_record(data, n003a, "N003a")}
        arms = {arm: {lead: metrics(predictions[arm][lead], data[q])
                      for lead, q in zip(LEADS, QUERY_INDICES)} for arm in predictions}
        patient_id = None if record_id in missing else str(patient_map[record_id])
        rows.append({"record_id": record_id, "patient_id": patient_id, "arms": arms})
        if (index + 1) % 50 == 0 or index + 1 == len(record_ids):
            print(f"evaluated={index + 1}/{len(record_ids)}", flush=True)

    summary = {
        "protocol": {"records_source": str(args.n002_records), "records_source_sha256": sha256(args.n002_records),
                     "n_records": len(rows),
                     "n_patients_mapped": len({row['patient_id'] for row in rows if row['patient_id'] is not None}),
                     "records_without_patient_id": missing,
                     "input_leads": ["I", "II", "V3"], "query_leads": LEADS,
                     "calibration": "none", "bootstrap_unit": "BDSPPatientID",
                     "n002_preprocessing": "observed-only min-max, inverted per record",
                     "n003a_preprocessing": "frozen [-4,4] mV clipped affine, inverted with x=8*y-4"},
        "checkpoints": {"N002": {"path": str(args.n002), "sha256": sha256(args.n002)},
                        "N003a": {"path": str(args.n003a), "sha256": sha256(args.n003a)}},
        "balanced_primary": {metric: paired_summary(rows, metric, None, args.bootstrap, 4200 + i)
                             for i, metric in enumerate(("mae_mv", "pearson_r", "ssim"))},
        "all_record_point_estimates": {metric: record_summary(rows, metric)
                                       for metric in ("mae_mv", "pearson_r", "ssim")},
        "per_lead": {lead: {metric: paired_summary(rows, metric, lead, args.bootstrap, 4300 + 10*i + j)
                            for j, metric in enumerate(("mae_mv", "pearson_r", "ssim"))}
                     for i, lead in enumerate(LEADS)},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "matched_clinical_records.json").write_text(json.dumps(rows) + "\n")
    (args.output / "matched_clinical_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
