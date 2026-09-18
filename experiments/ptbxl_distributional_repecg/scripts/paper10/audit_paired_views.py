from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from repecg.common import load_config
from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.common.preprocess import LeadScaler
from repecg.common.ptbxl import PTBXLStore
from repecg.paper10_interventional.paired_views import (
    apply_environment,
    environment_bank,
    fingerprint,
    preprocess_view,
    relative_l2,
    snr_db,
)


_STORE: PTBXLStore | None = None
_SCALER: LeadScaler | None = None
_WHITENING: WhiteningTransform | None = None
_MAPPING: NystromMap | None = None
_MASTER_SEED: int | None = None
_MINIMUM_CYCLES: int | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_frozen_map(path: Path) -> tuple[WhiteningTransform, NystromMap]:
    with np.load(path) as item:
        whitening = WhiteningTransform(item["whitening_mean"], item["whitening_components"], item["whitening_scales"])
        mapping = NystromMap(item["landmarks"], item["inverse_root"], float(item["c2"]))
    return whitening, mapping


def _coverage(frame: pd.DataFrame, labels: tuple[str, ...]) -> dict[str, object]:
    clean = frame[frame.environment == "clean"].set_index("ecg_id")
    result: dict[str, object] = {}
    for environment, group in frame.groupby("environment", sort=True):
        joined = group.set_index("ecg_id").join(clean[["eligible"]], rsuffix="_clean", how="inner")
        denominator = joined[joined.eligible_clean]
        entry: dict[str, object] = {
            "common_support_denominator": int(len(denominator)),
            "common_support_valid": int((denominator.eligible).sum()),
            "coverage": None if len(denominator) == 0 else float(denominator.eligible.mean()),
        }
        by_label: dict[str, object] = {}
        for index, label in enumerate(labels):
            positives = denominator[denominator[f"label_{index}"] == 1]
            by_label[label] = {
                "denominator": int(len(positives)),
                "valid": int(positives.eligible.sum()),
                "coverage": None if len(positives) == 0 else float(positives.eligible.mean()),
            }
        entry["diagnosis_stratified"] = by_label
        result[environment] = entry
    return result


def _initialize(config_path: str, scaler_path: str, kernel_fit_path: str, master_seed: int) -> None:
    global _STORE, _SCALER, _WHITENING, _MAPPING, _MASTER_SEED, _MINIMUM_CYCLES
    config = load_config(config_path)
    _STORE = PTBXLStore(config)
    payload = json.loads(Path(scaler_path).read_text())
    _SCALER = LeadScaler(np.asarray(payload["mean"]), np.asarray(payload["std"]))
    _WHITENING, _MAPPING = _load_frozen_map(Path(kernel_fit_path))
    _MASTER_SEED = master_seed
    _MINIMUM_CYCLES = int(config.raw["beats"]["min_full_cycles"])


def _audit_record(ecg_id: int) -> list[dict[str, object]]:
    assert _STORE is not None and _SCALER is not None and _WHITENING is not None and _MAPPING is not None
    assert _MASTER_SEED is not None and _MINIMUM_CYCLES is not None
    record = _STORE.read_record(ecg_id)
    raw_hash = fingerprint(record.signal_mv)
    label_hash = fingerprint(record.labels)
    clean = preprocess_view(record.signal_mv, scaler=_SCALER, whitening=_WHITENING, mapping=_MAPPING)
    rows: list[dict[str, object]] = []
    for environment in environment_bank():
        transformed, seed = apply_environment(
            record.signal_mv,
            environment,
            patient_id=record.patient_id,
            ecg_id=record.ecg_id,
            master_seed=_MASTER_SEED,
        )
        observed = clean if environment.kind == "clean" else preprocess_view(
            transformed, scaler=_SCALER, whitening=_WHITENING, mapping=_MAPPING
        )
        row: dict[str, object] = {
            "ecg_id": record.ecg_id,
            "patient_id": record.patient_id,
            "fold": record.fold,
            "environment": environment.name,
            "tier": environment.tier,
            "realization": environment.realization,
            "seed": seed,
            "raw_sha256": raw_hash,
            "label_sha256": label_hash,
            "view_sha256": fingerprint(transformed),
            "raw_relative_l2": relative_l2(record.signal_mv, transformed),
            "post_filter_relative_l2": relative_l2(clean["filtered"], observed["filtered"]),
            "raw_snr_db": None if environment.kind != "noise" else snr_db(record.signal_mv, transformed),
            "post_filter_snr_db": None if environment.kind != "noise" else snr_db(clean["filtered"], observed["filtered"]),
            "clean_rpeaks": int(len(clean["rpeaks"])),
            "rpeaks": int(len(observed["rpeaks"])),
            "rpeak_delta": int(len(observed["rpeaks"]) - len(clean["rpeaks"])),
            "clean_valid_cycles": int(clean["valid_cycles"]),
            "valid_cycles": int(observed["valid_cycles"]),
            "cycle_survival": None if clean["valid_cycles"] == 0 else float(observed["valid_cycles"] / clean["valid_cycles"]),
            "detector_lead": observed["detector_lead"],
            "eligible": bool(observed["valid_cycles"] >= _MINIMUM_CYCLES),
            "kme_drift": float(np.linalg.norm(clean["kme"] - observed["kme"])),
        }
        for index, value in enumerate(record.labels):
            row[f"label_{index}"] = int(value)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="G0/G1 raw paired-view provenance and survival audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development_train", "development_select"), required=True)
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    config = load_config(args.config)
    store = PTBXLStore(config)
    if not args.scaler.is_file() or not args.kernel_fit.is_file():
        raise FileNotFoundError("G0/G1 requires the frozen scaler and Phase-KME fit")
    ids = [int(value) for value in store.split_frame(args.split).index]
    if args.limit is not None:
        ids = ids[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    state = {
        "split": args.split,
        "records_requested": len(ids),
        "master_seed": args.master_seed,
        "config_sha256": _sha256(args.config),
        "scaler_sha256": _sha256(args.scaler),
        "kernel_fit_sha256": _sha256(args.kernel_fit),
        "environment_bank": [environment.__dict__ for environment in environment_bank()],
    }
    _atomic_json(args.output / "state.json", state)
    rows: list[dict[str, object]] = []
    initializer = (str(args.config), str(args.scaler), str(args.kernel_fit), args.master_seed)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initialize, initargs=initializer) as executor:
        for sequence, record_rows in enumerate(executor.map(_audit_record, ids, chunksize=1), start=1):
            rows.extend(record_rows)
            if sequence == 1 or sequence % 100 == 0 or sequence == len(ids):
                print(json.dumps({"record": sequence, "records": len(ids)}), flush=True)
    frame = pd.DataFrame(rows).sort_values(["ecg_id", "environment", "realization"])
    expected = len(ids) * len(environment_bank())
    if len(frame) != expected or frame.duplicated(["ecg_id", "environment", "realization"]).any():
        raise RuntimeError("paired-view manifest is incomplete or has duplicate identity keys")
    source = frame.groupby("ecg_id")[["patient_id", "fold", "raw_sha256", "label_sha256"]].nunique()
    if not (source == 1).all().all():
        raise RuntimeError("G0 provenance failure: an intervention changed source identity, fold, or labels")
    frame.to_csv(args.output / "paired_view_audit.csv", index=False)
    summary = {
        "kind": "paper10_g0_g1_paired_view_audit",
        "status": "OBSERVED_NOT_ADMISSIBILITY_DECIDED",
        "g0_provenance_pass": True,
        "g1_note": "Coverage and survival are reported; frozen admissibility thresholds must be supplied before a pass/fail claim.",
        "rows": int(len(frame)),
        "records": len(ids),
        "environment_coverage": _coverage(frame, config.classes),
        "stage_means": frame.groupby("environment", sort=True)[
            ["raw_relative_l2", "post_filter_relative_l2", "rpeak_delta", "valid_cycles", "kme_drift"]
        ].mean().to_dict(),
        "noise_snr_means": frame[frame.raw_snr_db.notna()].groupby("environment", sort=True)[
            ["raw_snr_db", "post_filter_snr_db"]
        ].mean().to_dict(),
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps({"output": str(args.output), "g0_provenance_pass": True, "status": summary["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
