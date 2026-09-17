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
from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import (
    LeadScaler,
    LeadScalerAccumulator,
    bandpass_ecg,
    independent_basis,
)
from repecg.common.ptbxl import PTBXLStore


_STORE: PTBXLStore | None = None
_SCALER: LeadScaler | None = None
_PRE_A001_CONFIG_SHA256 = "036933eaae2a95c136ecf80e97f004518bef275a5956d18c15778d1603e0249a"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _preprocessing_sha256(config: object) -> str:
    raw = config.raw
    payload = {name: raw[name] for name in ("data", "splits", "labels", "signal", "beats")}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _initialize(config_path: str, scaler_path: str | None) -> None:
    global _STORE, _SCALER
    _STORE = PTBXLStore(load_config(config_path))
    if scaler_path is not None:
        payload = json.loads(Path(scaler_path).read_text())
        _SCALER = LeadScaler(
            mean=np.asarray(payload["mean"], dtype=np.float64),
            std=np.asarray(payload["std"], dtype=np.float64),
        )


def _filtered_basis(ecg_id: int) -> tuple[object, np.ndarray]:
    assert _STORE is not None
    record = _STORE.read_record(ecg_id)
    return record, independent_basis(bandpass_ecg(record.signal_mv))


def _moments(ecg_id: int) -> tuple[int, np.ndarray, np.ndarray]:
    _, basis = _filtered_basis(ecg_id)
    mean = basis.mean(axis=0)
    return len(basis), mean, np.square(basis - mean).sum(axis=0)


def _prepare(ecg_id: int) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    assert _SCALER is not None
    record, basis = _filtered_basis(ecg_id)
    detection = detect_rpeaks(basis)
    beats = phase_normalize(_SCALER.transform(basis), detection.valid_intervals)
    minimum = int(_STORE.config.raw["beats"]["min_full_cycles"])
    eligible = len(beats) >= minimum
    row = {
        "ecg_id": record.ecg_id,
        "patient_id": record.patient_id,
        "fold": record.fold,
        "detector_lead": detection.lead,
        "rpeaks": len(detection.rpeaks),
        "valid_cycles": len(beats),
        "eligible": eligible,
        "reason": "" if eligible else f"fewer_than_{minimum}_valid_cycles",
        "artifact": f"records/{record.ecg_id}.npz",
    }
    return row, beats.astype(np.float32), record.labels.astype(np.float32)


def _pool(workers: int, config: Path, scaler: Path | None) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialize,
        initargs=(str(config), None if scaler is None else str(scaler)),
    )


def fit_scaler(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    store = PTBXLStore(config)
    ids = [int(value) for value in store.split_frame(args.split).index]
    if args.limit is not None:
        ids = ids[: args.limit]
    accumulator = LeadScalerAccumulator()
    with _pool(args.workers, args.config, None) as executor:
        for count, mean, m2 in executor.map(_moments, ids, chunksize=8):
            accumulator.merge(count, mean, m2)
    scaler = accumulator.finalize()
    payload = {
        "kind": "train_only_per_lead_scaler" if args.limit is None else "non_scientific_limited_scaler",
        "config_sha256": _sha256(args.config),
        "preprocessing_sha256": _preprocessing_sha256(config),
        "source_split": args.split,
        "records": len(ids),
        "samples": accumulator.count,
        "mean": scaler.mean.tolist(),
        "std": scaler.std.tolist(),
    }
    args.scaler.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.scaler.with_suffix(args.scaler.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.scaler)
    print(json.dumps({"records": len(ids), "samples": accumulator.count, "scaler": str(args.scaler)}))


def build_split(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    store = PTBXLStore(config)
    if args.split == "locked_test":
        raise PermissionError("production preparation cannot open locked_test")
    scaler_payload = json.loads(args.scaler.read_text())
    preprocessing_sha = _preprocessing_sha256(config)
    scaler_preprocessing_sha = scaler_payload.get("preprocessing_sha256")
    if scaler_preprocessing_sha is not None:
        if scaler_preprocessing_sha != preprocessing_sha:
            raise ValueError("scaler/preprocessing hash mismatch")
    elif scaler_payload.get("config_sha256") != _PRE_A001_CONFIG_SHA256:
        raise ValueError("legacy scaler is not the documented pre-A001 artifact")
    output = args.output / args.split
    records_dir = output / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(value) for value in store.split_frame(args.split).index]
    if args.limit is not None:
        ids = ids[: args.limit]
    state_path = output / "state.json"
    progress_path = output / "progress.jsonl"
    expected_state = {
        "preprocessing_sha256": preprocessing_sha,
        "scaler_sha256": _sha256(args.scaler),
        "split": args.split,
        "limit": args.limit,
    }
    if state_path.exists():
        observed_state = json.loads(state_path.read_text())
        legacy_state = {
            "config_sha256": _PRE_A001_CONFIG_SHA256,
            "scaler_sha256": _sha256(args.scaler),
            "split": args.split,
            "limit": args.limit,
        }
        if observed_state not in (expected_state, legacy_state):
            raise ValueError("existing cache state does not match this invocation")
    else:
        state_path.write_text(json.dumps(expected_state, indent=2, sort_keys=True) + "\n")
    rows = []
    if progress_path.exists():
        rows = [json.loads(line) for line in progress_path.read_text().splitlines() if line]
    retained = [row for row in rows if (output / str(row["artifact"])).is_file()]
    if len(retained) != len(rows):
        repair = progress_path.with_suffix(".repair.tmp")
        repair.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in retained))
        os.replace(repair, progress_path)
        rows = retained
    completed = {int(row["ecg_id"]) for row in rows}
    pending = [ecg_id for ecg_id in ids if ecg_id not in completed]
    with _pool(args.workers, args.config, args.scaler) as executor:
        for row, beats, labels in executor.map(_prepare, pending, chunksize=4):
            destination = output / str(row["artifact"])
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                beats=beats,
                labels=labels,
                patient_id=np.int64(row["patient_id"]),
                ecg_id=np.int64(row["ecg_id"]),
            )
            os.replace(temporary, destination)
            rows.append(row)
            with progress_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    qc = pd.DataFrame(rows).sort_values("ecg_id")
    qc.to_csv(output / "qc.csv", index=False)
    manifest = {
        "kind": "production_phase_cache" if args.limit is None else "non_scientific_limited_phase_cache",
        "config_sha256": _sha256(args.config),
        "preprocessing_sha256": preprocessing_sha,
        "scaler_sha256": _sha256(args.scaler),
        "split": args.split,
        "records": len(qc),
        "eligible": int(qc.eligible.sum()),
        "ineligible": int((~qc.eligible).sum()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fit-scaler", "build-split"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.command == "fit-scaler":
        fit_scaler(args)
    else:
        if args.output is None:
            parser.error("build-split requires --output")
        build_split(args)


if __name__ == "__main__":
    main()
