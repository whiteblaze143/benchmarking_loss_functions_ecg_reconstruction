from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from repecg.common import load_config
from repecg.common.beats import detect_rpeaks
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import bandpass_ecg, fit_lead_scaler, independent_basis
from repecg.common.ptbxl import PTBXLStore


def _config_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=128)
    args = parser.parse_args()
    if args.records <= 0:
        raise ValueError("records must be positive")

    config = load_config(args.config)
    store = PTBXLStore(config)
    frame = store.split_frame("development_train").iloc[: args.records]
    filtered = []
    source = []
    for ecg_id in frame.index:
        record = store.read_record(int(ecg_id))
        basis = independent_basis(bandpass_ecg(record.signal_mv))
        filtered.append(basis)
        source.append(record)
    scaler = fit_lead_scaler(filtered)

    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for record, basis_mv in zip(source, filtered, strict=True):
        detection = detect_rpeaks(basis_mv)
        standardized = scaler.transform(basis_mv)
        beats = phase_normalize(standardized, detection.valid_intervals)
        eligible = len(beats) >= config.raw["beats"]["min_full_cycles"]
        filename = f"{record.ecg_id}.npz"
        np.savez_compressed(
            args.output / filename,
            beats=beats.astype(np.float32),
            labels=record.labels.astype(np.float32),
            patient_id=np.int64(record.patient_id),
            ecg_id=np.int64(record.ecg_id),
        )
        rows.append(
            {
                "ecg_id": record.ecg_id,
                "patient_id": record.patient_id,
                "fold": record.fold,
                "detector_lead": detection.lead,
                "rpeaks": int(len(detection.rpeaks)),
                "valid_cycles": int(len(beats)),
                "eligible": bool(eligible),
                "reason": "" if eligible else "fewer_than_two_valid_cycles",
                "artifact": filename,
            }
        )
    manifest = {
        "kind": "non_scientific_m1_smoke_cache",
        "config_sha256": _config_hash(args.config),
        "records_requested": args.records,
        "records_written": len(rows),
        "eligible": sum(row["eligible"] for row in rows),
        "lead_mean": scaler.mean.tolist(),
        "lead_std": scaler.std.tolist(),
        "rows": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("kind", "records_written", "eligible")}, sort_keys=True))


if __name__ == "__main__":
    main()
