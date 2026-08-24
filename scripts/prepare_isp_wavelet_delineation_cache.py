#!/usr/bin/env python3
"""Build the strict 500-Hz wavelet-MTL cache from the ISP training partition."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.signal
import torch
import wfdb


ROOT = Path(__file__).resolve().parents[1]
LEADS = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
SOURCE_FS = 1000
TARGET_FS = 500
TARGET_LEN = 5000
WAVE_CLASS = {0: 1, 1: 2, 2: 3}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_split(record_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:isp_train:{record_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if value < 0.2 else "train"


def exact_split(rows: list[dict[str, str]], seed: int, validation_fraction: float) -> dict[str, str]:
    count = max(1, round(len(rows) * validation_fraction))
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:isp_train:{row['file_name']}".encode()).digest(),
    )
    validation = {row["file_name"] for row in ranked[:count]}
    return {row["file_name"]: ("val" if row["file_name"] in validation else "train") for row in rows}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"file_name", "age", "sex", "target"}:
        raise ValueError(f"unexpected ISP CSV schema: {path}")
    if len({row["file_name"] for row in rows}) != len(rows):
        raise ValueError("duplicate record IDs in ISP training CSV")
    return rows


def resample_record(record_path: Path) -> tuple[np.ndarray, dict[str, object]]:
    record = wfdb.rdrecord(str(record_path))
    if int(record.fs) != SOURCE_FS:
        raise ValueError(f"{record_path}: expected {SOURCE_FS} Hz, got {record.fs}")
    names = [str(name).lower() for name in record.sig_name]
    if set(names) != set(LEADS) or len(names) != 12:
        raise ValueError(f"{record_path}: unexpected leads {record.sig_name}")
    units = [str(unit).lower() for unit in record.units]
    if set(units) != {"mkv"}:
        raise ValueError(f"{record_path}: expected microvolt 'mkv' units, got {record.units}")
    signal = np.asarray(record.p_signal, dtype=np.float64)[:, [names.index(lead) for lead in LEADS]]
    if not np.isfinite(signal).all():
        raise ValueError(f"{record_path}: non-finite waveform")
    # WFDB exposes the source's `mkv` (microvolt) physical units.  The PTB-XL
    # reconstruction tensors use mV, so convert before the upstream resample.
    signal = signal / 1000.0
    signal = scipy.signal.resample_poly(signal, TARGET_FS, SOURCE_FS, axis=0)
    if signal.shape[0] < TARGET_LEN:
        signal = np.pad(signal, ((0, TARGET_LEN - signal.shape[0]), (0, 0)), mode="edge")
    signal = signal[:TARGET_LEN].T.astype(np.float32, copy=False)
    if signal.shape != (12, TARGET_LEN) or not np.isfinite(signal).all():
        raise ValueError(f"{record_path}: bad resampled waveform {signal.shape}")
    metadata = {
        "source_samples": int(record.sig_len), "source_fs_hz": int(record.fs),
        "source_units": units, "target_fs_hz": TARGET_FS, "target_samples": TARGET_LEN,
        "unit_conversion": "mkv (microvolt) / 1000 -> mV",
        "resampler": "scipy.signal.resample_poly(up=500,down=1000), edge-pad then crop to 5000",
    }
    return signal, metadata


def parse_intervals(target_text: str) -> list[tuple[int, int, int]]:
    intervals = ast.literal_eval(target_text)
    if not isinstance(intervals, list):
        raise ValueError("ISP target must be a list")
    parsed=[]
    for item in intervals:
        if not isinstance(item, tuple) or len(item) != 3:
            raise ValueError(f"bad interval {item!r}")
        parsed.append(tuple(map(int,item)))
    return parsed


def segmentation(target_text: str, source_samples: int = SOURCE_FS * 10) -> tuple[np.ndarray, int]:
    intervals=parse_intervals(target_text)
    labels = np.zeros(TARGET_LEN, dtype=np.int64)
    for item in intervals:
        wave, onset, offset = item
        if wave not in WAVE_CLASS or not (0 <= onset < offset <= source_samples):
            raise ValueError(f"bad ISP interval {item!r}")
        start = max(0, math.floor(onset * TARGET_FS / SOURCE_FS))
        end = min(TARGET_LEN, math.ceil(offset * TARGET_FS / SOURCE_FS))
        occupied = labels[start:end]
        if np.any((occupied != 0) & (occupied != WAVE_CLASS[wave])):
            raise ValueError(f"overlapping wave classes around {item!r}")
        labels[start:end] = WAVE_CLASS[wave]
    return np.broadcast_to(labels[None], (12, TARGET_LEN)).copy(), len(intervals)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(source_root: Path, output_root: Path, validation_fraction: float, split_seed: int) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output_root}")
    if not (0 < validation_fraction < 0.5):
        raise ValueError("validation fraction must be in (0,0.5)")
    csv_path = source_root / "train_isp_delineation_data.csv"
    source_rows = load_rows(csv_path)
    if len(source_rows) != 403:
        raise ValueError(f"expected the 403-record ISP training partition, found {len(source_rows)}")
    rows=[];exclusions=[]
    for row in source_rows:
        record_id=row["file_name"]
        source_samples=int(wfdb.rdheader(str(source_root/"train_data"/record_id)).sig_len)
        invalid=[item for item in parse_intervals(row["target"]) if not (0<=item[1]<item[2]<=source_samples)]
        if invalid:
            exclusions.append({"record_id":record_id,"source_samples":source_samples,
                               "reason":"annotation_outside_waveform","intervals":invalid})
        else:
            rows.append(row)
    assignments = exact_split(rows, split_seed, validation_fraction)
    estimate = len(rows) * 900_000 + 2 * 1024**3
    free = shutil.disk_usage(output_root.parent).free
    if free < estimate:
        raise RuntimeError(f"cache disk gate: need {estimate/1024**3:.2f} GiB, have {free/1024**3:.2f} GiB")
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.build.", dir=output_root.parent))
    records: list[dict[str, object]] = []
    class_counts = np.zeros(4, dtype=np.int64)
    try:
        for split in ("train", "val"):
            (temporary_root / split).mkdir(parents=True)
        for index, row in enumerate(rows, start=1):
            record_id = row["file_name"]
            split = assignments[record_id]
            record_path = source_root / "train_data" / record_id
            waveform, resample_metadata = resample_record(record_path)
            labels, interval_count = segmentation(row["target"],int(resample_metadata["source_samples"]))
            class_counts += np.bincount(labels[0], minlength=4)
            patient_id = f"isp_subject_{record_id}"
            output = temporary_root / split / f"isp_{record_id}.pt"
            payload: dict[str, object] = {
                "waveform": torch.from_numpy(waveform),
                "segmentation": torch.from_numpy(labels),
                "seg_valid": torch.ones(12, TARGET_LEN, dtype=torch.bool),
                "annotation_weight": 1.0,
                "annotation_type": "integrated",
                "record_id": f"isp_train_{record_id}",
                "patient_id": patient_id,
                "source_dataset": "ISP",
                "source_partition": "official_train",
                "source_record": str(record_path.resolve()),
                "source_header_sha256": sha256_file(record_path.with_suffix(".hea")),
                "source_signal_sha256": sha256_file(record_path.with_suffix(".dat")),
                "interval_count": interval_count,
                "resampling": resample_metadata,
            }
            atomic_torch_save(payload, output)
            records.append({
                "record_id": record_id, "patient_id": patient_id, "split": split,
                "output": output.name, "output_sha256": sha256_file(output),
                "source_header_sha256": payload["source_header_sha256"],
                "source_signal_sha256": payload["source_signal_sha256"],
                "interval_count": interval_count,
            })
            if index % 25 == 0 or index == len(rows):
                print(f"prepared {index}/{len(rows)} ISP records", flush=True)
        train_patients = {str(row["patient_id"]) for row in records if row["split"] == "train"}
        val_patients = {str(row["patient_id"]) for row in records if row["split"] == "val"}
        if train_patients & val_patients:
            raise RuntimeError("patient overlap in generated split")
        manifest: dict[str, object] = {
            "version": 1, "created_at": utc_now(),
            "source": {
                "name": "ISP ECG delineation dataset", "zenodo_doi": "10.5281/zenodo.14679837",
                "root": str(source_root.resolve()), "training_csv": str(csv_path.resolve()),
                "training_csv_sha256": sha256_file(csv_path), "official_test_used": False,
                "official_train_records":len(source_rows),"included_records":len(rows),
                "excluded_records":exclusions,
                "subject_identity_basis": "ISP reports one ECG per subject; official-train record ID is the subject key",
            },
            "split": {
                "method": "SHA-256 rank of official-train subject ID", "seed": split_seed,
                "validation_fraction": validation_fraction, "train_subjects": len(train_patients),
                "val_subjects": len(val_patients), "patient_overlap": 0,
            },
            "label_contract": {
                "type": "integrated timing duplicated across 12 leads (not lead-specific truth)",
                "classes": {"0": "background", "1": "P", "2": "QRS", "3": "T"},
                "interval_mapping": "floor(onset/2):ceil(offset/2)", "fiducials_present": False,
                "sample_rate_hz": TARGET_FS, "samples": TARGET_LEN,
            },
            "class_counts_per_integrated_stream": {
                "background": int(class_counts[0]), "P": int(class_counts[1]),
                "QRS": int(class_counts[2]), "T": int(class_counts[3]),
            },
            "records": records,
        }
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary_root, output_root)
        return manifest
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/isp_delineation_dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/delineation_cache")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260822)
    args = parser.parse_args()
    manifest = build(
        args.source_root.resolve(), args.output_root.resolve(),
        args.validation_fraction, args.split_seed,
    )
    print(json.dumps({"records": len(manifest["records"]), **manifest["split"]}, indent=2))


if __name__ == "__main__":
    main()
