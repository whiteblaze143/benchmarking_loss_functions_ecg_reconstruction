#!/usr/bin/env python3
"""Materialize the official SemiSegECG LUDB pickle contract from raw WFDB data."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb


SPLITS = (
    "LUDB_train_unlabeled.csv",
    "LUDB_train_labeled_1over2.csv",
    "LUDB_valid.csv",
    "LUDB_test.csv",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def parse_annotation(record_path: Path, lead: str) -> tuple[np.ndarray, int, int]:
    annotation = wfdb.rdann(str(record_path), extension=lead.lower())
    if len(annotation.sample) % 3:
        raise ValueError(f"malformed annotation triplets: {record_path.name}/{lead}")
    label = np.zeros(5000, dtype=np.float64)
    p_onsets, qrs_onsets, qrs_offsets, t_offsets = [], [], [], []
    for index in range(0, len(annotation.sample), 3):
        onset, _peak, offset = map(int, annotation.sample[index:index + 3])
        left, wave, right = annotation.symbol[index:index + 3]
        if left != "(" or right != ")" or wave not in {"p", "N", "t"}:
            raise ValueError(f"malformed annotation symbols: {record_path.name}/{lead}")
        class_id = {"p": 1, "N": 2, "t": 3}[wave]
        label[onset:offset] = class_id
        if wave == "p":
            p_onsets.append(onset)
        elif wave == "N":
            qrs_onsets.append(onset)
            qrs_offsets.append(offset)
        else:
            t_offsets.append(offset)
    support = p_onsets + qrs_onsets + qrs_offsets + t_offsets
    return label, min(support), max(support)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("/home/mithunmanivannan/data/ludb"))
    parser.add_argument("--index-root", type=Path, default=Path("external/semiseg/official_indices/ludb"))
    parser.add_argument("--output-root", type=Path, default=Path("external/semiseg/author_data/ludb"))
    args = parser.parse_args()

    frames = [pd.read_csv(args.index_root / name) for name in SPLITS]
    streams = sorted(set().union(*(frame.waveform.tolist() for frame in frames)))
    ecg_dir, label_dir = args.output_root / "ecg", args.output_root / "label"
    ecg_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    cached_records: dict[str, object] = {}
    for position, filename in enumerate(streams, start=1):
        stem = Path(filename).stem
        record_id, lead = stem.split("_lead_", 1)
        record = cached_records.get(record_id)
        if record is None:
            record = wfdb.rdrecord(str(args.raw_root / record_id), physical=True)
            cached_records[record_id] = record
        lookup = {name.lower(): index for index, name in enumerate(record.sig_name)}
        signal = np.asarray(record.p_signal[:5000, lookup[lead.lower()]], dtype=np.float64).copy()
        label, label_start, label_end = parse_annotation(args.raw_root / record_id, lead)
        # Exact LUDB.ipynb behavior: replace unannotated edge signal with the
        # signal values at the first/last annotation support coordinates.
        signal[:label_start] = signal[label_start]
        signal[label_end:] = signal[label_end]
        with (ecg_dir / filename).open("wb") as handle:
            pickle.dump(signal, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with (label_dir / filename).open("wb") as handle:
            pickle.dump(label, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if position % 250 == 0 or position == len(streams):
            print(json.dumps({"event": "progress", "streams": position, "total": len(streams)}), flush=True)

    all_labeled = pd.read_csv(args.index_root / "LUDB_train_unlabeled.csv")
    all_labeled.to_csv(args.index_root / "LUDB_train_labeled_all.csv", index=False)
    manifest = {
        "protocol": "semiseg_ludb_author_data_v1",
        "streams": len(streams),
        "records": len(cached_records),
        "raw_root": str(args.raw_root.resolve()),
        "index_hashes": {name: digest(args.index_root / name) for name in SPLITS},
        "train_labeled_all_sha256": digest(args.index_root / "LUDB_train_labeled_all.csv"),
        "signal_dtype": "float64",
        "label_dtype": "float64",
        "label_dependent_edge_padding": True,
    }
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **manifest}), flush=True)


if __name__ == "__main__":
    main()
