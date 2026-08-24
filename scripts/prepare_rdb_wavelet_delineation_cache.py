#!/usr/bin/env python3
"""Build a provenance-pinned RDB cache for wavelet/delineation training.

The mapping-flagged duplicate is excluded.  Splits are deterministic,
Chapman-record-disjoint, and stratified by the mapped Chapman rhythm.  The
test partition is materialized for audit but is never consumed by the sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rdb_oracle import (
    LEADS,
    RAW_UNITS_PER_MV,
    TARGET_SAMPLES,
    read_annotation,
    read_mapping,
    sha256_file,
)


DEFAULT_SOURCE = ROOT / "data/rdb"
DEFAULT_OUTPUT = ROOT / "data/rdb_wavelet_delineation_cache"
FIDUCIAL_NAMES = (
    "P_onset", "P_offset", "QRS_onset", "QRS_offset", "T_onset", "T_offset",
)
BOUNDARY_CHANNELS = {0: (0, 1), 1: (2, 3), 2: (4, 5)}
FIDUCIAL_SIGMA_SAMPLES = 5.0  # 10 ms at 500 Hz


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def durable_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stratified_assignments(
    rows: list[dict[str, str | None]], seed: int, validation_fraction: float, test_fraction: float
) -> dict[str, str]:
    grouped: dict[str, list[dict[str, str | None]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["chapman_rhythm"])].append(row)
    assignments: dict[str, str] = {}
    for rhythm, group in sorted(grouped.items()):
        ordered = sorted(
            group,
            key=lambda row: hashlib.sha256(
                f"rdb-wavelet-v1:{seed}:{row['chapman_record_id']}".encode()
            ).hexdigest(),
        )
        test_count = round(len(ordered) * test_fraction)
        validation_count = round(len(ordered) * validation_fraction)
        if len(ordered) - test_count - validation_count < 1:
            raise ValueError(f"split leaves no training records for rhythm {rhythm}")
        for index, row in enumerate(ordered):
            split = "test" if index < test_count else (
                "val" if index < test_count + validation_count else "train"
            )
            assignments[str(row["rdb_record_id"])] = split
    return assignments


def dense_labels(
    source_root: Path, record_id: str, audit: Counter[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    labels = np.zeros((12, TARGET_SAMPLES), dtype=np.int64)
    valid = np.ones((12, TARGET_SAMPLES), dtype=np.bool_)
    # Float16 keeps the materialized cache compact. Values are promoted to
    # float32 by the strict training loader.
    fiducials = np.zeros((12, 6, TARGET_SAMPLES), dtype=np.float16)
    fiducial_valid = np.zeros((12, 6), dtype=np.bool_)
    annotation_hashes: list[str] = []
    ambiguous_samples = 0
    exclusion_keys = (
        "excluded_wrong_columns", "excluded_parse_error", "excluded_invalid_value",
        "excluded_reversed", "excluded_zero_length", "excluded_out_of_bounds",
    )
    def merge_audit(local: Counter[str]) -> None:
        for key, value in local.items():
            if key == "max_rounding_adjustment_microsamples":
                audit[key] = max(audit[key], value)
            else:
                audit[key] += value

    for lead_index, lead in enumerate(LEADS):
        path = source_root / "ann_txt" / f"{record_id}.{lead}.txt"
        local: Counter[str] = Counter()
        intervals = read_annotation(path, local, "lead")
        merge_audit(local)
        annotation_hashes.append(sha256_file(path))
        if any(local[key] for key in exclusion_keys):
            # An invalid source row cannot always be localized safely.  The
            # conservative contract excludes this lead from the supervised
            # segmentation loss while retaining its waveform for reconstruction.
            labels[lead_index] = -1
            valid[lead_index] = False
            audit["invalid_lead_streams"] += 1
            continue
        occupied = np.zeros(TARGET_SAMPLES, dtype=np.bool_)
        for kind, onset, offset in intervals:
            kind, onset, offset = int(kind), int(onset), int(offset)
            region = slice(int(onset), int(offset) + 1)
            proposed = kind + 1
            conflict = occupied[region] & (labels[lead_index, region] != proposed)
            if conflict.any():
                indices = np.flatnonzero(conflict) + int(onset)
                labels[lead_index, indices] = -1
                valid[lead_index, indices] = False
                ambiguous_samples += int(len(indices))
            fill = ~occupied[region]
            labels[lead_index, region][fill] = proposed
            occupied[region] = True
            onset_channel, offset_channel = BOUNDARY_CHANNELS[kind]
            for channel, center in ((onset_channel, onset), (offset_channel, offset)):
                left = max(0, int(center - 4 * FIDUCIAL_SIGMA_SAMPLES))
                right = min(TARGET_SAMPLES, int(center + 4 * FIDUCIAL_SIGMA_SAMPLES) + 1)
                positions = np.arange(left, right, dtype=np.float32)
                bump = np.exp(-0.5 * ((positions - center) / FIDUCIAL_SIGMA_SAMPLES) ** 2)
                fiducials[lead_index, channel, left:right] = np.maximum(
                    fiducials[lead_index, channel, left:right], bump.astype(np.float16)
                )
                fiducial_valid[lead_index, channel] = True
    consensus_path = source_root / "ann_txt" / f"{record_id}.all.txt"
    consensus_audit: Counter[str] = Counter()
    read_annotation(consensus_path, consensus_audit, "consensus")
    merge_audit(consensus_audit)
    annotation_hashes.append(sha256_file(consensus_path))
    return labels, valid, fiducials, fiducial_valid, annotation_hashes, ambiguous_samples


def build(
    source_root: Path,
    output_root: Path,
    split_seed: int,
    validation_fraction: float,
    test_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {output_root}")
    if validation_fraction <= 0 or test_fraction <= 0 or validation_fraction + test_fraction >= 0.5:
        raise ValueError("validation/test fractions must be positive and sum to less than 0.5")
    mapping_path = source_root / "rdb_chapman_mapping.xlsx"
    source_rows = read_mapping(mapping_path)
    excluded = [row for row in source_rows if row["note"] == "duplicate_rdb_record"]
    rows = [row for row in source_rows if row["note"] != "duplicate_rdb_record"]
    if len(rows) != 2398 or len({row["chapman_record_id"] for row in rows}) != 2398:
        raise ValueError("expected 2,398 unique Chapman sources after duplicate exclusion")
    assignments = stratified_assignments(rows, split_seed, validation_fraction, test_fraction)
    # Waveform, dense labels and six float16 boundary heatmap channels plus
    # serialization overhead. Keep the disk gate conservative.
    estimated_cache_bytes = len(rows) * 2_000_000
    required = estimated_cache_bytes + minimum_free_gib * 1024**3
    free = shutil.disk_usage(output_root.parent).free
    if free < required:
        raise RuntimeError(
            f"disk gate: {free/1024**3:.2f} GiB free; requires {required/1024**3:.2f} GiB"
        )

    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    audit: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    rhythm_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    class_counts: dict[str, np.ndarray] = {
        split: np.zeros(5, dtype=np.int64) for split in ("train", "val", "test")
    }
    records: list[dict[str, Any]] = []
    dataset_digest = hashlib.sha256()
    try:
        for path in (source_root / "README.md", mapping_path):
            digest = sha256_file(path)
            dataset_digest.update(path.name.encode())
            dataset_digest.update(digest.encode())
        for index, row in enumerate(rows, 1):
            record_id = str(row["rdb_record_id"])
            patient_id = str(row["chapman_record_id"])
            split = assignments[record_id]
            signal_path = source_root / "dat_csv" / f"{record_id}.csv"
            signal_hash = sha256_file(signal_path)
            signal = np.loadtxt(signal_path, delimiter=",", dtype=np.float32)
            if signal.shape != (TARGET_SAMPLES, 12) or not np.isfinite(signal).all():
                raise ValueError(f"invalid RDB signal: {record_id} {signal.shape}")
            waveform = (signal.T / RAW_UNITS_PER_MV).astype(np.float32)
            labels, valid, fiducials, fiducial_valid, annotation_hashes, ambiguous = dense_labels(
                source_root, record_id, audit
            )
            paths = [signal_path, *(
                source_root / "ann_txt" / f"{record_id}.{lead}.txt" for lead in (*LEADS, "all")
            )]
            hashes = [signal_hash, *annotation_hashes]
            bundle = hashlib.sha256()
            for path, digest in zip(paths, hashes, strict=True):
                dataset_digest.update(path.name.encode())
                dataset_digest.update(digest.encode())
                bundle.update(path.name.encode())
                bundle.update(digest.encode())
            output = temporary_root / split / f"rdb_{record_id}.pt"
            payload = {
                "waveform": torch.from_numpy(waveform),
                "segmentation": torch.from_numpy(labels),
                "seg_valid": torch.from_numpy(valid),
                "fiducial_heatmaps": torch.from_numpy(fiducials),
                "fiducial_valid": torch.from_numpy(fiducial_valid),
                "fiducial_names": FIDUCIAL_NAMES,
                "annotation_weight": 1.0,
                "annotation_type": "lead_specific",
                "record_id": record_id,
                "patient_id": patient_id,
                "source_dataset": "RDB",
                "released_rhythm": str(row["rdb_rhythm"]),
                "canonical_rhythm": str(row["chapman_rhythm"]),
                "split": split,
            }
            durable_torch_save(payload, output)
            output_hash = sha256_file(output)
            split_counts[split] += 1
            rhythm_split_counts[str(row["chapman_rhythm"])][split] += 1
            for klass in range(4):
                class_counts[split][klass] += int(np.count_nonzero(labels == klass))
            class_counts[split][4] += int(np.count_nonzero(labels == -1))
            records.append({
                "record_id": record_id,
                "patient_id": patient_id,
                "released_rhythm": str(row["rdb_rhythm"]),
                "canonical_rhythm": str(row["chapman_rhythm"]),
                "split": split,
                "source_bundle_sha256": bundle.hexdigest(),
                "output": str(output.relative_to(temporary_root)),
                "output_sha256": output_hash,
                "ambiguous_samples": ambiguous,
            })
            if index % 100 == 0 or index == len(rows):
                print(f"built {index}/{len(rows)} records", flush=True)
        manifest: dict[str, Any] = {
            "version": 1,
            "created_at": utc_now(),
            "source": {
                "name": "Resting ECG Segmentation Dataset (RDB)",
                "root": str(source_root.resolve()),
                "readme_sha256": sha256_file(source_root / "README.md"),
                "mapping_sha256": sha256_file(mapping_path),
                "dataset_sha256": dataset_digest.hexdigest(),
                "raw_signal_origin_doi": "10.6084/m9.figshare.4560497.v2",
                "annotation_method_paper_doi": "10.1016/j.eswa.2025.127955",
                "raw_units_per_mv": RAW_UNITS_PER_MV,
                "duplicate_policy": "exclude mapping-flagged duplicate_rdb_record",
                "excluded_records": excluded,
            },
            "split": {
                "method": "Chapman-record-disjoint rhythm-stratified SHA-256 ordering",
                "seed": split_seed,
                "validation_fraction": validation_fraction,
                "test_fraction": test_fraction,
                "counts": dict(split_counts),
                "rhythm_counts": {
                    rhythm: dict(counts) for rhythm, counts in sorted(rhythm_split_counts.items())
                },
                "test_role": "untouched; excluded from architecture selection and sweep training",
            },
            "label_contract": {
                "classes": {"0": "background", "1": "P", "2": "QRS", "3": "T", "-1": "invalid"},
                "source_intervals": "lead-specific, inclusive END, half-up coordinate rounding",
                "fiducials_available": True,
                "fiducial_names": list(FIDUCIAL_NAMES),
                "fiducial_heatmap_sigma_samples": FIDUCIAL_SIGMA_SAMPLES,
                "fiducial_source": "lead-specific annotated inclusive START/END",
                "peak_channel_policy": "no peak output channels; RDB releases only START/END",
                "invalid_source_row_policy": "exclude the affected lead stream from supervised segmentation",
                "cross_class_overlap_policy": "mark only ambiguous samples invalid",
            },
            "waveform_contract": {"shape": [12, 5000], "sample_rate_hz": 500, "units": "mV"},
            "audit": dict(audit),
            "class_counts": {
                split: dict(zip(("background", "P", "QRS", "T", "invalid"), map(int, counts), strict=True))
                for split, counts in class_counts.items()
            },
            "records": records,
        }
        atomic_json(temporary_root / "manifest.json", manifest)
        directory_fd = os.open(temporary_root, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary_root, output_root)
        parent_fd = os.open(output_root.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-seed", type=int, default=20260822)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--minimum-free-gib", type=float, default=8.0)
    args = parser.parse_args()
    manifest = build(
        args.source_root.resolve(), args.output_root.resolve(), args.split_seed,
        args.validation_fraction, args.test_fraction, args.minimum_free_gib,
    )
    print(json.dumps({
        "output": str(args.output_root.resolve()),
        "dataset_sha256": manifest["source"]["dataset_sha256"],
        "split_counts": manifest["split"]["counts"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
