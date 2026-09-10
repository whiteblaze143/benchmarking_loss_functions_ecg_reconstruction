#!/usr/bin/env python3
"""Audit generated Lead-I peaks against Lead II without using Lead II as input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.materialize_ptbxl_beat_bounds import atomic_text, detect_r_peaks


def matched_f1(candidate: np.ndarray, reference: np.ndarray, tolerance: int) -> tuple[float, list[int]]:
    used = np.zeros(len(reference), dtype=bool)
    errors = []
    for peak in candidate:
        distance = np.abs(reference - peak)
        distance[used] = np.iinfo(np.int64).max
        match = int(distance.argmin())
        if distance[match] <= tolerance:
            used[match] = True
            errors.append(int(distance[match]))
    precision = len(errors) / max(len(candidate), 1)
    recall = len(errors) / max(len(reference), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12), errors


def audit(tensor_root: Path, bounds_root: Path, records: int, fs: int = 500) -> dict:
    result = {"version": 1, "reference_lead": "II", "reference_used_for_generation": False, "splits": {}}
    tolerance = round(0.1 * fs)
    for split in ("train", "val", "test"):
        generated = {
            row["record_id"]: row
            for row in (json.loads(line) for line in (bounds_root / f"{split}.jsonl").read_text().splitlines())
        }
        files = sorted((tensor_root / split).glob("*.pt"), key=lambda path: int(path.stem))
        indices = np.linspace(0, len(files) - 1, min(records, len(files)), dtype=int)
        f1_values, count_deltas, errors, failures = [], [], [], []
        for index in indices:
            path = files[int(index)]
            if path.stem not in generated:
                failures.append({"record_id": path.stem, "error": "missing generated row"})
                continue
            try:
                lead_ii = torch.load(path, map_location="cpu", weights_only=True)[1].numpy()
                reference = detect_r_peaks(lead_ii, fs)
                candidate = np.asarray(generated[path.stem]["r_peaks"], dtype=np.int64)
                f1, timing = matched_f1(candidate, reference, tolerance)
                f1_values.append(f1)
                count_deltas.append(abs(len(candidate) - len(reference)))
                errors.extend(timing)
            except (ValueError, RuntimeError) as exc:
                failures.append({"record_id": path.stem, "error": str(exc)})
        split_result = {
            "records": len(indices),
            "failures": len(failures),
            "median_f1": float(np.median(f1_values)),
            "f1_p05": float(np.quantile(f1_values, 0.05)),
            "median_abs_count_delta": float(np.median(count_deltas)),
            "matched_peak_timing_median_ms": float(np.median(errors) * 1000 / fs),
            "matched_peak_timing_p95_ms": float(np.quantile(errors, 0.95) * 1000 / fs),
            "failure_examples": failures[:20],
        }
        result["splits"][split] = split_result
        if split_result["f1_p05"] < 0.90 or split_result["matched_peak_timing_p95_ms"] > 50:
            raise RuntimeError(f"{split} cross-lead agreement failed: {split_result}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--bounds-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-per-split", type=int, default=500)
    args = parser.parse_args()
    result = audit(args.tensor_root, args.bounds_root, args.records_per_split)
    atomic_text(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

