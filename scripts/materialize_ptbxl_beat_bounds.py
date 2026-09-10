#!/usr/bin/env python3
"""Materialize auditable Lead-I beat intervals for the ECG-AIM/LVCG pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_r_peaks(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Return polarity-invariant R peaks using the LVCG paper's SciPy recipe."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or signal.size < fs:
        raise ValueError("signal must be a one-dimensional recording of at least one second")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains non-finite values")
    signal = signal - signal.mean()
    scale = signal.std()
    if scale > 1e-6:
        signal = signal / scale

    filtered = sosfiltfilt(butter(3, (5.0, 20.0), btype="bandpass", fs=fs, output="sos"), signal)
    derivative_energy = np.diff(filtered, prepend=filtered[0]) ** 2
    window = max(1, round(0.08 * fs))
    envelope = np.convolve(derivative_energy, np.ones(window) / window, mode="same")
    distance = max(1, round(0.25 * fs))

    candidates = np.empty(0, dtype=np.int64)
    for quantile in (0.90, 0.75):
        candidates, _ = find_peaks(envelope, height=np.quantile(envelope, quantile), distance=distance)
        if candidates.size >= 2:
            break
    if candidates.size < 2:
        raise ValueError("fewer than two R-peak candidates")

    radius = round(0.08 * fs)
    refined = []
    for candidate in candidates:
        lo, hi = max(0, candidate - radius), min(signal.size, candidate + radius + 1)
        refined.append(lo + int(np.argmax(np.abs(signal[lo:hi]))))
    peaks = np.asarray(sorted(set(refined)), dtype=np.int64)
    keep = [int(peaks[0])]
    for peak in peaks[1:]:
        if int(peak) - keep[-1] >= distance:
            keep.append(int(peak))
        elif abs(signal[int(peak)]) > abs(signal[keep[-1]]):
            keep[-1] = int(peak)
    peaks = np.asarray(keep, dtype=np.int64)
    if peaks.size < 2:
        raise ValueError("fewer than two distinct physiological R peaks")
    return peaks


def beat_bounds(peaks: np.ndarray, length: int, fs: int = 500, max_beats: int = 40) -> list[list[int]]:
    """Create LVCG-style boundary and consecutive-peak intervals."""
    min_rr, max_rr = round(0.25 * fs), round(2.0 * fs)
    bounds = [[0, int(peaks[0])]]
    bounds.extend(
        [int(left), int(right)]
        for left, right in zip(peaks[:-1], peaks[1:])
        if min_rr <= int(right - left) <= max_rr
    )
    bounds.append([int(peaks[-1]), int(length)])
    bounds = [pair for pair in bounds if pair[1] > pair[0]]
    if not bounds:
        raise ValueError("no valid beat intervals")
    if len(bounds) > max_beats:
        raise ValueError(f"{len(bounds)} beat intervals exceed max_beats={max_beats}")
    return bounds


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize(input_root: Path, output_root: Path, fs: int, max_beats: int, limit: int | None) -> dict:
    source_manifest = Path(__file__).resolve().parents[1] / "refine-logs" / "ptbxl_tensor_content_manifest.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"missing PTB-XL tensor provenance: {source_manifest}")
    audit = {
        "version": 1,
        "detector": "scipy_bandpass_derivative_energy_v1",
        "reference_lead": "I",
        "sample_rate_hz": fs,
        "max_beats": max_beats,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256(source_manifest),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        files = sorted((input_root / split).glob("*.pt"), key=lambda path: int(path.stem))
        if limit is not None:
            files = files[:limit]
        if not files:
            raise FileNotFoundError(f"no tensors in {input_root / split}")
        rows, failures, counts = [], [], []
        for path in files:
            try:
                tensor = torch.load(path, map_location="cpu", weights_only=True)
                if not isinstance(tensor, torch.Tensor) or tensor.shape != (12, 5000):
                    raise ValueError(f"expected [12,5000], got {getattr(tensor, 'shape', None)}")
                peaks = detect_r_peaks(tensor[0].numpy(), fs)
                bounds = beat_bounds(peaks, tensor.shape[-1], fs, max_beats)
                counts.append(len(bounds))
                rows.append({"record_id": path.stem, "beat_bounds": bounds, "r_peaks": peaks.tolist()})
            except (ValueError, RuntimeError) as exc:
                failures.append({"record_id": path.stem, "error": str(exc)})
        failure_rate = len(failures) / len(files)
        audit["splits"][split] = {
            "records": len(files),
            "successes": len(rows),
            "failures": len(failures),
            "failure_rate": failure_rate,
            "beat_count_min": min(counts) if counts else None,
            "beat_count_median": float(np.median(counts)) if counts else None,
            "beat_count_max": max(counts) if counts else None,
            "failure_examples": failures[:20],
        }
        payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
        atomic_text(output_root / f"{split}.jsonl", payload)
        if failure_rate > 0.01:
            atomic_text(output_root / "audit.json", json.dumps(audit, indent=2) + "\n")
            raise RuntimeError(f"{split} failure rate {failure_rate:.3%} exceeds 1% gate")
    atomic_text(output_root / "audit.json", json.dumps(audit, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=500)
    parser.add_argument("--max-beats", type=int, default=40)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()
    audit = materialize(args.input_root, args.output_root, args.sample_rate, args.max_beats, args.limit_per_split)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

