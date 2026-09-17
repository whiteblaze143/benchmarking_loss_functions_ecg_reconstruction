#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt

from repecg.common.beats import detect_rpeaks
from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.common.phase import phase_normalize
from repecg.common.preprocess import LeadScaler
from repecg.evaluation.kingston import zero_padded_basis
from repecg.paper02_kernel_mean.controls import exact_moment_matched_gaussian, mean_covariance_features


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _filter_and_resample(signal: np.ndarray, source_fs: float) -> np.ndarray:
    sos = butter(4, [0.5, 40.0], btype="bandpass", fs=source_fs, output="sos")
    filtered = sosfiltfilt(sos, signal, axis=0)
    if source_fs == 500.0:
        return filtered
    ratio = Fraction(500.0 / source_fs).limit_denominator(1000)
    return resample_poly(filtered, ratio.numerator, ratio.denominator, axis=0)


def _rhythm_label(value: str) -> str:
    mapping = {"SINUS": "SINUS", "AFIB/AFLT": "AFIB_AFLT"}
    try:
        return mapping[str(value)]
    except KeyError as error:
        raise ValueError(f"unsupported Kingston rhythm label: {value!r}") from error


def _embed(
    beats: np.ndarray,
    whitening: WhiteningTransform,
    mapping: NystromMap,
    generator: torch.Generator,
) -> dict[str, np.ndarray]:
    device = torch.device("cuda")
    cells = torch.as_tensor(
        beats.reshape(len(beats), 16, 16, 8).transpose(1, 0, 2, 3).reshape(16, len(beats) * 16, 8),
        device=device,
        dtype=torch.float32,
    )
    white_mean = torch.as_tensor(whitening.mean, device=device, dtype=torch.float32)
    components = torch.as_tensor(whitening.components, device=device, dtype=torch.float32)
    scales = torch.as_tensor(whitening.scales, device=device, dtype=torch.float32)
    anchors = torch.as_tensor(mapping.landmarks, device=device, dtype=torch.float32)
    inverse_root = torch.as_tensor(mapping.inverse_root, device=device, dtype=torch.float32)

    def embed(values: torch.Tensor) -> torch.Tensor:
        white = (values.float() - white_mean) @ components * scales
        feature = torch.rsqrt(torch.cdist(white, anchors.expand(len(values), -1, -1)).square() + mapping.c2)
        return (feature @ inverse_root).mean(dim=1)

    with torch.inference_mode():
        matched = exact_moment_matched_gaussian(cells, generator=generator, tolerance=1e-5)
        return {
            "kernel": embed(cells).cpu().numpy(),
            "gaussian": embed(matched).cpu().numpy(),
            "linear": (((cells - white_mean) @ components) * scales).mean(dim=1).cpu().numpy(),
            "moments": mean_covariance_features(cells.double()).float().cpu().numpy(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("Train", "Test"), default="Test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    metadata_path = args.dataset / "metadata.csv"
    metadata = pd.read_csv(metadata_path)
    selected = metadata.loc[metadata["TrainOrTest"] == args.split].copy()
    if args.max_records is not None:
        if args.max_records < 1:
            raise ValueError("--max-records must be positive")
        selected = selected.iloc[: args.max_records].copy()
    if selected.empty:
        raise RuntimeError(f"Kingston split {args.split!r} has no records")
    with np.load(args.kernel_fit) as item:
        whitening = WhiteningTransform(item["whitening_mean"], item["whitening_components"], item["whitening_scales"])
        mapping = NystromMap(item["landmarks"], item["inverse_root"], float(item["c2"]))
    scaler_payload = json.loads(args.scaler.read_text())
    scaler = LeadScaler(np.asarray(scaler_payload["mean"]), np.asarray(scaler_payload["std"]))

    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    rows: list[dict[str, object]] = []
    representations: dict[str, list[np.ndarray]] = {key: [] for key in ("kernel", "gaussian", "linear", "moments")}
    for item in selected.itertuples(index=False):
        record_path = args.dataset / str(item.ECG)
        rhythm = _rhythm_label(item.Rhythm)
        try:
            record = wfdb.rdrecord(str(record_path))
            source_leads = tuple(record.sig_name)
            if source_leads != ("I", "II", "III", "V"):
                raise ValueError(f"unexpected Kingston leads: {source_leads}")
            filtered = _filter_and_resample(np.asarray(record.p_signal), float(record.fs))
            basis = zero_padded_basis(filtered, source_leads, scaler)
            detection = detect_rpeaks(basis.physical_mv, fs=500, fallback_leads=("I",))
            beats = phase_normalize(basis.standardized, detection.valid_intervals)
            if len(beats) < 2:
                rows.append({"record_id": record_path.name, "eligible": False, "reason": "fewer_than_two_cycles", "rhythm": rhythm})
                continue
            embedded = _embed(beats, whitening, mapping, generator)
            for key, value in embedded.items():
                representations[key].append(value)
            rows.append({"record_id": record_path.name, "eligible": True, "reason": "", "rhythm": rhythm})
        except Exception as error:
            rows.append({"record_id": record_path.name, "eligible": False, "reason": f"{type(error).__name__}:{error}", "rhythm": rhythm})

    eligible_rows = [row for row in rows if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no eligible Kingston records")
    args.output.mkdir(parents=True, exist_ok=True)
    split_slug = args.split.lower()
    archive = args.output / f"representation_negative_control_kingston_icu_{split_slug}.npz"
    np.savez_compressed(
        archive,
        **{key: np.stack(value) for key, value in representations.items()},
        ecg_ids=np.asarray([row["record_id"] for row in eligible_rows], dtype=str),
        rhythm_labels=np.asarray([row["rhythm"] for row in eligible_rows], dtype=str),
        observed_lead_mask=np.tile(np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8), (len(eligible_rows), 1)),
        n_observed_leads=np.full(len(eligible_rows), 2, dtype=np.uint8),
    )
    reconciliation = pd.DataFrame(rows)
    reconciliation.to_csv(args.output / f"kingston_reconciliation_{split_slug}.csv", index=False)
    manifest = {
        "kind": "missing_lead_zero_padding_negative_control",
        "evaluation_role": "negative_control",
        "dataset": "kingston-icu-af-dataset-1.0.0",
        "split": args.split,
        "source_records": int(len(selected)),
        "eligible_records": int(len(eligible_rows)),
        "ineligible_records": int(len(selected) - len(eligible_rows)),
        "source_fs_hz": 240,
        "target_fs_hz": 500,
        "source_leads": ["I", "II", "III", "V"],
        "admitted_leads": ["I", "II"],
        "excluded_leads": {"III": "derived_limb_lead", "V": "unspecified_precordial_location"},
        "canonical_independent_leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        "observed_lead_mask": [1, 1, 0, 0, 0, 0, 0, 0],
        "padding": "exact_zero_after_frozen_ptbxl_standardization",
        "task_id": "kingston_rhythm",
        "task_type": "binary_classification",
        "labels": ["SINUS", "AFIB_AFLT"],
        "ptbxl_five_class_status": "ineligible_task_mismatch",
        "source_native_label_counts": {"SINUS": int((selected["Rhythm"] == "SINUS").sum()), "AFIB_AFLT": int((selected["Rhythm"] == "AFIB/AFLT").sum())},
        "eligible_native_label_counts": {
            label: sum(row["rhythm"] == label for row in eligible_rows)
            for label in ("SINUS", "AFIB_AFLT")
        },
        "metadata_sha256": _sha256(metadata_path),
        "kernel_fit_sha256": _sha256(args.kernel_fit),
        "scaler_sha256": _sha256(args.scaler),
        "archive_sha256": _sha256(archive),
        "seed": args.seed,
        "max_records": args.max_records,
    }
    (args.output / f"kingston_manifest_{split_slug}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
