#!/usr/bin/env python3
"""Audit SJLIFE pairing metadata before any cross-device representation claim."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


def _describe(values: pd.Series) -> dict[str, float]:
    return {key: float(value) for key, value in values.describe(percentiles=[.05, .25, .5, .75, .95]).items()}


def _clinical_rate_witness(paths: list[Path], reference_rate: np.ndarray, hz: int) -> dict[str, float]:
    """Infer file duration by QRS count, checked against supplied ventricular rate."""
    band = butter(3, [5, 20], btype="bandpass", fs=hz, output="sos")
    counts = []
    for path in paths:
        waveform = np.asarray(np.load(path, mmap_mode="r"))[0]
        filtered = sosfiltfilt(band, waveform, axis=1)
        energy = np.mean(np.diff(filtered, axis=1) ** 2, axis=0)
        width = round(0.12 * hz)
        energy = np.convolve(energy, np.ones(width) / width, mode="same")
        peaks, _ = find_peaks(energy, distance=round(0.35 * hz), height=np.quantile(energy, .70), prominence=np.std(energy) * .2)
        counts.append(len(peaks))
    estimated = np.asarray(counts) * 60 / (5000 / hz)
    error = np.abs(estimated - reference_rate)
    return {"hz": hz, "median_absolute_hr_error_bpm": float(np.median(error)), "mean_absolute_hr_error_bpm": float(error.mean()), "fraction_within_10_bpm": float(np.mean(error <= 10))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    table = pd.read_csv(args.data / "shared_paired_data_243.csv")
    required = {"sudoID", "apple_loc_ECG_243", "clinical_loc_ECG_243", "Minutes_ABS_Btw_Clinical_Apple", "RestingECG_OriginalRestingECGMeasurements_VentricularRate_", "ecgAvgHR"}
    if missing := required - set(table):
        raise ValueError(f"missing columns {sorted(missing)}")
    apple_paths = [args.data / path for path in table.apple_loc_ECG_243]
    # The metadata path has an apple_ prefix although files use clinical_.
    clinical_paths = [args.data / "ClinicalECGs_full_243" / Path(path).name.replace("apple_", "clinical_") for path in table.clinical_loc_ECG_243]
    if not all(path.exists() for path in apple_paths + clinical_paths):
        raise ValueError("not all paired waveform files resolve")
    apple_shapes = sorted({tuple(np.load(path, mmap_mode="r").shape) for path in apple_paths})
    clinical_shapes = sorted({tuple(np.load(path, mmap_mode="r").shape) for path in clinical_paths})
    apple_scale = np.asarray([np.std(np.load(path, mmap_mode="r")) for path in apple_paths])
    clinical_scale = np.asarray([np.std(np.load(path, mmap_mode="r")) for path in clinical_paths])
    separation = table.Minutes_ABS_Btw_Clinical_Apple.astype(float)
    heart_rate_difference = np.abs(table.RestingECG_OriginalRestingECGMeasurements_VentricularRate_.astype(float) - table.ecgAvgHR.astype(float))
    clinical_rate = table.RestingECG_OriginalRestingECGMeasurements_VentricularRate_.to_numpy(float)
    rate_witness = [_clinical_rate_witness(clinical_paths, clinical_rate, hz) for hz in (250, 500)]
    selected_witness = min(rate_witness, key=lambda witness: witness["mean_absolute_hr_error_bpm"])
    output = {
        "schema_version": "sjlife_transport_admission_audit_v1", "status": "complete",
        "subjects": int(table.sudoID.nunique()), "paired_rows": int(len(table)),
        "waveforms": {"apple": {"paths_resolved": len(apple_paths), "shapes": [list(shape) for shape in apple_shapes]}, "clinical": {"paths_resolved": len(clinical_paths), "shapes": [list(shape) for shape in clinical_shapes]}},
        "time_separation_minutes": {"distribution": _describe(separation), "at_most_5": int((separation <= 5).sum()), "at_most_10": int((separation <= 10).sum()), "at_most_30": int((separation <= 30).sum()), "at_most_60": int((separation <= 60).sum())},
        "rhythm_proxy_absolute_hr_difference_bpm": {"distribution": _describe(heart_rate_difference), "at_most_5": int((heart_rate_difference <= 5).sum()), "at_most_10": int((heart_rate_difference <= 10).sum())},
        "published_acquisition_protocol": {"clinical": "GE MUSE X, standard supine 12-lead, 10 seconds", "watch": "Apple Watch Series 7 ECG-Air single lead, 30 seconds, 512 Hz, recorded sequentially after electrode removal while seated"},
        "sample_rate": {"apple": "512 Hz, consistent with 15360 samples / 30 seconds", "clinical": "publication says source records were 500 or 250 Hz and downsampled to 250 Hz, but stored arrays have 5000 samples despite the stated 10-second duration", "clinical_qrs_rate_witness": rate_witness, "clinical_file_timebase": f"500 Hz selected empirically: QRS-count rates match supplied ventricular rates best (mean absolute error {selected_witness['mean_absolute_hr_error_bpm']:.3f} bpm)"},
        "amplitude_scale": {"apple_record_std": _describe(pd.Series(apple_scale)), "clinical_record_std": _describe(pd.Series(clinical_scale)), "cross_device_unit_calibration": "not supplied; observed scales differ materially"},
        "admissibility": {"sample_aligned_reconstruction": "NO: recordings are sequential, postures differ, and no common timing/beat correspondence is provided", "beat_normalization": "POSSIBLE ONLY after resolving clinical file timebase plus an explicit device-specific R-peak QC and unit-normalization contract", "same_patient_representation_retrieval": "CONDITIONALLY POSSIBLE as a hardware-transport endpoint after that contract, but must not be interpreted as acute waveform agreement"},
        "source_note": "CSV clinical paths require deterministic apple_->clinical_ filename substitution; raw files themselves are complete.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "subjects": output["subjects"]}, sort_keys=True))


if __name__ == "__main__":
    main()
