"""Audit Zhejiang waveform/annotation coordinates independently of CAHC."""
from __future__ import annotations

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tit_ecg.src.dataset_adapters import ZhejiangAdapter
from tit_ecg.scripts.run_empirical_multi_dataset_benchmark import beat_ids_from_segmentation


WORST_RECORDS = ["1010096", "1003549", "1010240", "1003838"]
OUT_DIR = "tit_ecg/results/zhejiang_contract_audit"


def intervals(mask: np.ndarray, value: int) -> tuple[np.ndarray, np.ndarray]:
    selected = mask == value
    starts = np.flatnonzero(selected & ~np.r_[False, selected[:-1]])
    ends = np.flatnonzero(selected & ~np.r_[selected[1:], False])
    return starts, ends


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    adapter = ZhejiangAdapter()
    rows = []

    for record_id in adapter.list_records(max_records=10_000):
        label_path = os.path.join(adapter.label_dir, f"{record_id}.pkl")
        with open(label_path, "rb") as handle:
            native_label = np.asarray(pickle.load(handle))
        native_lead_lengths = []
        for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
            with open(os.path.join(adapter.ecg_dir, f"{record_id}_{lead}.pkl"), "rb") as handle:
                native_lead_lengths.append(len(np.asarray(pickle.load(handle))))

        record = adapter.load_record(record_id, n_samples=5000)
        mask = record["segmentation"]
        step_change = np.max(np.abs(np.diff(record["ecg"], axis=0)), axis=1)
        changing = np.flatnonzero(step_change > 1e-10)
        active_end = int(changing[-1] + 2) if len(changing) else 0
        labeled = np.flatnonzero(mask != 0)
        annotation_end = int(labeled[-1] + 1) if len(labeled) else 0
        beat_ids = beat_ids_from_segmentation(mask)
        qrs_starts, qrs_ends = intervals(mask, 2)
        qrs_centers = (qrs_starts + qrs_ends) / 2.0
        rr_ms = np.diff(qrs_centers) * 1000.0 / record["fs"]
        qrs_ms = (qrs_ends - qrs_starts + 1) * 1000.0 / record["fs"]
        qrs_ids = beat_ids[mask == 2]
        unique_qrs_ids = np.unique(qrs_ids)
        one_qrs_per_beat = len(unique_qrs_ids) == len(qrs_starts)

        rows.append({
            "record_id": record_id,
            "native_label_length": len(native_label),
            "native_lead_min_length": min(native_lead_lengths),
            "native_lead_max_length": max(native_lead_lengths),
            "corrected_length": len(mask),
            "duration_seconds": len(mask) / record["fs"],
            "active_signal_duration_seconds": active_end / record["fs"],
            "annotation_coverage_duration_seconds": annotation_end / record["fs"],
            "tail_padding_seconds": (len(mask) - active_end) / record["fs"],
            "n_qrs": len(qrs_starts),
            "median_rr_ms": float(np.median(rr_ms)) if len(rr_ms) else np.nan,
            "median_qrs_duration_ms": float(np.median(qrs_ms)) if len(qrs_ms) else np.nan,
            "unlabeled_fraction": float(np.mean(mask == 0)),
            "p_fraction": float(np.mean(mask == 1)),
            "qrs_fraction": float(np.mean(mask == 2)),
            "t_fraction": float(np.mean(mask == 3)),
            "beat_ids_monotonic": bool(np.all(np.diff(beat_ids) >= 0)),
            "qrs_ids_contiguous": bool(np.array_equal(unique_qrs_ids, np.arange(1, len(unique_qrs_ids) + 1))),
            "one_qrs_occurrence_per_beat": bool(one_qrs_per_beat),
            "signal_label_lengths_match_native": bool(len(native_label) == min(native_lead_lengths) == max(native_lead_lengths)),
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "zhejiang_record_contracts.csv"), index=False)
    boolean_columns = [
        "beat_ids_monotonic",
        "qrs_ids_contiguous",
        "one_qrs_occurrence_per_beat",
        "signal_label_lengths_match_native",
    ]
    summary = {
        "n_records": len(table),
        "native_lengths": sorted(int(x) for x in table["native_label_length"].unique()),
        "corrected_lengths": sorted(int(x) for x in table["corrected_length"].unique()),
        "contract_failures": {column: int((~table[column]).sum()) for column in boolean_columns},
        "median_n_qrs": float(table["n_qrs"].median()),
        "median_rr_ms": float(table["median_rr_ms"].median()),
        "median_qrs_duration_ms": float(table["median_qrs_duration_ms"].median()),
        "median_unlabeled_fraction": float(table["unlabeled_fraction"].median()),
        "median_active_signal_duration_seconds": float(table["active_signal_duration_seconds"].median()),
        "median_annotation_coverage_duration_seconds": float(table["annotation_coverage_duration_seconds"].median()),
        "median_tail_padding_seconds": float(table["tail_padding_seconds"].median()),
        "source_rate_contract": "native 2000 Hz mapped 4:1 to 500 Hz",
        "unit_contract": "undocumented; division by 1000 remains provisional",
    }
    with open(os.path.join(OUT_DIR, "zhejiang_contract_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    fig, axes = plt.subplots(len(WORST_RECORDS), 1, figsize=(14, 10), constrained_layout=True)
    colors = {1: "#377eb8", 2: "#e41a1c", 3: "#4daf4a"}
    for axis, record_id in zip(axes, WORST_RECORDS):
        record = adapter.load_record(record_id, n_samples=5000)
        signal = record["ecg"][:, 1]
        mask = record["segmentation"]
        time = record["time"]
        axis.plot(time, signal, color="black", linewidth=0.7)
        for value, color in colors.items():
            starts, ends = intervals(mask, value)
            for start, end in zip(starts, ends):
                axis.axvspan(time[start], time[end], color=color, alpha=0.16)
        axis.set_title(record_id)
        axis.set_ylabel("Lead II (provisional mV)")
    axes[-1].set_xlabel("Time (s), corrected 500 Hz coordinate")
    fig.savefig(os.path.join(OUT_DIR, "worst_record_annotation_overlay.png"), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
