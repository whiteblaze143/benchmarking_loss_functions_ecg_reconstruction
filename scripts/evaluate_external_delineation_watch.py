#!/usr/bin/env python3
"""Generation-bound, low-resource external delineation evaluation watcher.

The watcher discovers release-compatible factorial checkpoints through the
checkpoint SQLite catalog and compatibility audit.  It materializes one exact
checkpoint generation at a time, evaluates it on LUDB, ISP, and Zhejiang using
CPU-only inference, writes per-record artifacts atomically, evicts its private
checkpoint cache, and sleeps when the current compatible cohort is complete.

LUDB has a documented 500 Hz/mV contract.  The local ISP and Zhejiang copies
do not include authoritative voltage provenance; their numeric ranges are
microvolt-like.  Those datasets therefore use an explicit provisional
``stored_value / 1000 -> mV`` adapter and remain exploratory until provenance
is supplied.  The assumption is embedded in every output identity.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import math
import os
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import neurokit2 as nk
import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import resample_poly


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.checkpoint_store import (  # noqa: E402
    DEFAULT_DB,
    connect,
    load_checkpoint_with_identity,
    prune_cache,
    store_lock,
)
from scripts.train_mcma_3lead import MCMAModel  # noqa: E402


LEADS = ("i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6")
DISPLAY_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
OBSERVED_LEADS = (0, 1, 7)
MISSING_LEADS = (2, 3, 4, 5, 6, 8, 9, 10, 11)
BOUNDARIES = ("P_Onset", "P_Offset", "R_Onset", "R_Offset", "T_Onset", "T_Offset")
WAVES = ("P", "QRS", "T")
TARGET_FS = 500
TARGET_SAMPLES = 5000
TOLERANCE_MS = 150.0


@dataclasses.dataclass
class ExternalRecord:
    dataset: str
    split: str
    record_id: str
    signal_mv: np.ndarray
    bounds_by_lead: dict[str, dict[str, np.ndarray]]
    masks_by_lead: dict[str, dict[str, np.ndarray]]


@dataclasses.dataclass(frozen=True)
class DatasetContract:
    dataset: str
    evidence_level: str
    source_root: str
    source_tree_sha256: str
    source_files: int
    source_bytes: int
    source_sampling_rate: str
    source_unit: str
    preprocessing: str
    annotation_semantics: str


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def tree_digest(root: Path, patterns: Iterable[str]) -> tuple[int, int, str]:
    files = sorted(
        {path for pattern in patterns for path in root.rglob(pattern) if path.is_file()},
        key=lambda path: str(path.relative_to(root)),
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(sha256_file(path).encode())
        digest.update(b"\n")
    return len(files), total_bytes, digest.hexdigest()


def require_memory(minimum_gib: float) -> None:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.strip().split()[0]) * 1024
    available = values["MemAvailable"]
    if available < minimum_gib * 1024**3:
        raise RuntimeError(
            f"memory safety gate: {available / 1024**3:.2f} GiB available "
            f"< {minimum_gib:.2f} GiB required"
        )


def wait_for_resource_gate(max_load: float, minimum_memory_gib: float, poll_seconds: int) -> None:
    while True:
        load = os.getloadavg()[0]
        try:
            require_memory(minimum_memory_gib)
            memory_ready = True
        except RuntimeError:
            memory_ready = False
        if load <= max_load and memory_ready:
            return
        print(
            json.dumps(
                {
                    "event": "resource_gate_wait",
                    "at": utc_now(),
                    "load_1m": load,
                    "max_load_1m": max_load,
                    "memory_ready": memory_ready,
                }
            ),
            flush=True,
        )
        time.sleep(poll_seconds)


def fixed_length(signal: np.ndarray, samples: int = TARGET_SAMPLES) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] != 12:
        raise ValueError(f"expected [12,time] signal, got {signal.shape}")
    if signal.shape[1] > samples:
        signal = signal[:, :samples]
    elif signal.shape[1] < samples:
        signal = np.pad(signal, ((0, 0), (0, samples - signal.shape[1])))
    if signal.shape != (12, samples) or not np.isfinite(signal).all():
        raise ValueError("external signal violates finite 12x5000 contract")
    return signal


def resample_exact(signal: np.ndarray, up: int, down: int) -> np.ndarray:
    """Polyphase resample, then correct odd source lengths on normalized time."""
    result = resample_poly(signal, up=up, down=down, axis=1)
    if result.shape[1] == TARGET_SAMPLES:
        return result
    source_positions = np.linspace(0.0, 1.0, result.shape[1])
    target_positions = np.linspace(0.0, 1.0, TARGET_SAMPLES)
    return np.stack(
        [np.interp(target_positions, source_positions, lead) for lead in result],
        axis=0,
    )


def scale_indices(values: Iterable[int], source_samples: int) -> np.ndarray:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return np.array([], dtype=np.int64)
    scaled = np.rint(values * TARGET_SAMPLES / source_samples).astype(np.int64)
    return np.unique(np.clip(scaled, 0, TARGET_SAMPLES - 1))


def empty_bounds() -> dict[str, np.ndarray]:
    return {name: np.array([], dtype=np.int64) for name in BOUNDARIES}


def masks_from_bounds(bounds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {wave: np.zeros(TARGET_SAMPLES, dtype=bool) for wave in WAVES}
    pairs = {
        "P": ("P_Onset", "P_Offset"),
        "QRS": ("R_Onset", "R_Offset"),
        "T": ("T_Onset", "T_Offset"),
    }
    for wave, (onset_name, offset_name) in pairs.items():
        onsets = np.asarray(bounds[onset_name], dtype=np.int64)
        offsets = np.asarray(bounds[offset_name], dtype=np.int64)
        for onset, offset in zip(onsets, offsets):
            if 0 <= onset <= offset < TARGET_SAMPLES:
                result[wave][onset : offset + 1] = True
    return result


def parse_ludb_bounds(record_path: Path, lead: str) -> dict[str, np.ndarray]:
    annotation = wfdb.rdann(str(record_path), extension=lead)
    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample, dtype=np.int64)
    values: dict[str, list[int]] = {name: [] for name in BOUNDARIES}
    mapping = {
        "p": ("P_Onset", "P_Offset"),
        "N": ("R_Onset", "R_Offset"),
        "t": ("T_Onset", "T_Offset"),
    }
    for index in range(1, len(symbols) - 1):
        if symbols[index] not in mapping:
            continue
        if symbols[index - 1] != "(" or symbols[index + 1] != ")":
            continue
        onset_name, offset_name = mapping[symbols[index]]
        values[onset_name].append(int(samples[index - 1]))
        values[offset_name].append(int(samples[index + 1]))
    return {name: np.asarray(items, dtype=np.int64) for name, items in values.items()}


def load_ludb(root: Path, max_records: int) -> tuple[list[ExternalRecord], DatasetContract]:
    source_files, source_bytes, source_digest = tree_digest(
        root,
        (
            "*.hea", "*.dat", "*.i", "*.ii", "*.iii", "*.avr", "*.avl",
            "*.avf", "*.v1", "*.v2", "*.v3", "*.v4", "*.v5", "*.v6",
            "README", "LICENSE.txt",
        ),
    )
    record_ids = sorted({path.stem for path in root.glob("*.dat")}, key=int)
    if max_records:
        record_ids = record_ids[:max_records]
    records: list[ExternalRecord] = []
    for record_id in record_ids:
        path = root / record_id
        record = wfdb.rdrecord(str(path), physical=True)
        if record.fs != TARGET_FS or record.p_signal.shape != (TARGET_SAMPLES, 12):
            raise ValueError(f"LUDB {record_id} violates 500 Hz 12x5000 contract")
        channel_lookup = {name.lower(): index for index, name in enumerate(record.sig_name)}
        if set(channel_lookup) != set(LEADS) or set(record.units) != {"mV"}:
            raise ValueError(f"LUDB {record_id} lead/unit contract mismatch")
        signal = np.stack(
            [record.p_signal[:, channel_lookup[lead]] for lead in LEADS], axis=0
        ).astype(np.float32)
        bounds_by_lead = {lead: parse_ludb_bounds(path, lead) for lead in LEADS}
        records.append(
            ExternalRecord(
                dataset="ludb",
                split="all",
                record_id=record_id,
                signal_mv=fixed_length(signal),
                bounds_by_lead=bounds_by_lead,
                masks_by_lead={lead: masks_from_bounds(bounds) for lead, bounds in bounds_by_lead.items()},
            )
        )
    return records, DatasetContract(
        dataset="ludb",
        evidence_level="documented_external_delineation",
        source_root=str(root.resolve()),
        source_tree_sha256=source_digest,
        source_files=source_files,
        source_bytes=source_bytes,
        source_sampling_rate="500 Hz documented in WFDB headers/source",
        source_unit="mV",
        preprocessing="lead reorder only; no resampling or amplitude scaling",
        annotation_semantics="lead-specific LUDB p/N/t onset and offset annotations",
    )


def bounds_from_intervals(intervals: list[tuple[int, int, int]], source_samples: int) -> dict[str, np.ndarray]:
    values: dict[str, list[int]] = {name: [] for name in BOUNDARIES}
    mapping = {
        0: ("P_Onset", "P_Offset"),
        1: ("R_Onset", "R_Offset"),
        2: ("T_Onset", "T_Offset"),
    }
    for class_id, onset, offset in intervals:
        if class_id not in mapping:
            raise ValueError(f"unexpected ISP class {class_id}")
        onset_name, offset_name = mapping[class_id]
        values[onset_name].append(onset)
        values[offset_name].append(offset)
    return {
        name: scale_indices(items, source_samples) for name, items in values.items()
    }


def load_isp(root: Path, max_records: int) -> tuple[list[ExternalRecord], DatasetContract]:
    source_files, source_bytes, source_digest = tree_digest(root, ("*.hea", "*.dat", "*.csv"))
    records: list[ExternalRecord] = []
    for split in ("train", "test"):
        frame = pd.read_csv(root / f"{split}_isp_delineation_data.csv")
        if max_records:
            frame = frame.head(max_records)
        for row in frame.itertuples(index=False):
            record_id = str(row.file_name)
            record = wfdb.rdrecord(str(root / f"{split}_data" / record_id), physical=True)
            if record.fs != 1000 or record.p_signal.shape[1] != 12:
                raise ValueError(f"ISP {split}/{record_id} violates header contract")
            channel_lookup = {name.lower(): index for index, name in enumerate(record.sig_name)}
            if set(channel_lookup) != set(LEADS) or set(record.units) != {"mkv"}:
                raise ValueError(f"ISP {split}/{record_id} lead/unit contract mismatch")
            raw = np.stack(
                [record.p_signal[:, channel_lookup[lead]] for lead in LEADS], axis=0
            )
            # Explicit provisional interpretation; retained in DatasetContract.
            signal_mv = resample_exact(raw / 1000.0, up=1, down=2)
            bounds = bounds_from_intervals(ast.literal_eval(row.target), record.p_signal.shape[0])
            bounds_by_lead = {lead: bounds for lead in LEADS}
            masks = masks_from_bounds(bounds)
            records.append(
                ExternalRecord(
                    dataset="isp",
                    split=split,
                    record_id=record_id,
                    signal_mv=fixed_length(signal_mv),
                    bounds_by_lead=bounds_by_lead,
                    masks_by_lead={lead: masks for lead in LEADS},
                )
            )
    return records, DatasetContract(
        dataset="isp",
        evidence_level="exploratory_unresolved_provenance",
        source_root=str(root.resolve()),
        source_tree_sha256=source_digest,
        source_files=source_files,
        source_bytes=source_bytes,
        source_sampling_rate="1000 Hz declared by local WFDB headers",
        source_unit="nonstandard 'mkv'; provisionally interpreted as microvolt-like",
        preprocessing="polyphase 1000->500 Hz; stored values divided by 1000 to provisional mV",
        annotation_semantics="provisional local convention class 0=P, 1=QRS, 2=T; source unresolved",
    )


def bounds_from_mask(mask: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    mask = np.asarray(mask)
    if mask.ndim != 1 or not set(np.unique(mask)).issubset({0, 1, 2, 3}):
        raise ValueError("Zhejiang mask violates {0,1,2,3} one-dimensional contract")
    if len(mask) == 20000:
        mask_5k = mask[::4]
        mask = np.pad(mask_5k, (0, TARGET_SAMPLES - len(mask_5k)), mode="constant", constant_values=0)
    elif len(mask) != TARGET_SAMPLES:
        positions = np.rint(np.linspace(0, len(mask) - 1, TARGET_SAMPLES)).astype(int)
        mask = mask[positions]
    bounds = empty_bounds()
    masks: dict[str, np.ndarray] = {}
    definitions = {
        "P": (1, "P_Onset", "P_Offset"),
        "QRS": (2, "R_Onset", "R_Offset"),
        "T": (3, "T_Onset", "T_Offset"),
    }
    for wave, (value, onset_name, offset_name) in definitions.items():
        binary = mask == value
        difference = np.diff(binary.astype(np.int8))
        onsets = np.where(difference == 1)[0] + 1
        offsets = np.where(difference == -1)[0]
        if binary[0]:
            onsets = np.insert(onsets, 0, 0)
        if binary[-1]:
            offsets = np.append(offsets, len(binary) - 1)
        bounds[onset_name] = onsets.astype(np.int64)
        bounds[offset_name] = offsets.astype(np.int64)
        masks[wave] = binary
    return bounds, masks


def load_zhejiang(root: Path, max_records: int) -> tuple[list[ExternalRecord], DatasetContract]:
    source_files, source_bytes, source_digest = tree_digest(root, ("*.pkl",))
    lead_names = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
    label_paths = sorted((root / "label").glob("*.pkl"), key=lambda path: path.stem)
    if max_records:
        label_paths = label_paths[:max_records]
    records: list[ExternalRecord] = []
    for label_path in label_paths:
        record_id = label_path.stem
        signals = []
        for lead in lead_names:
            with (root / "ecg" / f"{record_id}_{lead}.pkl").open("rb") as handle:
                signal = np.asarray(pickle.load(handle), dtype=np.float64)
            if signal.shape != (20000,) or not np.isfinite(signal).all():
                raise ValueError(f"Zhejiang {record_id}/{lead} violates signal contract")
            signals.append(signal)
        with label_path.open("rb") as handle:
            mask = np.asarray(pickle.load(handle))
        bounds, masks = bounds_from_mask(mask)
        # Native 4:1 index conversion is verified locally; 2000 Hz and voltage
        # interpretations remain explicitly provisional.
        signal_mv = resample_exact(np.stack(signals) / 1000.0, up=1, down=4)
        bounds_by_lead = {lead: bounds for lead in LEADS}
        records.append(
            ExternalRecord(
                dataset="zhejiang",
                split="all",
                record_id=record_id,
                signal_mv=fixed_length(signal_mv),
                bounds_by_lead=bounds_by_lead,
                masks_by_lead={lead: masks for lead in LEADS},
            )
        )
    return records, DatasetContract(
        dataset="zhejiang",
        evidence_level="exploratory_unresolved_provenance",
        source_root=str(root.resolve()),
        source_tree_sha256=source_digest,
        source_files=source_files,
        source_bytes=source_bytes,
        source_sampling_rate="native 20000 samples mapped 4:1 to model length; 2000 Hz source claim unresolved",
        source_unit="undocumented; provisionally interpreted as microvolt-like",
        preprocessing="polyphase native 20000->5000 samples; stored values divided by 1000 to provisional mV",
        annotation_semantics="provisional local convention 0=background, 1=P, 2=QRS, 3=T; source unresolved",
    )


def finite_sorted(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return np.unique(array[np.isfinite(array)].astype(np.int64))


def monotonic_match(reference: Iterable[int], predicted: Iterable[int], tolerance_samples: int) -> list[tuple[int, int]]:
    """Maximize one-to-one matches, then minimize total absolute timing error."""
    left = finite_sorted(reference)
    right = finite_sorted(predicted)
    # Each cell stores (count, negative_error, pairs); lexicographic max gives
    # maximum cardinality and then minimum total absolute error.
    dynamic: list[list[tuple[int, int, tuple[tuple[int, int], ...]]]] = [
        [(0, 0, ()) for _ in range(len(right) + 1)] for _ in range(len(left) + 1)
    ]
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            candidates = [dynamic[i - 1][j], dynamic[i][j - 1]]
            error = abs(int(left[i - 1]) - int(right[j - 1]))
            if error <= tolerance_samples:
                count, negative_error, pairs = dynamic[i - 1][j - 1]
                candidates.append((count + 1, negative_error - error, pairs + ((i - 1, j - 1),)))
            dynamic[i][j] = max(candidates, key=lambda item: (item[0], item[1]))
    return [(int(left[i]), int(right[j])) for i, j in dynamic[-1][-1][2]]


def delineate(signal: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
    try:
        cleaned = nk.ecg_clean(np.asarray(signal, dtype=float), sampling_rate=TARGET_FS)
        _, peak_info = nk.ecg_peaks(cleaned, sampling_rate=TARGET_FS)
        peaks = np.asarray(peak_info.get("ECG_R_Peaks", []), dtype=np.int64)
        if len(peaks) < 2:
            return empty_bounds(), "insufficient_r_peaks"
        _, waves = nk.ecg_delineate(
            cleaned, peaks, sampling_rate=TARGET_FS, method="dwt"
        )
        mapping = {
            "P_Onset": "ECG_P_Onsets",
            "P_Offset": "ECG_P_Offsets",
            "R_Onset": "ECG_R_Onsets",
            "R_Offset": "ECG_R_Offsets",
            "T_Onset": "ECG_T_Onsets",
            "T_Offset": "ECG_T_Offsets",
        }
        return {
            name: finite_sorted(waves.get(source, []))
            for name, source in mapping.items()
        }, "ok"
    except Exception as error:  # detector failures are data, not watcher crashes
        return empty_bounds(), f"detector_error:{type(error).__name__}"


def boundary_row(
    record: ExternalRecord,
    model_id: str,
    checkpoint_sha256: str,
    lead_index: int,
    boundary: str,
    reference: np.ndarray,
    predicted: np.ndarray,
    detector_status: str,
    tolerance_samples: int,
) -> dict[str, Any]:
    matches = monotonic_match(reference, predicted, tolerance_samples)
    errors = np.asarray([abs(real - estimate) for real, estimate in matches], dtype=float)
    true_positive = len(matches)
    false_negative = len(finite_sorted(reference)) - true_positive
    false_positive = len(finite_sorted(predicted)) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "dataset": record.dataset,
        "split": record.split,
        "record_id": record.record_id,
        "lead_index": lead_index,
        "lead": DISPLAY_LEADS[lead_index],
        "lead_role": "observed" if lead_index in OBSERVED_LEADS else "missing",
        "boundary": boundary,
        "detector_status": detector_status,
        "reference_events": len(finite_sorted(reference)),
        "predicted_events": len(finite_sorted(predicted)),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": (2 * true_positive / denominator) if denominator else float("nan"),
        "timing_mae_samples": float(errors.mean()) if len(errors) else float("nan"),
        "timing_median_samples": float(np.median(errors)) if len(errors) else float("nan"),
        "timing_p95_samples": float(np.quantile(errors, 0.95)) if len(errors) else float("nan"),
        "timing_mae_ms_header_derived": float(errors.mean() * 2.0) if len(errors) else float("nan"),
        "tolerance_samples": tolerance_samples,
        "tolerance_ms_header_derived": tolerance_samples * 2.0,
    }


def dice(reference: np.ndarray, predicted: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    denominator = int(reference.sum() + predicted.sum())
    return 2.0 * int((reference & predicted).sum()) / denominator if denominator else 1.0


def model_reconstruct(model: MCMAModel, signals: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            batch = torch.from_numpy(signals[start : start + batch_size]).float()
            padded = torch.nn.functional.pad(batch, (0, 120))
            output = model(padded[:, OBSERVED_LEADS, :])[..., :TARGET_SAMPLES]
            if output.shape != batch.shape or not torch.isfinite(output).all():
                raise RuntimeError(f"model emitted invalid external reconstruction {tuple(output.shape)}")
            for reconstruction in output.cpu().numpy():
                yield reconstruction


def evaluate_dataset(
    records: list[ExternalRecord],
    model: MCMAModel | None,
    model_id: str,
    checkpoint_sha256: str,
    lead_indices: tuple[int, ...],
    batch_size: int,
    tolerance_samples: int,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signals = np.stack([record.signal_mv for record in records])
    reconstructions: Iterable[np.ndarray]
    if model is None:
        reconstructions = iter(signals)
    else:
        reconstructions = model_reconstruct(model, signals, batch_size)
    reconstruction_list = list(reconstructions)
    tasks = [
        reconstruction[lead_index]
        for reconstruction in reconstruction_list
        for lead_index in lead_indices
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        delineations = list(executor.map(delineate, tasks, chunksize=1))
    delineation_index = 0
    event_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    wave_pairs = {
        "P": ("P_Onset", "P_Offset"),
        "QRS": ("R_Onset", "R_Offset"),
        "T": ("T_Onset", "T_Offset"),
    }
    for record, reconstruction in zip(records, reconstruction_list, strict=True):
        for lead_index in lead_indices:
            lead = LEADS[lead_index]
            predicted_bounds, detector_status = delineations[delineation_index]
            delineation_index += 1
            reference_bounds = record.bounds_by_lead[lead]
            for boundary in BOUNDARIES:
                event_rows.append(
                    boundary_row(
                        record,
                        model_id,
                        checkpoint_sha256,
                        lead_index,
                        boundary,
                        reference_bounds[boundary],
                        predicted_bounds[boundary],
                        detector_status,
                        tolerance_samples,
                    )
                )
            predicted_masks = masks_from_bounds(predicted_bounds)
            for wave, _ in wave_pairs.items():
                overlap_rows.append(
                    {
                        "model_id": model_id,
                        "checkpoint_sha256": checkpoint_sha256,
                        "dataset": record.dataset,
                        "split": record.split,
                        "record_id": record.record_id,
                        "lead_index": lead_index,
                        "lead": DISPLAY_LEADS[lead_index],
                        "lead_role": "observed" if lead_index in OBSERVED_LEADS else "missing",
                        "wave": wave,
                        "detector_status": detector_status,
                        "dice": dice(record.masks_by_lead[lead][wave], predicted_masks[wave]),
                    }
                )
    return pd.DataFrame(event_rows), pd.DataFrame(overlap_rows)


def summarize(events: pd.DataFrame, overlaps: pd.DataFrame) -> pd.DataFrame:
    event_summary = (
        events.groupby(["dataset", "split", "lead_role", "boundary"], dropna=False)
        .agg(
            record_lead_rows=("record_id", "size"),
            detector_success_rate=("detector_status", lambda values: float((values == "ok").mean())),
            reference_events=("reference_events", "sum"),
            predicted_events=("predicted_events", "sum"),
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
            macro_f1=("f1", "mean"),
            median_record_f1=("f1", "median"),
            timing_mae_ms_header_derived=("timing_mae_ms_header_derived", "mean"),
            timing_p95_ms_header_derived=("timing_mae_ms_header_derived", lambda values: values.quantile(0.95)),
        )
        .reset_index()
    )
    denominator = 2 * event_summary.true_positive + event_summary.false_positive + event_summary.false_negative
    event_summary["micro_f1"] = np.where(
        denominator > 0, 2 * event_summary.true_positive / denominator, np.nan
    )
    event_summary["metric_family"] = "boundary"
    overlap_summary = (
        overlaps.groupby(["dataset", "split", "lead_role", "wave"], dropna=False)
        .agg(
            record_lead_rows=("record_id", "size"),
            detector_success_rate=("detector_status", lambda values: float((values == "ok").mean())),
            macro_dice=("dice", "mean"),
            median_record_dice=("dice", "median"),
        )
        .reset_index()
    )
    overlap_summary["metric_family"] = "wave_overlap"
    return pd.concat([event_summary, overlap_summary], ignore_index=True, sort=False)


def eligible_models(audit_path: Path, db_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit = json.loads(audit_path.read_text())
    compatible = {
        row["model_id"]: row for row in audit["models"] if row.get("compatible") is True
    }
    connection = connect(db_path)
    try:
        rows = connection.execute(
            """
            SELECT model_id, factorial_mask, seed, sha256, size_bytes, status
            FROM checkpoints
            WHERE status IN ('local', 'remote_verified', 'cached')
            """
        ).fetchall()
    finally:
        connection.close()
    result = []
    for row in rows:
        identity = compatible.get(row["model_id"])
        if identity is None or identity["checkpoint_sha256"] != row["sha256"]:
            continue
        result.append({**dict(row), **identity})
    return audit, sorted(result, key=lambda row: (int(row["seed"]), row["factorial_mask"]))


def completed_for_generation(metadata_path: Path, identity: dict[str, Any], evaluation_sha256: str, dataset_contracts: dict[str, DatasetContract]) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    return (
        metadata.get("status") == "complete"
        and metadata.get("checkpoint_sha256") == identity["checkpoint_sha256"]
        and metadata.get("evaluation_code_sha256") == evaluation_sha256
        and metadata.get("dataset_contract_sha256")
        == sha256_text(json.dumps(json_safe({key: dataclasses.asdict(value) for key, value in dataset_contracts.items()}), sort_keys=True))
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, default=DEFAULT_DB)
    result.add_argument(
        "--compatibility-audit",
        type=Path,
        default=ROOT / "results/checkpoint_store/compatibility_audit.json",
    )
    result.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/factorial_mixed_level/external_delineation_generation_bound",
    )
    result.add_argument(
        "--cache-dir", type=Path, default=ROOT / "checkpoints/external_delineation_cache"
    )
    result.add_argument("--poll-seconds", type=int, default=1200)
    result.add_argument("--resource-poll-seconds", type=int, default=60)
    result.add_argument("--max-load-1m", type=float, default=6.5)
    result.add_argument("--min-available-memory-gib", type=float, default=3.0)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Delineation workers; production launch is affinity-limited to two CPU cores",
    )
    result.add_argument("--max-attempts", type=int, default=2)
    result.add_argument("--model-id")
    result.add_argument("--once", action="store_true")
    result.add_argument("--max-records-per-dataset", type=int, default=0)
    result.add_argument(
        "--lead-indices",
        default=",".join(map(str, MISSING_LEADS)),
        help="Comma-separated lead indices; defaults to the nine missing leads",
    )
    result.add_argument("--skip-ceiling", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if (
        args.poll_seconds < 1
        or args.batch_size < 1
        or args.max_attempts < 1
        or args.workers not in (1, 2)
    ):
        raise ValueError(
            "poll seconds, batch size, and max attempts must be positive; "
            "workers must be 1 or 2"
        )
    lead_indices = tuple(int(value) for value in args.lead_indices.split(","))
    if not lead_indices or set(lead_indices) - set(range(12)):
        raise ValueError("lead indices must be a nonempty subset of 0..11")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("external delineation watcher requires CUDA_VISIBLE_DEVICES='' or '-1'")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_sha256 = sha256_file(Path(__file__))
    print(json.dumps({"event": "dataset_load_started", "at": utc_now()}), flush=True)
    datasets: dict[str, list[ExternalRecord]] = {}
    contracts: dict[str, DatasetContract] = {}
    loaders = {
        "ludb": (load_ludb, ROOT / "data/ludb"),
        "isp": (load_isp, ROOT / "data/isp_delineation_dataset"),
        "zhejiang": (load_zhejiang, ROOT / "data/zhejiang"),
    }
    for name, (loader, root) in loaders.items():
        wait_for_resource_gate(args.max_load_1m, args.min_available_memory_gib, args.resource_poll_seconds)
        records, contract = loader(root, args.max_records_per_dataset)
        if not records:
            raise RuntimeError(f"{name} loader returned no records")
        datasets[name] = records
        contracts[name] = contract
        print(json.dumps({"event": "dataset_loaded", "dataset": name, "records": len(records), "at": utc_now()}), flush=True)
    contract_payload = {key: dataclasses.asdict(value) for key, value in contracts.items()}
    contract_sha256 = sha256_text(json.dumps(contract_payload, sort_keys=True))
    atomic_json(
        args.output_dir / "dataset_contracts.json",
        {
            "schema_version": 1,
            "generated_at": utc_now(),
            "dataset_contract_sha256": contract_sha256,
            "contracts": contract_payload,
            "lead_indices": list(lead_indices),
            "lead_names": [DISPLAY_LEADS[index] for index in lead_indices],
            "tolerance_ms": TOLERANCE_MS,
        },
    )

    if not args.skip_ceiling:
        for name, records in datasets.items():
            ceiling_dir = args.output_dir / "ground_truth_ceiling" / name
            ceiling_meta = ceiling_dir / "metadata.json"
            ceiling_identity = {
                "checkpoint_sha256": "ground_truth",
            }
            single_contract = {name: contracts[name]}
            if completed_for_generation(ceiling_meta, ceiling_identity, evaluation_sha256, single_contract):
                continue
            wait_for_resource_gate(args.max_load_1m, args.min_available_memory_gib, args.resource_poll_seconds)
            started = time.perf_counter()
            events, overlaps = evaluate_dataset(
                records,
                None,
                "ground_truth_ceiling",
                "ground_truth",
                lead_indices,
                args.batch_size,
                int(round(TOLERANCE_MS * TARGET_FS / 1000)),
                args.workers,
            )
            summary = summarize(events, overlaps)
            atomic_text(ceiling_dir / "per_record_boundary.csv", events.to_csv(index=False))
            atomic_text(ceiling_dir / "per_record_wave_overlap.csv", overlaps.to_csv(index=False))
            atomic_text(ceiling_dir / "summary.csv", summary.to_csv(index=False))
            single_contract_sha = sha256_text(
                json.dumps({name: dataclasses.asdict(contracts[name])}, sort_keys=True)
            )
            atomic_json(
                ceiling_meta,
                {
                    "schema_version": 1,
                    "status": "complete",
                    "completed_at": utc_now(),
                    "model_id": "ground_truth_ceiling",
                    "checkpoint_sha256": "ground_truth",
                    "evaluation_code_sha256": evaluation_sha256,
                    "dataset_contract_sha256": single_contract_sha,
                    "dataset": name,
                    "records": len(records),
                    "duration_seconds": time.perf_counter() - started,
                    "event_rows": len(events),
                    "overlap_rows": len(overlaps),
                },
            )

    while True:
        audit, eligible = eligible_models(args.compatibility_audit, args.db)
        if args.model_id:
            eligible = [row for row in eligible if row["model_id"] == args.model_id]
            if not eligible:
                raise ValueError(f"requested model is not DB-backed and compatible: {args.model_id}")
        complete = 0
        exhausted_failures = 0
        errors_this_pass = 0
        for position, identity in enumerate(eligible, start=1):
            model_id = identity["model_id"]
            model_dir = args.output_dir / model_id
            metadata_path = model_dir / "metadata.json"
            if completed_for_generation(metadata_path, identity, evaluation_sha256, contracts):
                complete += 1
                continue
            attempts = 0
            try:
                previous = json.loads(metadata_path.read_text())
                if (
                    previous.get("checkpoint_sha256") == identity["checkpoint_sha256"]
                    and previous.get("evaluation_code_sha256") == evaluation_sha256
                ):
                    attempts = int(previous.get("attempts", 0))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
            if attempts >= args.max_attempts:
                exhausted_failures += 1
                continue
            attempts += 1
            wait_for_resource_gate(args.max_load_1m, args.min_available_memory_gib, args.resource_poll_seconds)
            atomic_json(
                args.output_dir / "watch_state.json",
                {
                    "schema_version": 1,
                    "status": "evaluating",
                    "updated_at": utc_now(),
                    "eligible_models": len(eligible),
                    "completed_models_before_pass": complete,
                    "current_model": model_id,
                    "current_position": position,
                    "poll_seconds": args.poll_seconds,
                },
            )
            started_at = utc_now()
            started_clock = time.perf_counter()
            try:
                payload, materialized = load_checkpoint_with_identity(
                    model_id,
                    db_path=args.db,
                    cache_dir=args.cache_dir,
                    map_location="cpu",
                    weights_only=False,
                )
                if materialized["sha256"] != identity["checkpoint_sha256"]:
                    raise RuntimeError(f"{model_id} DB/audit checkpoint digest mismatch")
                model = MCMAModel(in_channels=3, out_channels=12)
                model.load_state_dict(payload.get("model_state_dict", payload), strict=True)
                model.eval()
                dataset_metadata: dict[str, Any] = {}
                for dataset_name, records in datasets.items():
                    wait_for_resource_gate(args.max_load_1m, args.min_available_memory_gib, args.resource_poll_seconds)
                    dataset_started = time.perf_counter()
                    events, overlaps = evaluate_dataset(
                        records,
                        model,
                        model_id,
                        identity["checkpoint_sha256"],
                        lead_indices,
                        args.batch_size,
                        int(round(TOLERANCE_MS * TARGET_FS / 1000)),
                        args.workers,
                    )
                    summary = summarize(events, overlaps)
                    dataset_dir = model_dir / dataset_name
                    atomic_text(dataset_dir / "per_record_boundary.csv", events.to_csv(index=False))
                    atomic_text(dataset_dir / "per_record_wave_overlap.csv", overlaps.to_csv(index=False))
                    atomic_text(dataset_dir / "summary.csv", summary.to_csv(index=False))
                    dataset_metadata[dataset_name] = {
                        "status": "complete",
                        "records": len(records),
                        "event_rows": len(events),
                        "overlap_rows": len(overlaps),
                        "duration_seconds": time.perf_counter() - dataset_started,
                        "per_record_boundary_sha256": sha256_file(dataset_dir / "per_record_boundary.csv"),
                        "per_record_wave_overlap_sha256": sha256_file(dataset_dir / "per_record_wave_overlap.csv"),
                        "summary_sha256": sha256_file(dataset_dir / "summary.csv"),
                    }
                    atomic_json(
                        metadata_path,
                        json_safe(
                            {
                                "schema_version": 1,
                                "status": "partial",
                                "started_at": started_at,
                                "updated_at": utc_now(),
                                "attempts": attempts,
                                "model_id": model_id,
                                "factorial_mask": identity["factorial_mask"],
                                "seed": int(identity["seed"]),
                                "checkpoint_sha256": identity["checkpoint_sha256"],
                                "checkpoint_size_bytes": int(identity["checkpoint_size_bytes"]),
                                "source_bundle_sha256": identity["source_bundle_sha256"],
                                "training_contract_id": audit["contract"]["contract_id"],
                                "evaluation_code_sha256": evaluation_sha256,
                                "dataset_contract_sha256": contract_sha256,
                                "datasets": dataset_metadata,
                            }
                        ),
                    )
                del model, payload
                gc.collect()
                atomic_json(
                    metadata_path,
                    json_safe(
                        {
                            "schema_version": 1,
                            "status": "complete",
                            "started_at": started_at,
                            "completed_at": utc_now(),
                            "duration_seconds": time.perf_counter() - started_clock,
                            "attempts": attempts,
                            "model_id": model_id,
                            "factorial_mask": identity["factorial_mask"],
                            "seed": int(identity["seed"]),
                            "checkpoint_sha256": identity["checkpoint_sha256"],
                            "checkpoint_size_bytes": int(identity["checkpoint_size_bytes"]),
                            "source_bundle_sha256": identity["source_bundle_sha256"],
                            "training_contract_id": audit["contract"]["contract_id"],
                            "evaluation_code_sha256": evaluation_sha256,
                            "dataset_contract_sha256": contract_sha256,
                            "lead_indices": list(lead_indices),
                            "datasets": dataset_metadata,
                        }
                    ),
                )
                complete += 1
            except Exception as error:
                errors_this_pass += 1
                atomic_json(
                    metadata_path,
                    {
                        "schema_version": 1,
                        "status": "error",
                        "updated_at": utc_now(),
                        "attempts": attempts,
                        "model_id": model_id,
                        "checkpoint_sha256": identity["checkpoint_sha256"],
                        "evaluation_code_sha256": evaluation_sha256,
                        "dataset_contract_sha256": contract_sha256,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                print(json.dumps({"event": "model_error", "model_id": model_id, "error": repr(error)}), flush=True)
            finally:
                with store_lock(args.db):
                    connection = connect(args.db)
                    try:
                        prune_cache(connection, args.cache_dir, 0.0)
                    finally:
                        connection.close()
        caught_up = complete + exhausted_failures == len(eligible)
        state = {
            "schema_version": 1,
            "status": "caught_up" if caught_up else "backlogged",
            "updated_at": utc_now(),
            "eligible_models": len(eligible),
            "completed_models": complete,
            "failed_models_at_retry_limit": exhausted_failures,
            "errors_this_pass": errors_this_pass,
            "pending_models": len(eligible) - complete - exhausted_failures,
            "poll_seconds": args.poll_seconds,
            "next_poll_not_before": (
                dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=args.poll_seconds)
            ).isoformat(),
        }
        atomic_json(args.output_dir / "watch_state.json", state)
        print(json.dumps({"event": "pass_complete", **state}), flush=True)
        if args.once or args.model_id:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("external delineation watcher stopped cleanly", file=sys.stderr)
