#!/usr/bin/env python3
"""Author-faithful SemiSegECG ViT-Tiny Mean Teacher evaluation on LUDB.

The official notebook-parity path and the deployable no-label-padding path are
kept as distinct, content-bound protocols.  Both use the released model,
official subject splits, vendor preprocessing, and lead-specific LUDB labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "external/semiseg/semi-seg-ecg"
VENDOR_SRC = VENDOR / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

import models.backbones as vendor_backbones  # noqa: E402
import models.decode_heads as vendor_heads  # noqa: E402
from models.encoder_decoder import EncoderDecoder  # noqa: E402
import utils.transforms as vendor_transforms  # noqa: E402


CHECKPOINT = VENDOR / "vit_tiny-mean_teacher.pth"
INDEX_ROOT = ROOT / "external/semiseg/official_indices/ludb"
LUDB_ROOT = ROOT / "data/ludb"
OUTPUT_DB = ROOT / "results/semiseg_ludb/semiseg_ludb.sqlite"
CLASS_NAMES = ("background", "P", "QRS", "T")
BOUNDARIES = ("P_onset", "P_offset", "QRS_onset", "QRS_offset", "T_onset", "T_offset")
NATIVE_FS = 250
NATIVE_SAMPLES = 2500
SOURCE_FS = 500
SOURCE_SAMPLES = 5000


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_model(checkpoint: dict, state_name: str, device: torch.device) -> nn.Module:
    """Construct through released modules while mapping the checkpoint's old alias."""
    config = copy.deepcopy(checkpoint["config"])
    backbone_name, backbone_kwargs = next(iter(config["backbone"].items()))
    if backbone_name == "vit_seg_tiny":
        backbone_name = "vit_tiny"
    if not hasattr(vendor_backbones, backbone_name):
        raise RuntimeError(f"Unsupported vendor backbone {backbone_name!r}")
    head_name, head_kwargs = next(iter(config["decode_head"].items()))
    if not hasattr(vendor_heads, head_name):
        raise RuntimeError(f"Unsupported vendor head {head_name!r}")
    model = EncoderDecoder(
        backbone=getattr(vendor_backbones, backbone_name)(**backbone_kwargs),
        decode_head=getattr(vendor_heads, head_name)(**head_kwargs),
        decode_head_loss=nn.CrossEntropyLoss(),
        use_latent_projection=bool(config.get("use_latent_projection", False)),
        projection_in_dim=config.get("projection_in_dim"),
        projection_out_dim=config.get("projection_out_dim"),
    )
    if state_name not in {"model", "model_ema"}:
        raise ValueError("state must be model or model_ema")
    model.load_state_dict(checkpoint[state_name], strict=True)
    return model.float().to(device).eval()


def strict_triplet_mask(record_path: Path, lead: str) -> np.ndarray:
    """Reproduce LUDB.ipynb triplet parsing and its half-open segment masks."""
    annotation = wfdb.rdann(str(record_path), extension=lead.lower())
    samples, symbols = annotation.sample, annotation.symbol
    if len(samples) % 3:
        raise ValueError(f"{record_path.name}/{lead}: annotation length is not divisible by three")
    result = np.zeros(SOURCE_SAMPLES, dtype=np.int64)
    class_for_symbol = {"p": 1, "N": 2, "t": 3}
    for index in range(0, len(samples), 3):
        onset, _peak, offset = map(int, samples[index:index + 3])
        left, wave, right = symbols[index:index + 3]
        if left != "(" or right != ")" or wave not in class_for_symbol:
            raise ValueError(f"{record_path.name}/{lead}: non-triplet annotation at {index}")
        result[onset:offset] = class_for_symbol[wave]
    return result


def label_extent(mask: np.ndarray) -> tuple[int, int]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("label mask contains no waves")
    return int(indices[0]), int(indices[-1] + 1)


def evaluation_view(
    reference: np.ndarray, predicted: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU arrays and a full-clock prediction for boundary scoring.

    LUDB does not annotate partial waves outside the first-to-last labeled
    region. ``annotated`` excludes those regions from scoring without changing
    the model input, unlike the author's label-dependent signal padding.
    """
    if mode == "full":
        return reference, predicted, predicted
    if mode != "annotated":
        raise ValueError("evaluation window must be full or annotated")
    start, stop = label_extent(reference)
    bounded_prediction = predicted.copy()
    bounded_prediction[:start] = 0
    bounded_prediction[stop:] = 0
    return reference[start:stop], bounded_prediction[start:stop], bounded_prediction


class AuthorPreprocessor:
    def __init__(self) -> None:
        self.signal_resample = vendor_transforms.Resample(target_length=NATIVE_SAMPLES)
        self.label_resample = vendor_transforms.Resample(
            target_length=NATIVE_SAMPLES, method="interp", kind="zero"
        )
        self.highpass = vendor_transforms.HighpassFilter(fs=NATIVE_FS, cutoff=0.67)
        self.lowpass = vendor_transforms.LowpassFilter(fs=NATIVE_FS, cutoff=40)
        self.standardize = vendor_transforms.Standardize(axis=(-1, -2))

    def __call__(self, signal: np.ndarray, label: np.ndarray, padding: str) -> tuple[np.ndarray, np.ndarray]:
        signal = np.asarray(signal, dtype=np.float64)[None, :]
        label = np.asarray(label, dtype=np.int64)[None, :]
        if padding == "notebook":
            start, stop = label_extent(label[0])
            signal[0, :start] = signal[0, start]
            signal[0, stop:] = signal[0, stop - 1]
        elif padding != "none":
            raise ValueError("padding must be none or notebook")
        signal = self.signal_resample(signal)
        label = self.label_resample(label)
        signal = self.lowpass(self.highpass(signal))
        signal = self.standardize(signal).astype(np.float32)
        label = np.rint(label).astype(np.int64)
        if signal.shape != (1, NATIVE_SAMPLES) or label.shape != (1, NATIVE_SAMPLES):
            raise RuntimeError("author preprocessing shape drift")
        return signal, label[0]


def components(mask: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    """Return inclusive connected components for each foreground class."""
    result = {1: [], 2: [], 3: []}
    for class_id in result:
        active = mask == class_id
        padded = np.pad(active.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1
        result[class_id] = [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]
    return result


def boundary_arrays(mask: np.ndarray) -> dict[str, np.ndarray]:
    regions = components(mask)
    result: dict[str, np.ndarray] = {}
    for class_id, wave in ((1, "P"), (2, "QRS"), (3, "T")):
        result[f"{wave}_onset"] = np.asarray([a for a, _ in regions[class_id]], dtype=np.int64)
        result[f"{wave}_offset"] = np.asarray([b for _, b in regions[class_id]], dtype=np.int64)
    return result


def monotonic_match(reference: Iterable[int], predicted: Iterable[int], tolerance: int) -> list[tuple[int, int]]:
    """Maximum-cardinality ordered match, then minimum total absolute error."""
    left, right = np.asarray(list(reference), dtype=int), np.asarray(list(predicted), dtype=int)
    dynamic = [[(0, 0, ()) for _ in range(len(right) + 1)] for _ in range(len(left) + 1)]
    for i in range(1, len(left) + 1):
        for j in range(1, len(right) + 1):
            choices = [dynamic[i - 1][j], dynamic[i][j - 1]]
            error = abs(int(left[i - 1] - right[j - 1]))
            if error <= tolerance:
                count, negative_error, pairs = dynamic[i - 1][j - 1]
                choices.append((count + 1, negative_error - error, pairs + ((i - 1, j - 1),)))
            dynamic[i][j] = max(choices, key=lambda item: (item[0], item[1]))
    return list(dynamic[-1][-1][2])


def notebook_interval_values(mask: np.ndarray) -> dict[str, float]:
    """Reproduce perf_eval.ipynb, including its last-beat and zero conventions."""
    bounds = boundary_arrays(mask)
    intervals = {"PR": [], "QRS": [], "QT": []}
    p_onsets, qrs_onsets = bounds["P_onset"], bounds["QRS_onset"]
    qrs_offsets, t_offsets = bounds["QRS_offset"], bounds["T_offset"]
    for name, onsets, candidates in (
        ("PR", p_onsets, qrs_onsets), ("QRS", qrs_onsets, qrs_offsets), ("QT", qrs_onsets, t_offsets)
    ):
        if len(onsets) == 1:
            windows = [(onsets[0], math.inf)]
        else:
            windows = list(zip(onsets[:-1], onsets[1:]))
        for onset, next_onset in windows:
            endpoint = next((int(x) for x in candidates if onset < x < next_onset), None)
            if endpoint is not None:
                intervals[name].append((endpoint - int(onset)) * 1000.0 / NATIVE_FS)
    return {name: float(np.median(values)) if values else 0.0 for name, values in intervals.items()}


def parse_filename(filename: str) -> tuple[str, str]:
    stem = Path(filename).stem
    record_id, lead = stem.split("_lead_", 1)
    return record_id, lead


def load_streams(
    split: str,
    padding: str,
    max_streams: int = 0,
    *,
    index_root: Path = INDEX_ROOT,
    ludb_root: Path = LUDB_ROOT,
):
    frame = pd.read_csv(index_root / f"LUDB_{split}.csv")
    if max_streams:
        frame = frame.iloc[:max_streams]
    preprocessor = AuthorPreprocessor()
    for row in frame.itertuples(index=False):
        record_id, lead = parse_filename(row.waveform)
        path = ludb_root / record_id
        record = wfdb.rdrecord(str(path), physical=True)
        lookup = {name.lower(): index for index, name in enumerate(record.sig_name)}
        signal = record.p_signal[:SOURCE_SAMPLES, lookup[lead.lower()]]
        label = strict_triplet_mask(path, lead)
        processed, native_label = preprocessor(signal, label, padding)
        yield record_id, lead, processed, native_label


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS runs(
          run_id TEXT PRIMARY KEY, split TEXT, state_name TEXT, padding TEXT,
          checkpoint_sha256 TEXT, index_sha256 TEXT, evaluator_sha256 TEXT,
          status TEXT, streams INTEGER, subjects INTEGER, duration_seconds REAL,
          summary_json TEXT, completed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS stream_metrics(
          run_id TEXT, record_id TEXT, lead TEXT, miou REAL,
          background_iou REAL, p_iou REAL, qrs_iou REAL, t_iou REAL,
          notebook_pr_error_ms REAL, notebook_qrs_error_ms REAL, notebook_qt_error_ms REAL,
          PRIMARY KEY(run_id,record_id,lead)
        );
        CREATE TABLE IF NOT EXISTS boundary_metrics(
          run_id TEXT, record_id TEXT, lead TEXT, boundary TEXT,
          reference_events INTEGER, predicted_events INTEGER, true_positive INTEGER,
          false_positive INTEGER, false_negative INTEGER, errors_ms_json TEXT,
          PRIMARY KEY(run_id,record_id,lead,boundary)
        );
        """
    )
    return connection


def safe_iou(intersection: int, union: int) -> float:
    return float(intersection / union) if union else float("nan")


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(payload, args.state, device)
    index_path = args.index_root / f"LUDB_{args.split}.csv"
    identity = {
        "protocol": "semiseg_ludb_v2",
        "split": args.split,
        "state": args.state,
        "padding": args.padding,
        "evaluation_window": args.evaluation_window,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "index_sha256": sha256_file(index_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "tolerance_ms": args.tolerance_ms,
        "max_streams": args.max_streams,
    }
    run_id = sha256_json(identity)
    connection = connect_db(args.output_db)
    existing = connection.execute("SELECT status,summary_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if existing and existing[0] == "complete" and not args.force:
        result = json.loads(existing[1]); print(json.dumps({"event": "already_complete", **result})); return result
    connection.execute("DELETE FROM stream_metrics WHERE run_id=?", (run_id,))
    connection.execute("DELETE FROM boundary_metrics WHERE run_id=?", (run_id,))
    connection.execute("DELETE FROM runs WHERE run_id=?", (run_id,)); connection.commit()

    started = time.perf_counter()
    intersections = np.zeros(4, dtype=np.int64); unions = np.zeros(4, dtype=np.int64)
    stream_rows, boundary_rows, subjects = [], [], set()
    batch = []

    def consume(items):
        inputs = torch.from_numpy(np.concatenate([item[2] for item in items], axis=0)[:, None, :]).to(device)
        with torch.inference_mode():
            logits = model(inputs, return_loss=False)["seg_logits"]
            predictions = logits.argmax(dim=1).cpu().numpy()
        for (record_id, lead, _signal, reference), predicted in zip(items, predictions, strict=True):
            subjects.add(record_id)
            reference_iou, predicted_iou, predicted_boundaries = evaluation_view(
                reference, predicted, args.evaluation_window
            )
            class_ious = []
            for class_id in range(4):
                ref, pred = reference_iou == class_id, predicted_iou == class_id
                intersection, union = int(np.logical_and(ref, pred).sum()), int(np.logical_or(ref, pred).sum())
                intersections[class_id] += intersection; unions[class_id] += union
                class_ious.append(safe_iou(intersection, union))
            ref_intervals = notebook_interval_values(reference)
            pred_intervals = notebook_interval_values(predicted_boundaries)
            stream_rows.append((run_id, record_id, lead, float(np.nanmean(class_ious)), *class_ious,
                                *(abs(pred_intervals[x] - ref_intervals[x]) for x in ("PR", "QRS", "QT"))))
            ref_bounds, pred_bounds = boundary_arrays(reference), boundary_arrays(predicted_boundaries)
            tolerance = round(args.tolerance_ms * NATIVE_FS / 1000)
            for boundary in BOUNDARIES:
                pairs = monotonic_match(ref_bounds[boundary], pred_bounds[boundary], tolerance)
                errors = [(int(pred_bounds[boundary][j]) - int(ref_bounds[boundary][i])) * 1000.0 / NATIVE_FS for i, j in pairs]
                tp, nr, npred = len(pairs), len(ref_bounds[boundary]), len(pred_bounds[boundary])
                boundary_rows.append((run_id, record_id, lead, boundary, nr, npred, tp, npred - tp, nr - tp, json.dumps(errors)))

    for item in load_streams(
        args.split,
        args.padding,
        args.max_streams,
        index_root=args.index_root,
        ludb_root=args.ludb_root,
    ):
        batch.append(item)
        if len(batch) == args.batch_size:
            consume(batch); batch = []
    if batch:
        consume(batch)
    connection.executemany("INSERT INTO stream_metrics VALUES(?,?,?,?,?,?,?,?,?,?,?)", stream_rows)
    connection.executemany("INSERT INTO boundary_metrics VALUES(?,?,?,?,?,?,?,?,?,?)", boundary_rows)

    class_iou = [safe_iou(int(intersections[i]), int(unions[i])) for i in range(4)]
    summaries = {}
    for boundary in BOUNDARIES:
        selected = [row for row in boundary_rows if row[3] == boundary]
        reference_events, predicted_events, tp = (sum(row[i] for row in selected) for i in (4, 5, 6))
        errors = np.asarray([value for row in selected for value in json.loads(row[9])], dtype=float)
        summaries[boundary] = {
            "reference_events": reference_events, "predicted_events": predicted_events,
            "sensitivity": tp / reference_events if reference_events else None,
            "ppv": tp / predicted_events if predicted_events else None,
            "bias_ms": float(errors.mean()) if len(errors) else None,
            "sd_ms": float(errors.std(ddof=1)) if len(errors) > 1 else None,
            "mae_ms": float(np.abs(errors).mean()) if len(errors) else None,
            "p95_abs_ms": float(np.quantile(np.abs(errors), .95)) if len(errors) else None,
        }
    interval_errors = np.asarray([row[8:11] for row in stream_rows], dtype=float)
    result = {
        **identity, "run_id": run_id, "streams": len(stream_rows), "subjects": len(subjects),
        "class_iou": dict(zip(CLASS_NAMES, class_iou)), "miou_including_background": float(np.nanmean(class_iou)),
        "miou_foreground": float(np.nanmean(class_iou[1:])),
        "notebook_interval_mae_ms": dict(zip(("PR", "QRS", "QT"), np.mean(interval_errors, axis=0).tolist())),
        "boundaries": summaries, "duration_seconds": time.perf_counter() - started,
    }
    connection.execute(
        "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))",
        (run_id, args.split, args.state, args.padding, identity["checkpoint_sha256"], identity["index_sha256"],
         identity["evaluator_sha256"], "complete", len(stream_rows), len(subjects), result["duration_seconds"], json.dumps(result)),
    )
    connection.commit(); connection.execute("PRAGMA wal_checkpoint(TRUNCATE)"); connection.close()
    print(json.dumps({"event": "complete", **result}, allow_nan=False))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    result.add_argument("--index-root", type=Path, default=INDEX_ROOT)
    result.add_argument("--ludb-root", type=Path, default=LUDB_ROOT)
    result.add_argument("--output-db", type=Path, default=OUTPUT_DB)
    result.add_argument("--split", choices=("valid", "test"), required=True)
    result.add_argument("--state", choices=("model", "model_ema"), required=True)
    result.add_argument("--padding", choices=("none", "notebook"), default="none")
    result.add_argument("--evaluation-window", choices=("full", "annotated"), default="full")
    result.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--tolerance-ms", type=float, default=150.0)
    result.add_argument("--max-streams", type=int, default=0)
    result.add_argument("--force", action="store_true")
    return result


if __name__ == "__main__":
    evaluate(parser().parse_args())
