#!/home/mithunmanivannan/.venv/bin/python3
"""Evaluate temporal MMD regression to the mean for ECG reconstruction on factorial models.

This script tests the hypothesis that dynamic temporal clustering (K-Means MMD) 
preserves clinical variances (T-wave amplitude, QT interval) better than global MMD.
It runs continuously, processing new checkpoints as they appear.
"""

import argparse
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import sys
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import neurokit2 as nk
from scipy import stats
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from torch.utils.data import Dataset, DataLoader
from scripts.evaluate_comprehensive_registry import load_adapter
from scripts.checkpoint_store import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DB,
    connect,
    materialize,
    prune_cache,
    store_lock,
)
from unified_latents.engineering.src.evaluation.fiducial_bland_altman import (
    bland_altman_for_fiducial
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

CLINICAL_FEATURES = (
    "P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp", "QT_Interval_ms"
)
PAIRING_TOLERANCE_MS = 100.0
PAIRING_SENSITIVITY_MS = (25.0, 50.0, 75.0, 100.0, 150.0)
RBF_MMD_SAMPLE_CAP = 512
RBF_MMD_ESTIMATOR = "biased_squared_rbf_mmd"
RBF_MMD_BANDWIDTH_RULE = "median_nonzero_pairwise_absolute_distance"
RBF_MMD_SKETCH_RULE = "sorted_evenly_spaced_quantile_indices"


def distribution_rbf_mmd2(real_values, recon_values, sample_cap=RBF_MMD_SAMPLE_CAP):
    """Deterministic bounded-memory RBF MMD² on two finite 1-D samples."""
    if sample_cap < 2:
        raise ValueError("sample_cap must be at least 2")

    def sketch(values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("MMD inputs must be one-dimensional with >=2 values")
        if not np.isfinite(values).all():
            raise ValueError("MMD inputs must be finite")
        ordered = np.sort(values)
        if len(ordered) <= sample_cap:
            return ordered
        indices = np.rint(np.linspace(0, len(ordered) - 1, sample_cap)).astype(int)
        return ordered[indices]

    real = sketch(real_values)
    recon = sketch(recon_values)
    combined = np.concatenate([real, recon])
    pairwise_distance = np.abs(combined[:, None] - combined[None, :])
    positive_distance = pairwise_distance[pairwise_distance > 0]
    if not len(positive_distance):
        bandwidth = 1.0
    else:
        bandwidth = float(np.median(positive_distance))

    denominator = 2.0 * bandwidth * bandwidth
    kernel_real = np.exp(-((real[:, None] - real[None, :]) ** 2) / denominator)
    kernel_recon = np.exp(-((recon[:, None] - recon[None, :]) ** 2) / denominator)
    kernel_cross = np.exp(-((real[:, None] - recon[None, :]) ** 2) / denominator)
    mmd2 = float(kernel_real.mean() + kernel_recon.mean() - 2.0 * kernel_cross.mean())
    # The biased estimator is non-negative analytically; clamp round-off only.
    mmd2 = max(mmd2, 0.0)
    return {
        "mmd2": mmd2,
        "bandwidth": bandwidth,
        "real_samples": len(real),
        "recon_samples": len(recon),
    }

class SimpleECGDataset(Dataset):
    def __init__(self, data_dir):
        self.files = sorted(Path(data_dir).glob("*.pt"), key=lambda p: int(p.stem))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        path = self.files[index]
        signal = torch.load(path, weights_only=True).float()
        return (signal, int(path.stem))


def extract_qt_events(qrs_onsets, r_peaks, t_offsets, fs=500):
    """Associate QRS onset, R peak, and T offset within one detected beat."""
    qrs_onsets = np.unique(np.asarray(qrs_onsets, dtype=np.int64))
    r_peaks = np.unique(np.asarray(r_peaks, dtype=np.int64))
    t_offsets = np.unique(np.asarray(t_offsets, dtype=np.int64))
    anchors = []
    intervals = []
    for index, qrs_onset in enumerate(qrs_onsets):
        next_qrs_onset = (
            qrs_onsets[index + 1] if index + 1 < len(qrs_onsets) else np.inf
        )
        candidate_r_peaks = r_peaks[
            (r_peaks >= qrs_onset) & (r_peaks < next_qrs_onset)
        ]
        if not len(candidate_r_peaks):
            continue
        r_peak = candidate_r_peaks[0]
        later_r_peaks = r_peaks[r_peaks > r_peak]
        next_r_peak = later_r_peaks[0] if len(later_r_peaks) else np.inf
        beat_end = min(next_qrs_onset, next_r_peak)
        candidates = t_offsets[
            (t_offsets > r_peak) & (t_offsets < beat_end)
        ]
        if len(candidates):
            anchors.append(qrs_onset)
            intervals.append((candidates[0] - qrs_onset) / fs * 1000.0)
    return {
        "sample_index": np.asarray(anchors, dtype=np.int64),
        "value": np.asarray(intervals, dtype=float),
    }

def extract_clinical_features(signal: np.ndarray, fs: int = 500):
    """Extract feature values and their detector-event sample indices."""
    try:
        signals, info = nk.ecg_process(signal, sampling_rate=fs)
        
        features = {
            "P_Amp": "ECG_P_Peaks",
            "Q_Amp": "ECG_Q_Peaks",
            "R_Amp": "ECG_R_Peaks",
            "S_Amp": "ECG_S_Peaks",
            "T_Amp": "ECG_T_Peaks"
        }
        
        clinical = {}
        for short_k, feature in features.items():
            if feature in signals.columns:
                indices = np.where(signals[feature] == 1)[0]
                if len(indices) > 0:
                    clinical[short_k] = {
                        "sample_index": indices.astype(np.int64),
                        "value": np.asarray(signal[indices], dtype=float),
                    }
                else:
                    clinical[short_k] = {
                        "sample_index": np.array([], dtype=np.int64),
                        "value": np.array([], dtype=float),
                    }
                    
        # Extract QT Interval: T_offset - Q_onset
        if all(
            column in signals.columns
            for column in ("ECG_T_Offsets", "ECG_R_Onsets", "ECG_R_Peaks")
        ):
            t_offsets = np.where(signals["ECG_T_Offsets"] == 1)[0]
            q_onsets = np.where(signals["ECG_R_Onsets"] == 1)[0]
            r_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
            
            qt_events = extract_qt_events(q_onsets, r_peaks, t_offsets, fs=fs)
            if len(qt_events["value"]) > 0:
                clinical["QT_Interval_ms"] = qt_events
                
        return clinical
    except Exception as e:
        return {}

import concurrent.futures


def target_cache_lookup(frame):
    """Index immutable target features by record, lead, and feature."""
    if frame.empty:
        return {}
    result = {}
    for (record_id, lead, feature), group in frame.groupby(
        ["record_id", "lead", "clinical_feature"], sort=False
    ):
        ordered = group.sort_values(["sample_index", "beat_index"])
        result[(str(record_id), lead, feature)] = {
            "sample_index": ordered["sample_index"].to_numpy(dtype=np.int64),
            "value": ordered["value"].to_numpy(dtype=float),
        }
    return result


def empty_events():
    return {
        "sample_index": np.array([], dtype=np.int64),
        "value": np.array([], dtype=float),
    }


def match_events_by_time(real_events, recon_events, tolerance_samples):
    """Monotonic 1:1 matching: maximize count, then minimize total time error."""
    real_times = np.asarray(real_events["sample_index"], dtype=np.int64)
    real_values = np.asarray(real_events["value"], dtype=float)
    recon_times = np.asarray(recon_events["sample_index"], dtype=np.int64)
    recon_values = np.asarray(recon_events["value"], dtype=float)
    if len(real_times) != len(real_values) or len(recon_times) != len(recon_values):
        raise ValueError("event times and values have unequal lengths")
    if tolerance_samples < 0:
        raise ValueError("tolerance_samples must be nonnegative")
    if np.any(np.diff(real_times) < 0) or np.any(np.diff(recon_times) < 0):
        raise ValueError("event times must be sorted")

    # Each cell stores (match_count, total_absolute_error, index_pairs).
    dp = [[(0, 0, ()) for _ in range(len(recon_times) + 1)]
          for _ in range(len(real_times) + 1)]
    for i in range(1, len(real_times) + 1):
        for j in range(1, len(recon_times) + 1):
            candidates = [dp[i - 1][j], dp[i][j - 1]]
            difference = abs(int(real_times[i - 1]) - int(recon_times[j - 1]))
            if difference <= tolerance_samples:
                count, error, pairs = dp[i - 1][j - 1]
                candidates.append(
                    (count + 1, error + difference, pairs + ((i - 1, j - 1),))
                )
            dp[i][j] = max(candidates, key=lambda item: (item[0], -item[1]))
    pairs = dp[-1][-1][2]
    real_indices = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    recon_indices = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    return {
        "real_values": real_values[real_indices],
        "recon_values": recon_values[recon_indices],
        "absolute_time_error_samples": np.abs(
            real_times[real_indices] - recon_times[recon_indices]
        ),
        "matched": len(pairs),
        "unmatched_real": len(real_times) - len(pairs),
        "unmatched_recon": len(recon_times) - len(pairs),
    }


def load_or_build_target_cache(
    dataset_name,
    loader,
    executor,
    leads_to_evaluate,
    out_dir,
    data_content_root_sha256,
    extractor_sha256,
):
    """Cache target delineation once; targets are identical across all models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / f"_target_{dataset_name}_features.parquet"
    metadata_path = out_dir / f"_target_{dataset_name}_features.json"
    expected = {
        "schema_version": 2,
        "dataset": dataset_name,
        "data_content_root_sha256": data_content_root_sha256,
        "extractor_sha256": extractor_sha256,
        "sample_rate_hz": 500,
        "leads": leads_to_evaluate,
    }
    if parquet_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if all(metadata.get(key) == value for key, value in expected.items()):
            if metadata.get("parquet_sha256") == hashlib.sha256(parquet_path.read_bytes()).hexdigest():
                frame = pd.read_parquet(parquet_path)
                logging.info("Reusing content-bound %s target cache with %d rows", dataset_name, len(frame))
                return target_cache_lookup(frame), metadata

    logging.info("Building one-time target feature cache for %s", dataset_name)
    rows = []
    records = 0
    lead_names = list(leads_to_evaluate)
    for target, record_ids in loader:
        target_np = target.numpy()
        signals = [
            target_np[b, leads_to_evaluate[lead_name]]
            for b in range(target_np.shape[0])
            for lead_name in lead_names
        ]
        results = list(executor.map(extract_clinical_features, signals))
        index = 0
        for b in range(target_np.shape[0]):
            record_id = str(int(record_ids[b]))
            records += 1
            for lead_name in lead_names:
                features = results[index]
                index += 1
                for feature, events in features.items():
                    for beat_index, (sample_index, value) in enumerate(zip(
                        events["sample_index"], events["value"]
                    )):
                        if np.isfinite(value):
                            rows.append({
                                "record_id": record_id,
                                "lead": lead_name,
                                "clinical_feature": feature,
                                "beat_index": beat_index,
                                "sample_index": int(sample_index),
                                "value": float(value),
                            })
    frame = pd.DataFrame(rows)
    temporary = parquet_path.with_name(f".{parquet_path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, parquet_path)
    metadata = {
        **expected,
        "records": records,
        "rows": len(frame),
        "parquet": parquet_path.name,
        "parquet_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_tmp = metadata_path.with_name(f".{metadata_path.name}.{os.getpid()}.tmp")
    meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(meta_tmp, metadata_path)
    logging.info("Published %s target cache with %d feature rows", dataset_name, len(frame))
    return target_cache_lookup(frame), metadata


def evaluate_model(ckpt_path, model_identity, contract, datasets, target_caches, device, executor, out_dir, evaluation_sha256):
    """Evaluate one exact checkpoint and atomically publish a bound artifact."""
    evaluation_started_at = datetime.now(timezone.utc)
    evaluation_started_monotonic = time.monotonic()
    model_id = model_identity["model_id"]
    match = re.search(r'f_(\d{7})_s(\d+)$', model_id)
    if not match:
        raise ValueError(f"Invalid model id: {model_id}")
    mask = match.group(1)
    seed = int(match.group(2))
    
    spec = {
        "kind": "unet",
        "observed_leads": [0, 1, 7],
        "checkpoint": str(ckpt_path)
    }
    
    try:
        logging.info(f"Loading adapter for {ckpt_path}...")
        adapter = load_adapter(spec, device)
        logging.info("Adapter loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load {ckpt_path}: {e}")
        return
        
    leads_to_evaluate = {"V3": 8, "V6": 11}
    pairing_tolerance_samples = int(round(PAIRING_TOLERANCE_MS / 1000.0 * 500))
    summary_rows = []
    per_record_rows = []
    sensitivity_stats = {}
    
    for dataset_name, loader in datasets.items():
        logging.info(f"Evaluating {mask} on {dataset_name} (batch 0)...")
        target_lookup, target_metadata = target_caches[dataset_name]
        
        real_features = {lead: {} for lead in leads_to_evaluate}
        recon_features = {lead: {} for lead in leads_to_evaluate}
        coverage = {lead: {} for lead in leads_to_evaluate}
        
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                target = batch[0].to(device)
                record_ids = batch[1]
                try:
                    recon = adapter.reconstruct(target)
                except Exception as e:
                    raise RuntimeError(
                        f"Reconstruction failed for {model_id}, dataset={dataset_name}, "
                        f"batch_index={batch_index}"
                    ) from e
                
                target_np = target.cpu().numpy()
                recon_np = recon.cpu().numpy()
                
                logging.info(f"  Processed batch of size {target.shape[0]}, extracting features...")
                
                B = target.shape[0]
                flat_recon = []
                lead_names_order = list(leads_to_evaluate.keys())
                
                for b in range(B):
                    for lead_name in lead_names_order:
                        lead_idx = leads_to_evaluate[lead_name]
                        flat_recon.append(recon_np[b, lead_idx])

                recon_results = list(executor.map(extract_clinical_features, flat_recon))
                
                idx = 0
                for b in range(B):
                    record_id = str(int(record_ids[b]))
                    for lead_name in lead_names_order:
                        recon_fid = recon_results[idx]
                        idx += 1

                        for k in CLINICAL_FEATURES:
                            recon_k = recon_fid.get(k, empty_events())
                            real_k = target_lookup.get(
                                (record_id, lead_name, k), empty_events()
                            )
                            if k not in real_features[lead_name]:
                                real_features[lead_name][k] = []
                                recon_features[lead_name][k] = []
                                coverage[lead_name][k] = {
                                    "records_total": 0,
                                    "records_real_detected": 0,
                                    "records_recon_detected": 0,
                                    "records_paired": 0,
                                    "real_beats_detected": 0,
                                    "recon_beats_detected": 0,
                                    "paired_beats": 0,
                                    "unmatched_real_events": 0,
                                    "unmatched_recon_events": 0,
                                    "absolute_time_error_samples": [],
                                }

                            sensitivity_matches = {}
                            for sensitivity_ms in PAIRING_SENSITIVITY_MS:
                                tolerance_samples = int(round(
                                    sensitivity_ms / 1000.0 * 500
                                ))
                                matched_at_tolerance = match_events_by_time(
                                    real_k, recon_k, tolerance_samples
                                )
                                sensitivity_matches[sensitivity_ms] = matched_at_tolerance
                                sensitivity_finite = np.isfinite(
                                    matched_at_tolerance["real_values"]
                                ) & np.isfinite(matched_at_tolerance["recon_values"])
                                sensitivity_real = matched_at_tolerance["real_values"][
                                    sensitivity_finite
                                ]
                                sensitivity_recon = matched_at_tolerance["recon_values"][
                                    sensitivity_finite
                                ]
                                sensitivity_key = (sensitivity_ms, lead_name, k)
                                if sensitivity_key not in sensitivity_stats:
                                    sensitivity_stats[sensitivity_key] = {
                                        "records_total": 0,
                                        "records_paired": 0,
                                        "real_events": 0,
                                        "recon_events": 0,
                                        "paired_events": 0,
                                        "finite_paired_events": 0,
                                        "unmatched_real_events": 0,
                                        "unmatched_recon_events": 0,
                                        "sum_real": 0.0,
                                        "sum_recon": 0.0,
                                        "sum_sq_real": 0.0,
                                        "sum_sq_recon": 0.0,
                                    }
                                sensitivity_counts = sensitivity_stats[sensitivity_key]
                                sensitivity_counts["records_total"] += 1
                                sensitivity_counts["records_paired"] += int(
                                    matched_at_tolerance["matched"] > 0
                                )
                                sensitivity_counts["real_events"] += len(real_k["value"])
                                sensitivity_counts["recon_events"] += len(recon_k["value"])
                                sensitivity_counts["paired_events"] += matched_at_tolerance["matched"]
                                sensitivity_counts["finite_paired_events"] += len(sensitivity_real)
                                sensitivity_counts["unmatched_real_events"] += matched_at_tolerance["unmatched_real"]
                                sensitivity_counts["unmatched_recon_events"] += matched_at_tolerance["unmatched_recon"]
                                sensitivity_counts["sum_real"] += float(sensitivity_real.sum())
                                sensitivity_counts["sum_recon"] += float(sensitivity_recon.sum())
                                sensitivity_counts["sum_sq_real"] += float(np.square(sensitivity_real).sum())
                                sensitivity_counts["sum_sq_recon"] += float(np.square(sensitivity_recon).sum())

                            matched = sensitivity_matches[PAIRING_TOLERANCE_MS]
                            n_beats = matched["matched"]
                            counts = coverage[lead_name][k]
                            counts["records_total"] += 1
                            counts["records_real_detected"] += int(len(real_k["value"]) > 0)
                            counts["records_recon_detected"] += int(len(recon_k["value"]) > 0)
                            counts["records_paired"] += int(n_beats > 0)
                            counts["real_beats_detected"] += len(real_k["value"])
                            counts["recon_beats_detected"] += len(recon_k["value"])
                            counts["paired_beats"] += n_beats
                            counts["unmatched_real_events"] += matched["unmatched_real"]
                            counts["unmatched_recon_events"] += matched["unmatched_recon"]
                            finite = np.isfinite(matched["real_values"]) & np.isfinite(
                                matched["recon_values"]
                            )
                            finite_real = matched["real_values"][finite]
                            finite_recon = matched["recon_values"][finite]
                            finite_time_errors_ms = (
                                matched["absolute_time_error_samples"][finite]
                                / 500.0 * 1000.0
                            )
                            real_features[lead_name][k].extend(
                                finite_real
                            )
                            recon_features[lead_name][k].extend(
                                finite_recon
                            )
                            counts["absolute_time_error_samples"].extend(
                                matched["absolute_time_error_samples"][finite]
                            )
                            real_detected = len(real_k["value"]) > 0
                            recon_detected = len(recon_k["value"]) > 0
                            if real_detected and recon_detected:
                                detector_state = "both_detected"
                            elif real_detected:
                                detector_state = "target_only"
                            elif recon_detected:
                                detector_state = "reconstruction_only"
                            else:
                                detector_state = "neither_detected"
                            per_record_rows.append({
                                "model_id": model_id,
                                "checkpoint_sha256": model_identity["checkpoint_sha256"],
                                "evaluation_code_sha256": evaluation_sha256,
                                "target_feature_cache_sha256": target_metadata["parquet_sha256"],
                                "dataset": dataset_name,
                                "record_id": record_id,
                                "lead": lead_name,
                                "clinical_feature": k,
                                "detector_state": detector_state,
                                "target_detected": real_detected,
                                "reconstruction_detected": recon_detected,
                                "n_real_events": len(real_k["value"]),
                                "n_recon_events": len(recon_k["value"]),
                                "n_paired_events": n_beats,
                                "n_finite_paired_events": len(finite_real),
                                "n_unmatched_real_events": matched["unmatched_real"],
                                "n_unmatched_recon_events": matched["unmatched_recon"],
                                "paired_sum_real": float(finite_real.sum()),
                                "paired_sum_recon": float(finite_recon.sum()),
                                "paired_sum_sq_real": float(np.square(finite_real).sum()),
                                "paired_sum_sq_recon": float(np.square(finite_recon).sum()),
                                "paired_mean_real": (
                                    float(finite_real.mean()) if len(finite_real) else np.nan
                                ),
                                "paired_mean_recon": (
                                    float(finite_recon.mean()) if len(finite_recon) else np.nan
                                ),
                                "paired_mean_difference": (
                                    float((finite_recon - finite_real).mean())
                                    if len(finite_real) else np.nan
                                ),
                                "paired_mae": (
                                    float(np.abs(finite_recon - finite_real).mean())
                                    if len(finite_real) else np.nan
                                ),
                                "median_abs_pairing_error_ms": (
                                    float(np.median(finite_time_errors_ms))
                                    if len(finite_time_errors_ms) else np.nan
                                ),
                                "p95_abs_pairing_error_ms": (
                                    float(np.percentile(finite_time_errors_ms, 95))
                                    if len(finite_time_errors_ms) else np.nan
                                ),
                            })
                                    
        for lead_name in leads_to_evaluate:
            for fid_name in real_features[lead_name]:
                real_arr = np.array(real_features[lead_name][fid_name])
                recon_arr = np.array(recon_features[lead_name][fid_name])
                
                if len(real_arr) < 5:
                    continue
                    
                ba_stats = bland_altman_for_fiducial(real_arr, recon_arr)
                distribution_mmd = distribution_rbf_mmd2(real_arr, recon_arr)
                var_real = np.var(real_arr, ddof=1)
                var_recon = np.var(recon_arr, ddof=1)
                counts = coverage[lead_name][fid_name]
                time_errors_ms = (
                    np.asarray(counts["absolute_time_error_samples"], dtype=float)
                    / 500.0
                    * 1000.0
                )
                
                summary_rows.append({
                    "model_id": model_id,
                    "model_mask": mask,
                    "seed": seed,
                    "architecture": "unet",
                    "mmd_kernel": int(mask[6]),
                    "checkpoint_sha256": model_identity["checkpoint_sha256"],
                    "checkpoint_size_bytes": int(model_identity["checkpoint_size_bytes"]),
                    "contract_id": contract["contract_id"],
                    "source_bundle_sha256": model_identity["source_bundle_sha256"],
                    "test_content_root_sha256": contract["split_content_roots"]["test"]["content_root_sha256"],
                    "evaluation_code_sha256": evaluation_sha256,
                    "target_feature_cache_sha256": target_metadata["parquet_sha256"],
                    "dataset": dataset_name,
                    "sample_rate_hz": 500,
                    "detector": "neurokit2.ecg_process_default",
                    "feature_pairing_rule": "within_record_monotonic_max_cardinality_min_abs_time",
                    "pairing_tolerance_ms": PAIRING_TOLERANCE_MS,
                    "observed_lead_indices": "0,1,7",
                    "lead": lead_name,
                    "evaluated_lead_index": leads_to_evaluate[lead_name],
                    "clinical_feature": fid_name,
                    "n_beats": len(real_arr),
                    "n_records_total": counts["records_total"],
                    "n_records_real_detected": counts["records_real_detected"],
                    "n_records_recon_detected": counts["records_recon_detected"],
                    "n_records_paired": counts["records_paired"],
                    "record_pair_coverage": (
                        counts["records_paired"] / max(counts["records_total"], 1)
                    ),
                    "n_real_beats_detected": counts["real_beats_detected"],
                    "n_recon_beats_detected": counts["recon_beats_detected"],
                    "n_paired_beats_before_finite_filter": counts["paired_beats"],
                    "n_unmatched_real_events": counts["unmatched_real_events"],
                    "n_unmatched_recon_events": counts["unmatched_recon_events"],
                    "median_abs_pairing_error_ms": np.median(time_errors_ms),
                    "p95_abs_pairing_error_ms": np.percentile(time_errors_ms, 95),
                    "mean_real": np.mean(real_arr),
                    "mean_recon": np.mean(recon_arr),
                    "variance_ratio": var_recon / max(var_real, 1e-12),
                    "ba_robust_slope": ba_stats.robust_slope,
                    "distribution_rbf_mmd2": distribution_mmd["mmd2"],
                    "distribution_rbf_bandwidth": distribution_mmd["bandwidth"],
                    "distribution_mmd_real_samples": distribution_mmd["real_samples"],
                    "distribution_mmd_recon_samples": distribution_mmd["recon_samples"],
                    "distribution_mmd_estimator": RBF_MMD_ESTIMATOR,
                    "distribution_mmd_bandwidth_rule": RBF_MMD_BANDWIDTH_RULE,
                    "distribution_mmd_sketch_rule": RBF_MMD_SKETCH_RULE,
                    "distribution_mmd_sample_cap": RBF_MMD_SAMPLE_CAP,
                })
                
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        per_record = pd.DataFrame(per_record_rows)
        sensitivity_rows = []
        for (tolerance_ms, lead_name, feature), counts in sorted(
            sensitivity_stats.items()
        ):
            n = counts["finite_paired_events"]
            if n < 2:
                raise RuntimeError(
                    f"Too few finite pairs for tolerance sensitivity: "
                    f"{lead_name}/{feature}/{tolerance_ms}ms"
                )
            mean_real = counts["sum_real"] / n
            mean_recon = counts["sum_recon"] / n
            variance_real = (
                counts["sum_sq_real"] - counts["sum_real"] ** 2 / n
            ) / (n - 1)
            variance_recon = (
                counts["sum_sq_recon"] - counts["sum_recon"] ** 2 / n
            ) / (n - 1)
            sensitivity_rows.append({
                "model_id": model_id,
                "checkpoint_sha256": model_identity["checkpoint_sha256"],
                "evaluation_code_sha256": evaluation_sha256,
                "target_feature_cache_sha256": target_metadata["parquet_sha256"],
                "dataset": "ptb_xl",
                "lead": lead_name,
                "clinical_feature": feature,
                "pairing_tolerance_ms": tolerance_ms,
                "n_records_total": counts["records_total"],
                "n_records_paired": counts["records_paired"],
                "record_pair_coverage": (
                    counts["records_paired"] / counts["records_total"]
                ),
                "n_real_events": counts["real_events"],
                "n_recon_events": counts["recon_events"],
                "n_paired_events": counts["paired_events"],
                "n_finite_paired_events": n,
                "n_unmatched_real_events": counts["unmatched_real_events"],
                "n_unmatched_recon_events": counts["unmatched_recon_events"],
                "mean_real": mean_real,
                "mean_recon": mean_recon,
                "variance_ratio": variance_recon / max(variance_real, 1e-12),
            })
        sensitivity = pd.DataFrame(sensitivity_rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        sensitivity_path = out_dir / f"{model_id}_tolerance_sensitivity.csv"
        sensitivity_temporary = out_dir / f".{model_id}.{os.getpid()}.sensitivity.tmp"
        sensitivity.to_csv(sensitivity_temporary, index=False)
        os.replace(sensitivity_temporary, sensitivity_path)
        sensitivity_sha256 = hashlib.sha256(sensitivity_path.read_bytes()).hexdigest()
        per_record_path = out_dir / f"{model_id}_per_record.parquet"
        per_record_temporary = out_dir / f".{model_id}.{os.getpid()}.per_record.tmp"
        per_record.to_parquet(per_record_temporary, index=False)
        os.replace(per_record_temporary, per_record_path)
        per_record_sha256 = hashlib.sha256(per_record_path.read_bytes()).hexdigest()
        out_csv = out_dir / f"{model_id}.csv"
        temporary = out_dir / f".{model_id}.{os.getpid()}.csv.tmp"
        df.to_csv(temporary, index=False)
        os.replace(temporary, out_csv)
        csv_sha256 = hashlib.sha256(out_csv.read_bytes()).hexdigest()
        completed_at = datetime.now(timezone.utc)
        metadata = {
            "schema_version": 1,
            "model_id": model_id,
            "rows": len(df),
            "checkpoint_sha256": model_identity["checkpoint_sha256"],
            "checkpoint_size_bytes": int(model_identity["checkpoint_size_bytes"]),
            "contract_id": contract["contract_id"],
            "source_bundle_sha256": model_identity["source_bundle_sha256"],
            "test_content_root_sha256": contract["split_content_roots"]["test"]["content_root_sha256"],
            "evaluation_code_sha256": evaluation_sha256,
            "target_feature_cache_sha256": sorted({
                metadata["parquet_sha256"] for _, metadata in target_caches.values()
            }),
            "detector": "neurokit2.ecg_process_default",
            "feature_pairing_rule": "within_record_monotonic_max_cardinality_min_abs_time",
            "pairing_tolerance_ms": PAIRING_TOLERANCE_MS,
            "pairing_sensitivity_ms": list(PAIRING_SENSITIVITY_MS),
            "distribution_mmd_estimator": RBF_MMD_ESTIMATOR,
            "distribution_mmd_bandwidth_rule": RBF_MMD_BANDWIDTH_RULE,
            "distribution_mmd_sketch_rule": RBF_MMD_SKETCH_RULE,
            "distribution_mmd_sample_cap": RBF_MMD_SAMPLE_CAP,
            "observed_lead_indices": [0, 1, 7],
            "evaluated_leads": {"V3": 8, "V6": 11},
            "datasets": {
                name: {"records_expected": len(loader.dataset)}
                for name, loader in datasets.items()
            },
            "reconstruction_batch_failures": 0,
            "started_at": evaluation_started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": time.monotonic() - evaluation_started_monotonic,
            "per_record_rows": len(per_record),
            "per_record_parquet": per_record_path.name,
            "per_record_parquet_sha256": per_record_sha256,
            "tolerance_sensitivity_rows": len(sensitivity),
            "tolerance_sensitivity_csv": sensitivity_path.name,
            "tolerance_sensitivity_csv_sha256": sensitivity_sha256,
            "csv": out_csv.name,
            "csv_sha256": csv_sha256,
        }
        meta_path = out_dir / f"{model_id}.json"
        meta_tmp = out_dir / f".{model_id}.{os.getpid()}.json.tmp"
        meta_tmp.write_text(json.dumps(metadata, indent=2) + "\n")
        os.replace(meta_tmp, meta_path)
        logging.info(f"Published generation-bound results for {model_id} to {out_csv}")
        return metadata
    raise RuntimeError(f"No valid temporal-MMD rows produced for {model_id}")

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Poll for newly archived compatible models")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--workers", type=int, default=2, help="Bound CPU feature-extraction workers")
    parser.add_argument("--model-id", help="Evaluate only this compatible model")
    parser.add_argument("--target-cache-only", action="store_true", help="Build/verify immutable target features, then exit")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("results/factorial_v4/temporal_mmd_generation_bound"))
    return parser.parse_args()


def compatible_models(audit_path):
    audit = json.loads(audit_path.read_text())
    return audit, {m["model_id"]: m for m in audit["models"] if m.get("compatible")}


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    project_root = Path(__file__).resolve().parents[1]
    
    # Datasets
    ptb_test = project_root / "data/ptb_xl/tensors/test"
    if not ptb_test.is_dir():
        raise FileNotFoundError(f"PTB-XL test tensors unavailable: {ptb_test}")
    datasets = {
        "ptb_xl": DataLoader(SimpleECGDataset(ptb_test), batch_size=32, shuffle=False, num_workers=0)
    }
    sunnybrook = project_root / "data/sunnybrook/tensors/clinical"
    if sunnybrook.is_dir() and any(sunnybrook.glob("*.pt")):
        datasets["sunnybrook"] = DataLoader(SimpleECGDataset(sunnybrook), batch_size=32, shuffle=False, num_workers=0)

    out_dir = (project_root / args.out_dir) if not args.out_dir.is_absolute() else args.out_dir
    audit_path = project_root / "results/checkpoint_store/compatibility_audit.json"
    evaluation_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    extractor_sha256 = hashlib.sha256(
        (
            inspect.getsource(extract_qt_events)
            + inspect.getsource(extract_clinical_features)
            + "|fs=500|V3=8|V6=11"
        ).encode()
    ).hexdigest()
    
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
    initial_audit, _ = compatible_models(audit_path)
    leads_to_evaluate = {"V3": 8, "V6": 11}
    target_caches = {}
    for dataset_name, loader in datasets.items():
        if dataset_name != "ptb_xl":
            raise RuntimeError(
                f"No content-root contract is defined for target cache {dataset_name}"
            )
        target_caches[dataset_name] = load_or_build_target_cache(
            dataset_name,
            loader,
            executor,
            leads_to_evaluate,
            out_dir,
            initial_audit["contract"]["split_content_roots"]["test"]["content_root_sha256"],
            extractor_sha256,
        )
    if args.target_cache_only:
        logging.info("Target feature cache is complete; exiting as requested")
        return
    logging.info("Starting generation-bound evaluation on %s", device)
    while True:
        audit, eligible = compatible_models(audit_path)
        if args.model_id:
            if args.model_id not in eligible:
                raise ValueError(f"Model is not current-contract compatible: {args.model_id}")
            eligible = {args.model_id: eligible[args.model_id]}
        for model_id, identity in sorted(eligible.items()):
            metadata_path = out_dir / f"{model_id}.json"
            if metadata_path.exists():
                previous = json.loads(metadata_path.read_text())
                if (previous.get("checkpoint_sha256") == identity["checkpoint_sha256"]
                        and previous.get("evaluation_code_sha256") == evaluation_sha256):
                    continue
            with store_lock(args.db):
                connection = connect(args.db)
                try:
                    checkpoint = materialize(connection, model_id, args.cache_dir)
                finally:
                    connection.close()
            try:
                evaluate_model(checkpoint, identity, audit["contract"], datasets, target_caches, device, executor, out_dir, evaluation_sha256)
            finally:
                with store_lock(args.db):
                    connection = connect(args.db)
                    try:
                        prune_cache(connection, args.cache_dir, 0.0)
                    finally:
                        connection.close()
        if not args.watch or args.model_id:
            break
        time.sleep(args.poll_seconds)

if __name__ == "__main__":
    main()
