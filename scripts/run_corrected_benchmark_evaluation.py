#!/usr/bin/env python3
"""
Corrected Clinical Benchmark & Evaluation Engine for HRS 2027 Study.
Standardized across:
- ArNet2 (AIM-Lab Official SavedModel, predict_fixed)
- RawECGNet / ArNetECG (AIM-Lab Official SavedModel, predict_fixed)
- SetOperator + RR (Multimodal Morphology-Rhythm Fusion)
- Baseline + Standard SQI Rejection
- Quality-Aware SetOperator (Selective Chow Abstention)
- Causal Moving-Average Postprocessing (k=9 windows)

Evaluates:
- All 98 rhythm-annotated recordings in SHDB-AF (2,350+ continuous hours)
- All 33 non-AF controls in SHDB-AF (Tier A: 20 beat-annotated, Tier B: 13 report-only)
- Noise Holter results imported from record-group-disjoint OOF predictions; patient independence is unverified
- Full threshold sweep tau in [0.10, ..., 0.90] for complete FROC curves
- Quality threshold sweep tau_qual in [0.10, ..., 0.90] for Risk-Coverage curves
- Generates paired_episode_predictions.parquet for all reference episodes
"""
from __future__ import annotations

import os
import argparse
import hashlib
import json
import sys
import time
import glob
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats, signal
import wfdb

# Add workspace and prototype to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "prototype"))

from prototype.metrics_engine import (
    extract_continuous_episodes,
    match_episodes,
    compute_clinical_false_alerts_per_24h,
    compute_burden_metrics,
)
from scripts.hrs2027_protocol import (
    add_ablation_dates, extract_features, quality_mask,
    parse_atr_gold_standard_episodes,
)
from scripts.hrs2027_cpu_features import initialize,prepare_window


def causal_moving_average(probs: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """Causal moving average smoothing over current and preceding windows."""
    if kernel_size <= 1:
        return probs.copy()
    n = len(probs)
    smoothed = np.zeros(n, dtype=np.float32)
    for i in range(n):
        s_idx = max(0, i - kernel_size + 1)
        valid = probs[s_idx : i + 1]
        valid = valid[~np.isnan(valid)]
        smoothed[i] = np.mean(valid) if len(valid) > 0 else probs[i]
    return smoothed


def run_corrected_benchmark(
    shdb_dir: str = "/data/mithunmanivannan/datasets/shdb_af",
    shdb_metadata: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/AdditionalData.csv",
    output_dir: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/corrected_package",
    device: str = "cuda",
    checkpoint: str | None = None,
    prepare_only: bool = False,
    shard_index: int = 0,
    shard_count: int = 1,
    start_index: int = 0,
):
    import tensorflow as tf
    import torch
    from prototype.models import SetOperatorPlusRR
    from scripts.hrs2027_gpu_features import batch_phase_kme
    tf.get_logger().setLevel("ERROR")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu,True)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.enabled=True
    if (shard_count > 1 or start_index > 0) and not prepare_only:
        raise ValueError('Only input preparation may be sharded; clinical scoring requires the complete cohort')
    os.makedirs(output_dir, exist_ok=True)
    if not torch.cuda.is_available() or not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('A100 evaluation requires both framework GPU runtimes; use the documented CUDA library path')
    print(f"=== Starting Corrected Benchmark Evaluation (PyTorch Device: {device}) ===")

    # 1. Load Pretrained Author Baselines using predict_fixed
    print("Loading AIM-Lab official pre-trained models from AFtoolkit...")
    arnet_ecg_dir = str(ROOT_DIR / "external/AFtoolkit/detection/ArNetECG_exported_model")
    arnet2_dir = str(ROOT_DIR / "external/AFtoolkit/detection/exported_model")

    model_rawecg = tf.saved_model.load(arnet_ecg_dir)
    infer_rawecg = model_rawecg.signatures["predict_fixed"]

    model_arnet2 = tf.saved_model.load(arnet2_dir)
    infer_arnet2 = model_arnet2.signatures["predict_fixed"]
    print("Author baselines loaded with predict_fixed signature.")

    # 2. Load Multimodal SetOperator Checkpoint
    if not prepare_only:
        if checkpoint is None:
            raise ValueError("Explicit full-corpus checkpoint required")
        checkpoint_hash=hashlib.file_digest(open(checkpoint,'rb'),'sha256').hexdigest()
        training=json.loads((Path(output_dir)/'training_provenance.json').read_text())
        if checkpoint_hash!=training['checkpoint_sha256'] or training['n_segments']!=6130:
            raise ValueError('Checkpoint does not match the full-corpus training provenance')
        setop_model = SetOperatorPlusRR()
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        setop_model.load_state_dict(state)
        setop_model.to(device)
        setop_model.eval()
    
    anchors = np.load(Path(output_dir)/'anchors.npy')
    if anchors.shape!=(128,16):
        raise ValueError('Invalid training anchors')
    cpu_executor=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),
                                     initializer=initialize,initargs=(anchors,))

    # 3. Read SHDB-AF Manifest
    df_meta = add_ablation_dates(pd.read_csv(shdb_metadata))
    df_meta.to_csv(Path(output_dir) / "shdb_cohort_manifest.csv", index=False)
    prediction_dir = Path(output_dir) / "window_predictions"
    prediction_dir.mkdir(exist_ok=True)
    prepared_dir = Path(output_dir) / "prepared_inputs"
    prepared_dir.mkdir(exist_ok=True)
    protocol_hash = hashlib.file_digest(open(ROOT_DIR / "scripts/hrs2027_protocol.py","rb"),"sha256").hexdigest()
    reference_rows = []
    print(f"Loaded SHDB-AF metadata: {len(df_meta)} recordings across {df_meta['Subject_ID'].nunique()} patients.")

    # Models to benchmark
    model_names = [
        "ArNet2_Raw",
        "ArNet2_MA9",
        "ArNet2_SQI",
        "RawECGNet_Raw",
        "RawECGNet_MA9",
        "RawECGNet_SQI",
        "SetOperator_Raw",
        "SetOperator_MA9",
        "QualityAware_SetOperator",
    ]

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Data collection containers
    rhythm_burden_rows = []
    patient_alert_stats = []
    paired_episode_records = []
    
    # Structure for threshold sweep: sweep_stats[tau][model]
    sweep_data = {tau: {m: {"ep6m_tot": 0, "ep6m_det": 0, "ep30s_tot": 0, "ep30s_det": 0, "alerts_tier_a": [], "burden_diffs": []} for m in model_names} for tau in thresholds}

    # Process all 128 recordings
    total_recs = len(df_meta)
    t_start = time.time()

    for idx, row in df_meta.iterrows():
        if idx < start_index:
            continue
        if idx % shard_count != shard_index:
            continue
        did = int(row["Data_ID"])
        rec_id = f"{did:03d}"
        subj_id = int(row["Subject_ID"])
        is_ann = bool(row["Annotated"])
        is_ablation = bool(row["Previous_AF_Ablation"])
        af_type = str(row["AF_Type"])
        is_post_blanking = bool(row["post_blanking"])

        rec_path = os.path.join(shdb_dir, rec_id)
        if not os.path.exists(f"{rec_path}.hea") or not os.path.exists(f"{rec_path}.dat"):
            raise FileNotFoundError(f"Missing .hea or .dat: {rec_id}")

        header = wfdb.rdheader(rec_path)
        fs = float(header.fs)
        sig_len = header.sig_len
        dur_hrs = (sig_len / fs) / 3600.0

        qrs = wfdb.rdann(rec_path, "qrs") if os.path.exists(f"{rec_path}.qrs") else None
        atr = wfdb.rdann(rec_path, "atr") if (is_ann and os.path.exists(f"{rec_path}.atr")) else None

        all_qrs = np.asarray(qrs.sample, dtype=int) if qrs else np.array([], dtype=int)

        # Ground truth episodes and sample mask
        gt_episodes = parse_atr_gold_standard_episodes(atr, sig_len=sig_len, fs=fs) if atr else []
        gt_af_episodes_30s = [ep for ep in gt_episodes if ep["is_af"] and ep["duration_sec"] >= 30.0]
        gt_af_episodes_6min = [ep for ep in gt_episodes if ep["is_af"] and ep["duration_sec"] >= 360.0]
        for episode_id, ep in enumerate(gt_af_episodes_30s):
            reference_rows.append(dict(record_id=rec_id, subject_id=subj_id, episode_id=episode_id,
                                       previous_af_ablation=is_ablation, post_blanking=is_post_blanking, **ep))

        gt_sample_mask = np.zeros(sig_len, dtype=bool)
        if atr:
            for ep in gt_episodes:
                if ep["is_af"]:
                    gt_sample_mask[ep["start_sample"] : ep["end_sample"]] = True
        
        gt_af_sec = float(np.sum(gt_sample_mask)) / fs
        gt_burden_pct = (gt_af_sec / (sig_len / fs)) * 100.0 if sig_len > 0 else 0.0

        # Segment into 30-second windows
        window_sec = 30.0
        win_samples = int(fs * window_sec)
        n_windows = int(np.ceil(sig_len / win_samples))
        window_durations = np.minimum(win_samples, sig_len - np.arange(n_windows) * win_samples) / fs
        gt_window_af_sec = np.array([gt_sample_mask[w*win_samples:min(sig_len,(w+1)*win_samples)].sum()/fs
                                     for w in range(n_windows)], dtype=np.float32)

        cache_path = prepared_dir / f"{rec_id}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            if str(cached["protocol_sha256"]) != protocol_hash:
                raise ValueError(f"Feature protocol changed: {cache_path}")
            if prepare_only:
                print(f"PREPARED {idx+1}/{total_recs} record={rec_id} cached",flush=True)
                continue
            kme_windows = cached["kme"]
            setop_rr_windows = cached["rr"]
            hrv_windows = cached["hrv"]
            sqi_scores = cached["sqi"]
            probs_rawecg = cached["rawecg"]
            probs_arnet2 = cached["arnet2"]
        else:
            stage_start = time.perf_counter()
            rec = wfdb.rdrecord(rec_path)
            sig_ecg = rec.p_signal[:,0].astype(np.float32)
            raw_windows = np.zeros((n_windows, 6000), dtype=np.float32)
            hrv_windows = np.zeros((n_windows, 8), dtype=np.float32)
            kme_windows = np.zeros((n_windows, 16, 128), dtype=np.float32)
            sqi_scores = np.zeros(n_windows, dtype=np.float32)
            setop_rr_windows = np.zeros((n_windows, 60), dtype=np.float32)
    
            payloads=((sig_ecg[w*win_samples:min((w+1)*win_samples,sig_len)],fs) for w in range(n_windows))
            feature_results = list(cpu_executor.map(prepare_window,payloads,chunksize=32))
            kme_windows = batch_phase_kme([result[5] for result in feature_results],anchors)
    
            for w in range(n_windows):
                s_idx = w * win_samples
                e_idx = min(s_idx + win_samples, sig_len)
                w_sig = sig_ecg[s_idx:e_idx]
                if len(w_sig) == 6000:
                    raw_windows[w] = w_sig
                else:
                    raw_windows[w] = np.pad(w_sig, (0, 6000 - len(w_sig)), mode="edge")
    
                _, setop_rr, hrv, sqi, _, _ = feature_results[w]
                sqi_scores[w] = sqi
                hrv_windows[w] = hrv
    
                # Match the training signal-only feature path for SetOperator.
                setop_rr_windows[w] = setop_rr
    
            # Batch Neural Inference
            print(f"FEATURE_TIMING record={rec_id} windows={n_windows} seconds={time.perf_counter()-stage_start:.2f}",flush=True)
            stage_start = time.perf_counter()
            prec = np.arange(n_windows, dtype=np.int32)
            succ = (n_windows - 1 - np.arange(n_windows)).astype(np.int32)
            glob = np.ones(n_windows, dtype=np.float32)
            ids = np.repeat(rec_id,n_windows)
    
            out_raw = infer_rawecg(
                x=tf.constant(raw_windows),
                prec_windows=tf.constant(prec),
                succ_windows=tf.constant(succ),
                glob_lab=tf.constant(glob),
                ids=tf.constant(ids, dtype=tf.string),
            )
            probs_rawecg = out_raw["probs"].numpy()
    
            # Author protocol: consecutive, non-overlapping 60-RR sequences.
            # Score on the common grid by overlap duration; unscored edge time remains unavailable.
            rr_count = (len(all_qrs)-1)//60
            if rr_count < 1:
                raise ValueError(f"Insufficient beat detections for ArNet2: {rec_id}")
            rr_sequence = (np.diff(all_qrs)/fs)[:rr_count*60].reshape(rr_count,60).astype(np.float32)
            out_rr = infer_arnet2(
                x=tf.constant(rr_sequence),
                prec_windows=tf.constant(np.arange(rr_count,dtype=np.int32)),
                glob_lab=tf.constant(np.ones(rr_count,dtype=np.float32)),
                ids=tf.constant(np.repeat(rec_id,rr_count), dtype=tf.string),
            )
            native_rr_probs = out_rr["probs"].numpy().ravel()
            rr_starts = all_qrs[np.arange(rr_count)*60]/fs
            rr_ends = all_qrs[(np.arange(rr_count)+1)*60]/fs
            window_starts = np.arange(n_windows)*30
            overlap = np.maximum(0,np.minimum(window_starts[:,None]+window_durations[:,None],rr_ends[None,:])
                                   -np.maximum(window_starts[:,None],rr_starts[None,:]))
            overlap_total = overlap.sum(1)
            probs_arnet2 = np.divide(overlap@native_rr_probs,overlap_total,
                                     out=np.full(n_windows,np.nan),where=overlap_total>0)
            print(f"BASELINE_TIMING record={rec_id} seconds={time.perf_counter()-stage_start:.2f}",flush=True)
    
            cache_temp = cache_path.with_name(f"{rec_id}.{os.getpid()}.tmp.npz")
            np.savez_compressed(cache_temp,kme=kme_windows,rr=setop_rr_windows,hrv=hrv_windows,
                                sqi=sqi_scores,rawecg=probs_rawecg,arnet2=probs_arnet2,
                                protocol_sha256=protocol_hash)
            os.replace(cache_temp,cache_path)
        if prepare_only:
            print(f"PREPARED {idx+1}/{total_recs} record={rec_id}",flush=True)
            continue

        with torch.no_grad():
            probs_setop = np.concatenate([
                setop_model.predict_proba(torch.from_numpy(kme_windows[s:s+1024]).to(device),
                                          torch.from_numpy(setop_rr_windows[s:s+1024]).to(device),
                                          torch.from_numpy(hrv_windows[s:s+1024]).to(device)).cpu().numpy().ravel()
                for s in range(0,n_windows,1024)])

        # Quality Abstention: Chow's Rule (SQI < 0.55 or high predictive entropy)
        abstain_mask = quality_mask(probs_setop,sqi_scores)
        probs_setop_qa = np.where(abstain_mask, np.nan, probs_setop)

        # Apply the same SQI threshold to all baselines.
        baseline_sqi_abstain = sqi_scores < 0.55
        probs_arnet2_sqi = np.where(baseline_sqi_abstain, np.nan, probs_arnet2)
        probs_rawecg_sqi = np.where(baseline_sqi_abstain, np.nan, probs_rawecg)

        # Causal Moving-Average Smoothing (k=9 windows)
        probs_arnet2_ma9 = causal_moving_average(probs_arnet2, kernel_size=9)
        probs_rawecg_ma9 = causal_moving_average(probs_rawecg, kernel_size=9)
        probs_setop_ma9 = causal_moving_average(probs_setop, kernel_size=9)
        probs_setop_qa_ma9 = causal_moving_average(probs_setop_qa, kernel_size=9)
        probs_setop_qa_ma9 = np.where(abstain_mask, np.nan, probs_setop_qa_ma9)

        model_probs = {
            "ArNet2_Raw": probs_arnet2,
            "ArNet2_MA9": probs_arnet2_ma9,
            "ArNet2_SQI": probs_arnet2_sqi,
            "RawECGNet_Raw": probs_rawecg,
            "RawECGNet_MA9": probs_rawecg_ma9,
            "RawECGNet_SQI": probs_rawecg_sqi,
            "SetOperator_Raw": probs_setop,
            "SetOperator_MA9": probs_setop_ma9,
            "QualityAware_SetOperator": probs_setop_qa_ma9,
        }
        np.savez_compressed(prediction_dir / f"{rec_id}.npz", **model_probs,
                            duration_sec=window_durations,gt_af_sec=gt_window_af_sec,
                            annotated=is_ann,sqi=sqi_scores,subject_id=subj_id,
                            checkpoint_sha256=checkpoint_hash,protocol_sha256=protocol_hash)

        # Negative Control classifications
        is_tier_a = (af_type == "non-AF") and is_ann  # Beat-annotated pure sinus (20 Holters, 480h)
        is_tier_b = (af_type == "non-AF") and (not is_ann)  # Clinical report only (13 Holters, 312h)
        is_pure_sinus = (af_type == "non-AF") or (is_ann and gt_burden_pct == 0.0)

        # ---------------------------------------------------------------------
        # Evaluation at Primary Operating Point (tau = 0.50)
        # ---------------------------------------------------------------------
        if is_ann:
            for m_name, m_p in model_probs.items():
                bin_mask = (m_p >= 0.50) & (~np.isnan(m_p))
                pred_eps_30s = extract_continuous_episodes(bin_mask, window_sec=30.0, min_duration_sec=30.0,record_duration_sec=sig_len/fs)
                pred_eps_6min = extract_continuous_episodes(bin_mask, window_sec=30.0, min_duration_sec=360.0,record_duration_sec=sig_len/fs)

                match_30s = match_episodes(gt_af_episodes_30s, pred_eps_30s, min_overlap_sec=15.0)
                match_6min = match_episodes(gt_af_episodes_6min, pred_eps_6min, min_overlap_sec=60.0)

                accepted = ~np.isnan(m_p)
                pred_af_sec = float(np.sum(window_durations[(m_p >= .5) & accepted]))
                eval_dur_sec = float(window_durations.sum())
                pred_burden_pct = pred_af_sec / eval_dur_sec * 100
                accepted_sec = float(window_durations[accepted].sum())

                rhythm_burden_rows.append({
                    "record_id": rec_id,
                    "subject_id": subj_id,
                    "previous_af_ablation": is_ablation,
                    "post_blanking": is_post_blanking,
                    "model": m_name,
                    "gt_burden_pct": round(gt_burden_pct, 2),
                    "pred_burden_pct": round(pred_burden_pct, 2),
                    "burden_estimand": "detected AF seconds / total recording seconds; abstention counted as no detection",
                    "accepted_seconds": accepted_sec,
                    "total_seconds": eval_dur_sec,
                    "accepted_gt_burden_pct": 100*gt_window_af_sec[accepted].sum()/accepted_sec if accepted_sec else np.nan,
                    "accepted_pred_burden_pct": 100*pred_af_sec/accepted_sec if accepted_sec else np.nan,
                    "burden_error_pts": round(abs(pred_burden_pct - gt_burden_pct), 2),
                    "episodes_ge_30s_total": len(gt_af_episodes_30s),
                    "episodes_ge_30s_detected": match_30s["detected_ref"],
                    "episodes_ge_6min_total": len(gt_af_episodes_6min),
                    "episodes_ge_6min_detected": match_6min["detected_ref"],
                    "false_episode_alerts": match_30s["false_pred"],
                    "abstention_rate_pct": round(100.0 * (1-accepted_sec/eval_dur_sec), 2),
                })

                # Build Paired Episode Predictions for Parquet Export
                matched_dict = {pair[0]: (pair[1], pair[2]) for pair in match_30s["matched_pairs"]}
                for ref_idx, ref_ep in enumerate(gt_af_episodes_30s):
                    is_det = ref_idx in matched_dict
                    p_idx, overlap = matched_dict[ref_idx] if is_det else (None, 0.0)
                    p_start = pred_eps_30s[p_idx]["start_sec"] if is_det else np.nan
                    p_end = pred_eps_30s[p_idx]["end_sec"] if is_det else np.nan
                    p_dur = pred_eps_30s[p_idx]["duration_sec"] if is_det else np.nan
                    jitter = (p_start - ref_ep["start_sec"]) if is_det else np.nan

                    paired_episode_records.append({
                        "record_id": rec_id,
                        "subject_id": subj_id,
                        "previous_af_ablation": is_ablation,
                        "post_blanking": is_post_blanking,
                        "ref_episode_id": ref_idx,
                        "ref_start_sec": ref_ep["start_sec"],
                        "ref_end_sec": ref_ep["end_sec"],
                        "ref_duration_sec": ref_ep["duration_sec"],
                        "ref_is_ge_30s": True,
                        "ref_is_ge_6min": (ref_ep["duration_sec"] >= 360.0),
                        "model": m_name,
                        "detected": is_det,
                        "detected_ge_6min": any(pair[0] == next((j for j, e in enumerate(gt_af_episodes_6min)
                                                                 if e["start_sec"] == ref_ep["start_sec"]), -1)
                                                    for pair in match_6min["matched_pairs"]),
                        "pred_start_sec": p_start,
                        "pred_end_sec": p_end,
                        "pred_duration_sec": p_dur,
                        "overlap_sec": overlap,
                        "temporal_jitter_sec": jitter,
                    })

        # False Alerts / 24h on Sinus Controls
        if is_pure_sinus:
            for m_name, m_p in model_probs.items():
                fp_mask = (m_p >= 0.50) & (~np.isnan(m_p))
                alert_dict = compute_clinical_false_alerts_per_24h(
                    fp_mask,
                    monitored_hours=dur_hrs,
                    window_sec=30.0,
                    refractory_period_sec=3600.0,
                )
                patient_alert_stats.append({
                    "record_id": rec_id,
                    "subject_id": subj_id,
                    "af_type": af_type,
                    "is_tier_a_annotated": is_tier_a,
                    "is_tier_b_report_only": is_tier_b,
                    "monitored_hours": dur_hrs,
                    "accepted_hours": float(window_durations[~np.isnan(m_p)].sum())/3600,
                    "model": m_name,
                    "raw_false_alarms": alert_dict["raw_fa"],
                    "raw_fa_per_24h": alert_dict["raw_fa_per_24h"],
                    "clinical_alerts_1h_refractory": alert_dict["clinical_alerts"],
                    "clinical_alerts_per_24h": alert_dict["alerts_per_24h"],
                    "clinical_alerts_per_24h_accepted": 24*alert_dict["clinical_alerts"]/(window_durations[~np.isnan(m_p)].sum()/3600)
                    if (~np.isnan(m_p)).any() else np.nan,
                    "zero_false_alert_patient": (alert_dict["clinical_alerts"] == 0),
                    "abstention_rate_pct": round(100.0 * float(np.mean(np.isnan(m_p))), 2),
                })

        # ---------------------------------------------------------------------
        # Threshold Sweep for Full FROC Curve (tau in [0.10, ..., 0.90])
        # ---------------------------------------------------------------------
        for tau in thresholds:
            for m_name, m_p in model_probs.items():
                b_mask = (m_p >= tau) & (~np.isnan(m_p))
                
                if is_ann:
                    p_eps_30s = extract_continuous_episodes(b_mask, window_sec=30.0, min_duration_sec=30.0,record_duration_sec=sig_len/fs)
                    p_eps_6m = extract_continuous_episodes(b_mask, window_sec=30.0, min_duration_sec=360.0,record_duration_sec=sig_len/fs)
                    
                    m30 = match_episodes(gt_af_episodes_30s, p_eps_30s, min_overlap_sec=15.0)
                    m6m = match_episodes(gt_af_episodes_6min, p_eps_6m, min_overlap_sec=60.0)

                    sweep_data[tau][m_name]["ep30s_tot"] += len(gt_af_episodes_30s)
                    sweep_data[tau][m_name]["ep30s_det"] += m30["detected_ref"]
                    sweep_data[tau][m_name]["ep6m_tot"] += len(gt_af_episodes_6min)
                    sweep_data[tau][m_name]["ep6m_det"] += m6m["detected_ref"]

                    p_sec = float(window_durations[(m_p >= tau) & ~np.isnan(m_p)].sum())
                    t_sec = float(window_durations.sum())
                    p_burd = (p_sec / t_sec) * 100.0 if t_sec > 0 else 0.0
                    sweep_data[tau][m_name]["burden_diffs"].append(abs(p_burd - gt_burden_pct))

                if is_tier_a:
                    al = compute_clinical_false_alerts_per_24h(b_mask, monitored_hours=dur_hrs, refractory_period_sec=3600.0)
                    sweep_data[tau][m_name]["alerts_tier_a"].append(al["alerts_per_24h"])

        if (idx + 1) % 10 == 0 or (idx + 1) == total_recs:
            elapsed = time.time() - t_start
            pace = elapsed / (idx + 1)
            eta = pace * (total_recs - (idx + 1))
            print(f"  Processed [{idx+1:3d}/{total_recs}] Holters ({dur_hrs:.1f}h) - {rec_id} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")

    # -------------------------------------------------------------------------
    # 4. SAVE AUDITED RESULTS & COMPUTATIONS
    # -------------------------------------------------------------------------
    cpu_executor.shutdown()
    if prepare_only:
        print("PREPARATION_COMPLETE",flush=True)
        return
    df_burden = pd.DataFrame(rhythm_burden_rows)
    df_alerts = pd.DataFrame(patient_alert_stats)
    df_paired_episodes = pd.DataFrame(paired_episode_records)
    pd.DataFrame(reference_rows).to_csv(Path(output_dir) / "reference_episodes.csv",index=False)

    burden_csv_path = os.path.join(output_dir, "rhythm_burden_comparison.csv")
    df_burden.to_csv(burden_csv_path, index=False)
    print(f"Saved: {burden_csv_path}")

    alerts_csv_path = os.path.join(output_dir, "patient_level_alert_metrics.csv")
    df_alerts.to_csv(alerts_csv_path, index=False)
    print(f"Saved: {alerts_csv_path}")

    # Parquet export
    parquet_path = os.path.join(output_dir, "paired_episode_predictions.parquet")
    df_paired_episodes.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"Saved: {parquet_path} ({len(df_paired_episodes)} rows)")

    # Threshold Sweep FROC Data
    froc_rows = []
    for tau in thresholds:
        for m_name in model_names:
            s = sweep_data[tau][m_name]
            sens_6m = (s["ep6m_det"] / s["ep6m_tot"] * 100.0) if s["ep6m_tot"] > 0 else 0.0
            sens_30s = (s["ep30s_det"] / s["ep30s_tot"] * 100.0) if s["ep30s_tot"] > 0 else 0.0
            mean_fa = float(np.mean(s["alerts_tier_a"])) if len(s["alerts_tier_a"]) > 0 else 0.0
            spec_tier_a = float(np.mean(np.array(s["alerts_tier_a"]) == 0.0) * 100.0) if len(s["alerts_tier_a"]) > 0 else 0.0
            mae = float(np.mean(s["burden_diffs"])) if len(s["burden_diffs"]) > 0 else 0.0

            froc_rows.append({
                "model": m_name,
                "threshold": tau,
                "sensitivity_ge_6min_pct": round(sens_6m, 2),
                "sensitivity_ge_30s_pct": round(sens_30s, 2),
                "false_alerts_per_24h_tier_a": round(mean_fa, 3),
                "patient_specificity_tier_a_pct": round(spec_tier_a, 2),
                "burden_mae_pts": round(mae, 2),
            })

    df_froc = pd.DataFrame(froc_rows)
    froc_csv_path = os.path.join(output_dir, "froc_curve_data.csv")
    df_froc.to_csv(froc_csv_path, index=False)
    print(f"Saved: {froc_csv_path}")

    # Primary Model Comparison Summary Table (tau = 0.50)
    print("\n=== SUMMARY OF EPISODE SENSITIVITY & BURDEN MAE ACROSS MODELS (tau = 0.50) ===")
    summary_rows = []
    for m in model_names:
        sub = df_burden[df_burden["model"] == m]
        tot_30s = sub["episodes_ge_30s_total"].sum()
        det_30s = sub["episodes_ge_30s_detected"].sum()
        sens_30s = (det_30s / tot_30s * 100.0) if tot_30s > 0 else 0.0

        tot_6m = sub["episodes_ge_6min_total"].sum()
        det_6m = sub["episodes_ge_6min_detected"].sum()
        sens_6m = (det_6m / tot_6m * 100.0) if tot_6m > 0 else 0.0

        b_metrics = compute_burden_metrics(sub["gt_burden_pct"].values, sub["pred_burden_pct"].values)

        sub_a = df_alerts[(df_alerts["model"] == m) & (df_alerts["is_tier_a_annotated"] == True)]
        mean_alerts_24h = 24*sub_a["clinical_alerts_1h_refractory"].sum()/sub_a["monitored_hours"].sum()
        patient_counts=sub_a.groupby('subject_id').clinical_alerts_1h_refractory.sum()
        patient_spec = 100*patient_counts.eq(0).mean()
        recording_spec=100*sub_a.zero_false_alert_patient.mean()
        mean_abstain = 100*(1-sub.accepted_seconds.sum()/sub.total_seconds.sum())

        summary_rows.append({
            "Model Architecture": m,
            "Episodes >=30s Detected": f"{det_30s}/{tot_30s}",
            "Sensitivity (>=30s) %": round(sens_30s, 2),
            "Episodes >=6min Detected": f"{det_6m}/{tot_6m}",
            "Sensitivity (>=6min) %": round(sens_6m, 2),
            "Burden MAE (% pts)": b_metrics["mae"],
            "Pearson r": b_metrics["pearson_r"],
            "Spearman rho": b_metrics["spearman_rho"],
            "False Alerts / 24h (Tier A)": round(mean_alerts_24h, 2),
            "Patient Specificity % (Tier A)": round(patient_spec, 2),
            "Recording Specificity % (Tier A)": round(recording_spec, 2),
            "Tier A Patients": len(patient_counts),
            "Tier A Recordings": len(sub_a),
            "Diagnostic Abstention %": round(mean_abstain, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    summary_csv_path = os.path.join(output_dir, "model_comparison_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")

    # Use recording-group OOF predictions; daily-record prefixes do not verify patient independence.
    df_sweep_rc = pd.read_csv(Path(output_dir) / "noise_risk_coverage_sweep.csv")
    df_sweep_rc[df_sweep_rc.quality_threshold == .55].to_csv(
        Path(output_dir) / "risk_coverage_by_quality.csv",index=False)
    evaluation_sources=[Path(__file__),ROOT_DIR/'scripts/hrs2027_protocol.py',
                        ROOT_DIR/'scripts/hrs2027_gpu_features.py',ROOT_DIR/'scripts/hrs2027_cpu_features.py',
                        ROOT_DIR/'prototype/metrics_engine.py']
    provenance=dict(checkpoint_sha256=checkpoint_hash,feature_protocol_sha256=protocol_hash,
                    tensorflow_version=tf.__version__,torch_version=torch.__version__,
                    cudnn_version=torch.backends.cudnn.version(),device=torch.cuda.get_device_name(),
                    n_recordings=len(df_meta),n_annotated_recordings=int(df_meta.Annotated.sum()),
                    metadata_sha256=hashlib.file_digest(open(shdb_metadata,'rb'),'sha256').hexdigest(),
                    code_sha256={str(p.relative_to(ROOT_DIR)):hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
                                 for p in evaluation_sources})
    (Path(output_dir)/'evaluation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

    print("\n=== Corrected Benchmark Evaluation Successfully Completed! ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--shard-index',type=int,default=0)
    parser.add_argument('--shard-count',type=int,default=1)
    parser.add_argument('--start-index',type=int,default=0)
    args = parser.parse_args()
    run_corrected_benchmark(output_dir=args.output_dir,checkpoint=args.checkpoint,prepare_only=args.prepare_only,
                            shard_index=args.shard_index,shard_count=args.shard_count,start_index=args.start_index)
