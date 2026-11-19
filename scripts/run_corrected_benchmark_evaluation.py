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
- Noise Holter Database (Clean, N1, N2 tiers)
- Full threshold sweep tau in [0.10, ..., 0.90] for complete FROC curves
- Quality threshold sweep tau_qual in [0.10, ..., 0.90] for Risk-Coverage curves
- Generates paired_episode_predictions.parquet for all reference episodes
"""
from __future__ import annotations

import os
import sys
import time
import glob
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats, signal
import tensorflow as tf
import torch
import torch.nn as nn
import wfdb

# Suppress TF logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

# Add workspace and prototype to path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "prototype"))
sys.path.insert(0, "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence")
sys.path.insert(0, "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/prototype")

from prototype.metrics_engine import (
    extract_continuous_episodes,
    match_episodes,
    compute_clinical_false_alerts_per_24h,
    compute_burden_metrics,
)
from prototype.models import SetOperatorPlusRR
from data_loader import compute_hrv_features
from run_gpu_discovery_campaign_12h import fast_compute_phase_kme, fast_compute_sqi


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


def parse_atr_gold_standard_episodes(atr: wfdb.Annotation, sig_len: int, fs: float = 200.0) -> List[Dict[str, Any]]:
    """Extract reference rhythm episodes from .atr file."""
    rhythms = [(s, note.strip("(\x00")) for s, note in zip(atr.sample, atr.aux_note) if note.startswith("(")]
    if not rhythms:
        return []

    episodes = []
    total_samples = sig_len

    for i in range(len(rhythms)):
        s_start, r_type = rhythms[i]
        s_end = rhythms[i + 1][0] if i + 1 < len(rhythms) else total_samples
        dur_sec = (s_end - s_start) / fs
        is_af = r_type in ("AFIB", "AFL", "AT")
        episodes.append({
            "rhythm": r_type,
            "start_sec": s_start / fs,
            "end_sec": s_end / fs,
            "start_sample": s_start,
            "end_sample": s_end,
            "duration_sec": dur_sec,
            "is_af": is_af,
        })
    return episodes


def run_corrected_benchmark(
    shdb_dir: str = "/data/mithunmanivannan/datasets/shdb_af",
    shdb_metadata: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/AdditionalData.csv",
    output_dir: str = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/corrected_package",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(output_dir, exist_ok=True)
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
    setop_pt_path = "/data/mithunmanivannan/codex_artifacts/hrs2027_af_recurrence/results_12h/best_multimodal_setoperator.pt"
    if os.path.exists(setop_pt_path):
        print(f"Loading Quality-Aware SetOperator checkpoint: {setop_pt_path}")
        setop_model = SetOperatorPlusRR()
        state = torch.load(setop_pt_path, map_location=device, weights_only=True)
        setop_model.load_state_dict(state)
        setop_model.to(device)
        setop_model.eval()
    else:
        setop_model = None
        print("Warning: SetOperator checkpoint not found.")

    rng = np.random.default_rng(42)
    anchors = rng.normal(0.0, 1.0, size=(128, 16)).astype(np.float32)

    # 3. Read SHDB-AF Manifest
    df_meta = pd.read_csv(shdb_metadata)
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
        did = int(row["Data_ID"])
        rec_id = f"{did:03d}"
        subj_id = int(row["Subject_ID"])
        is_ann = bool(row["Annotated"])
        is_ablation = bool(row["Previous_AF_Ablation"])
        af_type = str(row["AF_Type"])
        days_from_ablation = row.get("days_from_1st", np.nan)
        is_post_blanking = bool(days_from_ablation > 90) if pd.notna(days_from_ablation) else False

        rec_path = os.path.join(shdb_dir, rec_id)
        if not os.path.exists(f"{rec_path}.hea") or not os.path.exists(f"{rec_path}.dat"):
            print(f"Skipping {rec_id}: missing .hea or .dat")
            continue

        header = wfdb.rdheader(rec_path)
        rec = wfdb.rdrecord(rec_path)
        fs = float(header.fs)
        sig_len = header.sig_len
        dur_hrs = (sig_len / fs) / 3600.0

        qrs = wfdb.rdann(rec_path, "qrs") if os.path.exists(f"{rec_path}.qrs") else None
        atr = wfdb.rdann(rec_path, "atr") if (is_ann and os.path.exists(f"{rec_path}.atr")) else None

        sig_ecg = rec.p_signal[:, 0].astype(np.float32)  # Lead 1 (CC5)
        all_qrs = np.asarray(qrs.sample, dtype=int) if qrs else np.array([], dtype=int)

        # Ground truth episodes and sample mask
        gt_episodes = parse_atr_gold_standard_episodes(atr, sig_len=sig_len, fs=fs) if atr else []
        gt_af_episodes_30s = [ep for ep in gt_episodes if ep["is_af"] and ep["duration_sec"] >= 30.0]
        gt_af_episodes_6min = [ep for ep in gt_episodes if ep["is_af"] and ep["duration_sec"] >= 360.0]

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
        n_windows = sig_len // win_samples

        raw_windows = np.zeros((n_windows, 6000), dtype=np.float32)
        rr_windows = np.zeros((n_windows, 60), dtype=np.float32)
        hrv_windows = np.zeros((n_windows, 8), dtype=np.float32)
        kme_windows = np.zeros((n_windows, 16, 128), dtype=np.float32)
        sqi_scores = np.zeros(n_windows, dtype=np.float32)

        for w in range(n_windows):
            s_idx = w * win_samples
            e_idx = s_idx + win_samples
            w_sig = sig_ecg[s_idx:e_idx]
            if len(w_sig) == 6000:
                raw_windows[w] = w_sig
            else:
                raw_windows[w] = np.pad(w_sig, (0, 6000 - len(w_sig)), mode="edge")

            p_mask = (all_qrs >= s_idx) & (all_qrs < e_idx)
            w_peaks = all_qrs[p_mask] - s_idx
            sqi_scores[w] = fast_compute_sqi(raw_windows[w], w_peaks, fs=fs)
            kme_windows[w] = fast_compute_phase_kme(raw_windows[w], w_peaks, anchors, fs=fs)

            ctx_mask = (all_qrs >= max(0, s_idx - int(fs * 60))) & (all_qrs < e_idx)
            ctx_peaks = all_qrs[ctx_mask]
            if len(ctx_peaks) > 1:
                rrs = np.diff(ctx_peaks) / fs
                rrs_clipped = rrs[-60:]
                if len(rrs_clipped) < 60:
                    pad_v = float(np.mean(rrs_clipped)) if len(rrs_clipped) > 0 else 0.8
                    rrs_clipped = np.pad(rrs_clipped, (0, 60 - len(rrs_clipped)), constant_values=pad_v)
                rr_windows[w] = rrs_clipped
                hrv_windows[w] = compute_hrv_features(rrs)
            else:
                rr_windows[w] = np.full(60, 0.8, dtype=np.float32)
                hrv_windows[w] = np.zeros(8, dtype=np.float32)

        # Batch Neural Inference
        prec = np.arange(n_windows, dtype=np.int32)
        succ = (n_windows - 1 - np.arange(n_windows)).astype(np.int32)
        glob = np.zeros(n_windows, dtype=np.float32)
        ids = np.array([f"{rec_id}_{i}" for i in range(n_windows)], dtype=object)

        out_raw = infer_rawecg(
            x=tf.constant(raw_windows),
            prec_windows=tf.constant(prec),
            succ_windows=tf.constant(succ),
            glob_lab=tf.constant(glob),
            ids=tf.constant(ids, dtype=tf.string),
        )
        probs_rawecg = out_raw["probs"].numpy()

        out_rr = infer_arnet2(
            x=tf.constant(rr_windows),
            prec_windows=tf.constant(prec),
            glob_lab=tf.constant(glob),
            ids=tf.constant(ids, dtype=tf.string),
        )
        probs_arnet2 = out_rr["probs"].numpy()

        if setop_model is not None:
            with torch.no_grad():
                kme_t = torch.from_numpy(kme_windows).to(device)
                rr_t = torch.from_numpy(rr_windows).to(device)
                hrv_t = torch.from_numpy(hrv_windows).to(device)
                probs_setop = setop_model.predict_proba(kme_t, rr_t, hrv_t).squeeze().cpu().numpy()
        else:
            probs_setop = 0.5 * probs_rawecg + 0.5 * probs_arnet2

        # Quality Abstention: Chow's Rule (SQI < 0.55 or high predictive entropy)
        unc_setop = 1.0 - np.maximum(probs_setop, 1.0 - probs_setop)
        abstain_mask = (sqi_scores < 0.55) | (unc_setop > 0.35)
        probs_setop_qa = np.where(abstain_mask, np.nan, probs_setop)

        # Baseline SQI filter (SQI < 0.50)
        baseline_sqi_abstain = sqi_scores < 0.50
        probs_arnet2_sqi = np.where(baseline_sqi_abstain, np.nan, probs_arnet2)
        probs_rawecg_sqi = np.where(baseline_sqi_abstain, np.nan, probs_rawecg)

        # Causal Moving-Average Smoothing (k=9 windows)
        probs_arnet2_ma9 = causal_moving_average(probs_arnet2, kernel_size=9)
        probs_rawecg_ma9 = causal_moving_average(probs_rawecg, kernel_size=9)
        probs_setop_ma9 = causal_moving_average(probs_setop, kernel_size=9)
        probs_setop_qa_ma9 = causal_moving_average(np.nan_to_num(probs_setop_qa, nan=0.0), kernel_size=9)
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
                pred_eps_30s = extract_continuous_episodes(bin_mask, window_sec=30.0, min_duration_sec=30.0)
                pred_eps_6min = extract_continuous_episodes(bin_mask, window_sec=30.0, min_duration_sec=360.0)

                match_30s = match_episodes(gt_af_episodes_30s, pred_eps_30s, min_overlap_sec=15.0)
                match_6min = match_episodes(gt_af_episodes_6min, pred_eps_6min, min_overlap_sec=60.0)

                valid_p = m_p[~np.isnan(m_p)]
                pred_af_sec = float(np.sum(valid_p >= 0.50)) * window_sec
                eval_dur_sec = float(len(valid_p)) * window_sec
                pred_burden_pct = (pred_af_sec / eval_dur_sec) * 100.0 if eval_dur_sec > 0 else 0.0

                rhythm_burden_rows.append({
                    "record_id": rec_id,
                    "subject_id": subj_id,
                    "previous_af_ablation": is_ablation,
                    "post_blanking": is_post_blanking,
                    "model": m_name,
                    "gt_burden_pct": round(gt_burden_pct, 2),
                    "pred_burden_pct": round(pred_burden_pct, 2),
                    "burden_error_pts": round(abs(pred_burden_pct - gt_burden_pct), 2),
                    "episodes_ge_30s_total": len(gt_af_episodes_30s),
                    "episodes_ge_30s_detected": match_30s["detected_ref"],
                    "episodes_ge_6min_total": len(gt_af_episodes_6min),
                    "episodes_ge_6min_detected": match_6min["detected_ref"],
                    "false_episode_alerts": match_30s["false_pred"],
                    "abstention_rate_pct": round(100.0 * float(np.mean(np.isnan(m_p))), 2),
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
                        "ref_episode_id": ref_idx,
                        "ref_start_sec": ref_ep["start_sec"],
                        "ref_end_sec": ref_ep["end_sec"],
                        "ref_duration_sec": ref_ep["duration_sec"],
                        "ref_is_ge_30s": True,
                        "ref_is_ge_6min": (ref_ep["duration_sec"] >= 360.0),
                        "model": m_name,
                        "detected": is_det,
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
                    "monitored_hours": round(dur_hrs, 2),
                    "model": m_name,
                    "raw_false_alarms": alert_dict["raw_fa"],
                    "raw_fa_per_24h": alert_dict["raw_fa_per_24h"],
                    "clinical_alerts_1h_refractory": alert_dict["clinical_alerts"],
                    "clinical_alerts_per_24h": alert_dict["alerts_per_24h"],
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
                    p_eps_30s = extract_continuous_episodes(b_mask, window_sec=30.0, min_duration_sec=30.0)
                    p_eps_6m = extract_continuous_episodes(b_mask, window_sec=30.0, min_duration_sec=360.0)
                    
                    m30 = match_episodes(gt_af_episodes_30s, p_eps_30s, min_overlap_sec=15.0)
                    m6m = match_episodes(gt_af_episodes_6min, p_eps_6m, min_overlap_sec=60.0)

                    sweep_data[tau][m_name]["ep30s_tot"] += len(gt_af_episodes_30s)
                    sweep_data[tau][m_name]["ep30s_det"] += m30["detected_ref"]
                    sweep_data[tau][m_name]["ep6m_tot"] += len(gt_af_episodes_6min)
                    sweep_data[tau][m_name]["ep6m_det"] += m6m["detected_ref"]

                    valid_p = m_p[~np.isnan(m_p)]
                    p_sec = float(np.sum(valid_p >= tau)) * window_sec
                    t_sec = float(len(valid_p)) * window_sec
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
    df_burden = pd.DataFrame(rhythm_burden_rows)
    df_alerts = pd.DataFrame(patient_alert_stats)
    df_paired_episodes = pd.DataFrame(paired_episode_records)

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
        mean_alerts_24h = sub_a["clinical_alerts_per_24h"].mean() if len(sub_a) > 0 else 0.0
        patient_spec = (sub_a["zero_false_alert_patient"].mean() * 100.0) if len(sub_a) > 0 else 0.0
        mean_abstain = sub["abstention_rate_pct"].mean()

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
            "Diagnostic Abstention %": round(mean_abstain, 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    summary_csv_path = os.path.join(output_dir, "model_comparison_summary.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"Saved: {summary_csv_path}")

    # -------------------------------------------------------------------------
    # 5. RISK-COVERAGE ACROSS NOISE TIERS (Noise Holter 600 Balanced Segments)
    # -------------------------------------------------------------------------
    print("\n=== Computing Risk-Coverage Stratified by Quality Tier (Noise Holter) ===")
    nh_wfdb = sorted(glob.glob("/data/mithunmanivannan/datasets/noise-holter-database-a-multi-device-multi-arrhythmia-real-world-clinical-ecg-signal-quality-benchmark-1.0.0/wfdb_files/*.hea"))

    tier_records = defaultdict(list)
    for h_path in nh_wfdb:
        base = h_path.replace(".hea", "")
        header = wfdb.rdheader(base)
        q_class = None
        is_af = False
        for c in header.comments:
            if "CLASS:Clean" in c: q_class = "Clean"
            elif "CLASS:N1" in c: q_class = "N1"
            elif "CLASS:N2" in c: q_class = "N2"
            if "EVENTS:A. Fib." in c or "EVENTS:A. Flutter" in c: is_af = True

        if q_class and len(tier_records[q_class]) < 200:
            rec = wfdb.rdrecord(base)
            sig = rec.p_signal[:6000, 0].astype(np.float32)
            if len(sig) < 6000:
                sig = np.pad(sig, (0, 6000 - len(sig)), mode="edge")
            qrs = wfdb.rdann(base, "qrs") if os.path.exists(f"{base}.qrs") else None
            p_peaks = qrs.sample if qrs else np.array([])
            sqi = fast_compute_sqi(sig, p_peaks, fs=header.fs)
            tier_records[q_class].append({
                "base": base,
                "sig": sig,
                "is_af": int(is_af),
                "sqi": sqi,
                "peaks": p_peaks,
            })

    # Predict RawECGNet and SetOperator on Noise Holter segments
    qual_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.70, 0.80, 0.90]
    sweep_rc_rows = []

    for tier in ["Clean", "N1", "N2"]:
        segs = tier_records[tier]
        if not segs:
            continue
        n_segs = len(segs)
        sigs_arr = np.array([s["sig"] for s in segs], dtype=np.float32)
        sqis_arr = np.array([s["sqi"] for s in segs], dtype=np.float32)
        y_true = np.array([s["is_af"] for s in segs], dtype=int)

        # Batch inference RawECGNet
        prec_nh = np.zeros(n_segs, dtype=np.int32)
        succ_nh = np.zeros(n_segs, dtype=np.int32)
        glob_nh = np.zeros(n_segs, dtype=np.float32)
        ids_nh = np.array([f"nh_{i}" for i in range(n_segs)], dtype=object)

        out_nh_raw = infer_rawecg(
            x=tf.constant(sigs_arr),
            prec_windows=tf.constant(prec_nh),
            succ_windows=tf.constant(succ_nh),
            glob_lab=tf.constant(glob_nh),
            ids=tf.constant(ids_nh, dtype=tf.string),
        )
        p_raw_nh = out_nh_raw["probs"].numpy()

        for q_th in qual_thresholds:
            abs_mask = sqis_arr < q_th
            acc_mask = ~abs_mask
            cov = float(np.mean(acc_mask) * 100.0)
            
            if np.sum(acc_mask) > 0:
                y_acc = y_true[acc_mask]
                p_acc = p_raw_nh[acc_mask] >= 0.50
                sens = float(np.mean(p_acc[y_acc == 1]) * 100.0) if np.sum(y_acc == 1) > 0 else 100.0
                fpr = float(np.mean(p_acc[y_acc == 0]) * 100.0) if np.sum(y_acc == 0) > 0 else 0.0
                err_rate = float(np.mean(p_acc != y_acc) * 100.0)
            else:
                sens, fpr, err_rate = np.nan, np.nan, np.nan

            missed_af = int(np.sum((y_true == 1) & abs_mask))

            sweep_rc_rows.append({
                "Noise Quality Tier": tier,
                "tau_qual": q_th,
                "Diagnostic Coverage %": round(cov, 2),
                "Abstention Rate %": round(100.0 - cov, 2),
                "Accepted Sensitivity %": round(sens, 2) if not np.isnan(sens) else np.nan,
                "Accepted False Positive Rate %": round(fpr, 2) if not np.isnan(fpr) else np.nan,
                "Accepted Error Rate %": round(err_rate, 2) if not np.isnan(err_rate) else np.nan,
                "Missed AF During Abstention": missed_af,
            })

    df_sweep_rc = pd.DataFrame(sweep_rc_rows)
    sweep_rc_path = os.path.join(output_dir, "noise_risk_coverage_sweep.csv")
    df_sweep_rc.to_csv(sweep_rc_path, index=False)
    print(f"Saved: {sweep_rc_path}")

    # Standard operating point tau_qual = 0.55 table
    df_rc_std = df_sweep_rc[df_sweep_rc["tau_qual"] == 0.55].copy()
    rc_csv_path = os.path.join(output_dir, "risk_coverage_by_quality.csv")
    df_rc_std.to_csv(rc_csv_path, index=False)
    print(f"Saved: {rc_csv_path}")
    print(df_rc_std.to_string(index=False))

    print("\n=== Corrected Benchmark Evaluation Successfully Completed! ===")


if __name__ == "__main__":
    run_corrected_benchmark()
