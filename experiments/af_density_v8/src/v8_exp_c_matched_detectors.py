"""Experiment C (A100 GPU): Matched Detector Benchmark on Complete Endpoints Across R0, R2, R3.

Evaluates identical comparator rosters on:
  1) All 182 full recordings (LTAFDB 84 + SHDB-AF 98) for Fixed Oracles (10s, 30s, 60s, 60-beat) + RR-GBDT (Ref & AutoQRS)
  2) Exact Matched-720 gaps (identical gaps) + Exact Matched-36 x 6h timelines (ALL comparators on ALL 36 records,
     including BOTH ECGFounder_ZNorm_10s AND ECGFounder_Raw_10s cleanly!)
Evaluates against R0_Native, R2_VT_SBR60_Unknown, and R3_gap30s:
  - Gap Precision, Recall, F1, IoU, Start/End Boundary Errors, Splitting/Merging Counts, False-Gap Rate
  - 3x3 Rhythm State Confusion Matrix (ZERO_AF, MIXED, ALL_AF) & Coverage
  - Charitos Density & Butkuviene AF Aggregation MAE/MedAE with B=2,000 patient bootstrap 95% CIs
  - longest_share Concealment Threshold Sensitivity across {0.15, 0.20, 0.30, 0.40}.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wfdb

sys.path.insert(0, "experiments/af_density")
from rr_features import features as rr_features, layout as rr_layout
sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
from foundation_adapters import get_model_adapter
from waveform_loader import WaveformLoader
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import clip_ivs_to_known, intervals_overlap_mask, gap_bin_label, gpu_resample_batch
from v8_m2_qrs_and_oracles import detect_qrs_one_record
from v8_m3_ecgfounder_matched_eval import fit_ecgfounder_head, gpu_resample_and_optionally_znorm
from v8_exp_a_reference_layers import metrics_from_intervals, project_known_horizon, apply_r3_rule
from v8_exp_b_oracles_and_bounds import fixed_oracle_grid, fixed_oracle_60beat, interval_intersection_len, interval_union_len

OUT_RES = Path("experiments/af_density_v8/results")
OUT_FIG = Path("experiments/af_density_v8/figures")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
RR_BASE = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/rr_baseline_v1")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
NON_BEAT_LTAF = {'+', '"', '~', '|', 's', 't', 'p', 'u', '='}


def eval_beats_edge_assigned_rr(q: np.ndarray, fs: float, first_k: float, raw_T: float, rr_model, thr: float) -> List[Tuple[float, float]]:
    """Run frozen RR-GBDT with identical edge-interval assignment as Fixed Oracle_60BeatRR."""
    q = np.sort(q)
    q = q[np.r_[True, np.diff(q) > 0]]
    q = q[q >= first_k]
    if len(q) < 62:
        return []
    rr_ms = np.diff(q).astype(float) * 1000.0 / fs
    starts, assigned, ends = rr_layout(len(rr_ms), 0)
    if len(starts) == 0:
        return []
    X, valid = rr_features(rr_ms[starts[:, None] + np.arange(60)])
    scores = np.full(len(starts), np.nan)
    if valid.any():
        scores[valid] = rr_model.predict_proba(X[valid].astype(np.float32))[:, 1]
    ivs = []
    for i, (lo, hi) in enumerate(zip(assigned, ends)):
        if valid[i] and scores[i] >= thr:
            t0 = (first_k / fs) if i == 0 else float(q[lo]) / fs
            t1 = raw_T if i == len(starts) - 1 else float(q[hi]) / fs
            if t1 > t0:
                ivs.append((t0, t1))
    return merge_intervals(ivs)


def gap_detection_endpoints(ref_af: List[Tuple[float, float]], pred_af: List[Tuple[float, float]], T: float) -> Dict[str, Any]:
    """Compute true vs predicted INTERNAL non-AF gap precision, recall, F1, IoU, boundary errors, splits, merges, false-gap rate."""
    def internal_gaps(af_ivs):
        af_s = merge_intervals([(max(0.0, a), min(T, b)) for a, b in af_ivs])
        gs = []
        for i in range(len(af_s) - 1):
            ga, gb = af_s[i][1], af_s[i + 1][0]
            if gb > ga + 1e-6:
                gs.append((ga, gb))
        return gs

    tgaps = internal_gaps(ref_af)
    pgaps = internal_gaps(pred_af)
    af_total_h = sum(b - a for a, b in ref_af) / 3600.0

    # Overlap matching (any overlap >= 0.5s or >= 10% of smaller interval)
    matched_t = 0
    matched_p = set()
    ious = []
    start_errs, abs_start_errs = [], []
    end_errs, abs_end_errs = [], []
    n_splits = 0
    for ta, tb in tgaps:
        ov_ps = []
        for pj, (pa, pb) in enumerate(pgaps):
            ov = min(tb, pb) - max(ta, pa)
            if ov > 0.5:
                ov_ps.append((pj, pa, pb, ov))
        if ov_ps:
            matched_t += 1
            if len(ov_ps) >= 2:
                n_splits += 1
            for pj, _, _, _ in ov_ps:
                matched_p.add(pj)
            # Best overlapping predicted gap for IoU / boundary alignment
            pj_best, pa_b, pb_b, _ = max(ov_ps, key=lambda x: x[3])
            inter = min(tb, pb_b) - max(ta, pa_b)
            union = max(tb, pb_b) - min(ta, pa_b)
            ious.append(inter / union if union > 0 else 0.0)
            start_errs.append(pa_b - ta)
            abs_start_errs.append(abs(pa_b - ta))
            end_errs.append(pb_b - tb)
            abs_end_errs.append(abs(pb_b - tb))

    # Count false gaps (predicted internal gaps strictly inside true AF with zero true non-AF overlap)
    true_nonaf = subtract_exclusions([(0.0, T)], ref_af)
    n_false_gaps = 0
    n_merges = 0
    for pj, (pa, pb) in enumerate(pgaps):
        ov_true_nonaf = sum(max(0.0, min(pb, nb) - max(pa, na)) for na, nb in true_nonaf)
        if ov_true_nonaf <= 1e-6:
            n_false_gaps += 1
        ov_tg = sum(1 for ta, tb in tgaps if min(tb, pb) - max(ta, pa) > 0.5)
        if ov_tg >= 2:
            n_merges += 1

    prec = float(len(matched_p) / len(pgaps)) if pgaps else np.nan
    rec = float(matched_t / len(tgaps)) if tgaps else np.nan
    f1 = (2.0 * prec * rec / (prec + rec)) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else (0.0 if (tgaps or pgaps) else np.nan)
    return {
        "n_true_gaps": len(tgaps), "n_pred_gaps": len(pgaps),
        "n_tp_true_gaps": matched_t, "n_tp_pred_gaps": len(matched_p),
        "n_false_gaps": n_false_gaps, "false_gaps_per_af_h": (n_false_gaps / af_total_h) if af_total_h > 0.05 else np.nan,
        "gap_precision": prec, "gap_recall": rec, "gap_f1": f1,
        "mean_gap_iou": float(np.mean(ious)) if ious else np.nan,
        "med_abs_start_err_s": float(np.median(abs_start_errs)) if abs_start_errs else np.nan,
        "med_signed_start_err_s": float(np.median(start_errs)) if start_errs else np.nan,
        "med_abs_end_err_s": float(np.median(abs_end_errs)) if abs_end_errs else np.nan,
        "n_true_gaps_split": n_splits, "n_pred_gaps_merging_ge2": n_merges,
    }


def eval_one_full_rec_cpu(args):
    ds, rid, pid, raw_T, fs, known_ivs, af_r0, ex_r2, rr_model_path, rr_thr = args
    rr_model = joblib.load(rr_model_path)
    _, _, _, q_auto, q_ref, _, _, _, sens_q, ppv_q = detect_qrs_one_record((ds, rid))
    first_k = float(known_ivs[0][0]) * fs
    rr_ref_raw = eval_beats_edge_assigned_rr(q_ref, fs, first_k, raw_T, rr_model, rr_thr)
    rr_auto_raw = eval_beats_edge_assigned_rr(q_auto, fs, first_k, raw_T, rr_model, rr_thr)
    _, _, _, _ = None, None, None, None
    ora60_raw, _, _, _ = fixed_oracle_60beat(q_ref, fs, first_k, raw_T, af_r0, known_ivs)

    out_rows = []
    for ref_lbl, ref_ivs, ex_ivs in [
        ("R0_Native", af_r0, []),
        ("R2_VT_SBR60_Unknown", af_r0, ex_r2),
        ("R3_gap30s", apply_r3_rule(known_ivs, af_r0, sum(b - a for a, b in known_ivs), bridge_gap_lt_s=30.0), []),
    ]:
        eff_known = subtract_exclusions(known_ivs, ex_ivs) if ex_ivs else known_ivs
        tot_k = sum(b - a for a, b in eff_known)
        if tot_k <= 60.0:
            continue
        if ref_lbl == "R3_gap30s":
            # Map known intervals linearly to [0, tot_k] cleanly matching apply_r3_rule
            rk, Tk = ref_ivs, sum(b - a for a, b in merge_intervals(known_ivs))
        else:
            rk, Tk = clip_ivs_to_known(subtract_exclusions(ref_ivs, ex_ivs) if ex_ivs else ref_ivs, eff_known)

        mr = metrics_from_intervals(rk, Tk)
        # Oracles & RR models clipped identically onto eff_known!
        cands = [
            ("Oracle_10sGrid_Fixed", fixed_oracle_grid(rk, Tk, 10.0)),
            ("Oracle_30sGrid_Fixed", fixed_oracle_grid(rk, Tk, 30.0)),
            ("Oracle_60sGrid_Fixed", fixed_oracle_grid(rk, Tk, 60.0)),
            ("Oracle_60BeatRR_Fixed", clip_ivs_to_known(ora60_raw, eff_known)[0]),
            ("RR_GBDT_RefBeats", clip_ivs_to_known(rr_ref_raw, eff_known)[0]),
            ("RR_GBDT_AutoQRS", clip_ivs_to_known(rr_auto_raw, eff_known)[0]),
        ]
        for cname, pk in cands:
            mp = metrics_from_intervals(pk, Tk)
            tp = interval_intersection_len(rk, pk)
            fp = max(0.0, interval_union_len(pk) - tp)
            fn = max(0.0, interval_union_len(rk) - tp)
            gstats = gap_detection_endpoints(rk, pk, Tk)
            row = {
                "scope": "Full182", "ref_layer": ref_lbl, "comparator": cname,
                "dataset": ds, "record_id": rid, "patient_id": pid,
                "ref_state": mr["state"], "pred_state": mp["state"],
                "ref_defined": mr["is_defined"], "pred_defined": mp["is_defined"],
                "both_defined": bool(mr["is_defined"] and mp["is_defined"]),
                "ref_burden": mr["burden"], "pred_burden": mp["burden"],
                "abs_burden_err": abs(mp["burden"] - mr["burden"]),
                "delta_disagreement": (fp + fn) / Tk,
                "ref_density": mr["charitos_density"], "pred_density": mp["charitos_density"],
                "abs_dens_err": abs(mp["charitos_density"] - mr["charitos_density"]) if (mr["is_defined"] and mp["is_defined"]) else np.nan,
                "ref_agg": mr["af_aggregation"], "pred_agg": mp["af_aggregation"],
                "abs_agg_err": abs(mp["af_aggregation"] - mr["af_aggregation"]) if (mr["is_defined"] and mp["is_defined"]) else np.nan,
                "abs_agg_zeroext_err": abs(mp["af_aggregation_zeroext"] - mr["af_aggregation_zeroext"]),
                "ref_ls": mr["longest_share"], "pred_ls": mp["longest_share"],
                "abs_ls_err": abs(mp["longest_share"] - mr["longest_share"]) if (np.isfinite(mr["longest_share"]) and np.isfinite(mp["longest_share"])) else np.nan,
                **gstats,
            }
            out_rows.append(row)
    return out_rows


def bootstrap_ci(vals: np.ndarray, pts: np.ndarray, stat_fn=np.mean, n_boot: int = 2000, seed: int = 42):
    m = np.isfinite(vals)
    v, p = vals[m], pts[m]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    obs = float(stat_fn(v))
    upts = np.unique(p)
    if len(upts) <= 1:
        return obs, obs, obs
    pt_idx = [np.where(p == pt)[0] for pt in upts]
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n_boot):
        samp = rng.choice(len(upts), size=len(upts), replace=True)
        idx = np.concatenate([pt_idx[k] for k in samp])
        bs.append(float(stat_fn(v[idx])))
    return obs, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    t0 = time.time()
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    ext = inv[inv["is_annotated"] & inv["dataset"].isin(["LTAFDB", "SHDB-AF"])].copy()
    rr_model_path = str(RR_BASE / "model.joblib")
    rr_model = joblib.load(rr_model_path)
    rr_thr = 0.48

    # Build recording metadata including R2 exclusions
    rec_info = {}
    cpu_tasks = []
    all_gaps = []
    for rec in ext.itertuples():
        ds, rid, pid, fs = rec.dataset, str(rec.record_id), rec.patient_id, float(rec.fs)
        sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
        known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
        af_r0 = merge_intervals(list(zip(
            runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["start_time_s"],
            runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["end_time_s"],
        )))
        vt_sub = sub_r[sub_r["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])]
        sbr60 = sub_r[(sub_r["rhythm_label_native"].astype(str) == "SBR") & ((sub_r["end_time_s"] - sub_r["start_time_s"]) < 60.0)]
        ex_r2 = merge_intervals(list(zip(vt_sub["start_time_s"], vt_sub["end_time_s"])) + list(zip(sbr60["start_time_s"], sbr60["end_time_s"])))
        raw_T = float(rec.raw_samples / fs)
        ref_k, T_k = clip_ivs_to_known(af_r0, known_ivs)
        m_r0 = metrics_from_intervals(ref_k, T_k)
        rec_info[(ds, rid)] = (pid, fs, raw_T, known_ivs, af_r0, ex_r2, ref_k, T_k, m_r0)
        cpu_tasks.append((ds, rid, pid, raw_T, fs, known_ivs, af_r0, ex_r2, rr_model_path, rr_thr))
        if len(ref_k) >= 2:
            for i in range(len(ref_k) - 1):
                ga, gb = ref_k[i][1], ref_k[i + 1][0]
                gd = gb - ga
                if gd > 1e-6:
                    all_gaps.append({"dataset": ds, "record_id": rid, "high_burden_ge80": m_r0["burden"] >= 0.80,
                                     "gap_start_k": ga, "gap_end_k": gb, "true_gap_s": gd, "gap_bin": gap_bin_label(gd)})

    # 1. Full 182 recordings on Fixed Oracles + RR-GBDT (Ref & AutoQRS) across R0, R2, R3
    print(f"=== Step C1: Evaluating Fixed Oracles + RR-GBDT Across All {len(cpu_tasks)} Recordings (R0, R2, R3) ===")
    full182_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(eval_one_full_rec_cpu, cpu_tasks):
            full182_rows.extend(res)
    f182 = pd.DataFrame(full182_rows)
    f182.to_parquet(OUT_RES / "V8_exp_c_full182_oracles_rr.parquet", index=False)

    # 2. A100 GPU ECGFounder (matched ZNorm & Raw) on:
    #    (a) Exact Matched-720 gaps
    #    (b) Exact Matched-25 x 6h timelines (ALL comparators on ALL 25 records!)
    print("\n=== Step C2: GPU ECGFounder Matched-720 Gaps + Matched-25 x 6h Complete Endpoint Benchmark ===")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = get_model_adapter("ECGFOUNDER", "experiments/af_density_prediction_v7p/checkpoints/1_lead_ECGFounder.pth", dev)
    sc_z, clf_z, thr_z, info_z = fit_ecgfounder_head(adapter, True, dev)
    sc_r, clf_r, thr_r, info_r = fit_ecgfounder_head(adapter, False, dev)

    def infer_probs(wav_list: List[np.ndarray], src_fs: float, znorm: bool):
        if not wav_list:
            return np.array([])
        arr = np.asarray(wav_list, dtype=np.float32)
        sc, clf = (sc_z, clf_z) if znorm else (sc_r, clf_r)
        probs = []
        with torch.no_grad():
            for i in range(0, len(arr), 1024):
                x500 = gpu_resample_and_optionally_znorm(arr[i:i + 1024], src_fs, znorm, dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    emb = adapter(x500).float().cpu().numpy()
                probs.append(clf.predict_proba(sc.transform(emb))[:, 1])
        return np.concatenate(probs)

    # Reconstruct exact same stratified 720 gaps
    gdf_all = pd.DataFrame(all_gaps)
    picks = []
    for b in sorted(gdf_all["gap_bin"].unique()):
        sub_b = gdf_all[gdf_all["gap_bin"] == b]
        hi = sub_b[sub_b["high_burden_ge80"]]
        lo = sub_b[~sub_b["high_burden_ge80"]]
        p_hi = hi.sample(n=min(60, len(hi)), random_state=42) if len(hi) else hi
        rem = max(0, 120 - len(p_hi))
        p_lo = lo.sample(n=min(rem, len(lo)), random_state=42) if len(lo) and rem else lo.iloc[0:0]
        picks.append(pd.concat([p_hi, p_lo], ignore_index=True))
    sg_df = pd.concat(picks, ignore_index=True)

    wl = WaveformLoader(cache_max_records=32)

    def known_to_raw(known_ivs, ga, gb):
        t_acc = 0.0
        for ka, kb in known_ivs:
            se = t_acc + (kb - ka)
            if gb > t_acc and ga < se:
                return ka + max(0.0, ga - t_acc), ka + min(kb - ka, gb - t_acc)
            t_acc = se
        return None, None

    gap_rows = []
    for (ds, rid), grp in sg_df.groupby(["dataset", "record_id"]):
        pid, fs, raw_T, known_ivs, af_r0, ex_r2, ref_k, T_k, _ = rec_info[(ds, rid)]
        sig_info = wl.load_record_full(ds, rid)
        if sig_info is None:
            continue
        sig, fs_real = sig_info
        _, _, _, q_auto, q_ref, _, _, _, _, _ = detect_qrs_one_record((ds, rid))
        first_k = known_ivs[0][0] * fs_real if known_ivs else 0.0
        rr_ref_raw = eval_beats_edge_assigned_rr(q_ref, fs_real, first_k, raw_T, rr_model, rr_thr)
        rr_auto_raw = eval_beats_edge_assigned_rr(q_auto, fs_real, first_k, raw_T, rr_model, rr_thr)
        ora60_raw, _, _, _ = fixed_oracle_60beat(q_ref, fs_real, first_k, raw_T, af_r0, known_ivs)
        rr_ref_k, _ = clip_ivs_to_known(rr_ref_raw, known_ivs)
        rr_auto_k, _ = clip_ivs_to_known(rr_auto_raw, known_ivs)
        ora60_k, _ = clip_ivs_to_known(ora60_raw, known_ivs)
        ora10_k = fixed_oracle_grid(ref_k, T_k, 10.0)
        ora30_k = fixed_oracle_grid(ref_k, T_k, 30.0)
        ora60s_k = fixed_oracle_grid(ref_k, T_k, 60.0)

        wlen = int(round(10.0 * fs_real))
        g_wavs, g_spans, sl_wavs, sl_spans, gmeta = [], [], [], [], []
        for g in grp.itertuples():
            ra, rb = known_to_raw(known_ivs, g.gap_start_k, g.gap_end_k)
            if ra is None:
                continue
            gdur = rb - ra
            step_g = 10.0 if gdur < 600.0 else 60.0
            ig0 = len(g_wavs)
            for ts in np.arange(max(0.0, ra - 30.0), min(len(sig) / fs_real - 10.0, rb + 20.0) + 1e-6, step_g):
                s0 = int(round(ts * fs_real))
                if 0 <= s0 and s0 + wlen <= len(sig):
                    g_wavs.append(sig[s0:s0 + wlen]); g_spans.append((ts, ts + step_g))
            ig1 = len(g_wavs)
            is0 = len(sl_wavs)
            stride = 2.0 if gdur < 120.0 else (10.0 if gdur < 360.0 else 60.0)
            for ts in np.arange(max(0.0, ra - 10.0), min(len(sig) / fs_real - 10.0, rb + 5.0) + 1e-6, stride):
                s0 = int(round(ts * fs_real))
                if 0 <= s0 and s0 + wlen <= len(sig):
                    sl_wavs.append(sig[s0:s0 + wlen]); sl_spans.append((ts, ts + 10.0, stride))
            is1 = len(sl_wavs)
            gmeta.append((g, ra, rb, gdur, ig0, ig1, is0, is1, stride))

        pg_z = infer_probs(g_wavs, fs_real, True)
        pg_r = infer_probs(g_wavs, fs_real, False)
        ps_z = infer_probs(sl_wavs, fs_real, True)

        for g, ra, rb, gdur, ig0, ig1, is0, is1, stride in gmeta:
            for cname, pk in [
                ("Oracle_10sGrid_Fixed", ora10_k), ("Oracle_30sGrid_Fixed", ora30_k),
                ("Oracle_60sGrid_Fixed", ora60s_k), ("Oracle_60BeatRR_Fixed", ora60_k),
                ("RR_GBDT_RefBeats", rr_ref_k), ("RR_GBDT_AutoQRS", rr_auto_k),
            ]:
                p_nonaf = subtract_exclusions([(0.0, T_k)], pk)
                ov = intervals_overlap_mask([(g.gap_start_k, g.gap_end_k)], p_nonaf)[0]
                gap_rows.append({
                    "patient_id": pid, "comparator": cname, "gap_bin": g.gap_bin, "true_gap_s": gdur,
                    "pred_overlap_s": ov, "det_any": int(ov > 0.5), "det_ge50": int(ov >= 0.5 * gdur),
                    "gap_iou": ov / max(gdur, 1e-6) if ov > 0.5 else 0.0,  # bounded overlap fraction on local target
                })
            for cname, probs_arr, thr_u in [("ECGFounder_ZNorm_10s", pg_z, thr_z), ("ECGFounder_Raw_10s", pg_r, thr_r)]:
                nonaf = merge_intervals([g_spans[k] for k in range(ig0, ig1) if probs_arr[k] < thr_u])
                ov = intervals_overlap_mask([(ra, rb)], nonaf)[0]
                gap_rows.append({
                    "patient_id": pid, "comparator": cname, "gap_bin": g.gap_bin, "true_gap_s": gdur,
                    "pred_overlap_s": ov, "det_any": int(ov > 0.5), "det_ge50": int(ov >= 0.5 * gdur),
                    "gap_iou": ov / max(gdur, 1e-6) if ov > 0.5 else 0.0,
                })
            nonaf_sl = merge_intervals([(sl_spans[k][0] + 5.0 - stride * 0.5, sl_spans[k][0] + 5.0 + stride * 0.5)
                                        for k in range(is0, is1) if ps_z[k] < thr_z])
            ov_sl = intervals_overlap_mask([(ra, rb)], nonaf_sl)[0]
            any_win = int(any((ps_z[k] < thr_z) and (sl_spans[k][1] > ra and sl_spans[k][0] < rb) for k in range(is0, is1)))
            gap_rows.append({
                "patient_id": pid, "comparator": "ECGFounder_ZNorm_FullSpanSlide", "gap_bin": g.gap_bin, "true_gap_s": gdur,
                "pred_overlap_s": ov_sl, "det_any": int(any_win or ov_sl > 0.5), "det_ge50": int(ov_sl >= 0.5 * gdur),
                "gap_iou": ov_sl / max(gdur, 1e-6) if ov_sl > 0.5 else 0.0,
            })

    gap_df = pd.DataFrame(gap_rows)
    gap_sum = gap_df.groupby(["comparator", "gap_bin"]).agg(
        n_gaps=("true_gap_s", "count"), sens_any=("det_any", "mean"),
        sens_ge50pct=("det_ge50", "mean"), bridge_rate=("det_any", lambda s: 1.0 - float(s.mean())),
        med_overlap_s=("pred_overlap_s", "median"),
    ).reset_index()
    gap_sum.to_csv(OUT_RES / "V8_exp_c_matched720_gaps.csv", index=False)

    # Exact Matched-25 x 6h reconstruction across ALL 25 records for EVERY comparator (including ECGFounder_Raw_10s on all 25!)
    H6 = 6 * 3600.0
    sel25 = [("LTAFDB", r) for r in ["15", "25", "33", "62", "74", "30", "38", "07", "111", "200", "60", "75", "00"]] + \
            [("SHDB-AF", r) for r in ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "014", "018"]]
    sel25 = [(ds, r) for ds, r in sel25 if (ds, r) in rec_info]

    m25_rows = []
    for ds, rid in sel25:
        pid, fs, raw_T, known_ivs, af_r0, ex_r2, ref_k_full, _, _ = rec_info[(ds, rid)]
        sig_info = wl.load_record_full(ds, rid)
        if sig_info is None:
            continue
        sig, fs_real = sig_info
        _, _, _, q_auto, q_ref, _, _, _, _, _ = detect_qrs_one_record((ds, rid))
        first_k = known_ivs[0][0] * fs_real if known_ivs else 0.0
        rr_ref_raw = eval_beats_edge_assigned_rr(q_ref, fs_real, first_k, raw_T, rr_model, rr_thr)
        rr_auto_raw = eval_beats_edge_assigned_rr(q_auto, fs_real, first_k, raw_T, rr_model, rr_thr)
        ora60_raw, _, _, _ = fixed_oracle_60beat(q_ref, fs_real, first_k, raw_T, af_r0, known_ivs)

        # Slice contiguous 10s ECG windows across first 6h known (exact trailing partial bin assigned to last full bin if needed)
        rem, t_acc, win_k_spans, win_wavs, win_raw_spans = H6, 0.0, [], [], []
        wlen = int(round(10.0 * fs_real))
        for ka, kb in known_ivs:
            if rem <= 1e-6:
                break
            take = min(kb - ka, rem)
            n_b = int(take // 10.0)
            for wi in range(n_b):
                s0 = int(round((ka + wi * 10.0) * fs_real))
                if s0 + wlen <= len(sig):
                    win_wavs.append(sig[s0:s0 + wlen])
                    win_k_spans.append((t_acc + wi * 10.0, t_acc + (wi + 1) * 10.0))
                    win_raw_spans.append((ka + wi * 10.0, ka + (wi + 1) * 10.0))
            t_acc += take; rem -= take

        pz = infer_probs(win_wavs, fs_real, True)
        pr = infer_probs(win_wavs, fs_real, False)
        ef_z_raw_ivs = merge_intervals([win_raw_spans[i] for i in range(len(pz)) if pz[i] >= thr_z])
        ef_r_raw_ivs = merge_intervals([win_raw_spans[i] for i in range(len(pr)) if pr[i] >= thr_r])

        # Evaluate identical 6h window across R0_Native, R2_VT_SBR60_Unknown, and R3_gap30s!
        k6 = []
        rem = H6
        for ka, kb in known_ivs:
            if rem <= 1e-6:
                break
            take = min(kb - ka, rem)
            k6.append((ka, ka + take))
            rem -= take
        k6 = merge_intervals(k6)

        for ref_lbl in ["R0_Native", "R2_VT_SBR60_Unknown", "R3_gap30s"]:
            eff_k = subtract_exclusions(k6, ex_r2) if ref_lbl == "R2_VT_SBR60_Unknown" else k6
            eff_T = sum(b - a for a, b in eff_k)
            if eff_T <= 60.0:
                continue
            if ref_lbl == "R3_gap30s":
                rk = apply_r3_rule(k6, af_r0, eff_T, bridge_gap_lt_s=30.0)
                Tk = eff_T
            else:
                rk, Tk = clip_ivs_to_known(subtract_exclusions(af_r0, ex_r2) if ref_lbl == "R2_VT_SBR60_Unknown" else af_r0, eff_k)

            mr = metrics_from_intervals(rk, Tk)
            comp_list = [
                ("Oracle_10sGrid_Fixed", fixed_oracle_grid(rk, Tk, 10.0)),
                ("Oracle_30sGrid_Fixed", fixed_oracle_grid(rk, Tk, 30.0)),
                ("Oracle_60sGrid_Fixed", fixed_oracle_grid(rk, Tk, 60.0)),
                ("Oracle_60BeatRR_Fixed", clip_ivs_to_known(ora60_raw, eff_k)[0]),
                ("RR_GBDT_RefBeats", clip_ivs_to_known(rr_ref_raw, eff_k)[0]),
                ("RR_GBDT_AutoQRS", clip_ivs_to_known(rr_auto_raw, eff_k)[0]),
                ("ECGFounder_ZNorm_10s", clip_ivs_to_known(ef_z_raw_ivs, eff_k)[0]),
                ("ECGFounder_Raw_10s", clip_ivs_to_known(ef_r_raw_ivs, eff_k)[0]),
            ]
            for cname, pk in comp_list:
                mp = metrics_from_intervals(pk, Tk)
                tp = interval_intersection_len(rk, pk)
                fp = max(0.0, interval_union_len(pk) - tp)
                fn = max(0.0, interval_union_len(rk) - tp)
                gs = gap_detection_endpoints(rk, pk, Tk)
                m25_rows.append({
                    "scope": "Matched25_6h", "ref_layer": ref_lbl, "comparator": cname,
                    "dataset": ds, "record_id": rid, "patient_id": pid,
                    "ref_state": mr["state"], "pred_state": mp["state"],
                    "ref_defined": mr["is_defined"], "pred_defined": mp["is_defined"],
                    "both_defined": bool(mr["is_defined"] and mp["is_defined"]),
                    "ref_burden": mr["burden"], "pred_burden": mp["burden"],
                    "abs_burden_err": abs(mp["burden"] - mr["burden"]),
                    "delta_disagreement": (fp + fn) / Tk,
                    "ref_density": mr["charitos_density"], "pred_density": mp["charitos_density"],
                    "abs_dens_err": abs(mp["charitos_density"] - mr["charitos_density"]) if (mr["is_defined"] and mp["is_defined"]) else np.nan,
                    "ref_agg": mr["af_aggregation"], "pred_agg": mp["af_aggregation"],
                    "abs_agg_err": abs(mp["af_aggregation"] - mr["af_aggregation"]) if (mr["is_defined"] and mp["is_defined"]) else np.nan,
                    "abs_agg_zeroext_err": abs(mp["af_aggregation_zeroext"] - mr["af_aggregation_zeroext"]),
                    "ref_ls": mr["longest_share"], "pred_ls": mp["longest_share"],
                    "abs_ls_err": abs(mp["longest_share"] - mr["longest_share"]) if (np.isfinite(mr["longest_share"]) and np.isfinite(mp["longest_share"])) else np.nan,
                    **gs,
                })

    wl.close()
    m25 = pd.DataFrame(m25_rows)
    m25.to_parquet(OUT_RES / "V8_exp_c_matched25_6h_full.parquet", index=False)

    # Aggregate complete endpoint tables with B=2000 patient-level bootstrap 95% CIs!
    def summarize_benchmark(df_in: pd.DataFrame) -> pd.DataFrame:
        out = []
        for (rlbl, cname), grp in df_in.groupby(["ref_layer", "comparator"]):
            ref_def = grp[grp["ref_defined"]].copy()
            pts_all = grp["patient_id"].values
            pts_def = ref_def["patient_id"].values
            d_mae, d_lo, d_hi = bootstrap_ci(ref_def["abs_dens_err"].values, pts_def, np.mean)
            a_mae, a_lo, a_hi = bootstrap_ci(ref_def["abs_agg_err"].values, pts_def, np.mean)
            az_mae, az_lo, az_hi = bootstrap_ci(grp["abs_agg_zeroext_err"].values, pts_all, np.mean)
            b_mae, b_lo, b_hi = bootstrap_ci(grp["abs_burden_err"].values, pts_all, np.mean)
            dl_m, _, _ = bootstrap_ci(grp["delta_disagreement"].values, pts_all, np.mean)
            tp_sum = grp["n_tp_true_gaps"].sum()
            tp_p_sum = grp["n_tp_pred_gaps"].sum()
            tg_sum = grp["n_true_gaps"].sum()
            pg_sum = grp["n_pred_gaps"].sum()
            prec = float(tp_p_sum / pg_sum) if pg_sum > 0 else np.nan
            rec = float(tp_sum / tg_sum) if tg_sum > 0 else np.nan
            f1 = (2 * prec * rec / (prec + rec)) if (np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0) else np.nan
            # Longest share concealment sensitivity across {0.15, 0.20, 0.30, 0.40}
            bd = ref_def[ref_def["both_defined"]]
            conc = {}
            for thr_c in [0.15, 0.20, 0.30, 0.40]:
                if len(bd):
                    concealed = ((bd["abs_dens_err"] >= thr_c) & (bd["abs_ls_err"] < 0.10)).values
                    conc[f"conc_pct_{int(thr_c*100)}"] = round(100.0 * float(concealed.mean()), 1)
                else:
                    conc[f"conc_pct_{int(thr_c*100)}"] = np.nan
            out.append({
                "ref_layer": rlbl, "comparator": cname, "n_recs": len(grp),
                "n_ref_all_af": int((grp["ref_state"] == "ALL_AF").sum()),
                "spurious_allaf_to_mixed": int(((grp["ref_state"] == "ALL_AF") & (grp["pred_state"] == "MIXED")).sum()),
                "n_ref_def": len(ref_def), "n_both_def": int(ref_def["both_defined"].sum()),
                "pct_refdef_lost_to_undef": round(100.0 * (1.0 - float(ref_def["both_defined"].mean())), 1) if len(ref_def) else np.nan,
                "gap_precision": round(prec, 4) if np.isfinite(prec) else np.nan,
                "gap_recall": round(rec, 4) if np.isfinite(rec) else np.nan,
                "gap_f1": round(f1, 4) if np.isfinite(f1) else np.nan,
                "mean_gap_iou": round(float(grp["mean_gap_iou"].mean()), 4),
                "med_abs_start_err_s": round(float(grp["med_abs_start_err_s"].median()), 2),
                "n_false_gaps_total": int(grp["n_false_gaps"].sum()),
                "mean_false_gaps_per_af_h": round(float(grp["false_gaps_per_af_h"].mean()), 3),
                "n_gaps_split": int(grp["n_true_gaps_split"].sum()),
                "n_gaps_merged": int(grp["n_pred_gaps_merging_ge2"].sum()),
                "mean_abs_burden_err": round(b_mae, 4),
                "mean_delta": round(dl_m, 4),
                "dens_mae": round(d_mae, 4), "dens_mae_ci95": f"[{d_lo:.4f}, {d_hi:.4f}]",
                "agg_mae": round(a_mae, 4), "agg_mae_ci95": f"[{a_lo:.4f}, {a_hi:.4f}]",
                "agg_zeroext_mae": round(az_mae, 4),
                **conc,
            })
        return pd.DataFrame(out)

    s182 = summarize_benchmark(f182)
    s25 = summarize_benchmark(m25)
    s182.to_csv(OUT_RES / "V8_exp_c_full182_summary.csv", index=False)
    s25.to_csv(OUT_RES / "V8_exp_c_matched25_summary.csv", index=False)
    # 3x3 State Confusion Matrix across Full182 (R0_Native) and Matched25 (R0_Native)
    cm182 = f182[f182["ref_layer"] == "R0_Native"].groupby(["comparator", "ref_state", "pred_state"]).size().reset_index(name="n_recs")
    cm25 = m25[m25["ref_layer"] == "R0_Native"].groupby(["comparator", "ref_state", "pred_state"]).size().reset_index(name="n_recs")
    cm182.to_csv(OUT_RES / "V8_exp_c_full182_state_confusion.csv", index=False)
    cm25.to_csv(OUT_RES / "V8_exp_c_matched25_state_confusion.csv", index=False)

    print("\nMatched-25 x 6h Complete Endpoint Summary (R0_Native vs R2 vs R3):")
    print(s25[["ref_layer", "comparator", "n_recs", "n_both_def", "gap_precision", "gap_recall", "gap_f1", "mean_gap_iou", "n_false_gaps_total", "dens_mae", "agg_mae", "conc_pct_20"]].to_string(index=False))
    print("\nFull-182 Summary (Fixed Oracles + RR-GBDT Across R0, R2, R3):")
    print(s182[["ref_layer", "comparator", "spurious_allaf_to_mixed", "pct_refdef_lost_to_undef", "gap_precision", "gap_recall", "gap_f1", "dens_mae", "agg_mae", "conc_pct_15", "conc_pct_20", "conc_pct_30"]].to_string(index=False))
    with (OUT_RES / "V8_exp_c_fit_info.json").open("w") as f:
        json.dump({"znorm": info_z, "raw": info_r, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nCompleted Experiment C in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
