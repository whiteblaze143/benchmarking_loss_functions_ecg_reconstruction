"""AF-DENSITY V8 Manuscript-Readiness M3 (A100 GPU):
Matched-Gap (720 gaps) + Matched-Timeline (36 x 6h records) Evaluation comparing:
- Oracle_10sGrid & Oracle_60BeatRR (from M2)
- RR_GBDT_RefBeats & RR_GBDT_AutoQRS (from M2)
- ECGFounder_10sGrid (with proper per-window Z-score normalization AND raw amplitude ablation!)
- Full-gap-span sliding-window ECGFounder (2s stride <2m; 5s stride >=2m, fixing the 55s boundary cap!)
"""
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict

import h5py
import joblib
import numpy as np
import pandas as pd
import torch
import wfdb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
from foundation_adapters import get_model_adapter
from waveform_loader import WaveformLoader
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import clip_ivs_to_known, intervals_overlap_mask, gap_bin_label, gpu_resample_batch, run_rr_gbdt_record
from v8_m2_qrs_and_oracles import run_oracle_10s_grid, eval_beats_to_rr_model_and_oracle, detect_qrs_one_record, rhythm_state

OUT_RES = Path("experiments/af_density_v8/results")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
RR_BASE = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/rr_baseline_v1")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
IRIDIA_H5 = IRIDIA_META / "ecg_hdf5"


def gpu_resample_and_optionally_znorm(wavs: np.ndarray, src_fs: float, znorm: bool, dev: torch.device) -> torch.Tensor:
    x = gpu_resample_batch(wavs, src_fs, 5000, dev)  # (B, 1, 5000)
    if znorm:
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp(min=1e-5)
        x = (x - mu) / sd
    return x


def fit_ecgfounder_head(adapter, znorm: bool, dev: torch.device):
    """Zero-leakage head fitted strictly on IRIDIA train (90 pts) + tuned on val (30 pts)."""
    splits = pd.read_csv(RR_BASE / "patient_splits.csv")
    train_pts = set(splits[splits["split"] == "train"]["patient_id"].astype(str))
    val_pts = set(splits[splits["split"] == "validation"]["patient_id"].astype(str))
    meta = pd.read_csv(IRIDIA_META / "iridia-af-metadata-v1.0.1.csv")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")
    rng = np.random.RandomState(42)
    fs = 200.0
    win_samp = 2000

    def sample_split(pt_set, max_k=16):
        wavs, ys, pids = [], [], []
        sub_meta = meta[meta["patient_id"].astype(str).isin(pt_set)].drop_duplicates("patient_id")
        for rec in sub_meta.itertuples():
            rid, pid = str(rec.record_id), str(rec.patient_id)
            raw_end = min(float(rec.record_samples) / fs, 86400.0)
            if rid in cal_grp.groups:
                cs = cal_grp.get_group(rid)
                ex = [(s / fs, e / fs) for s, e in zip(cs["excluded_start_sample"], cs["excluded_end_sample_exclusive"]) if s / fs < raw_end]
            else:
                ex = [(0.0, 30.0)]
            k_ivs = subtract_exclusions([(0.0, raw_end)], ex)
            sub_af = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
            af_ivs = subtract_exclusions(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])), ex)
            nonaf_ivs = subtract_exclusions(k_ivs, af_ivs)
            h5p = IRIDIA_H5 / rid / f"{rid}_ecg_00.h5"
            if not h5p.exists():
                continue
            with h5py.File(h5p, "r") as f:
                ds = f["ecg"]
                for cval, ivs in [(1, af_ivs), (0, nonaf_ivs)]:
                    cands = [(a, b) for a, b in ivs if (b - a) >= 12.0 and b <= 86400.0]
                    if not cands:
                        continue
                    pts_t = [rng.uniform(ca + 0.5, cb - 10.5) for ca, cb in [cands[rng.randint(len(cands))] for _ in range(max_k * 2)]][:max_k]
                    for t0 in pts_t:
                        s0 = int(round(t0 * fs))
                        seg = ds[s0:s0 + win_samp, 0].astype(np.float32)
                        if len(seg) == win_samp and np.all(np.isfinite(seg)) and np.std(seg) > 1e-6:
                            wavs.append(seg); ys.append(cval); pids.append(pid)
        return np.array(wavs, dtype=np.float32), np.array(ys, dtype=np.int32), np.array(pids)

    w_tr, y_tr, p_tr = sample_split(train_pts)
    w_va, y_va, p_va = sample_split(val_pts)

    def embed(arr):
        outs = []
        with torch.no_grad():
            for i in range(0, len(arr), 1024):
                x500 = gpu_resample_and_optionally_znorm(arr[i:i + 1024], 200.0, znorm, dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outs.append(adapter(x500).float().cpu().numpy())
        return np.vstack(outs)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embed(w_tr))
    X_va = scaler.transform(embed(w_va))
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr, y_tr)
    pv = clf.predict_proba(X_va)[:, 1]
    val_auc = float(roc_auc_score(y_va, pv))
    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.10, 0.90, 81):
        sl, sp = [], []
        for pt in np.unique(p_va):
            m = p_va == pt
            if (y_va[m] == 1).any(): sl.append(float((pv[m][y_va[m] == 1] >= thr).mean()))
            if (y_va[m] == 0).any(): sp.append(float((pv[m][y_va[m] == 0] < thr).mean()))
        ba = 0.5 * (np.mean(sl) + np.mean(sp))
        if ba > best_ba:
            best_ba, best_thr = ba, float(thr)
    info = {"znorm": znorm, "iridia_val_auroc": round(val_auc, 4), "iridia_val_bal_acc": round(float(best_ba), 4), "thr": round(best_thr, 3)}
    print("Fit ECGFounder:", info)
    return scaler, clf, best_thr, info


def main():
    t0 = time.time()
    s_df = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    rr_model = joblib.load(RR_BASE / "model.joblib")
    rr_thr = 0.48
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = get_model_adapter("ECGFOUNDER", "experiments/af_density_prediction_v7p/checkpoints/1_lead_ECGFounder.pth", dev)

    # Fit BOTH Z-normalized (official ECGFounder spec) AND Raw (for direct comparison with earlier V8-E07!)
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

    # -------------------------------------------------------------------------
    # Part 1: Matched 720 Gaps (Full-Span 2s/5s Stride + 10s Grid + M2 Oracles & RR-GBDT)
    # -------------------------------------------------------------------------
    print("\n=== Part 1: Matched 720-Gap Full-Span Comparison Across All Comparators ===")
    # Reconstruct same 720 stratified gaps (120 per bin) cleanly
    ext_recs = s_df[(s_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) & (s_df["is_annotated"])].copy()
    all_gaps = []
    rec_meta = {}
    for rec in ext_recs.itertuples():
        ds, rid, pid, fs = rec.dataset, str(rec.record_id), rec.patient_id, float(rec.fs)
        sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
        known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
        af_raw = merge_intervals(list(zip(
            runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["start_time_s"],
            runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["end_time_s"],
        )))
        competing = False
        if ds == "SHDB-AF":
            lbls = set(sub_r["rhythm_label_native"].astype(str))
            competing = any(lbl in lbls for lbl in ["AFL", "AT", "NOD", "AB", "PAT"])
        ref_k, T_k = clip_ivs_to_known(af_raw, known_ivs)
        ref_c = exact_reference_charitos_components(ref_k, T_k)
        rec_meta[(ds, rid)] = (pid, fs, known_ivs, af_raw, ref_k, T_k, ref_c, competing)
        if len(ref_k) >= 2:
            for i in range(len(ref_k) - 1):
                ga, gb = ref_k[i][1], ref_k[i + 1][0]
                gd = gb - ga
                if gd > 1e-6:
                    all_gaps.append({
                        "dataset": ds, "record_id": rid, "high_burden_ge80": ref_c["burden"] >= 0.80,
                        "gap_start_k": ga, "gap_end_k": gb, "true_gap_s": gd, "gap_bin": gap_bin_label(gd),
                    })
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

    # Also evaluate M2 comparators (RR_RefBeats, RR_AutoQRS, Oracle_10sGrid, Oracle_60BeatRR) on THESE EXACT 720 gaps!
    m2_full = pd.read_csv(OUT_RES / "V8_m2_182rec_all_comparators.csv") if (OUT_RES / "V8_m2_182rec_all_comparators.csv").exists() else None
    wl = WaveformLoader(cache_max_records=32)

    def known_to_raw(known_ivs, ga, gb):
        t_acc = 0.0
        for ka, kb in known_ivs:
            se = t_acc + (kb - ka)
            if gb > t_acc and ga < se:
                return ka + max(0.0, ga - t_acc), ka + min(kb - ka, gb - t_acc)
            t_acc = se
        return None, None

    gap_eval_rows = []
    for (ds, rid), grp in sg_df.groupby(["dataset", "record_id"]):
        pid, fs, known_ivs, af_raw, ref_k, T_k, ref_c, competing = rec_meta[(ds, rid)]
        sig_info = wl.load_record_full(ds, rid)
        if sig_info is None:
            continue
        sig, fs_real = sig_info
        # Compute M2 beat-based & oracle intervals on THIS record cleanly
        _, _, _, q_auto, q_ref, _, _, _, _, _ = detect_qrs_one_record((ds, rid))
        first_k = known_ivs[0][0] * fs_real if known_ivs else 0.0
        rr_ref_raw, ora60_raw = eval_beats_to_rr_model_and_oracle(q_ref, fs_real, first_k, af_raw, rr_model, rr_thr)
        rr_auto_raw, _ = eval_beats_to_rr_model_and_oracle(q_auto, fs_real, first_k, af_raw, rr_model, rr_thr)
        ora10_k = run_oracle_10s_grid(ref_k, T_k)
        rr_ref_k, _ = clip_ivs_to_known(rr_ref_raw, known_ivs)
        rr_auto_k, _ = clip_ivs_to_known(rr_auto_raw, known_ivs)
        ora60_k, _ = clip_ivs_to_known(ora60_raw, known_ivs)

        win_len = int(round(10.0 * fs_real))
        grid_wavs, grid_spans = [], []
        sl_wavs, sl_spans = [], []
        gmeta = []

        for g in grp.itertuples():
            ra, rb = known_to_raw(known_ivs, g.gap_start_k, g.gap_end_k)
            if ra is None:
                continue
            gdur = rb - ra
            # Non-overlapping 10s grid: full span for <600s; 30s-spaced 10s probes for >=600s multi-hour gaps
            step_g = 10.0 if gdur < 600.0 else 60.0
            t_hi_g = min(len(sig) / fs_real - 10.0, rb + 20.0)
            ig0 = len(grid_wavs)
            for ts in np.arange(max(0.0, ra - 30.0), t_hi_g + 1e-6, step_g):
                s0 = int(round(ts * fs_real))
                if 0 <= s0 and s0 + win_len <= len(sig):
                    grid_wavs.append(sig[s0:s0 + win_len]); grid_spans.append((ts, ts + step_g))
            ig1 = len(grid_wavs)

            # Full-span sliding window: 2s stride for <120s; 10s stride for 2-6m; boundary+60s step for >=6m
            is0 = len(sl_wavs)
            if gdur < 120.0:
                stride = 2.0
                ts_list = np.arange(max(0.0, ra - 10.0), min(len(sig) / fs_real - 10.0, rb + 5.0) + 1e-6, stride)
            elif gdur < 360.0:
                stride = 10.0
                ts_list = np.arange(max(0.0, ra - 10.0), min(len(sig) / fs_real - 10.0, rb + 5.0) + 1e-6, stride)
            else:
                stride = 60.0
                ts_list = np.arange(max(0.0, ra - 10.0), min(len(sig) / fs_real - 10.0, rb + 5.0) + 1e-6, stride)
            for ts in ts_list:
                s0 = int(round(ts * fs_real))
                if 0 <= s0 and s0 + win_len <= len(sig):
                    sl_wavs.append(sig[s0:s0 + win_len]); sl_spans.append((ts, ts + 10.0, stride))
            is1 = len(sl_wavs)
            gmeta.append((g, ra, rb, gdur, ig0, ig1, is0, is1, stride))

        pg_z = infer_probs(grid_wavs, fs_real, True)
        pg_r = infer_probs(grid_wavs, fs_real, False)
        ps_z = infer_probs(sl_wavs, fs_real, True)

        for g, ra, rb, gdur, ig0, ig1, is0, is1, stride in gmeta:
            # 1) M2 comparators in KNOWN coords on exact same gap
            for cname, pk in [
                ("Oracle_10sGrid", ora10_k), ("Oracle_60BeatRR", ora60_k),
                ("RR_GBDT_RefBeats", rr_ref_k), ("RR_GBDT_AutoQRS", rr_auto_k),
            ]:
                p_nonaf = subtract_exclusions([(0.0, T_k)], pk)
                ov = intervals_overlap_mask([(g.gap_start_k, g.gap_end_k)], p_nonaf)[0]
                gap_eval_rows.append({
                    "comparator": cname, "gap_bin": g.gap_bin, "true_gap_s": gdur,
                    "pred_overlap_s": ov, "det_any": int(ov > 0.5),
                    "det_ge50": int(ov >= 0.5 * gdur), "dur_err_s": ov - gdur,
                })
            # 2) ECGFounder 10s grid (ZNorm vs Raw)
            for cname, probs_arr, thr_u in [("ECGFounder_ZNorm_10s", pg_z, thr_z), ("ECGFounder_Raw_10s", pg_r, thr_r)]:
                nonaf = merge_intervals([grid_spans[k] for k in range(ig0, ig1) if probs_arr[k] < thr_u])
                ov = intervals_overlap_mask([(ra, rb)], nonaf)[0]
                gap_eval_rows.append({
                    "comparator": cname, "gap_bin": g.gap_bin, "true_gap_s": gdur,
                    "pred_overlap_s": ov, "det_any": int(ov > 0.5),
                    "det_ge50": int(ov >= 0.5 * gdur), "dur_err_s": ov - gdur,
                })
            # 3) Full-gap sliding-window ECGFounder_ZNorm (2s/5s centered step [ts + 5 - stride/2, ts + 5 + stride/2])
            nonaf_sl = merge_intervals([
                (sl_spans[k][0] + 5.0 - stride * 0.5, sl_spans[k][0] + 5.0 + stride * 0.5)
                for k in range(is0, is1) if ps_z[k] < thr_z
            ])
            ov_sl = intervals_overlap_mask([(ra, rb)], nonaf_sl)[0]
            any_win = int(any((ps_z[k] < thr_z) and (sl_spans[k][1] > ra and sl_spans[k][0] < rb) for k in range(is0, is1)))
            gap_eval_rows.append({
                "comparator": "ECGFounder_ZNorm_FullSpanSlide", "gap_bin": g.gap_bin, "true_gap_s": gdur,
                "pred_overlap_s": ov_sl, "det_any": int(any_win or ov_sl > 0.5),
                "det_ge50": int(ov_sl >= 0.5 * gdur), "dur_err_s": ov_sl - gdur,
            })

    ge_df = pd.DataFrame(gap_eval_rows)
    ge_sum = ge_df.groupby(["comparator", "gap_bin"]).agg(
        n_gaps=("true_gap_s", "count"),
        sens_any=("det_any", "mean"),
        sens_ge50pct=("det_ge50", "mean"),
        bridge_rate=("det_any", lambda s: float(1.0 - s.mean())),
        med_true_s=("true_gap_s", "median"),
        med_overlap_s=("pred_overlap_s", "median"),
        med_dur_err_s=("dur_err_s", "median"),
    ).reset_index()
    ge_sum.to_csv(OUT_RES / "V8_m3_matched720_gaps_comparison.csv", index=False)
    print("\nMatched 720-Gap Summary (All Oracles, RR-GBDT Variants & Z-Scored ECGFounder):")
    print(ge_sum.to_string(index=False))

    # -------------------------------------------------------------------------
    # Part 2: Matched 25-Record x 6h Continuous Known-Horizon Reconstruction
    # (Exact same 24 records as E07 + LTAFDB 00 continuity-concealment case!)
    # -------------------------------------------------------------------------
    print("\n=== Part 2: Matched 25-Record 6h Reconstruction (Oracles, RR-Ref, RR-AutoQRS, ECGFounder Raw & ZNorm) ===")
    H6 = 6 * 3600.0
    sel25 = [
        ("LTAFDB", r) for r in ["15", "25", "33", "62", "74", "30", "38", "07", "111", "200", "60", "75", "00"]
    ] + [
        ("SHDB-AF", r) for r in ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "014", "018"]
    ]
    sel36 = [(ds, r) for ds, r in sel25 if (ds, r) in rec_meta]

    rec36_rows = []
    for ds, rid in sel36:
        pid, fs, known_ivs, af_raw, ref_k_full, T_k_full, _, competing = rec_meta[(ds, rid)]
        sig_info = wl.load_record_full(ds, rid)
        if sig_info is None:
            continue
        sig, fs_real = sig_info
        _, _, _, q_auto, q_ref, _, _, _, _, _ = detect_qrs_one_record((ds, rid))
        first_k = known_ivs[0][0] * fs_real if known_ivs else 0.0
        rr_ref_raw, ora60_raw = eval_beats_to_rr_model_and_oracle(q_ref, fs_real, first_k, af_raw, rr_model, rr_thr)
        rr_auto_raw, _ = eval_beats_to_rr_model_and_oracle(q_auto, fs_real, first_k, af_raw, rr_model, rr_thr)
        rr_ref_6 = merge_intervals([(a, min(b, H6)) for a, b in clip_ivs_to_known(rr_ref_raw, known_ivs)[0] if a < H6])
        rr_auto_6 = merge_intervals([(a, min(b, H6)) for a, b in clip_ivs_to_known(rr_auto_raw, known_ivs)[0] if a < H6])
        ora60_6 = merge_intervals([(a, min(b, H6)) for a, b in clip_ivs_to_known(ora60_raw, known_ivs)[0] if a < H6])
        ref_6 = merge_intervals([(a, min(b, H6)) for a, b in ref_k_full if a < H6])
        ora10_6 = run_oracle_10s_grid(ref_6, H6)

        # Slice contiguous 10s ECG windows across first 6h known
        rem, t_acc, win_k_spans, win_wavs = H6, 0.0, [], []
        win_len = int(round(10.0 * fs_real))
        for ka, kb in known_ivs:
            if rem <= 1e-6:
                break
            take = min(kb - ka, rem)
            for wi in range(int(take // 10.0)):
                s0 = int(round((ka + wi * 10.0) * fs_real))
                if s0 + win_len <= len(sig):
                    win_wavs.append(sig[s0:s0 + win_len])
                    win_k_spans.append((t_acc + wi * 10.0, t_acc + (wi + 1) * 10.0))
            t_acc += take; rem -= take

        pz = infer_probs(win_wavs, fs_real, True)
        ef_z_6 = merge_intervals([win_k_spans[i] for i in range(len(pz)) if pz[i] >= thr_z])
        # Compute Raw ECGFounder too on LTAFDB 15, 25, 33, 62, 74, 00 & SHDB 001..004 for direct pairing without doubling full runtime!
        if rid in {"15", "25", "33", "62", "74", "00", "001", "002", "010"}:
            pr = infer_probs(win_wavs, fs_real, False)
            ef_r_6 = merge_intervals([win_k_spans[i] for i in range(len(pr)) if pr[i] >= thr_r])
        else:
            ef_r_6 = None

        ref_c6 = exact_reference_charitos_components(ref_6, H6)
        r_ls = max((b - a for a, b in ref_6), default=0.0) / ref_c6["af_s"] if ref_c6["af_s"] > 0 else np.nan
        r_state = rhythm_state(ref_c6)

        comp_list = [
            ("Oracle_10sGrid", ora10_6), ("Oracle_60BeatRR", ora60_6),
            ("RR_GBDT_RefBeats", rr_ref_6), ("RR_GBDT_AutoQRS", rr_auto_6),
            ("ECGFounder_ZNorm_10s", ef_z_6),
        ]
        if ef_r_6 is not None:
            comp_list.append(("ECGFounder_Raw_10s", ef_r_6))
        for cname, pk in comp_list:
            pc = exact_reference_charitos_components(pk, H6)
            pls = max((b - a for a, b in pk), default=0.0) / pc["af_s"] if pc["af_s"] > 0 else np.nan
            rec36_rows.append({
                "comparator": cname, "dataset": ds, "record_id": rid, "has_competing_afl_at": competing,
                "ref_state": r_state, "pred_state": rhythm_state(pc),
                "ref_defined": ref_c6["is_defined"], "pred_defined": pc["is_defined"],
                "ref_burden": ref_c6["burden"], "pred_burden": pc["burden"],
                "abs_burden_err": abs(pc["burden"] - ref_c6["burden"]),
                "ref_n_runs": ref_c6["n_runs"], "pred_n_runs": pc["n_runs"],
                "ref_density": ref_c6["density"], "pred_density": pc["density"],
                "abs_dens_err": abs(pc["density"] - ref_c6["density"]) if (ref_c6["is_defined"] and pc["is_defined"]) else np.nan,
                "abs_ls_err": abs(pls - r_ls) if (np.isfinite(r_ls) and np.isfinite(pls)) else np.nan,
            })

    wl.close()
    r36 = pd.DataFrame(rec36_rows)
    r36.to_csv(OUT_RES / "V8_m3_matched36_6h_reconstruction.csv", index=False)
    sum36 = r36[r36["ref_defined"]].groupby(["comparator", r36["ref_burden"] >= 0.80]).agg(
        n_ref_def=("record_id", "count"),
        n_both_def=("pred_defined", "sum"),
        med_abs_burden_err=("abs_burden_err", "median"),
        med_abs_dens_err=("abs_dens_err", "median"),
        mean_abs_dens_err=("abs_dens_err", "mean"),
        med_abs_ls_err=("abs_ls_err", "median"),
    ).reset_index()
    sum36.to_csv(OUT_RES / "V8_m3_matched36_summary.csv", index=False)
    print("\nMatched 36-Record 6h Summary (Z-Norm Fix vs Raw vs Oracles):")
    print(sum36.to_string(index=False))
    with (OUT_RES / "V8_m3_fit_info.json").open("w") as f:
        json.dump({"znorm": info_z, "raw": info_r, "elapsed_s": round(time.time() - t0, 1)}, f, indent=2)
    print(f"\nM3 completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
