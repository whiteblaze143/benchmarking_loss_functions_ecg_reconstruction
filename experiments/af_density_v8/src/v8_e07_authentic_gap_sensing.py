"""AF-DENSITY V8-E07: Authentic Sensing of Short AF Interruptions and Density Error on Real Ambulatory ECG."""
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

sys.path.insert(0, "experiments/af_density")
from rr_features import features as rr_features, layout as rr_layout
sys.path.insert(0, "experiments/af_density_prediction_v7p/src")
from foundation_adapters import get_model_adapter
from waveform_loader import WaveformLoader
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals, subtract_exclusions

OUT_RES = Path("experiments/af_density_v8/results")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
RR_BASE = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/rr_baseline_v1")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
IRIDIA_H5 = IRIDIA_META / "ecg_hdf5"

NON_BEAT_LTAF = {'+', '"', '~', '|', 's', 't', 'p', 'u', '='}
GAP_BINS = [
    (0.0, 10.0, "01_lt10s"),
    (10.0, 30.0, "02_10_30s"),
    (30.0, 60.0, "03_30_60s"),
    (60.0, 120.0, "04_1_2m"),
    (120.0, 360.0, "05_2_6m"),
    (360.0, 1e12, "06_ge6m"),
]


def gap_bin_label(dur_s: float) -> str:
    for lo, hi, lbl in GAP_BINS:
        if lo <= dur_s < hi:
            return lbl
    return "06_ge6m"


def clip_ivs_to_known(ivs: List[Tuple[float, float]], known_ivs: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], float]:
    """Map intervals inside known_ivs into contiguous known-coordinate timeline [0, T_known]."""
    sorted_k = merge_intervals(known_ivs)
    sorted_iv = merge_intervals(ivs)
    out = []
    t_acc = 0.0
    for ka, kb in sorted_k:
        for aa, ab in sorted_iv:
            oa, ob = max(aa, ka), min(ab, kb)
            if ob > oa + 1e-9:
                out.append((t_acc + oa - ka, t_acc + ob - ka))
        t_acc += (kb - ka)
    return merge_intervals(out), t_acc


def run_rr_gbdt_record(ds: str, rid: str, fs: float, known_ivs: List[Tuple[float, float]], rr_model, thr: float) -> Tuple[List[Tuple[float, float]], float]:
    """Run frozen RR-GBDT (model.joblib, float32) on physical beat timestamps and return predicted AF intervals in raw record seconds."""
    if ds == "SHDB-AF":
        q = wfdb.rdann(str(SHDB_DIR / rid), "qrs").sample
    else:
        ann = wfdb.rdann(str(LTAF_DIR / rid), "atr")
        mask = np.array([s not in NON_BEAT_LTAF for s in ann.symbol], dtype=bool)
        q = ann.sample[mask]
    q = np.sort(q)
    q = q[np.r_[True, np.diff(q) > 0]]
    if len(q) < 62:
        return [], 0.0
    first_k = known_ivs[0][0] * fs if known_ivs else 0.0
    q = q[q >= first_k]
    if len(q) < 62:
        return [], 0.0
    rr_ms = np.diff(q).astype(float) * 1000.0 / fs
    starts, assigned, ends = rr_layout(len(rr_ms), 0)
    if len(starts) == 0:
        return [], 0.0
    X, valid = rr_features(rr_ms[starts[:, None] + np.arange(60)])
    scores = np.full(len(starts), np.nan)
    if valid.any():
        scores[valid] = rr_model.predict_proba(X[valid].astype(np.float32))[:, 1]
    pred_ivs = []
    n_valid = 0
    for i, (lo, hi) in enumerate(zip(assigned, ends)):
        if valid[i]:
            n_valid += (hi - lo)
            if scores[i] >= thr:
                t0 = float(q[lo]) / fs
                t1 = float(q[hi]) / fs
                if t1 > t0:
                    pred_ivs.append((t0, t1))
    cov = float(n_valid / len(rr_ms)) if len(rr_ms) > 0 else 0.0
    return merge_intervals(pred_ivs), cov


def gpu_resample_batch(wavs: np.ndarray, src_fs: float, dst_len: int = 5000, device: torch.device = torch.device("cuda:0")) -> torch.Tensor:
    """Exact Fourier resampling of (B, L) numpy signals -> (B, 1, dst_len) tensor on GPU."""
    bx = torch.from_numpy(wavs).to(device=device, dtype=torch.float32)
    src_len = bx.shape[-1]
    if src_len == dst_len:
        return bx.unsqueeze(1)
    X = torch.fft.rfft(bx, dim=-1)
    out = torch.zeros(bx.shape[0], dst_len // 2 + 1, dtype=torch.complex64, device=device)
    keep = min(X.shape[-1], out.shape[-1])
    out[..., :keep] = X[..., :keep]
    res = torch.fft.irfft(out, n=dst_len, dim=-1) * (float(dst_len) / float(src_len))
    return res.unsqueeze(1)


def fit_ecgfounder_iridia_head(adapter, dev: torch.device) -> Tuple[StandardScaler, LogisticRegression, float, Dict]:
    """Fit linear head on ECGFounder 10s embeddings using ONLY IRIDIA-AF train split; tune threshold on val split."""
    print("=== Fitting Zero-Leakage ECGFounder AF Head on IRIDIA-AF Train/Val Only ===")
    splits = pd.read_csv(RR_BASE / "patient_splits.csv")
    train_pts = set(splits[splits["split"] == "train"]["patient_id"].astype(str))
    val_pts = set(splits[splits["split"] == "validation"]["patient_id"].astype(str))
    meta = pd.read_csv(IRIDIA_META / "iridia-af-metadata-v1.0.1.csv")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")

    rng = np.random.RandomState(42)
    win_s = 10.0
    fs = 200.0
    win_samp = int(win_s * fs)

    def sample_split_windows(pt_set, max_per_class_pt=12):
        wavs, ys, pids = [], [], []
        sub_meta = meta[meta["patient_id"].astype(str).isin(pt_set)].drop_duplicates("patient_id")
        for rec in sub_meta.itertuples():
            rid = str(rec.record_id)
            pid = str(rec.patient_id)
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
                for cls_val, iv_list in [(1, af_ivs), (0, nonaf_ivs)]:
                    cands = [(a, b) for a, b in iv_list if (b - a) >= 12.0 and b <= 86400.0]
                    if not cands:
                        continue
                    pts_t = []
                    for _ in range(max_per_class_pt * 3):
                        ca, cb = cands[rng.randint(len(cands))]
                        t0 = rng.uniform(ca + 0.5, cb - win_s - 0.5)
                        pts_t.append(t0)
                    pts_t = pts_t[:max_per_class_pt]
                    for t0 in pts_t:
                        s0 = int(round(t0 * fs))
                        seg = ds[s0:s0 + win_samp, 0].astype(np.float32)
                        if len(seg) == win_samp and np.all(np.isfinite(seg)):
                            wavs.append(seg)
                            ys.append(cls_val)
                            pids.append(pid)
        return np.array(wavs, dtype=np.float32), np.array(ys, dtype=np.int32), np.array(pids)

    w_tr, y_tr, p_tr = sample_split_windows(train_pts, max_per_class_pt=16)
    w_va, y_va, p_va = sample_split_windows(val_pts, max_per_class_pt=16)
    print(f"Extracted IRIDIA train={len(w_tr)} (AF={y_tr.sum()}), val={len(w_va)} (AF={y_va.sum()}); NEVER touching IRIDIA test!")

    def embed(wavs_arr):
        outs = []
        with torch.no_grad():
            for i in range(0, len(wavs_arr), 1024):
                x500 = gpu_resample_batch(wavs_arr[i:i + 1024], 200.0, 5000, dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    emb = adapter(x500).float().cpu().numpy()
                outs.append(emb)
        return np.vstack(outs)

    E_tr = embed(w_tr)
    E_va = embed(w_va)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(E_tr)
    X_va = scaler.transform(E_va)
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_tr, y_tr)
    p_val_prob = clf.predict_proba(X_va)[:, 1]
    val_auc = float(roc_auc_score(y_va, p_val_prob))

    best_thr, best_ba = 0.5, -1.0
    for thr in np.linspace(0.10, 0.90, 81):
        sens_l, spec_l = [], []
        for pt in np.unique(p_va):
            m = p_va == pt
            if (y_va[m] == 1).any():
                sens_l.append(float((p_val_prob[m][y_va[m] == 1] >= thr).mean()))
            if (y_va[m] == 0).any():
                spec_l.append(float((p_val_prob[m][y_va[m] == 0] < thr).mean()))
        ba = 0.5 * (np.mean(sens_l) + np.mean(spec_l))
        if ba > best_ba:
            best_ba = ba
            best_thr = float(thr)

    info = {
        "n_train_windows": int(len(w_tr)), "n_val_windows": int(len(w_va)),
        "iridia_val_auroc": round(val_auc, 4), "iridia_val_bal_acc": round(float(best_ba), 4),
        "chosen_threshold": round(best_thr, 3),
    }
    print("ECGFounder IRIDIA Train/Val Fit Info:", info)
    return scaler, clf, best_thr, info


def intervals_overlap_mask(gaps: List[Tuple[float, float]], pred_nonaf: List[Tuple[float, float]]):
    """For each true gap [ga, gb], calculate predicted non-AF overlap seconds."""
    overlaps = []
    # pred_nonaf is sorted merged intervals
    j = 0
    for ga, gb in gaps:
        ov = 0.0
        while j < len(pred_nonaf) and pred_nonaf[j][1] <= ga:
            j += 1
        k = j
        while k < len(pred_nonaf) and pred_nonaf[k][0] < gb:
            oa = max(ga, pred_nonaf[k][0])
            ob = min(gb, pred_nonaf[k][1])
            if ob > oa:
                ov += (ob - oa)
            k += 1
        overlaps.append(ov)
    return overlaps


def main():
    t_start = time.time()
    s_df = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    rr_model = joblib.load(RR_BASE / "model.joblib")
    rr_thr = 0.48

    ext_recs = s_df[(s_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) & (s_df["is_annotated"])].copy()

    # 1. FULL RECORDING RR-GBDT across ALL 84 LTAFDB + 98 SHDB-AF records!
    print("=== Part A: Frozen RR-GBDT Across All 182 LTAFDB+SHDB Full Recordings ===")
    rr_rec_rows = []
    rr_gap_rows = []
    rr_false_gap_rows = []
    rec_cache_meta = {}

    for rec in ext_recs.itertuples():
        ds, rid, pid, fs = rec.dataset, str(rec.record_id), rec.patient_id, float(rec.fs)
        sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
        known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
        af_raw = list(zip(runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["start_time_s"],
                          runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]["end_time_s"]))
        competing = False
        if ds == "SHDB-AF":
            lbls = set(sub_r["rhythm_label_native"].astype(str))
            competing = any(lbl in lbls for lbl in ["AFL", "AT", "NOD", "AB", "PAT"])

        ref_k, T_k = clip_ivs_to_known(af_raw, known_ivs)
        pred_raw, rr_cov = run_rr_gbdt_record(ds, rid, fs, known_ivs, rr_model, rr_thr)
        pred_k, _ = clip_ivs_to_known(pred_raw, known_ivs)

        ref_c = exact_reference_charitos_components(ref_k, T_k)
        pred_c = exact_reference_charitos_components(pred_k, T_k)
        rec_cache_meta[(ds, rid)] = (pid, fs, known_ivs, af_raw, ref_k, T_k, ref_c, competing, pred_k)

        ref_ls = (max((b - a for a, b in ref_k), default=0.0) / ref_c["af_s"]) if ref_c["af_s"] > 0 else np.nan
        pred_ls = (max((b - a for a, b in pred_k), default=0.0) / pred_c["af_s"]) if pred_c["af_s"] > 0 else np.nan

        rr_rec_rows.append({
            "detector": "RR_GBDT", "dataset": ds, "record_id": rid, "patient_id": pid,
            "horizon_h": round(T_k / 3600.0, 3), "valid_cov": round(rr_cov, 4),
            "has_competing_afl_at": competing,
            "ref_burden": ref_c["burden"], "pred_burden": pred_c["burden"],
            "burden_err": pred_c["burden"] - ref_c["burden"],
            "abs_burden_err": abs(pred_c["burden"] - ref_c["burden"]),
            "ref_n_runs": ref_c["n_runs"], "pred_n_runs": pred_c["n_runs"],
            "ep_count_err": pred_c["n_runs"] - ref_c["n_runs"],
            "abs_ep_count_err": abs(pred_c["n_runs"] - ref_c["n_runs"]),
            "ref_density": ref_c["density"], "pred_density": pred_c["density"],
            "ref_defined": ref_c["is_defined"], "pred_defined": pred_c["is_defined"],
            "pred_undef_reason": pred_c["undef_reason"],
            "signed_dens_err": (pred_c["density"] - ref_c["density"]) if (ref_c["is_defined"] and pred_c["is_defined"]) else np.nan,
            "abs_dens_err": abs(pred_c["density"] - ref_c["density"]) if (ref_c["is_defined"] and pred_c["is_defined"]) else np.nan,
            "ref_longest_share": ref_ls, "pred_longest_share": pred_ls,
            "abs_longest_share_err": abs(pred_ls - ref_ls) if (np.isfinite(ref_ls) and np.isfinite(pred_ls)) else np.nan,
        })

        # Internal gaps in KNOWN coordinates [0, T_k] (gaps strictly between AF episodes)
        if len(ref_k) >= 2:
            true_gaps = [(ref_k[i][1], ref_k[i + 1][0]) for i in range(len(ref_k) - 1) if (ref_k[i + 1][0] - ref_k[i][1]) > 1e-6]
            pred_nonaf = subtract_exclusions([(0.0, T_k)], pred_k)
            ovs = intervals_overlap_mask(true_gaps, pred_nonaf)
            for (ga, gb), ov in zip(true_gaps, ovs):
                gdur = gb - ga
                rr_gap_rows.append({
                    "detector": "RR_GBDT", "dataset": ds, "record_id": rid,
                    "ref_burden": ref_c["burden"], "high_burden_ge80": ref_c["burden"] >= 0.80,
                    "has_competing_afl_at": competing,
                    "gap_start_k": ga, "gap_end_k": gb, "true_gap_s": gdur,
                    "gap_bin": gap_bin_label(gdur),
                    "pred_nonaf_overlap_s": ov,
                    "detected_any": int(ov > 0.5),
                    "detected_ge50pct": int(ov >= 0.5 * gdur),
                    "dur_err_s": ov - gdur,
                })

        # False gaps inside prolonged continuous AF runs >= 600s
        prolonged_af = [(a, b) for a, b in ref_k if (b - a) >= 600.0]
        if prolonged_af:
            pred_nonaf = subtract_exclusions([(0.0, T_k)], pred_k)
            tot_prol_s = sum(b - a for a, b in prolonged_af)
            # Count distinct pred_nonaf intervals strictly or partially inside prolonged AF runs
            n_fgaps = 0
            fgap_s = 0.0
            for pa, pb in pred_nonaf:
                for aa, ab in prolonged_af:
                    oa, ob = max(pa, aa), min(pb, ab)
                    if ob > oa + 0.5:
                        n_fgaps += 1
                        fgap_s += (ob - oa)
            rr_false_gap_rows.append({
                "detector": "RR_GBDT", "dataset": ds, "record_id": rid,
                "ref_burden": ref_c["burden"], "high_burden_ge80": ref_c["burden"] >= 0.80,
                "has_competing_afl_at": competing,
                "prolonged_af_h": tot_prol_s / 3600.0,
                "n_false_gaps": n_fgaps, "false_nonaf_s": fgap_s,
            })

    rr_rec_df = pd.DataFrame(rr_rec_rows)
    rr_gap_df = pd.DataFrame(rr_gap_rows)
    rr_fg_df = pd.DataFrame(rr_false_gap_rows)
    print(f"RR-GBDT full 182-rec done in {time.time() - t_start:.1f}s: n_internal_gaps={len(rr_gap_df)}, n_prolonged_af_recs={len(rr_fg_df)}")

    # 2. ECGFOUNDER 1-LEAD ON A100 GPU
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = get_model_adapter("ECGFOUNDER", "experiments/af_density_prediction_v7p/checkpoints/1_lead_ECGFounder.pth", dev)
    scaler, clf, ef_thr, ef_fit_info = fit_ecgfounder_iridia_head(adapter, dev)

    def infer_windows_ecgfounder(wav_list: List[np.ndarray], src_fs: float) -> np.ndarray:
        if not wav_list:
            return np.array([])
        arr = np.asarray(wav_list, dtype=np.float32)
        probs = []
        with torch.no_grad():
            for i in range(0, len(arr), 1024):
                x500 = gpu_resample_batch(arr[i:i + 1024], src_fs, 5000, dev)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    emb = adapter(x500).float().cpu().numpy()
                probs.append(clf.predict_proba(scaler.transform(emb))[:, 1])
        return np.concatenate(probs)

    # 2a. Stratified Gap-Neighborhood Sensing (10s fixed grid & 2s stride centered on gap!)
    print("\n=== Part B: ECGFounder Transition-Neighborhood Sensing Across All 6 Gap Bins ===")
    wl = WaveformLoader(cache_max_records=32)
    rng = np.random.RandomState(42)

    # Sample up to 120 gaps per bin (prioritizing high-burden records and SHDB competing rhythms!)
    sampled_gaps = []
    for _, _, blbl in GAP_BINS:
        sub_g = rr_gap_df[rr_gap_df["gap_bin"] == blbl]
        # Take ALL high-burden >=80% gaps up to 60, plus random remainder up to 120 total
        hi_g = sub_g[sub_g["high_burden_ge80"]]
        lo_g = sub_g[~sub_g["high_burden_ge80"]]
        pick_hi = hi_g.sample(n=min(60, len(hi_g)), random_state=42) if len(hi_g) > 0 else hi_g
        rem_n = max(0, 120 - len(pick_hi))
        pick_lo = lo_g.sample(n=min(rem_n, len(lo_g)), random_state=42) if len(lo_g) > 0 and rem_n > 0 else lo_g.iloc[0:0]
        sampled_gaps.append(pd.concat([pick_hi, pick_lo], ignore_index=True))
    sg_df = pd.concat(sampled_gaps, ignore_index=True)

    # Map known coordinates back to raw timestamps cleanly per record to slice real ECG!
    def known_to_raw_intervals(known_ivs, gap_ka, gap_kb):
        # Returns raw [a, b] overlapping [gap_ka, gap_kb] inside known coordinates (almost always 1 contiguous span)
        t_acc = 0.0
        for ka, kb in known_ivs:
            seg_end = t_acc + (kb - ka)
            if gap_kb > t_acc and gap_ka < seg_end:
                ra = ka + max(0.0, gap_ka - t_acc)
                rb = ka + min(kb - ka, gap_kb - t_acc)
                return ra, rb
            t_acc = seg_end
        return None, None

    ef_gap_rows_grid = []
    ef_gap_rows_stride2 = []
    for (ds, rid), grp in sg_df.groupby(["dataset", "record_id"]):
        pid, fs, known_ivs, _, ref_k, T_k, ref_c, competing, _ = rec_cache_meta[(ds, rid)]
        rec_sig_info = wl.load_record_full(ds, rid)
        if rec_sig_info is None:
            continue
        sig, fs_real = rec_sig_info
        win_len = int(round(10.0 * fs_real))
        win_spans_grid = []
        win_wavs_grid = []
        win_spans_s2 = []
        win_wavs_s2 = []
        g_meta = []
        for g_row in grp.itertuples():
            ra, rb = known_to_raw_intervals(known_ivs, g_row.gap_start_k, g_row.gap_end_k)
            if ra is None:
                continue
            gdur = rb - ra
            # 10s grid across [ra - 30s, rb + 30s] (clipped to [ra - 30, rb + 30] and recording bounds)
            t_lo = max(0.0, ra - 30.0)
            t_hi = min(len(sig) / fs_real - 10.0, rb + 20.0)
            g_starts = np.arange(t_lo, t_hi + 1e-6, 10.0)
            idx0_g = len(win_wavs_grid)
            for ts in g_starts:
                s0 = int(round(ts * fs_real))
                if 0 <= s0 and s0 + win_len <= len(sig):
                    win_wavs_grid.append(sig[s0:s0 + win_len])
                    win_spans_grid.append((ts, ts + 10.0))
            idx1_g = len(win_wavs_grid)

            # 2s stride centered around gap [ra - 20s, min(ra+40, rb+10)] (limit huge gaps to first 60s of boundaries!)
            t_s_lo = max(0.0, ra - 15.0)
            t_s_hi = min(len(sig) / fs_real - 10.0, min(rb + 10.0, ra + 50.0))
            s2_starts = np.arange(t_s_lo, t_s_hi + 1e-6, 2.0)
            idx0_s = len(win_wavs_s2)
            for ts in s2_starts:
                s0 = int(round(ts * fs_real))
                if 0 <= s0 and s0 + win_len <= len(sig):
                    win_wavs_s2.append(sig[s0:s0 + win_len])
                    win_spans_s2.append((ts, ts + 10.0))
            idx1_s = len(win_wavs_s2)
            g_meta.append((g_row, ra, rb, gdur, idx0_g, idx1_g, idx0_s, idx1_s))

        probs_g = infer_windows_ecgfounder(win_wavs_grid, fs_real)
        probs_s = infer_windows_ecgfounder(win_wavs_s2, fs_real)

        for g_row, ra, rb, gdur, ig0, ig1, is0, is1 in g_meta:
            # 10s grid non-AF intervals (< ef_thr)
            nonaf_g = merge_intervals([win_spans_grid[k] for k in range(ig0, ig1) if probs_g[k] < ef_thr])
            ov_g = intervals_overlap_mask([(ra, rb)], nonaf_g)[0]
            ef_gap_rows_grid.append({
                "detector": "ECGFounder_10sGrid", "dataset": ds, "record_id": rid,
                "ref_burden": g_row.ref_burden, "high_burden_ge80": g_row.high_burden_ge80,
                "has_competing_afl_at": competing, "true_gap_s": gdur, "gap_bin": g_row.gap_bin,
                "pred_nonaf_overlap_s": ov_g, "detected_any": int(ov_g > 0.5),
                "detected_ge50pct": int(ov_g >= 0.5 * gdur), "dur_err_s": ov_g - gdur,
            })
            # 2s stride centered sensitivity (any centered window < ef_thr overlapping gap!)
            # Convert 2s stride centers [ts+4, ts+6] -> 2s steps!
            nonaf_s2 = merge_intervals([(win_spans_s2[k][0] + 4.0, win_spans_s2[k][0] + 6.0) for k in range(is0, is1) if probs_s[k] < ef_thr])
            ov_s2 = intervals_overlap_mask([(ra, rb)], nonaf_s2)[0]
            any_win_det = int(any(
                (probs_s[k] < ef_thr) and (win_spans_s2[k][1] > ra and win_spans_s2[k][0] < rb)
                for k in range(is0, is1)
            ))
            ef_gap_rows_stride2.append({
                "detector": "ECGFounder_2sStrideCenter", "dataset": ds, "record_id": rid,
                "ref_burden": g_row.ref_burden, "high_burden_ge80": g_row.high_burden_ge80,
                "has_competing_afl_at": competing, "true_gap_s": gdur, "gap_bin": g_row.gap_bin,
                "pred_nonaf_overlap_s": ov_s2, "detected_any": int(any_win_det or ov_s2 > 0.5),
                "detected_ge50pct": int(ov_s2 >= 0.5 * gdur), "dur_err_s": ov_s2 - gdur,
            })

    # Combine gap sensitivity table
    all_gaps_eval = pd.concat([rr_gap_df, pd.DataFrame(ef_gap_rows_grid), pd.DataFrame(ef_gap_rows_stride2)], ignore_index=True)
    gap_sum = all_gaps_eval.groupby(["detector", "gap_bin"]).agg(
        n_gaps=("true_gap_s", "count"),
        gap_sens_any=("detected_any", "mean"),
        gap_sens_ge50pct=("detected_ge50pct", "mean"),
        incorrect_bridge_rate=("detected_any", lambda s: float(1.0 - s.mean())),
        med_true_gap_s=("true_gap_s", "median"),
        med_pred_overlap_s=("pred_nonaf_overlap_s", "median"),
        med_dur_err_s=("dur_err_s", "median"),
    ).reset_index()
    gap_sum.to_csv(OUT_RES / "V8_e07_gap_sensitivity_by_bin.csv", index=False)
    print("\n=== V8-E07 Gap-Level Sensitivity & Bridging Error by Gap Duration Bin ===")
    print(gap_sum.to_string(index=False))

    # 2b. ECGFounder vs RR-GBDT Full Known-Horizon (6h) Reconstruction on 24 Diverse Multi-Run / Stratified Records!
    print("\n=== Part C: Matched 6h Full-Horizon Physical Reconstruction (Reference vs RR-GBDT vs ECGFounder-10s) ===")
    # Select 24 records (12 LTAFDB incl 33,74,62,15,25 + 12 SHDB-AF incl high burden & AFL/AT!) with >= 6h known time!
    cands_ltaf = rr_rec_df[(rr_rec_df["dataset"] == "LTAFDB") & (rr_rec_df["horizon_h"] >= 6.0)]
    worst5_rids = {"33", "74", "62", "15", "25"}
    pick_ltaf_w = cands_ltaf[cands_ltaf["record_id"].isin(worst5_rids)]
    pick_ltaf_other = cands_ltaf[~cands_ltaf["record_id"].isin(worst5_rids)].sort_values("ref_burden").iloc[
        np.linspace(0, len(cands_ltaf) - len(pick_ltaf_w) - 1, 7, dtype=int)
    ]
    cands_shdb = rr_rec_df[(rr_rec_df["dataset"] == "SHDB-AF") & (rr_rec_df["horizon_h"] >= 6.0)]
    pick_shdb_hi = cands_shdb[cands_shdb["ref_burden"] >= 0.50].head(5)
    pick_shdb_comp = cands_shdb[(cands_shdb["has_competing_afl_at"]) & (~cands_shdb["record_id"].isin(pick_shdb_hi["record_id"]))].head(4)
    pick_shdb_rem = cands_shdb[~cands_shdb["record_id"].isin(set(pick_shdb_hi["record_id"]) | set(pick_shdb_comp["record_id"]))].head(3)
    rec24 = pd.concat([pick_ltaf_w, pick_ltaf_other, pick_shdb_hi, pick_shdb_comp, pick_shdb_rem], ignore_index=True)

    H6 = 6 * 3600.0
    matched_rec_rows = []
    ef_fg_rows = []
    for row in rec24.itertuples():
        ds, rid = row.dataset, str(row.record_id)
        pid, fs, known_ivs, af_raw, ref_k_full, T_k_full, _, competing, rr_k_full = rec_cache_meta[(ds, rid)]
        sig_info = wl.load_record_full(ds, rid)
        if sig_info is None:
            continue
        sig, fs_real = sig_info
        # Slice non-overlapping 10s windows across raw recording up to first 6h known!
        # Map known_ivs up to H6
        rem = H6
        win_k_spans, win_wavs = [], []
        win_len = int(round(10.0 * fs_real))
        t_acc = 0.0
        for ka, kb in known_ivs:
            if rem <= 1e-6:
                break
            take = min(kb - ka, rem)
            n_w = int(take // 10.0)
            for wi in range(n_w):
                s0 = int(round((ka + wi * 10.0) * fs_real))
                if s0 + win_len <= len(sig):
                    win_wavs.append(sig[s0:s0 + win_len])
                    win_k_spans.append((t_acc + wi * 10.0, t_acc + (wi + 1) * 10.0))
            t_acc += take
            rem -= take
        probs = infer_windows_ecgfounder(win_wavs, fs_real)
        ef_af_k = merge_intervals([win_k_spans[i] for i in range(len(probs)) if probs[i] >= ef_thr])

        ref_6 = merge_intervals([(a, min(b, H6)) for a, b in ref_k_full if a < H6])
        rr_6 = merge_intervals([(a, min(b, H6)) for a, b in rr_k_full if a < H6])
        ref_c6 = exact_reference_charitos_components(ref_6, H6)
        rr_c6 = exact_reference_charitos_components(rr_6, H6)
        ef_c6 = exact_reference_charitos_components(ef_af_k, H6)

        # False gaps for ECGFounder inside prolonged AF >= 600s in [0, H6]
        prol_6 = [(a, b) for a, b in ref_6 if (b - a) >= 600.0]
        if prol_6:
            ef_nonaf = subtract_exclusions([(0.0, H6)], ef_af_k)
            nf, fsec = 0, 0.0
            for pa, pb in ef_nonaf:
                for aa, ab in prol_6:
                    oa, ob = max(pa, aa), min(pb, ab)
                    if ob > oa + 0.5:
                        nf += 1; fsec += (ob - oa)
            ef_fg_rows.append({
                "detector": "ECGFounder_10sGrid", "dataset": ds, "record_id": rid,
                "ref_burden": ref_c6["burden"], "high_burden_ge80": ref_c6["burden"] >= 0.80,
                "has_competing_afl_at": competing,
                "prolonged_af_h": sum(b - a for a, b in prol_6) / 3600.0,
                "n_false_gaps": nf, "false_nonaf_s": fsec,
            })

        for dname, dc, d_ivs in [("RR_GBDT", rr_c6, rr_6), ("ECGFounder_10sGrid", ef_c6, ef_af_k)]:
            r_ls = max((b - a for a, b in ref_6), default=0.0) / ref_c6["af_s"] if ref_c6["af_s"] > 0 else np.nan
            p_ls = max((b - a for a, b in d_ivs), default=0.0) / dc["af_s"] if dc["af_s"] > 0 else np.nan
            matched_rec_rows.append({
                "detector": dname, "dataset": ds, "record_id": rid, "has_competing_afl_at": competing,
                "ref_burden": ref_c6["burden"], "pred_burden": dc["burden"], "abs_burden_err": abs(dc["burden"] - ref_c6["burden"]),
                "ref_n_runs": ref_c6["n_runs"], "pred_n_runs": dc["n_runs"], "ep_count_err": dc["n_runs"] - ref_c6["n_runs"],
                "ref_density": ref_c6["density"], "pred_density": dc["density"],
                "ref_defined": ref_c6["is_defined"], "pred_defined": dc["is_defined"], "pred_undef_reason": dc["undef_reason"],
                "signed_dens_err": (dc["density"] - ref_c6["density"]) if (ref_c6["is_defined"] and dc["is_defined"]) else np.nan,
                "abs_dens_err": abs(dc["density"] - ref_c6["density"]) if (ref_c6["is_defined"] and dc["is_defined"]) else np.nan,
                "abs_longest_share_err": abs(p_ls - r_ls) if (np.isfinite(r_ls) and np.isfinite(p_ls)) else np.nan,
            })

    wl.close()

    # Export False Gaps Summary
    all_fg = pd.concat([rr_fg_df, pd.DataFrame(ef_fg_rows)], ignore_index=True)
    fg_sum = all_fg.groupby(["detector", "high_burden_ge80"]).agg(
        n_records=("record_id", "count"),
        total_prolonged_af_h=("prolonged_af_h", "sum"),
        total_false_gaps=("n_false_gaps", "sum"),
        false_gaps_per_af_hr=("n_false_gaps", lambda s: float(s.sum() / max(1e-6, all_fg.loc[s.index, "prolonged_af_h"].sum()))),
        false_nonaf_pct_of_prolonged_af=("false_nonaf_s", lambda s: float(s.sum() / (max(1e-6, all_fg.loc[s.index, "prolonged_af_h"].sum()) * 3600.0) * 100.0)),
    ).reset_index()
    fg_sum.to_csv(OUT_RES / "V8_e07_false_gaps_in_prolonged_af.csv", index=False)

    # Export Full 182-rec RR-GBDT + Matched 24-rec 6h Reconstruction
    rr_rec_df.to_csv(OUT_RES / "V8_e07_rr_full182_reconstruction.csv", index=False)
    m24_df = pd.DataFrame(matched_rec_rows)
    m24_df.to_csv(OUT_RES / "V8_e07_matched24_6h_reconstruction.csv", index=False)

    def summarize_rec(df, scope_lbl):
        sub = df[df["ref_defined"]].copy()
        sub["hi_burd"] = sub["ref_burden"] >= 0.80
        return sub.groupby(["detector", "hi_burd"]).agg(
            n_ref_defined=("record_id", "count"),
            n_pred_undef=("pred_defined", lambda s: int((~s).sum())),
            med_abs_ep_err=("abs_ep_count_err", "median") if "abs_ep_count_err" in sub.columns else ("ep_count_err", lambda s: float(np.median(np.abs(s)))),
            med_abs_burden_err=("abs_burden_err", "median"),
            mean_abs_dens_err=("abs_dens_err", "mean"),
            med_abs_dens_err=("abs_dens_err", "median"),
            med_abs_ls_err=("abs_longest_share_err", "median"),
        ).reset_index().assign(scope=scope_lbl)

    rec_sum = pd.concat([summarize_rec(rr_rec_df, "FullHorizon_182_Recs"), summarize_rec(m24_df, "Matched_24_Recs_6h")], ignore_index=True)
    rec_sum.to_csv(OUT_RES / "V8_e07_reconstruction_summary.csv", index=False)
    print("\n=== V8-E07 Full-Horizon Reconstruction Error Summary ===")
    print(rec_sum.to_string(index=False))

    # Representative & Worst Cases CSV
    cases = m24_df.sort_values(["detector", "abs_dens_err"], ascending=[True, False])
    cases.to_csv(OUT_RES / "V8_e07_representative_and_worst_cases.csv", index=False)
    with (OUT_RES / "V8_e07_ecgfounder_fit_info.json").open("w") as f:
        json.dump(ef_fit_info, f, indent=2)
    print(f"\nV8-E07 completed in {time.time() - t_start:.1f}s!")


if __name__ == "__main__":
    main()
