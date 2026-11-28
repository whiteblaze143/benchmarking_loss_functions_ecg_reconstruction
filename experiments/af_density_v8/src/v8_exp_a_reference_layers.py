"""Experiment A: Parallel Atrial-State-Aware Reference Layers (R0, R1, R2, R3).

Constructs without modifying native .atr files:
  R0 : Native harmonized (AFIB-only reference.
  R1 : Automated quantitative 2-lead ECG atrial-activity / QRST-cancelled waveform stratification
       (explicitly NOT human electrophysiologist blinded adjudication).
  R2 : Conservative UNKNOWN-masked reference sensitivity layers ((VT/(IVR/(B/(T and short (SBR).
  R3 : Operational minimum-interruption (<10s, <30s, <60s) and minimum-AF (<30s) rules.

Computes AF burden, Charitos AF density, Butkuviene/Petrenas AF aggregation ((1-B)*D = 2*Area),
n_runs, longest_share, state (ZERO_AF/MIXED/ALL_AF), and unknown fraction across:
  - All 19 high-burden (B >= 0.80) 12h LTAFDB patients
  - All 329 unique patients (12h) across LTAFDB (84), SHDB-AF (98), IRIDIA-AF (147)
  - 24h subset where observed.
"""
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, sosfiltfilt, welch

sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals, subtract_exclusions

OUT_RES = Path("experiments/af_density_v8/results")
OUT_FIG = Path("experiments/af_density_v8/figures")
OUT_RES.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")
NON_BEAT = {'+', '"', '~', '|', 's', 't', 'p', 'u', '='}


def metrics_from_intervals(intervals: List[Tuple[float, float]], T: float) -> Dict[str, Any]:
    """Return burden, Charitos density, Butkuviene AF aggregation, n_runs, longest_share, state."""
    ivs = merge_intervals([(max(0.0, a), min(T, b)) for a, b in intervals])
    comp = exact_reference_charitos_components(ivs, T)
    b = float(comp["burden"])
    if comp["is_defined"]:
        d = float(comp["density"])
        agg = float(comp["numerator_2A"])  # Continuous Butkuviene Eq. 4: 2 * Area = (1 - B) * D
        agg_z = agg
        state = "MIXED"
        durs = [e - s for s, e in ivs]
        lshare = float(max(durs) / sum(durs)) if durs else np.nan
    else:
        d = np.nan
        agg = np.nan
        agg_z = 0.0  # Continuous boundary limit as B->0+ or B->1-
        state = str(comp["undef_reason"])
        lshare = 1.0 if state == "ALL_AF" else np.nan
    return {
        "burden": b,
        "charitos_density": d,
        "af_aggregation": agg,
        "af_aggregation_zeroext": agg_z,
        "n_runs": int(comp["n_runs"]),
        "longest_share": lshare,
        "is_defined": bool(comp["is_defined"]),
        "state": state,
    }


def analyze_ltafdb_record_gaps(args) -> List[Dict[str, Any]]:
    """Compute quantitative 2-lead QRST-cancelled atrial & beat metrics for EVERY internal gap in one LTAFDB record."""
    rid, gap_rows = args
    rec = wfdb.rdrecord(str(LTAF_DIR / rid))
    ann = wfdb.rdann(str(LTAF_DIR / rid), "atr")
    fs = float(rec.fs)  # 128 Hz
    sig0 = np.nan_to_num(rec.p_signal[:, 0].astype(np.float64))
    sig1 = np.nan_to_num(rec.p_signal[:, 1].astype(np.float64)) if rec.p_signal.shape[1] > 1 else sig0.copy()
    sigs = [sig0, sig1]

    # Pre-filter bandpass 2.5 - 12 Hz for atrial fibrillatory residual & 25 - 55 Hz for HF muscle/noise
    sos_atrial = butter(3, [2.5, 12.0], btype="bandpass", fs=fs, output="sos")
    sos_hf = butter(2, [25.0, 55.0], btype="bandpass", fs=fs, output="sos")
    sig_hf0 = sosfiltfilt(sos_hf, sig0)
    sig_at = [sosfiltfilt(sos_atrial, s) for s in sigs]

    beat_mask = np.array([s not in NON_BEAT for s in ann.symbol], dtype=bool)
    b_samp = ann.sample[beat_mask]
    b_sym = np.array(ann.symbol)[beat_mask]
    pre_samp = int(round(0.070 * fs))
    post_samp = int(round(0.220 * fs))
    win_len = pre_samp + post_samp

    out = []
    n_samp = len(sig0)
    for g in gap_rows:
        ga, gb = float(g["gap_start_s"]), float(g["gap_end_s"])
        dur = gb - ga
        sa, sb = max(0, int(round(ga * fs))), min(n_samp, int(round(gb * fs)))
        pad_s = int(round(12.0 * fs))
        w_lo, w_hi = max(0, sa - pad_s), min(n_samp, sb + pad_s)

        # Beats strictly inside gap & in flanking 12s AF windows
        in_g = (b_samp >= sa) & (b_samp < sb)
        g_beats = b_samp[in_g]
        g_syms = b_sym[in_g]
        n_beats = int(len(g_beats))
        n_v = int(np.sum(g_syms == "V"))
        n_a = int(np.sum(g_syms == "A"))
        n_n = int(np.sum(g_syms == "N"))
        v_frac = float(n_v / n_beats) if n_beats > 0 else 0.0

        rr_gap = np.diff(g_beats) / fs if n_beats >= 2 else np.array([])
        med_rr = float(np.median(rr_gap)) if len(rr_gap) else np.nan
        cvrr_gap = float(np.std(rr_gap) / np.mean(rr_gap)) if len(rr_gap) >= 2 and np.mean(rr_gap) > 0 else np.nan

        flank_pre_b = b_samp[(b_samp >= w_lo) & (b_samp < sa)]
        flank_post_b = b_samp[(b_samp >= sb) & (b_samp < w_hi)]
        rr_pre = np.diff(flank_pre_b) / fs if len(flank_pre_b) >= 3 else np.array([])
        rr_post = np.diff(flank_post_b) / fs if len(flank_post_b) >= 3 else np.array([])
        flank_rr = np.concatenate([rr_pre, rr_post])
        flank_cvrr = float(np.std(flank_rr) / np.mean(flank_rr)) if len(flank_rr) >= 3 and np.mean(flank_rr) > 0 else np.nan

        # Signal quality / amplitude drop / HF burst check across Lead 0 & Lead 1
        g_s0 = sig0[sa:sb]
        fp0 = np.concatenate([sig0[w_lo:sa], sig0[sb:w_hi]])
        amp_ratio = (float(np.std(g_s0)) / (float(np.std(fp0)) + 1e-6)) if len(g_s0) >= 8 and len(fp0) >= 16 else 1.0
        hf_g = float(np.std(sig_hf0[sa:sb])) if sb - sa >= 8 else 0.0
        hf_f = float(np.std(np.concatenate([sig_hf0[w_lo:sa], sig_hf0[sb:w_hi]]))) if len(fp0) >= 16 else 1.0
        hf_ratio = hf_g / (hf_f + 1e-6)

        # 2-lead local adaptive median QRST cancellation over [w_lo, w_hi]
        loc_idx = (b_samp >= w_lo + pre_samp) & (b_samp <= w_hi - post_samp)
        loc_b = b_samp[loc_idx]
        lead_f_ratios = []
        lead_df_hz = np.nan
        lead_sc = np.nan
        lead_snr = np.nan

        if len(loc_b) >= 6 and (sb - sa) >= int(0.5 * fs):
            for l_i, s_raw in enumerate(sigs):
                # Subtract running local median QRST template on raw bandpass atrial+ventricular signal
                res = sig_at[l_i][w_lo:w_hi].copy()
                rel_b = loc_b - w_lo
                cpx = np.array([res[b - pre_samp : b + post_samp] for b in rel_b])
                n_loc = len(rel_b)
                for bi in range(n_loc):
                    k0, k1 = max(0, bi - 3), min(n_loc, bi + 4)
                    med_tpl = np.median(cpx[k0:k1], axis=0)
                    den = float(np.dot(med_tpl, med_tpl))
                    alpha = float(np.clip(np.dot(cpx[bi], med_tpl) / den, 0.5, 1.5)) if den > 1e-12 else 1.0
                    res[rel_b[bi] - pre_samp : rel_b[bi] + post_samp] -= alpha * med_tpl

                sa_r, sb_r = sa - w_lo, sb - w_lo
                res_gap = res[sa_r:sb_r]
                res_flk = np.concatenate([res[:sa_r], res[sb_r:]])
                if len(res_flk) >= int(2.0 * fs) and np.var(res_flk) > 1e-12:
                    lead_f_ratios.append(float(np.var(res_gap) / np.var(res_flk)))
                if l_i == 0 and len(res_gap) >= int(2.5 * fs):
                    nperseg = int(round(2.5 * fs))
                    freqs, psd = welch(res_gap, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, nfft=nperseg * 4)
                    bm = (freqs >= 3.0) & (freqs <= 9.0)
                    if bm.any():
                        bf, bp = freqs[bm], psd[bm]
                        pidx = int(np.argmax(bp))
                        df = float(bf[pidx])
                        pk = float(bp[pidx])
                        tpow = float(np.sum(bp))
                        sc_v = float(np.sum(psd[(freqs >= df - 0.5) & (freqs <= df + 0.5)]) / tpow) if tpow > 0 else 0.0
                        bg = float(np.median(bp))
                        lead_df_hz = df
                        lead_sc = sc_v
                        lead_snr = float(10.0 * np.log10(pk / bg)) if bg > 0 else 0.0

        f_power_ratio = float(np.nanmax(lead_f_ratios)) if lead_f_ratios else np.nan

        # Coherent multi-beat P-wave average across N-beats inside gap (suppresses random f-waves by sqrt(n_N))
        p_sync_snr = np.nan
        p_rms_ratio = np.nan
        n_pre_w = int(round(0.220 * fs))
        n_end_w = int(round(0.055 * fs))
        tp_a = int(round(0.320 * fs))
        tp_b = int(round(0.240 * fs))
        n_beats_list = [int(b) for b, sm in zip(g_beats, g_syms) if sm == "N" and (b - tp_a >= 0) and (b + 8 < n_samp)]
        if len(n_beats_list) >= 2:
            # Evaluate on both leads, take max coherent P-wave SNR across leads
            snrs, rms_r = [], []
            for s_raw in sigs:
                pr_segs = []
                tp_segs = []
                for b in n_beats_list:
                    tp = s_raw[b - tp_a : b - tp_b]
                    bl = float(np.median(tp))
                    pr_segs.append(s_raw[b - n_pre_w : b - n_end_w] - bl)
                    tp_segs.append(tp - bl)
                pr_arr = np.array(pr_segs)
                tp_arr = np.array(tp_segs)
                avg_pr = np.mean(pr_arr, axis=0)
                avg_tp = np.mean(tp_arr, axis=0)
                noise_tp = float(np.std(avg_tp)) + 1e-6
                snrs.append(float(np.max(np.abs(avg_pr)) / noise_tp))
                rms_r.append(float(np.std(avg_pr) / noise_tp))
            p_sync_snr = float(np.max(snrs))
            p_rms_ratio = float(np.max(rms_r))

        # Transparent, strictly automated quantitative waveform stratification (NOT clinical human EP reading)
        nt = g["native_tags"]
        if amp_ratio < 0.25 or hf_ratio > 2.5:
            r1_cls = "ARTIFACT_OR_DROPOUT"
        elif nt in ["VT", "IVR", "B", "T"] or "VT" in nt or "IVR" in nt or nt == "B" or v_frac >= 0.50:
            r1_cls = "VENTRICULAR_OVERRIDE_PROB_CONTINUATION"
        elif (n_n >= 3 and np.isfinite(cvrr_gap) and cvrr_gap < 0.12) or (np.isfinite(p_sync_snr) and p_sync_snr >= 3.0 and (not np.isfinite(f_power_ratio) or f_power_ratio < 0.60)):
            r1_cls = "ORGANIZED_NON_AF_PROB_TERMINATION"
        elif dur < 60.0 and ((np.isfinite(cvrr_gap) and cvrr_gap >= 0.15) or (n_beats <= 2 and np.isfinite(f_power_ratio) and f_power_ratio >= 0.50)) and (not np.isfinite(p_sync_snr) or p_sync_snr < 2.5):
            r1_cls = "IRREGULAR_FWAVE_PROB_CONTINUATION"
        else:
            r1_cls = "INDETERMINATE_SHORT_OR_MIXED"

        row_out = dict(g)
        row_out.update({
            "n_beats": n_beats, "n_N": n_n, "n_V": n_v, "n_A": n_a, "v_frac": round(v_frac, 4),
            "med_rr_s": round(med_rr, 4) if np.isfinite(med_rr) else np.nan,
            "cvrr_gap": round(cvrr_gap, 4) if np.isfinite(cvrr_gap) else np.nan,
            "flank_cvrr": round(flank_cvrr, 4) if np.isfinite(flank_cvrr) else np.nan,
            "amp_ratio": round(amp_ratio, 4), "hf_ratio": round(hf_ratio, 4),
            "f_power_ratio": round(f_power_ratio, 4) if np.isfinite(f_power_ratio) else np.nan,
            "df_hz": round(lead_df_hz, 3) if np.isfinite(lead_df_hz) else np.nan,
            "df_sc": round(lead_sc, 4) if np.isfinite(lead_sc) else np.nan,
            "df_snr_db": round(lead_snr, 3) if np.isfinite(lead_snr) else np.nan,
            "p_sync_snr": round(p_sync_snr, 4) if np.isfinite(p_sync_snr) else np.nan,
            "p_rms_ratio": round(p_rms_ratio, 4) if np.isfinite(p_rms_ratio) else np.nan,
            "r1_class": r1_cls,
        })
        out.append(row_out)
    return out


def project_known_horizon(known_ivs: List[Tuple[float, float]], af_ivs: List[Tuple[float, float]],
                          unknown_exclusions: List[Tuple[float, float]], horizon_s: float) -> Tuple[List[Tuple[float, float]], float, float]:
    """Subtract additional unknown_exclusions from known_ivs, then project first horizon_s of remaining KNOWN time.
    Returns (mapped_af, T_used, unknown_frac_over_raw_known).
    Note: preserves chronology without bridging across unknown gaps when computing layer effects."""
    base_known = merge_intervals(known_ivs)
    tot_base = sum(b - a for a, b in base_known)
    if horizon_s > tot_base + 1e-6:
        return [], 0.0, np.nan
    rem_known = subtract_exclusions(base_known, unknown_exclusions) if unknown_exclusions else base_known
    tot_rem = sum(b - a for a, b in rem_known)
    if tot_rem <= 1e-6:
        return [], 0.0, 1.0
    # Restrict to the same calendar time span [0, end_of_base_horizon] for strict apples-to-apples comparison!
    rem = float(horizon_s)
    end_calendar = base_known[-1][1]
    for ka, kb in base_known:
        if rem <= 1e-9:
            break
        take = min(kb - ka, rem)
        end_calendar = ka + take
        rem -= take
    clipped_base = [(ka, min(kb, end_calendar)) for ka, kb in base_known if ka < end_calendar - 1e-9]
    clipped_rem = subtract_exclusions(clipped_base, unknown_exclusions) if unknown_exclusions else clipped_base
    rem_dur = sum(b - a for a, b in clipped_rem)
    unk_frac = max(0.0, 1.0 - rem_dur / float(horizon_s))
    if rem_dur <= 1e-6:
        return [], 0.0, unk_frac
    # Map remaining known intervals sequentially onto [0, rem_dur]
    af_clean = subtract_exclusions(merge_intervals(af_ivs), unknown_exclusions) if unknown_exclusions else merge_intervals(af_ivs)
    mapped_af = []
    k_el = 0.0
    for ka, kb in clipped_rem:
        for aa, ab in af_clean:
            oa, ob = max(ka, aa), min(kb, ab)
            if ob > oa + 1e-9:
                mapped_af.append((k_el + (oa - ka), k_el + (ob - ka)))
        k_el += (kb - ka)
    return merge_intervals(mapped_af), rem_dur, unk_frac


def apply_r3_rule(known_ivs: List[Tuple[float, float]], af_ivs: List[Tuple[float, float]], horizon_s: float,
                  bridge_gap_lt_s: float = 0.0, min_af_s: float = 0.0) -> List[Tuple[float, float]]:
    """Build operational R3 AF intervals over exact 12h/24h known clock without bridging across unknown gaps."""
    base_known = merge_intervals(known_ivs)
    af_sorted = merge_intervals(af_ivs)
    rem = float(horizon_s)
    k_el = 0.0
    mapped_seg = []
    for sidx, (ka, kb) in enumerate(base_known):
        if rem <= 1e-9:
            break
        take = min(kb - ka, rem)
        seg_end = ka + take
        for aa, ab in af_sorted:
            oa, ob = max(aa, ka), min(ab, seg_end)
            if ob > oa + 1e-9:
                mapped_seg.append((k_el + (oa - ka), k_el + (ob - ka), sidx))
        k_el += take
        rem -= take
    if not mapped_seg:
        return []
    if bridge_gap_lt_s > 0.0:
        br = [[mapped_seg[0][0], mapped_seg[0][1], mapped_seg[0][2]]]
        for a, b, si in mapped_seg[1:]:
            if si == br[-1][2] and (a - br[-1][1]) < bridge_gap_lt_s:
                br[-1][1] = b
            else:
                br.append([a, b, si])
        out = merge_intervals([(a, b) for a, b, _ in br])
    else:
        out = merge_intervals([(a, b) for a, b, _ in mapped_seg])
    if min_af_s > 0.0:
        out = [(a, b) for a, b in out if (b - a) >= min_af_s]
    return out


def main():
    t0 = time.time()
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    cal = pd.read_csv(IRIDIA_META / "calibration_exclusion_intervals.csv")
    cal_grp = cal.groupby("record_id")
    ann = inv[inv["is_annotated"]].copy()

    # 1. Extract EVERY known internal non-AF gap in LTAFDB + run quantitative 2-lead waveform & beat analysis
    print("=== Step A1: Quantitative 2-Lead Waveform Characterization Across ALL LTAFDB Internal Gaps ===")
    ltaf_tasks = []
    ltaf_rhy = rhy[rhy["dataset"] == "LTAFDB"]
    ltaf_runs = runs[runs["dataset"] == "LTAFDB"]
    for rid, agrp in ltaf_runs.groupby("record_id"):
        rid = str(rid)
        agrp = agrp.sort_values("start_time_s")
        s, e = agrp["start_time_s"].values, agrp["end_time_s"].values
        sub_r = ltaf_rhy[(ltaf_rhy["record_id"].astype(str) == rid) & (ltaf_rhy["rhythm_label_harmonized"] != "AF")]
        g_rows = []
        for i in range(len(s) - 1):
            ga, gb = float(e[i]), float(s[i + 1])
            dur = gb - ga
            if dur <= 1e-6:
                continue
            ov = sub_r[(sub_r["end_time_s"] > ga + 1e-6) & (sub_r["start_time_s"] < gb - 1e-6)]
            if len(ov) and (~ov["rhythm_known"]).any():
                continue
            lbls = "+".join(sorted(ov["rhythm_label_native"].astype(str).unique())) if len(ov) else "NONE"
            g_rows.append({"record_id": rid, "gap_start_s": ga, "gap_end_s": gb, "dur_s": dur, "native_tags": lbls})
        if g_rows:
            ltaf_tasks.append((rid, g_rows))

    all_gaps = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(analyze_ltafdb_record_gaps, ltaf_tasks):
            all_gaps.extend(res)
    gdf = pd.DataFrame(all_gaps)
    gdf["dur_bin"] = pd.cut(gdf["dur_s"], bins=[0, 10, 30, 60, 120, 360, 1e12],
                            labels=["01_lt10s", "02_10_30s", "03_30_60s", "04_1_2m", "05_2_6m", "06_ge6m"])

    # Identify the 19 high-burden (B >= 0.80) 12h LTAFDB records explicitly
    e1_full = pd.read_parquet(OUT_RES / "V8_e01_e02_perturbations_full.parquet")
    hb19_df = e1_full[(e1_full["dataset"] == "LTAFDB") & (e1_full["horizon_h"] == 12.0) &
                      (e1_full["perturbation"] == "reference") & (e1_full["ref_burden"] >= 0.80)]
    hb19_ids = sorted(hb19_df["record_id"].astype(str).unique())
    print(f"Identified {len(hb19_ids)} high-burden (>=0.80) 12h LTAFDB records: {hb19_ids}")
    gdf["is_hb19_record"] = gdf["record_id"].isin(hb19_ids)
    gdf.to_parquet(OUT_RES / "V8_exp_a_ltafdb_all_gaps_waveform.parquet", index=False)

    # Gap cross-tab summary
    gs = gdf.groupby(["is_hb19_record", "dur_bin", "r1_class"], observed=True).agg(
        n_gaps=("dur_s", "count"),
        med_dur_s=("dur_s", "median"),
        med_f_power_ratio=("f_power_ratio", "median"),
        med_p_sync_snr=("p_sync_snr", "median"),
        med_cvrr_gap=("cvrr_gap", "median"),
        med_n_beats=("n_beats", "median"),
    ).reset_index()
    gs.to_csv(OUT_RES / "V8_exp_a_ltafdb_gap_strat_summary.csv", index=False)
    print("\nLTAFDB All Internal Gaps Automated 2-Lead Waveform Stratification (HB19 vs Others):")
    print(gs[gs["dur_bin"].isin(["01_lt10s", "02_10_30s", "03_30_60s"])].to_string(index=False))

    # Index per-record R1 gap subsets for constructing R1_StrictProbAF and R1_ConsUnknown
    gaps_by_rec = {rid: grp for rid, grp in gdf.groupby("record_id")}

    # 2. Construct R0, R1, R2, R3 across all 349 annotated recordings (12h & 24h)
    print("\n=== Step A2: Computing R0, R1, R2, R3 Reference Layers Across All Cohorts ===")
    rec_rows = []
    for rec in ann.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        if ds in ["LTAFDB", "SHDB-AF"]:
            sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
            known_ivs = list(zip(sub_r["start_time_s"], sub_r["end_time_s"]))
            sub_af = runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]
            af_ivs = list(zip(sub_af["start_time_s"], sub_af["end_time_s"]))
        else:
            raw_end = rec.raw_samples / rec.fs
            if rid in cal_grp.groups:
                cs = cal_grp.get_group(rid)
                ex_ivs = [(s / 200.0, e / 200.0) for s, e in zip(cs["excluded_start_sample"], cs["excluded_end_sample_exclusive"])]
            else:
                ex_ivs = [(0.0, min(raw_end, 30.0))]
            known_ivs = subtract_exclusions([(0.0, raw_end)], ex_ivs)
            sub_af = runs[(runs["dataset"] == "IRIDIA-AF") & (runs["record_id"].astype(str) == rid)]
            af_ivs = subtract_exclusions(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])), ex_ivs)
            sub_r = pd.DataFrame()

        tot_known_s = sum(b - a for a, b in merge_intervals(known_ivs))

        # Layer exclusions / modifications on the raw clock before horizon clipping
        vt_ivs, sbr10_ivs, sbr60_ivs, sub10_ivs = [], [], [], []
        if ds in ["LTAFDB", "SHDB-AF"] and len(sub_r):
            vt_sub = sub_r[sub_r["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])]
            vt_ivs = list(zip(vt_sub["start_time_s"], vt_sub["end_time_s"]))
            sbr_sub = sub_r[sub_r["rhythm_label_native"].astype(str) == "SBR"]
            sbr10_ivs = [(float(a), float(b)) for a, b in zip(sbr_sub["start_time_s"], sbr_sub["end_time_s"]) if (b - a) < 10.0]
            sbr60_ivs = [(float(a), float(b)) for a, b in zip(sbr_sub["start_time_s"], sbr_sub["end_time_s"]) if (b - a) < 60.0]
            nonaf_sub = sub_r[sub_r["rhythm_label_harmonized"] != "AF"]
            sub10_ivs = [(float(a), float(b)) for a, b in zip(nonaf_sub["start_time_s"], nonaf_sub["end_time_s"]) if (b - a) < 10.0]

        r1_prob_af_add, r1_art_ex, r1_cons_ex = [], [], []
        if ds == "LTAFDB" and rid in gaps_by_rec:
            rg = gaps_by_rec[rid]
            for gr in rg.itertuples():
                iv = (float(gr.gap_start_s), float(gr.gap_end_s))
                if gr.r1_class in ["VENTRICULAR_OVERRIDE_PROB_CONTINUATION", "IRREGULAR_FWAVE_PROB_CONTINUATION"]:
                    r1_prob_af_add.append(iv)
                    r1_cons_ex.append(iv)
                elif gr.r1_class in ["ARTIFACT_OR_DROPOUT", "INDETERMINATE_SHORT_OR_MIXED"]:
                    if gr.r1_class == "ARTIFACT_OR_DROPOUT":
                        r1_art_ex.append(iv)
                    r1_cons_ex.append(iv)

        for h_lbl, h_s in [("12h", 43200.0), ("24h", 86400.0)]:
            if tot_known_s + 1e-6 < h_s:
                continue
            # R0 baseline
            af_r0, t_r0, _ = project_known_horizon(known_ivs, af_ivs, [], h_s)
            m_r0 = metrics_from_intervals(af_r0, t_r0)

            layer_specs = {
                "R0_Native": (af_r0, t_r0, 0.0),
                "R3_gap10s": (apply_r3_rule(known_ivs, af_ivs, h_s, bridge_gap_lt_s=10.0), h_s, 0.0),
                "R3_gap30s": (apply_r3_rule(known_ivs, af_ivs, h_s, bridge_gap_lt_s=30.0), h_s, 0.0),
                "R3_gap60s": (apply_r3_rule(known_ivs, af_ivs, h_s, bridge_gap_lt_s=60.0), h_s, 0.0),
                "R3_ep30s": (apply_r3_rule(known_ivs, af_ivs, h_s, min_af_s=30.0), h_s, 0.0),
                "R3_comp30s": (apply_r3_rule(known_ivs, af_ivs, h_s, bridge_gap_lt_s=30.0, min_af_s=30.0), h_s, 0.0),
            }
            # R2 layers
            for l_name, ex in [
                ("R2_VT_Unknown", vt_ivs),
                ("R2_VT_SBR10_Unknown", vt_ivs + sbr10_ivs),
                ("R2_VT_SBR60_Unknown", vt_ivs + sbr60_ivs),
                ("R2_Cons_Sub10_Unknown", vt_ivs + sub10_ivs),
            ]:
                af_l, t_l, uf_l = project_known_horizon(known_ivs, af_ivs, ex, h_s)
                layer_specs[l_name] = (af_l, t_l, uf_l)

            # R1 layers (LTAFDB only; equal to R0 for SHDB/IRIDIA)
            af_r1_strict_raw = merge_intervals(af_ivs + r1_prob_af_add)
            af_r1s, t_r1s, uf_r1s = project_known_horizon(known_ivs, af_r1_strict_raw, r1_art_ex, h_s)
            layer_specs["R1_Waveform_ProbAF"] = (af_r1s, t_r1s, uf_r1s)

            af_r1c, t_r1c, uf_r1c = project_known_horizon(known_ivs, af_ivs, r1_cons_ex, h_s)
            layer_specs["R1_Waveform_ConsUnknown"] = (af_r1c, t_r1c, uf_r1c)

            for l_name, (af_l, t_l, uf_l) in layer_specs.items():
                if t_l <= 1e-6:
                    mm = {"burden": np.nan, "charitos_density": np.nan, "af_aggregation": np.nan,
                          "af_aggregation_zeroext": np.nan, "n_runs": 0, "longest_share": np.nan,
                          "is_defined": False, "state": "ALL_UNKNOWN"}
                else:
                    mm = metrics_from_intervals(af_l, t_l)
                rec_rows.append({
                    "dataset": ds, "record_id": rid, "patient_id": pid, "horizon": h_lbl,
                    "layer": l_name, "is_hb19": (ds == "LTAFDB" and rid in hb19_ids),
                    "r0_burden": m_r0["burden"], "r0_density": m_r0["charitos_density"],
                    "r0_aggregation": m_r0["af_aggregation"], "r0_state": m_r0["state"],
                    "r0_defined": m_r0["is_defined"], "t_known_eff_s": round(t_l, 3),
                    "unknown_masked_frac": round(float(uf_l), 5),
                    **mm,
                    "delta_density_vs_r0": (mm["charitos_density"] - m_r0["charitos_density"]) if (mm["is_defined"] and m_r0["is_defined"]) else np.nan,
                    "delta_agg_vs_r0": (mm["af_aggregation"] - m_r0["af_aggregation"]) if (mm["is_defined"] and m_r0["is_defined"]) else np.nan,
                })

    ldf = pd.DataFrame(rec_rows)
    ldf.to_parquet(OUT_RES / "V8_exp_a_all_layers_records.parquet", index=False)

    # Save exact 19-patient high-burden LTAFDB table (12h)
    hb_tbl = ldf[(ldf["is_hb19"]) & (ldf["horizon"] == "12h")].copy()
    hb_tbl.to_csv(OUT_RES / "V8_exp_a_highburden19_patient_table.csv", index=False)
    print("\n19 High-Burden (>=0.80) LTAFDB 12h Patients Across Reference Layers:")
    hb_sum = hb_tbl.groupby("layer").agg(
        n_recs=("record_id", "count"),
        n_mixed=("state", lambda s: int((s == "MIXED").sum())),
        n_all_af=("state", lambda s: int((s == "ALL_AF").sum())),
        n_all_unk=("state", lambda s: int((s == "ALL_UNKNOWN").sum())),
        med_burden=("burden", "median"),
        med_density=("charitos_density", "median"),
        iqr_density=("charitos_density", lambda s: float(s.quantile(0.75) - s.quantile(0.25)) if s.notna().sum() >= 2 else np.nan),
        med_agg=("af_aggregation", "median"),
        med_n_runs=("n_runs", "median"),
        mean_unk_frac=("unknown_masked_frac", "mean"),
    ).reset_index()
    print(hb_sum.to_string(index=False))

    # Patient-deduplicated 12h & 24h cohort summary across ALL 329 patients!
    pt_df = ldf.sort_values(["horizon", "layer", "dataset", "patient_id", "record_id"]).drop_duplicates(["horizon", "layer", "dataset", "patient_id"])
    pt_df["r0_stratum"] = np.where(pt_df["r0_burden"] >= 0.80, "HighBurden_ge80pct",
                          np.where(pt_df["r0_burden"] > 0.0, "LowModBurden_0_80pct", "ZeroAF"))
    csum = pt_df.groupby(["horizon", "dataset", "r0_stratum", "layer"]).agg(
        n_patients=("patient_id", "count"),
        pct_defined=("is_defined", lambda s: round(100.0 * s.mean(), 2)),
        n_mixed=("state", lambda s: int((s == "MIXED").sum())),
        n_all_af=("state", lambda s: int((s == "ALL_AF").sum())),
        n_zero_af=("state", lambda s: int((s == "ZERO_AF").sum())),
        med_burden=("burden", "median"),
        med_density=("charitos_density", "median"),
        med_abs_ddens=("delta_density_vs_r0", lambda s: float(np.nanmedian(np.abs(s))) if s.notna().any() else np.nan),
        med_agg=("af_aggregation", "median"),
        med_abs_dagg=("delta_agg_vs_r0", lambda s: float(np.nanmedian(np.abs(s))) if s.notna().any() else np.nan),
        mean_unk_masked=("unknown_masked_frac", "mean"),
    ).reset_index()
    csum.to_csv(OUT_RES / "V8_exp_a_cohort_layers_summary.csv", index=False)

    # Also overall cohort (all strata combined per dataset, 12h) for quick clarity
    overall12 = pt_df[pt_df["horizon"] == "12h"].groupby(["dataset", "layer"]).agg(
        n_patients=("patient_id", "count"),
        pct_defined=("is_defined", lambda s: round(100.0 * s.mean(), 2)),
        n_all_af=("state", lambda s: int((s == "ALL_AF").sum())),
        med_density=("charitos_density", "median"),
        med_agg=("af_aggregation", "median"),
        med_abs_ddens=("delta_density_vs_r0", lambda s: float(np.nanmedian(np.abs(s))) if s.notna().any() else np.nan),
        med_abs_dagg=("delta_agg_vs_r0", lambda s: float(np.nanmedian(np.abs(s))) if s.notna().any() else np.nan),
        mean_unk_masked=("unknown_masked_frac", "mean"),
    ).reset_index()
    overall12.to_csv(OUT_RES / "V8_exp_a_overall_12h_summary.csv", index=False)
    print("\n12h Cohort Summary (LTAFDB High-Burden >=80% Stratum):")
    print(csum[(csum["horizon"] == "12h") & (csum["r0_stratum"] == "HighBurden_ge80pct")].to_string(index=False))

    # Plot Figure A: 19 High-Burden LTAFDB trajectories across R0 -> R1 -> R2 -> R3
    layers_plot = ["R0_Native", "R2_VT_Unknown", "R2_VT_SBR60_Unknown", "R1_Waveform_ProbAF", "R3_gap10s", "R3_gap60s"]
    labels_plot = ["R0\nNative .atr", "R2a\nVT Masked", "R2b\nVT+SBR<60s Masked", "R1\n2-Lead Waveform", "R3a\nBridge <10s", "R3b\nBridge <60s"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    p_sub = hb_tbl[hb_tbl["layer"].isin(layers_plot)].copy()
    p_sub["layer"] = pd.Categorical(p_sub["layer"], categories=layers_plot, ordered=True)
    agg_p = p_sub.groupby("layer", observed=True).agg(
        pct_mixed=("state", lambda s: 100.0 * (s == "MIXED").mean()),
        pct_allaf=("state", lambda s: 100.0 * (s == "ALL_AF").mean()),
        med_dens=("charitos_density", "median"),
        med_agg=("af_aggregation_zeroext", "median"),
    ).reset_index()
    x = np.arange(len(layers_plot))
    w = 0.36
    ax1.bar(x - w / 2, agg_p["pct_mixed"], width=w, color="#4c72b0", label="MIXED (Density Defined)")
    ax1.bar(x + w / 2, agg_p["pct_allaf"], width=w, color="#dd8452", label="ALL_AF (Density Undefined)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels_plot, fontsize=8.5)
    ax1.set_ylabel("% of High-Burden (B≥0.80) Patients (n=19)", fontsize=10)
    ax1.set_title("A. High-Burden LTAFDB Regime Across Reference Layers", fontsize=10.5, fontweight="bold")
    ax1.set_ylim(0, 108); ax1.grid(axis="y", alpha=0.25); ax1.legend(fontsize=8.5)

    ax2.plot(x, agg_p["med_dens"], marker="o", lw=2.2, color="#c44e52", label="Median Charitos Density (Defined Only)")
    ax2.plot(x, agg_p["med_agg"], marker="s", lw=2.2, color="#2ca02c", label="Median Butkuvienė AF Aggregation ((1-B)D, 0-ext)")
    for xi, (dv, av) in enumerate(zip(agg_p["med_dens"], agg_p["med_agg"])):
        if np.isfinite(dv):
            ax2.annotate(f"{dv:.2f}", (xi, dv), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#c44e52")
        ax2.annotate(f"{av:.3f}", (xi, av), textcoords="offset points", xytext=(0, -13), ha="center", fontsize=8, color="#2ca02c")
    ax2.set_xticks(x); ax2.set_xticklabels(labels_plot, fontsize=8.5)
    ax2.set_ylabel("Metric Value", fontsize=10)
    ax2.set_title("B. Normalized Density Fragility vs AF Aggregation Stability", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(-0.08, 1.08); ax2.grid(alpha=0.25); ax2.legend(fontsize=8.5)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "V8_exp_a_reference_layers.png", dpi=240)
    fig.savefig(OUT_FIG / "V8_exp_a_reference_layers.pdf")
    plt.close(fig)
    print(f"\nCompleted Experiment A in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
