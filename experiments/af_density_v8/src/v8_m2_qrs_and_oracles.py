"""AF-DENSITY V8 Manuscript-Readiness M2:
Priorities 4, 5, 6, 7, 8 across ALL 182 full external recordings & ALL 7,992 internal gaps:
1. End-to-end NeuroKit2 automatic single-lead QRS detection -> frozen RR-GBDT (RR_GBDT_AutoQRS)
2. Frozen RR-GBDT using supplied beats (RR_GBDT_RefBeats)
3. Perfect-Annotation 10s Non-Overlapping Windowing Oracle (Oracle_10sGrid)
4. Perfect-Annotation 60-Beat RR Windowing Oracle (Oracle_60BeatRR)
Plus 3x3 rhythm-state confusion matrices, continuity concealment stats, and gap-level sensitivity.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, "experiments/af_density")
from rr_features import features as rr_features, layout as rr_layout
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import clip_ivs_to_known, intervals_overlap_mask, gap_bin_label

OUT_RES = Path("experiments/af_density_v8/results")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
RR_BASE = Path("/data/mithunmanivannan/codex_artifacts/af_density_20261120/rr_baseline_v1")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
NON_BEAT_LTAF = {'+', '"', '~', '|', 's', 't', 'p', 'u', '='}


def rhythm_state(comp: Dict) -> str:
    if comp["is_defined"]:
        return "MIXED"
    return comp["undef_reason"]  # ZERO_AF or ALL_AF


def run_oracle_10s_grid(ref_k: List[Tuple[float, float]], T_k: float) -> List[Tuple[float, float]]:
    """Discretize perfect reference AF intervals on contiguous 10s grid (>=5.0s AF inside bin)."""
    n_bins = int(T_k // 10.0)
    if n_bins <= 0 or not ref_k:
        return []
    out = []
    j = 0
    for wi in range(n_bins):
        ba, bb = wi * 10.0, (wi + 1) * 10.0
        while j < len(ref_k) and ref_k[j][1] <= ba:
            j += 1
        ov = 0.0
        k = j
        while k < len(ref_k) and ref_k[k][0] < bb:
            oa, ob = max(ba, ref_k[k][0]), min(bb, ref_k[k][1])
            if ob > oa:
                ov += (ob - oa)
            k += 1
        if ov >= 5.0:
            out.append((ba, bb))
    return merge_intervals(out)


def eval_beats_to_rr_model_and_oracle(q: np.ndarray, fs: float, first_k: float, ref_raw_af: List[Tuple[float, float]], rr_model, thr: float):
    """Run both frozen RR-GBDT and (if ref beats) 60-beat majority Oracle on beat array q."""
    q = np.sort(q)
    q = q[np.r_[True, np.diff(q) > 0]]
    q = q[q >= first_k]
    if len(q) < 62:
        return [], []
    rr_ms = np.diff(q).astype(float) * 1000.0 / fs
    starts, assigned, ends = rr_layout(len(rr_ms), 0)
    if len(starts) == 0:
        return [], []
    X, valid = rr_features(rr_ms[starts[:, None] + np.arange(60)])
    scores = np.full(len(starts), np.nan)
    if valid.any():
        scores[valid] = rr_model.predict_proba(X[valid].astype(np.float32))[:, 1]
    gbdt_ivs = []
    for i, (lo, hi) in enumerate(zip(assigned, ends)):
        if valid[i] and scores[i] >= thr:
            t0, t1 = float(q[lo]) / fs, float(q[hi]) / fs
            if t1 > t0:
                gbdt_ivs.append((t0, t1))

    # 60-beat majority oracle on the exact same beat timestamps (using exact endpoint rhythm label, matching rr_protocol)
    end_t = q[1:].astype(float) / fs
    truth = np.zeros(len(end_t), dtype=np.int8)
    for aa, ab in ref_raw_af:
        truth[(end_t >= aa) & (end_t < ab)] = 1
    pref = np.r_[0, np.cumsum(truth)]
    oracle_60_af = ((pref[starts + 60] - pref[starts]) >= 30)
    oracle_ivs = []
    for i, (lo, hi) in enumerate(zip(assigned, ends)):
        if oracle_60_af[i]:
            t0, t1 = float(q[lo]) / fs, float(q[hi]) / fs
            if t1 > t0:
                oracle_ivs.append((t0, t1))
    return merge_intervals(gbdt_ivs), merge_intervals(oracle_ivs)


def detect_qrs_one_record(args):
    ds, rid = args
    base = str((LTAF_DIR if ds == "LTAFDB" else SHDB_DIR) / rid)
    rec = wfdb.rdrecord(base, channels=[0])
    sig = np.nan_to_num(rec.p_signal[:, 0].astype(np.float32))
    fs = float(rec.fs)
    _, info = nk.ecg_peaks(sig, sampling_rate=int(round(fs)), method="neurokit")
    q_auto = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)

    # Supplied reference / source beats
    if ds == "SHDB-AF":
        q_ref = wfdb.rdann(base, "qrs").sample
    else:
        ann = wfdb.rdann(base, "atr")
        q_ref = ann.sample[np.array([s not in NON_BEAT_LTAF for s in ann.symbol], dtype=bool)]
    q_ref = np.sort(q_ref)
    q_ref = q_ref[np.r_[True, np.diff(q_ref) > 0]]

    # Match automatic vs reference beats within +/- 100 ms tolerance using two-pointer merge
    tol = int(round(0.10 * fs))
    tp = 0
    i, j = 0, 0
    while i < len(q_ref) and j < len(q_auto):
        d = int(q_auto[j]) - int(q_ref[i])
        if abs(d) <= tol:
            tp += 1; i += 1; j += 1
        elif q_auto[j] < q_ref[i]:
            j += 1
        else:
            i += 1
    sens = tp / len(q_ref) if len(q_ref) else np.nan
    ppv = tp / len(q_auto) if len(q_auto) else np.nan
    return ds, rid, fs, q_auto, q_ref, len(q_ref), len(q_auto), tp, sens, ppv


def main():
    t0 = time.time()
    s_df = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    rr_model = joblib.load(RR_BASE / "model.joblib")
    rr_thr = 0.48
    ext_recs = s_df[(s_df["dataset"].isin(["LTAFDB", "SHDB-AF"])) & (s_df["is_annotated"])].copy()

    tasks = [(r.dataset, str(r.record_id)) for r in ext_recs.itertuples()]
    print(f"=== Running NeuroKit2 Automatic Single-Lead QRS Detection Across All {len(tasks)} Full Recordings (8 workers) ===")
    qrs_cache = {}
    qrs_quality = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(detect_qrs_one_record, t) for t in tasks]
        for fut in as_completed(futs):
            ds, rid, fs, q_auto, q_ref, n_ref, n_auto, tp, sens, ppv = fut.result()
            qrs_cache[(ds, rid)] = (fs, q_auto, q_ref)
            qrs_quality.append({
                "dataset": ds, "record_id": rid, "n_ref_beats": n_ref, "n_auto_beats": n_auto,
                "tp_beats": tp, "qrs_sensitivity": round(float(sens), 4), "qrs_ppv": round(float(ppv), 4),
            })
    qq_df = pd.DataFrame(qrs_quality)
    qq_df.to_csv(OUT_RES / "V8_m2_autoqrs_quality.csv", index=False)
    print(f"Auto-QRS detection finished in {time.time() - t0:.1f}s:")
    print(qq_df.groupby("dataset")[["qrs_sensitivity", "qrs_ppv"]].agg(["median", "mean", "min"]).to_string())

    print("\n=== Evaluating All 4 Comparators Across Identical 182 Full Recordings & 7,992 Gaps ===")
    rec_rows = []
    gap_rows = []
    fg_rows = []

    for rec in ext_recs.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        fs, q_auto, q_ref = qrs_cache[(ds, rid)]
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
        ref_state = rhythm_state(ref_c)
        ref_ls = (max((b - a for a, b in ref_k), default=0.0) / ref_c["af_s"]) if ref_c["af_s"] > 0 else np.nan

        first_k = known_ivs[0][0] * fs if known_ivs else 0.0
        raw_rr_ref, raw_ora_60 = eval_beats_to_rr_model_and_oracle(q_ref, fs, first_k, af_raw, rr_model, rr_thr)
        raw_rr_auto, _ = eval_beats_to_rr_model_and_oracle(q_auto, fs, first_k, af_raw, rr_model, rr_thr)
        k_rr_ref, _ = clip_ivs_to_known(raw_rr_ref, known_ivs)
        k_rr_auto, _ = clip_ivs_to_known(raw_rr_auto, known_ivs)
        k_ora_60, _ = clip_ivs_to_known(raw_ora_60, known_ivs)
        k_ora_10 = run_oracle_10s_grid(ref_k, T_k)

        true_gaps = [(ref_k[i][1], ref_k[i + 1][0]) for i in range(len(ref_k) - 1) if (ref_k[i + 1][0] - ref_k[i][1]) > 1e-6] if len(ref_k) >= 2 else []
        prolonged_af = [(a, b) for a, b in ref_k if (b - a) >= 600.0]

        for cname, pred_k in [
            ("RR_GBDT_RefBeats", k_rr_ref),
            ("RR_GBDT_AutoQRS", k_rr_auto),
            ("Oracle_60BeatRR", k_ora_60),
            ("Oracle_10sGrid", k_ora_10),
        ]:
            pc = exact_reference_charitos_components(pred_k, T_k)
            pls = (max((b - a for a, b in pred_k), default=0.0) / pc["af_s"]) if pc["af_s"] > 0 else np.nan
            rec_rows.append({
                "comparator": cname, "dataset": ds, "record_id": rid, "patient_id": pid,
                "has_competing_afl_at": competing,
                "ref_state": ref_state, "pred_state": rhythm_state(pc),
                "ref_defined": ref_c["is_defined"], "pred_defined": pc["is_defined"],
                "ref_burden": ref_c["burden"], "pred_burden": pc["burden"],
                "abs_burden_err": abs(pc["burden"] - ref_c["burden"]),
                "ref_n_runs": ref_c["n_runs"], "pred_n_runs": pc["n_runs"],
                "ep_count_err": pc["n_runs"] - ref_c["n_runs"],
                "ref_density": ref_c["density"], "pred_density": pc["density"],
                "signed_dens_err": (pc["density"] - ref_c["density"]) if (ref_c["is_defined"] and pc["is_defined"]) else np.nan,
                "abs_dens_err": abs(pc["density"] - ref_c["density"]) if (ref_c["is_defined"] and pc["is_defined"]) else np.nan,
                "ref_ls": ref_ls, "pred_ls": pls,
                "abs_ls_err": abs(pls - ref_ls) if (np.isfinite(ref_ls) and np.isfinite(pls)) else np.nan,
            })

            pred_nonaf = subtract_exclusions([(0.0, T_k)], pred_k)
            if true_gaps:
                ovs = intervals_overlap_mask(true_gaps, pred_nonaf)
                for (ga, gb), ov in zip(true_gaps, ovs):
                    gd = gb - ga
                    gap_rows.append({
                        "comparator": cname, "dataset": ds, "record_id": rid,
                        "high_burden_ge80": ref_c["burden"] >= 0.80,
                        "true_gap_s": gd, "gap_bin": gap_bin_label(gd),
                        "pred_overlap_s": ov, "det_any": int(ov > 0.5),
                        "det_ge50": int(ov >= 0.5 * gd), "dur_err_s": ov - gd,
                    })

            if prolonged_af:
                nf, fsec = 0, 0.0
                for pa, pb in pred_nonaf:
                    for aa, ab in prolonged_af:
                        oa, ob = max(pa, aa), min(pb, ab)
                        if ob > oa + 0.5:
                            nf += 1; fsec += (ob - oa)
                fg_rows.append({
                    "comparator": cname, "dataset": ds, "record_id": rid,
                    "high_burden_ge80": ref_c["burden"] >= 0.80,
                    "prolonged_af_h": sum(b - a for a, b in prolonged_af) / 3600.0,
                    "n_false_gaps": nf, "false_nonaf_s": fsec,
                })

    rdf = pd.DataFrame(rec_rows)
    gdf = pd.DataFrame(gap_rows)
    fgdf = pd.DataFrame(fg_rows)
    rdf.to_csv(OUT_RES / "V8_m2_182rec_all_comparators.csv", index=False)

    # 3x3 Confusion matrices
    conf = rdf.groupby(["comparator", "ref_state", "pred_state"]).size().reset_index(name="n_records")
    conf.to_csv(OUT_RES / "V8_m2_state_confusion_matrices.csv", index=False)
    print("\n3x3 Rhythm-State Confusion Matrix (All 182 External Recordings):")
    print(conf.to_string(index=False))

    # Continuity concealment + high-burden error decomposition
    conc_rows = []
    for cname, grp in rdf.groupby("comparator"):
        both_def = grp[grp["ref_defined"] & grp["pred_defined"]]
        exact_bd = both_def[(both_def["abs_burden_err"] < 0.01) & (both_def["abs_dens_err"] < 0.01)]
        concealed = exact_bd[exact_bd["abs_ls_err"] > 0.30]
        for hi_b in [False, True]:
            sub = both_def[(both_def["ref_burden"] >= 0.80) == hi_b]
            conc_rows.append({
                "comparator": cname, "high_burden_ge80": hi_b,
                "n_both_def": len(sub),
                "n_ref_def": int((grp["ref_defined"] & ((grp["ref_burden"] >= 0.80) == hi_b)).sum()),
                "med_abs_burden_err": round(float(sub["abs_burden_err"].median()), 5),
                "med_abs_dens_err": round(float(sub["abs_dens_err"].median()), 4),
                "mean_abs_dens_err": round(float(sub["abs_dens_err"].mean()), 4),
                "med_abs_ls_err": round(float(sub["abs_ls_err"].median()), 4),
                "n_accurate_b_and_d_lt0.01_all": len(exact_bd),
                "n_concealed_ls_gt0.30_all": len(concealed),
                "pct_concealed_ls_gt0.30_all": round(100.0 * len(concealed) / max(1, len(exact_bd)), 1),
            })
    cdf = pd.DataFrame(conc_rows)
    cdf.to_csv(OUT_RES / "V8_m2_continuity_concealment.csv", index=False)
    print("\nReconstruction Errors & Longest-Share Concealment (All 182 Identical Records):")
    print(cdf.to_string(index=False))

    # Matched Gap Sensitivity Across ALL 7,992 Gaps
    gsum = gdf.groupby(["comparator", "gap_bin"]).agg(
        n_gaps=("true_gap_s", "count"),
        sens_any=("det_any", "mean"),
        sens_ge50pct=("det_ge50", "mean"),
        bridge_rate=("det_any", lambda s: float(1.0 - s.mean())),
        med_dur_err_s=("dur_err_s", "median"),
    ).reset_index()
    gsum.to_csv(OUT_RES / "V8_m2_gap_sensitivity_182rec.csv", index=False)
    print("\nIdentical 7,992-Gap Sensitivity (Oracles vs RefBeats RR-GBDT vs AutoQRS RR-GBDT):")
    print(gsum.to_string(index=False))

    fgsum = fgdf.groupby(["comparator", "high_burden_ge80"]).agg(
        n_recs=("record_id", "count"),
        tot_prol_af_h=("prolonged_af_h", "sum"),
        tot_false_gaps=("n_false_gaps", "sum"),
        false_gaps_per_af_h=("n_false_gaps", lambda s: float(s.sum() / max(1e-6, fgdf.loc[s.index, "prolonged_af_h"].sum()))),
        false_nonaf_pct=("false_nonaf_s", lambda s: float(s.sum() / (max(1e-6, fgdf.loc[s.index, "prolonged_af_h"].sum()) * 3600.0) * 100.0)),
    ).reset_index()
    fgsum.to_csv(OUT_RES / "V8_m2_false_gaps_182rec.csv", index=False)
    print("\nFalse Gaps in Prolonged AF (>=10m) Across All 182 Recordings:")
    print(fgsum.to_string(index=False))
    print(f"\nM2 completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
