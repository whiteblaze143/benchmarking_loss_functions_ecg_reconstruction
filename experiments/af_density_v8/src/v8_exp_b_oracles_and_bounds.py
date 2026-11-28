"""Experiment B: Oracle Boundary Bug Fixes, Multi-Scale Representations & delta Disagreement Analysis.

1. Fixes the Oracle_10sGrid and Oracle_60BeatRR boundary assignment bug so ALL_AF -> 100% ALL_AF
   and ZERO_AF -> 100% ZERO_AF (no spurious <10s/edge non-AF gaps).
2. Evaluates all 182 full recordings (LTAFDB 84 + SHDB 98) across Continuous Reference,
   Oracle_60BeatRR_EdgeAssigned, Oracle_60BeatRR_SharedClock, Oracle_10sGrid,
   Oracle_10s_FractionalMass, Oracle_30sGrid, Oracle_60sGrid, Oracle_5mGrid, Oracle_1hGrid
   for BOTH Charitos Density AND Butkuviene AF Aggregation.
3. Empirically & theoretically compares Total Rhythm Disagreement delta = (FalseAF + MissedAF)/T
   vs Net Burden Error |B_hat - B| across detectors/oracles and zero-net-burden swap experiments.
"""
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, "experiments/af_density")
from rr_features import layout as rr_layout
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import clip_ivs_to_known, intervals_overlap_mask
from v8_exp_a_reference_layers import metrics_from_intervals

OUT_RES = Path("experiments/af_density_v8/results")
OUT_FIG = Path("experiments/af_density_v8/figures")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
SHDB_DIR = Path("/data/mithunmanivannan/datasets/shdb_af")
NON_BEAT_LTAF = {'+', '"', '~', '|', 's', 't', 'p', 'u', '='}


def fixed_oracle_grid(ref_k: List[Tuple[float, float]], T_k: float, win_s: float) -> List[Tuple[float, float]]:
    """Fixed contiguous window majority oracle including trailing partial bin [n*win_s, T_k] by >=50% width."""
    if T_k <= 1e-9 or not ref_k:
        return []
    n_bins = int(math.ceil(T_k / win_s))
    out = []
    j = 0
    for wi in range(n_bins):
        ba = wi * win_s
        bb = min(T_k, (wi + 1) * win_s)
        w = bb - ba
        if w <= 1e-9:
            continue
        while j < len(ref_k) and ref_k[j][1] <= ba:
            j += 1
        ov = 0.0
        k = j
        while k < len(ref_k) and ref_k[k][0] < bb:
            oa, ob = max(ba, ref_k[k][0]), min(bb, ref_k[k][1])
            if ob > oa:
                ov += (ob - oa)
            k += 1
        if ov >= 0.5 * w - 1e-12:
            out.append((ba, bb))
    return merge_intervals(out)


def fractional_binned_metrics(ref_k: List[Tuple[float, float]], T_k: float, win_s: float) -> Dict[str, Any]:
    """Sub-grid fractional AF density & aggregation (assigning bin mass fraction cleanly along bin widths)."""
    if T_k <= 1e-9 or not ref_k:
        return metrics_from_intervals([], T_k)
    n_bins = int(math.ceil(T_k / win_s))
    # Represent each bin [ba, bb] with AF fraction f_i by centered contiguous AF chunk of length f_i * (bb - ba)
    # Wait: even cleaner and exact for Author Binned Density: concentrate AF mass inside each bin -> piecewise interval!
    # Wait: centered vs left-aligned within bin is within <= win_s; let's ALSO check direct fractional intervals [ba, ba + ov]
    # Note: if f_i == 1.0, [ba, bb] is full AF -> adjacent full-AF bins merge cleanly -> 100% ALL_AF!
    frac_ivs = []
    j = 0
    for wi in range(n_bins):
        ba = wi * win_s
        bb = min(T_k, (wi + 1) * win_s)
        w = bb - ba
        if w <= 1e-9:
            continue
        while j < len(ref_k) and ref_k[j][1] <= ba:
            j += 1
        ov = 0.0
        k = j
        while k < len(ref_k) and ref_k[k][0] < bb:
            oa, ob = max(ba, ref_k[k][0]), min(bb, ref_k[k][1])
            if ob > oa:
                ov += (ob - oa)
            k += 1
        if ov > 1e-9:
            if ov >= w - 1e-9:
                frac_ivs.append((ba, bb))
            else:
                frac_ivs.append((ba, min(bb, ba + ov)))
    return metrics_from_intervals(frac_ivs, T_k)


def fixed_oracle_60beat(q: np.ndarray, fs: float, first_k: float, raw_T: float,
                        ref_raw_af: List[Tuple[float, float]], known_ivs: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float, List[Tuple[float, float]]]:
    """Fixed 60-beat majority oracle with BOTH edge-assigned full timeline and common-beat-clock timeline."""
    q = np.sort(q)
    q = q[np.r_[True, np.diff(q) > 0]]
    q = q[q >= first_k]
    if len(q) < 62:
        return [], [], 0.0, []
    starts, assigned, ends = rr_layout(len(q) - 1, 0)
    if len(starts) == 0:
        return [], [], 0.0, []

    end_t = q[1:].astype(float) / fs
    truth = np.zeros(len(end_t), dtype=np.int8)
    for aa, ab in ref_raw_af:
        truth[(end_t >= aa) & (end_t < ab)] = 1
    pref = np.r_[0, np.cumsum(truth)]
    oracle_60_af = ((pref[starts + 60] - pref[starts]) >= 30)

    # 1. Edge-assigned on full raw clock (assigns [first_k/fs, q[assigned[0]]/fs] and [q[ends[-1]]/fs, raw_T] to boundary window)
    edge_ivs = []
    for i, (lo, hi) in enumerate(zip(assigned, ends)):
        if oracle_60_af[i]:
            t0 = (first_k / fs) if i == 0 else float(q[lo]) / fs
            t1 = raw_T if i == len(starts) - 1 else float(q[hi]) / fs
            if t1 > t0:
                edge_ivs.append((t0, t1))
    edge_ivs = merge_intervals(edge_ivs)

    # 2. Strict shared beat-clock span [span_lo, span_hi] clipped inside known_ivs
    span_lo, span_hi = float(q[assigned[0]]) / fs, float(q[ends[-1]]) / fs
    shared_known = [(max(ka, span_lo), min(kb, span_hi)) for ka, kb in merge_intervals(known_ivs) if min(kb, span_hi) > max(ka, span_lo) + 1e-6]
    shared_ivs = []
    for i, (lo, hi) in enumerate(zip(assigned, ends)):
        if oracle_60_af[i]:
            t0, t1 = float(q[lo]) / fs, float(q[hi]) / fs
            if t1 > t0:
                shared_ivs.append((t0, t1))
    shared_ivs = merge_intervals(shared_ivs)
    shared_ref_k, T_shared = clip_ivs_to_known(ref_raw_af, shared_known)
    shared_pred_k, _ = clip_ivs_to_known(shared_ivs, shared_known)
    return edge_ivs, shared_pred_k, T_shared, shared_ref_k


def interval_union_len(ivs: List[Tuple[float, float]]) -> float:
    return sum(b - a for a, b in merge_intervals(ivs))


def interval_intersection_len(a_ivs: List[Tuple[float, float]], b_ivs: List[Tuple[float, float]]) -> float:
    a = merge_intervals(a_ivs)
    b = merge_intervals(b_ivs)
    tot = 0.0
    i, j = 0, 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += (hi - lo)
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def eval_record_oracles(args):
    ds, rid, pid, raw_T, fs, known_ivs, ref_raw_af = args
    base = str((LTAF_DIR if ds == "LTAFDB" else SHDB_DIR) / rid)
    if ds == "SHDB-AF":
        q_ref = wfdb.rdann(base, "qrs").sample
    else:
        ann = wfdb.rdann(base, "atr")
        q_ref = ann.sample[np.array([s not in NON_BEAT_LTAF for s in ann.symbol], dtype=bool)]
    first_k = float(known_ivs[0][0]) * fs
    ref_k, T_k = clip_ivs_to_known(ref_raw_af, known_ivs)
    m_ref = metrics_from_intervals(ref_k, T_k)

    rows = []
    # Fixed Grid Oracles
    for oname, w_s in [
        ("Oracle_10sGrid_Fixed", 10.0),
        ("Oracle_30sGrid_Fixed", 30.0),
        ("Oracle_60sGrid_Fixed", 60.0),
        ("Oracle_5mGrid_Fixed", 300.0),
        ("Oracle_1hGrid_Fixed", 3600.0),
    ]:
        p_k = fixed_oracle_grid(ref_k, T_k, w_s)
        m_p = metrics_from_intervals(p_k, T_k)
        tp = interval_intersection_len(ref_k, p_k)
        fp = max(0.0, interval_union_len(p_k) - tp)
        fn = max(0.0, interval_union_len(ref_k) - tp)
        delta = (fp + fn) / T_k
        rows.append({
            "dataset": ds, "record_id": rid, "patient_id": pid, "representation": oname,
            "ref_state": m_ref["state"], "pred_state": m_p["state"],
            "ref_burden": m_ref["burden"], "pred_burden": m_p["burden"],
            "abs_burden_err": abs(m_p["burden"] - m_ref["burden"]),
            "total_disagreement_delta": delta, "fp_frac": fp / T_k, "fn_frac": fn / T_k,
            "ref_density": m_ref["charitos_density"], "pred_density": m_p["charitos_density"],
            "abs_density_err": abs(m_p["charitos_density"] - m_ref["charitos_density"]) if (m_ref["is_defined"] and m_p["is_defined"]) else np.nan,
            "ref_agg": m_ref["af_aggregation"], "pred_agg": m_p["af_aggregation"],
            "abs_agg_err": abs(m_p["af_aggregation"] - m_ref["af_aggregation"]) if (m_ref["is_defined"] and m_p["is_defined"]) else np.nan,
            "abs_agg_zeroext_err": abs(m_p["af_aggregation_zeroext"] - m_ref["af_aggregation_zeroext"]),
            "ref_n_runs": m_ref["n_runs"], "pred_n_runs": m_p["n_runs"],
        })

    # 10s Fractional Mass (continuous representation inside bins; delta measured relative to ref interval mass)
    m_fr = fractional_binned_metrics(ref_k, T_k, 10.0)
    rows.append({
        "dataset": ds, "record_id": rid, "patient_id": pid, "representation": "Oracle_10s_FractionalMass",
        "ref_state": m_ref["state"], "pred_state": m_fr["state"],
        "ref_burden": m_ref["burden"], "pred_burden": m_fr["burden"],
        "abs_burden_err": abs(m_fr["burden"] - m_ref["burden"]),
        "total_disagreement_delta": np.nan, "fp_frac": np.nan, "fn_frac": np.nan,
        "ref_density": m_ref["charitos_density"], "pred_density": m_fr["charitos_density"],
        "abs_density_err": abs(m_fr["charitos_density"] - m_ref["charitos_density"]) if (m_ref["is_defined"] and m_fr["is_defined"]) else np.nan,
        "ref_agg": m_ref["af_aggregation"], "pred_agg": m_fr["af_aggregation"],
        "abs_agg_err": abs(m_fr["af_aggregation"] - m_ref["af_aggregation"]) if (m_ref["is_defined"] and m_fr["is_defined"]) else np.nan,
        "abs_agg_zeroext_err": abs(m_fr["af_aggregation_zeroext"] - m_ref["af_aggregation_zeroext"]),
        "ref_n_runs": m_ref["n_runs"], "pred_n_runs": m_fr["n_runs"],
    })

    # Fixed 60-beat Oracles (Edge-Assigned on full T_k & Shared-Clock on [q0, qend])
    edge_raw, sh_pred, T_sh, sh_ref = fixed_oracle_60beat(q_ref, fs, first_k, raw_T, ref_raw_af, known_ivs)
    edge_k, _ = clip_ivs_to_known(edge_raw, known_ivs)
    for cname, rk, pk, tlen in [
        ("Oracle_60BeatRR_EdgeAssigned", ref_k, edge_k, T_k),
        ("Oracle_60BeatRR_SharedClock", sh_ref, sh_pred, T_sh),
    ]:
        mr = metrics_from_intervals(rk, tlen)
        mp = metrics_from_intervals(pk, tlen)
        tp = interval_intersection_len(rk, pk)
        fp = max(0.0, interval_union_len(pk) - tp)
        fn = max(0.0, interval_union_len(rk) - tp)
        delta = (fp + fn) / tlen if tlen > 0 else np.nan
        rows.append({
            "dataset": ds, "record_id": rid, "patient_id": pid, "representation": cname,
            "ref_state": mr["state"], "pred_state": mp["state"],
            "ref_burden": mr["burden"], "pred_burden": mp["burden"],
            "abs_burden_err": abs(mp["burden"] - mr["burden"]),
            "total_disagreement_delta": delta, "fp_frac": fp / tlen if tlen > 0 else np.nan, "fn_frac": fn / tlen if tlen > 0 else np.nan,
            "ref_density": mr["charitos_density"], "pred_density": mp["charitos_density"],
            "abs_density_err": abs(mp["charitos_density"] - mr["charitos_density"]) if (mr["is_defined"] and mp["is_defined"]) else np.nan,
            "ref_agg": mr["af_aggregation"], "pred_agg": mp["af_aggregation"],
            "abs_agg_err": abs(mp["af_aggregation"] - mr["af_aggregation"]) if (mr["is_defined"] and mp["is_defined"]) else np.nan,
            "abs_agg_zeroext_err": abs(mp["af_aggregation_zeroext"] - mr["af_aggregation_zeroext"]),
            "ref_n_runs": mr["n_runs"], "pred_n_runs": mp["n_runs"],
        })
    return rows


def run_balanced_swap_verification() -> pd.DataFrame:
    """Controlled zero-net-burden perturbation family (|B_hat - B| == 0, delta > 0) verifying delta vs |B_hat - B| bounds."""
    T = 43200.0  # 12h
    rows = []
    for B in [0.10, 0.50, 0.80, 0.90, 0.95, 0.99]:
        eps = 1.0 - B
        for K in [5, 25, 100]:
            # Construct true pattern A: K equal AF blocks separated by K equal gaps (sum gap = eps*T) -> D_ref = 1/K
            af_len = B * T / K
            gap_len = eps * T / K
            step = af_len + gap_len
            ref_ivs = [(i * step, i * step + af_len) for i in range(K)]
            # Construct zero-burden-error perturbed pattern A_hat: consolidate K blocks into 1 single block of exact mass B*T!
            # Wait: |B_hat - B| = 0 identically! Meanwhile delta = 2 * overlap_mismatch / T!
            hat_ivs = [(0.0, B * T)]
            m_r = metrics_from_intervals(ref_ivs, T)
            m_h = metrics_from_intervals(hat_ivs, T)
            tp = interval_intersection_len(ref_ivs, hat_ivs)
            delta = (interval_union_len(ref_ivs) + interval_union_len(hat_ivs) - 2.0 * tp) / T
            d_d = abs(m_h["charitos_density"] - m_r["charitos_density"])
            d_a = abs(m_h["af_aggregation"] - m_r["af_aggregation"])
            rows.append({
                "burden_B": B, "epsilon": eps, "K_episodes": K,
                "abs_burden_err": abs(m_h["burden"] - m_r["burden"]),
                "total_disagreement_delta": round(delta, 6),
                "ref_density": round(m_r["charitos_density"], 6),
                "hat_density": round(m_h["charitos_density"], 6),
                "abs_density_err": round(d_d, 6),
                "ratio_dD_over_delta": round(d_d / delta, 3) if delta > 0 else np.nan,
                "ref_agg": round(m_r["af_aggregation"], 6),
                "hat_agg": round(m_h["af_aggregation"], 6),
                "abs_agg_err": round(d_a, 6),
                "ratio_dAgg_over_delta": round(d_a / delta, 3) if delta > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    ext = inv[inv["is_annotated"] & inv["dataset"].isin(["LTAFDB", "SHDB-AF"])].copy()

    tasks = []
    for rec in ext.itertuples():
        ds, rid, pid = rec.dataset, str(rec.record_id), rec.patient_id
        sub_r = rhy[(rhy["dataset"] == ds) & (rhy["record_id"].astype(str) == rid) & (rhy["rhythm_known"])]
        known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
        sub_af = runs[(runs["dataset"] == ds) & (runs["record_id"].astype(str) == rid)]
        ref_raw_af = merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])))
        tasks.append((ds, rid, pid, float(rec.raw_samples / rec.fs), float(rec.fs), known_ivs, ref_raw_af))

    print(f"=== Step B1: Evaluating Fixed Oracles Across All {len(tasks)} Recordings ===")
    all_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(eval_record_oracles, tasks):
            all_rows.extend(res)
    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_RES / "V8_exp_b_oracles_182rec.parquet", index=False)

    # State preservation verification (specifically ALL_AF -> ALL_AF and ZERO_AF -> ZERO_AF!)
    summary = df.groupby("representation").agg(
        n_recs=("record_id", "count"),
        ref_all_af=("ref_state", lambda s: int((s == "ALL_AF").sum())),
        pred_all_af_on_ref_all_af=("pred_state", lambda s: int(((df.loc[s.index, "ref_state"] == "ALL_AF") & (s == "ALL_AF")).sum())),
        spurious_mixed_from_all_af=("pred_state", lambda s: int(((df.loc[s.index, "ref_state"] == "ALL_AF") & (s == "MIXED")).sum())),
        spurious_mixed_from_zero_af=("pred_state", lambda s: int(((df.loc[s.index, "ref_state"] == "ZERO_AF") & (s == "MIXED")).sum())),
        mean_abs_burden_err=("abs_burden_err", "mean"),
        mean_delta=("total_disagreement_delta", "mean"),
        med_abs_dens_err=("abs_density_err", "median"),
        mean_abs_dens_err=("abs_density_err", "mean"),
        max_abs_dens_err=("abs_density_err", "max"),
        med_abs_agg_err=("abs_agg_err", "median"),
        mean_abs_agg_err=("abs_agg_err", "mean"),
        max_abs_agg_err=("abs_agg_err", "max"),
    ).reset_index()
    summary.to_csv(OUT_RES / "V8_exp_b_oracles_summary.csv", index=False)
    print(summary.to_string(index=False))

    print("\n=== Step B2: Zero-Net-Burden Balanced Swap Verification (|B_hat-B|==0, delta>0) ===")
    swap_df = run_balanced_swap_verification()
    swap_df.to_csv(OUT_RES / "V8_exp_b_zero_burden_swap_bounds.csv", index=False)
    print(swap_df[swap_df["K_episodes"] == 25].to_string(index=False))

    # Plot Figure B: Oracle Error Across Resolutions & Zero-Burden-Error Disagreement Bound
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    order = ["Oracle_10s_FractionalMass", "Oracle_10sGrid_Fixed", "Oracle_30sGrid_Fixed",
             "Oracle_60BeatRR_EdgeAssigned", "Oracle_60sGrid_Fixed", "Oracle_5mGrid_Fixed", "Oracle_1hGrid_Fixed"]
    labels = ["10s Frac", "10s Maj", "30s Maj", "60-Beat", "60s Maj", "5m Maj", "1h Maj"]
    s_plot = summary.set_index("representation").loc[order]
    x = np.arange(len(order))
    w = 0.36
    ax1.bar(x - w/2, s_plot["mean_abs_dens_err"], width=w, color="#c44e52", label="Charitos Density Mean |ΔD|")
    ax1.bar(x + w/2, s_plot["mean_abs_agg_err"], width=w, color="#2ca02c", label="Butkuvienė Aggregation Mean |ΔAgg|")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Mean Absolute Error vs Continuous Exact (n=182)", fontsize=9.5)
    ax1.set_title("A. Fixed-Boundary Multi-Scale Oracle Discretization Error", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="y", alpha=0.25); ax1.legend(fontsize=8.5)

    for K, col in [(5, "#1f77b4"), (25, "#d62728"), (100, "#9467bd")]:
        sk = swap_df[swap_df["K_episodes"] == K]
        ax2.plot(sk["burden_B"], sk["ratio_dD_over_delta"], marker="o", lw=2, color=col, label=f"Charitos |ΔD|/δ (K={K})")
        ax2.plot(sk["burden_B"], sk["ratio_dAgg_over_delta"], marker="s", ls="--", lw=1.6, color=col, alpha=0.7, label=f"Agg |ΔAgg|/δ (K={K})")
    ax2.set_yscale("log")
    ax2.set_xlabel("AF Burden B (with exact |B_hat - B| = 0.000!)", fontsize=9.5)
    ax2.set_ylabel("Error Amplification Ratio per Unit δ (log)", fontsize=9.5)
    ax2.set_title("B. Zero-Burden-Error Perturbation (|ΔB|=0, δ>0): Density vs Aggregation", fontsize=10.5, fontweight="bold")
    ax2.grid(alpha=0.25, which="both"); ax2.legend(fontsize=7.5, ncol=2)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "V8_exp_b_oracles_and_delta_bounds.png", dpi=240)
    fig.savefig(OUT_FIG / "V8_exp_b_oracles_and_delta_bounds.pdf")
    plt.close(fig)
    print(f"\nCompleted Experiment B in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
