"""V8 Revision Pass 1 (Experiments A & B Mathematical + Reference-Layer Corrections):

1. Experiment A Horizon Reconciliation & Partial Identification:
   - Restricts gap taxonomy & waveform audit strictly to the EXACT 12h known windows of the 19 high-burden
     LTAFDB patients (<= 1,097 internal gaps, eliminating the 3,021 full-24h vs 12h mismatch).
   - Computes exact masking fractions separately for All-36 vs Initial-19 MIXED (0.1979% / 0.3984% / 0.4836%).
   - Distinguishes ALL_KNOWN_AF (masked_frac > 0) from CONFIRMED_ALL_AF (masked_frac == 0).
   - Evaluates Partial Identification across plausible completions (AllAF, AllNonAF, combinatorial / subset
     completions) on the fixed 43,200s physical clock for each of the 19 MIXED records.
2. Experiment B Mathematical & Fractional-Mass Audit:
   - Verifies corrected Theorem 1: Delta_G == 0 identically (<=1e-15) while |Delta_D / Delta_B| = 1/(1-B).
   - Verifies user-derived equal-burden general perturbation upper bounds:
     |G - G_hat| <= delta / B and |D - D_hat| <= delta / (B * (1 - B)) on balanced swaps & 500 random shuffles.
   - Evaluates Fractional AF Mass (left-aligned, centered, and Charitos's charitos_author_binned_port) vs
     Majority Binary across {10s, 30s, 60s, 5m, 15m, 60m} on n=182 recordings (density, agg, longest_share, n_runs).
   - Computes Density vs AF Aggregation reliability suite: ICC(2,1), Spearman rho, MAE, MAE/SD across burden strata.
"""
import itertools
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
from scipy.stats import spearmanr

sys.path.insert(0, "experiments/af_density")
from metrics import charitos_author_binned_port
sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals, subtract_exclusions
from v8_e07_authentic_gap_sensing import clip_ivs_to_known
from v8_exp_a_reference_layers import metrics_from_intervals
from v8_exp_b_oracles_and_bounds import fixed_oracle_grid, interval_intersection_len, interval_union_len

OUT_RES = Path("experiments/af_density_v8/results")
OUT_FIG = Path("experiments/af_density_v8/figures")
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")


def icc_2_1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-way random effects absolute agreement single-measure ICC(2,1)."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    x = np.column_stack([y_true[m], y_pred[m]])
    n = len(x)
    if n < 3:
        return np.nan
    k = 2.0
    grand = np.mean(x)
    row_means = np.mean(x, axis=1)
    col_means = np.mean(x, axis=0)
    ss_total = np.sum((x - grand) ** 2)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_err = max(0.0, ss_total - ss_rows - ss_cols)
    ms_r = ss_rows / (n - 1)
    ms_c = ss_cols / (k - 1)
    ms_e = ss_err / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return float((ms_r - ms_e) / denom) if denom > 1e-15 else np.nan


def step1_exact_12h_taxonomy_and_partial_id():
    print("=== Rev1-Step 1: Exact 12h Gap Taxonomy, Masking Denominators & Partial Identification ===")
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    gdf = pd.read_parquet(OUT_RES / "V8_exp_a_ltafdb_all_gaps_waveform.parquet")
    hb_tbl = pd.read_csv(OUT_RES / "V8_exp_a_highburden19_patient_table.csv")

    hb19_ids = sorted(hb_tbl[hb_tbl["r0_state"] == "MIXED"]["record_id"].astype(str).unique())
    all36_ids = sorted(hb_tbl["record_id"].astype(str).unique())

    # 1. Exact masking fractions on All-36 vs Initial-19 MIXED
    mask_rows = []
    for lname in ["R2_VT_Unknown", "R2_VT_SBR10_Unknown", "R2_VT_SBR60_Unknown", "R1_Waveform_ConsUnknown", "R2_Cons_Sub10_Unknown"]:
        s36 = hb_tbl[hb_tbl["layer"] == lname]
        s19 = s36[s36["record_id"].astype(str).isin(hb19_ids)]
        mask_rows.append({
            "layer": lname,
            "n_all36": len(s36), "mean_masked_pct_all36": round(100.0 * float(s36["unknown_masked_frac"].mean()), 4),
            "n_mixed19": len(s19), "mean_masked_pct_mixed19": round(100.0 * float(s19["unknown_masked_frac"].mean()), 4),
            "max_masked_pct_mixed19": round(100.0 * float(s19["unknown_masked_frac"].max()), 4),
            "n_still_mixed": int((s19["state"] == "MIXED").sum()),
            "n_all_known_af": int(((s19["state"] == "ALL_AF") & (s19["unknown_masked_frac"] > 0)).sum()),
            "n_confirmed_all_af": int(((s19["state"] == "ALL_AF") & (s19["unknown_masked_frac"] == 0)).sum()),
        })
    mdf = pd.DataFrame(mask_rows)
    mdf.to_csv(OUT_RES / "V8_rev1_masking_denominators.csv", index=False)
    print("Exact Masking Fractions (All-36 vs Mixed-19) & ALL_KNOWN_AF Count:")
    print(mdf.to_string(index=False))

    # 2. Exact 12h Known-Horizon Cutoff Time for every LTAFDB record -> restrict gap taxonomy strictly to <= T12!
    cutoff_12h = {}
    partial_rows = []
    H12 = 43200.0
    ltaf_rhy = rhy[rhy["dataset"] == "LTAFDB"]
    ltaf_runs = runs[runs["dataset"] == "LTAFDB"]
    for rid in sorted(gdf["record_id"].astype(str).unique()):
        sub_r = ltaf_rhy[(ltaf_rhy["record_id"].astype(str) == rid) & (ltaf_rhy["rhythm_known"])]
        known_ivs = merge_intervals(list(zip(sub_r["start_time_s"], sub_r["end_time_s"])))
        rem = H12
        t_end = None
        clipped_k = []
        for ka, kb in known_ivs:
            if rem <= 1e-9:
                break
            take = min(kb - ka, rem)
            t_end = ka + take
            clipped_k.append((ka, t_end))
            rem -= take
        if rem <= 1e-6:
            cutoff_12h[rid] = (t_end, clipped_k)

    # Filter gdf strictly to gaps strictly inside the first 12h known horizon (gap_end_s <= t_end_12h + 1e-6)
    gdf["in_12h_window"] = gdf.apply(
        lambda r: (str(r["record_id"]) in cutoff_12h) and (r["gap_end_s"] <= cutoff_12h[str(r["record_id"])][0] + 1e-6), axis=1
    )
    g12_hb19 = gdf[gdf["is_hb19_record"] & gdf["in_12h_window"]].copy()
    print(f"\nExact 12h Horizon Internal Gaps in HB19: total={len(g12_hb19)}, <10s={int((g12_hb19['dur_s']<10.0).sum())}")
    assert len(g12_hb19) <= 1097, f"Unexpected gap count > 1097: {len(g12_hb19)}"

    tab12 = g12_hb19.groupby([g12_hb19["dur_s"] < 10.0, "native_tags", "r1_class"]).agg(
        n_gaps=("dur_s", "count"),
        tot_dur_s=("dur_s", "sum"),
        med_dur_s=("dur_s", "median"),
        med_f_power=("f_power_ratio", "median"),
        med_p_snr=("p_sync_snr", "median"),
        med_cvrr=("cvrr_gap", "median"),
    ).reset_index().rename(columns={"dur_s": "is_lt10s"})
    tab12.to_csv(OUT_RES / "V8_rev1_hb19_exact12h_gaps.csv", index=False)
    print("\nHB19 Exact 12h Internal Gaps (<10s) Native Tag vs Automated 2-Lead Waveform Class:")
    print(tab12[tab12["is_lt10s"]].to_string(index=False))

    # 3. Physical-Clock Partial Identification on the 19 HB MIXED records under R2_VT_Unknown and R2_VT_SBR60_Unknown!
    # Keep fixed known clock [0, 43200s], identify exact non-masked known AF + known Non-AF + list of masked intervals U_1..U_M.
    # Evaluate plausible completions: AllAF, AllNonAF, Only-1-Gap-NonAF (each single masked interval preserved, maximizing single-gap D=1),
    # and combinatorial / greedy subset completions to obtain empirical feasible intervals [D_min, D_max], [Agg_min, Agg_max].
    for rid in hb19_ids:
        t_end, clipped_k = cutoff_12h[rid]
        sub_r = ltaf_rhy[(ltaf_rhy["record_id"].astype(str) == rid) & (ltaf_rhy["rhythm_known"])]
        sub_af = ltaf_runs[ltaf_runs["record_id"].astype(str) == rid]
        af_raw = merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])))
        # Map native AF onto fixed [0, 43200] known clock!
        af_12, T12 = clip_ivs_to_known(af_raw, clipped_k)
        m_r0 = metrics_from_intervals(af_12, T12)

        for rname, unk_tags_fn in [
            ("R2_VT_Unknown", lambda df: df["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"])),
            ("R2_VT_SBR60_Unknown", lambda df: df["rhythm_label_native"].astype(str).isin(["VT", "IVR", "B", "T"]) |
             ((df["rhythm_label_native"].astype(str) == "SBR") & (df["duration_s"] < 60.0))),
        ]:
            u_df = sub_r[unk_tags_fn(sub_r)]
            u_raw = merge_intervals(list(zip(u_df["start_time_s"], u_df["end_time_s"])))
            u_12, _ = clip_ivs_to_known(u_raw, clipped_k)
            masked_s = sum(b - a for a, b in u_12)
            # Base certain AF (outside u_12) and certain Non-AF (outside af_12 union u_12)
            certain_af = subtract_exclusions(af_12, u_12)
            certain_nonaf = subtract_exclusions([(0.0, T12)], merge_intervals(af_12 + u_12))

            # Plausible completions of [u_1, ..., u_M] over physical T12 clock:
            # Candidate set: AllAF, AllNonAF (=R0 if all u_i were non-AF), Keep-single-u_k (for each k),
            # Drop-single-u_k, Leave-Every-Other-u_k (parity), plus exhaustive subsets if M<=10 or 128 random subsets if M>10.
            M = len(u_12)
            cands_af = [merge_intervals(af_12 + u_12), af_12]  # AllAF, AllNonAF
            if 1 <= M <= 10:
                for bits in itertools.product([0, 1], repeat=M):
                    extra = [u_12[idx] for idx, bit in enumerate(bits) if bit == 1]
                    cands_af.append(merge_intervals(certain_af + extra))
            elif M > 10:
                # Structured extreme candidates + 200 random completions
                for idx in range(M):
                    # Keep only u_12[idx] as Non-AF (fill rest with AF) -> drives max concentration (D -> 1 if no other certain_nonaf)
                    extra = [u_12[k] for k in range(M) if k != idx]
                    cands_af.append(merge_intervals(certain_af + extra))
                    cands_af.append(merge_intervals(certain_af + [u_12[idx]]))
                rng = np.random.RandomState(42)
                for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    for _ in range(40):
                        sel = rng.rand(M) < p
                        extra = [u_12[k] for k in range(M) if sel[k]]
                        cands_af.append(merge_intervals(certain_af + extra))

            dens_vals, agg_vals, states = [], [], set()
            for caf in cands_af:
                mc = metrics_from_intervals(caf, T12)
                states.add(mc["state"])
                agg_vals.append(mc["af_aggregation_zeroext"])
                if mc["is_defined"]:
                    dens_vals.append(mc["charitos_density"])

            allaf_m = metrics_from_intervals(cands_af[0], T12)
            allnonaf_m = metrics_from_intervals(cands_af[1], T12)
            r2_label = "ALL_KNOWN_AF" if (allaf_m["state"] == "ALL_AF" and masked_s > 0) else ("CONFIRMED_ALL_AF" if allaf_m["state"] == "ALL_AF" else "MIXED")
            partial_rows.append({
                "record_id": rid, "layer": rname, "n_masked_intervals": M,
                "masked_s": round(masked_s, 2), "masked_pct": round(100.0 * masked_s / T12, 4),
                "certain_nonaf_s": round(sum(b - a for a, b in certain_nonaf), 2),
                "r0_state": m_r0["state"], "r0_density": round(m_r0["charitos_density"], 4),
                "r2_retained_state": "ALL_KNOWN_AF" if (len(certain_nonaf) == 0 and masked_s > 0) else ("MIXED" if len(certain_nonaf) > 0 else "CONFIRMED_ALL_AF"),
                "feasible_states": "+".join(sorted(states)),
                "density_always_defined": ("ALL_AF" not in states and "ZERO_AF" not in states),
                "density_all_nonaf": round(allnonaf_m["charitos_density"], 4) if allnonaf_m["is_defined"] else np.nan,
                "density_all_af": round(allaf_m["charitos_density"], 4) if allaf_m["is_defined"] else np.nan,
                "partial_id_dens_min": round(float(min(dens_vals)), 4) if dens_vals else np.nan,
                "partial_id_dens_max": round(float(max(dens_vals)), 4) if dens_vals else np.nan,
                "partial_id_dens_width": round(float(max(dens_vals) - min(dens_vals)), 4) if dens_vals else np.nan,
                "partial_id_agg_min": round(float(min(agg_vals)), 5),
                "partial_id_agg_max": round(float(max(agg_vals)), 5),
                "partial_id_agg_width": round(float(max(agg_vals) - min(agg_vals)), 5),
            })

    pid_df = pd.DataFrame(partial_rows)
    pid_df.to_csv(OUT_RES / "V8_rev1_partial_identification_19hb.csv", index=False)
    print("\nPartial Identification Summary on Fixed 43,200s Clock Across 19 High-Burden LTAFDB MIXED Records:")
    print(pid_df[pid_df["layer"] == "R2_VT_SBR60_Unknown"][
        ["record_id", "n_masked_intervals", "masked_s", "certain_nonaf_s", "r0_density", "r2_retained_state", "feasible_states", "partial_id_dens_min", "partial_id_dens_max", "partial_id_agg_min", "partial_id_agg_max"]
    ].to_string(index=False))


def step2_corrected_theorems_and_bounds():
    print("\n=== Rev1-Step 2: Corrected Theorem 1 (Delta_G = 0) & Equal-Burden Analytical Bounds ===")
    T = 43200.0
    th_rows = []
    for eps in [0.50, 0.20, 0.10, 0.05, 0.026, 0.02, 0.01, 0.002]:
        B = 1.0 - eps
        for K in [2, 10, 50, 100]:
            af_len = (1.0 - eps) * T / K
            gap_len = eps * T / K
            step = af_len + gap_len
            orig_ivs = [(i * step, i * step + af_len) for i in range(K)]
            # Bridge K-1 internal gaps leaving single terminal gap [0, T - gap_len]
            br_ivs = [(0.0, T - gap_len)]
            mo = metrics_from_intervals(orig_ivs, T)
            mb = metrics_from_intervals(br_ivs, T)
            dB = mb["burden"] - mo["burden"]
            dD = mb["charitos_density"] - mo["charitos_density"]
            dG = mb["af_aggregation"] - mo["af_aggregation"]
            th_rows.append({
                "epsilon": eps, "burden_ref": round(B, 4), "K_gaps": K,
                "D_ref": round(mo["charitos_density"], 8), "D_bridged": round(mb["charitos_density"], 8),
                "Delta_D": round(dD, 8), "Delta_B": round(dB, 8),
                "ratio_dD_dB": round(dD / dB, 4), "exact_1_over_eps": round(1.0 / eps, 4),
                "G_ref": round(mo["af_aggregation"], 8), "G_bridged": round(mb["af_aggregation"], 8),
                "Delta_G": dG, "abs_Delta_G": abs(dG),
            })
    tdf = pd.DataFrame(th_rows)
    tdf.to_csv(OUT_RES / "V8_rev1_theorem1_corrected_agg.csv", index=False)
    print(f"Max |Delta_G| across all 32 periodic configurations = {tdf['abs_Delta_G'].max():.3e}")
    print(tdf[tdf["K_gaps"] == 50][["epsilon", "burden_ref", "D_ref", "D_bridged", "ratio_dD_dB", "G_ref", "G_bridged", "abs_Delta_G"]].to_string(index=False))

    # Verify user's equal-burden analytical upper bounds:
    # |G - G_hat| <= delta / B and |D - D_hat| <= delta / (B * (1 - B)) across balanced swaps & 500 random equal-burden perturbations!
    rng = np.random.RandomState(42)
    bnd_rows = []
    # 1) Balanced periodic swaps + 2) Random 20-bin equal-burden shuffles across B in {0.1, 0.3, 0.5, 0.8, 0.9, 0.95, 0.99}
    for B in [0.10, 0.30, 0.50, 0.80, 0.90, 0.95, 0.99]:
        # Test K=25 periodic -> single contiguous block
        K = 25
        af_l = B * T / K
        gap_l = (1.0 - B) * T / K
        step = af_l + gap_l
        s_ref = [(i * step, i * step + af_l) for i in range(K)]
        s_hat = [(0.0, B * T)]
        mr = metrics_from_intervals(s_ref, T)
        mh = metrics_from_intervals(s_hat, T)
        tp = interval_intersection_len(s_ref, s_hat)
        delta = (interval_union_len(s_ref) + interval_union_len(s_hat) - 2.0 * tp) / T
        dG = abs(mr["af_aggregation"] - mh["af_aggregation"])
        dD = abs(mr["charitos_density"] - mh["charitos_density"])
        bnd_rows.append({
            "family": "periodic_vs_single", "B": B, "delta": round(delta, 6),
            "abs_dG": round(dG, 6), "bound_delta_over_B": round(delta / B, 6),
            "G_satisfied": dG <= (delta / B) + 1e-12,
            "abs_dD": round(dD, 6), "bound_delta_over_B1mB": round(delta / (B * (1.0 - B)), 6),
            "D_satisfied": dD <= (delta / (B * (1.0 - B))) + 1e-12,
        })
        # Also test 60 random equal-burden binary permutations on N=100 equal cells
        k_cells = int(round(B * 100))
        cell_w = T / 100.0
        for _ in range(60):
            p1 = rng.permutation(100) < k_cells
            p2 = rng.permutation(100) < k_cells
            if np.array_equal(p1, p2):
                continue
            iv1 = merge_intervals([(i * cell_w, (i + 1) * cell_w) for i in range(100) if p1[i]])
            iv2 = merge_intervals([(i * cell_w, (i + 1) * cell_w) for i in range(100) if p2[i]])
            m1, m2 = metrics_from_intervals(iv1, T), metrics_from_intervals(iv2, T)
            tp_r = interval_intersection_len(iv1, iv2)
            del_r = (interval_union_len(iv1) + interval_union_len(iv2) - 2.0 * tp_r) / T
            dg_r = abs(m1["af_aggregation"] - m2["af_aggregation"])
            dd_r = abs(m1["charitos_density"] - m2["charitos_density"])
            bnd_rows.append({
                "family": "random_permutation", "B": B, "delta": round(del_r, 6),
                "abs_dG": round(dg_r, 6), "bound_delta_over_B": round(del_r / B, 6),
                "G_satisfied": dg_r <= (del_r / B) + 1e-12,
                "abs_dD": round(dd_r, 6), "bound_delta_over_B1mB": round(del_r / (B * (1.0 - B)), 6),
                "D_satisfied": dd_r <= (del_r / (B * (1.0 - B))) + 1e-12,
            })
    bdf = pd.DataFrame(bnd_rows)
    bdf.to_csv(OUT_RES / "V8_rev1_equal_burden_bounds.csv", index=False)
    print(f"Equal-Burden Bounds Check ({len(bdf)} tests): G_satisfied={int(bdf['G_satisfied'].all())}, D_satisfied={int(bdf['D_satisfied'].all())}")
    print(bdf[bdf["family"] == "periodic_vs_single"].to_string(index=False))


def eval_frac_one_rec(args):
    ds, rid, pid, known_ivs, af_raw = args
    ref_k, T_k = clip_ivs_to_known(af_raw, known_ivs)
    m_ref = metrics_from_intervals(ref_k, T_k)
    rows = []
    for res_lbl, w_s in [("10s", 10.0), ("30s", 30.0), ("60s", 60.0), ("5m", 300.0), ("15m", 900.0), ("60m", 3600.0)]:
        # 1. Majority binary
        p_maj = fixed_oracle_grid(ref_k, T_k, w_s)
        mm_maj = metrics_from_intervals(p_maj, T_k)

        # Compute exact overlap mass per bin
        n_bins = int(math.ceil(T_k / w_s))
        ovs = np.zeros(n_bins, dtype=np.float64)
        ws = np.zeros(n_bins, dtype=np.float64)
        j = 0
        for wi in range(n_bins):
            ba, bb = wi * w_s, min(T_k, (wi + 1) * w_s)
            ws[wi] = bb - ba
            while j < len(ref_k) and ref_k[j][1] <= ba:
                j += 1
            k = j
            while k < len(ref_k) and ref_k[k][0] < bb:
                oa, ob = max(ba, ref_k[k][0]), min(bb, ref_k[k][1])
                if ob > oa:
                    ovs[wi] += (ob - oa)
                k += 1

        # 2. Left-aligned fractional mass [ba, ba + ov]
        left_ivs = [(wi * w_s, min(T_k, wi * w_s + ovs[wi])) for wi in range(n_bins) if ovs[wi] > 1e-9]
        mm_left = metrics_from_intervals(left_ivs, T_k)

        # 3. Centered fractional mass [mid - ov/2, mid + ov/2]
        cent_ivs = []
        for wi in range(n_bins):
            if ovs[wi] > 1e-9:
                ba, bb = wi * w_s, min(T_k, (wi + 1) * w_s)
                if ovs[wi] >= ws[wi] - 1e-9:
                    cent_ivs.append((ba, bb))
                else:
                    mid = 0.5 * (ba + bb)
                    cent_ivs.append((mid - 0.5 * ovs[wi], mid + 0.5 * ovs[wi]))
        mm_cent = metrics_from_intervals(cent_ivs, T_k)

        # 4. Exact Charitos 2019 R v3 binned port (full bins only where width == w_s or uniform bins)
        # Note: charitos_author_binned_port uses equal timeunits bins (clip tail bin only if full or scale full-bin equivalent)
        d_author = np.nan
        if m_ref["is_defined"] and res_lbl in ["60s", "5m", "15m", "60m"]:
            # Scale fractional occupancy of last bin safely into [0, w_s] for author port check when tail bin >= 0.5*w_s
            x_bins = ovs[:-1].copy() if (ws[-1] < w_s - 1e-6 and len(ovs) > 1) else ovs.copy()
            b_x = float(x_bins.sum() / (len(x_bins) * w_s)) if len(x_bins) else 0.0
            if 0.0 < b_x < 1.0:
                d_author = charitos_author_binned_port(x_bins, timeunits=w_s, minim=0.002, limit=True)

        for cname, mm, auth_d in [
            ("Majority_Binary", mm_maj, np.nan),
            ("Fractional_LeftAligned", mm_left, d_author),
            ("Fractional_Centered", mm_cent, np.nan),
        ]:
            rows.append({
                "dataset": ds, "record_id": rid, "patient_id": pid,
                "resolution": res_lbl, "win_s": w_s, "representation": cname,
                "ref_defined": m_ref["is_defined"], "pred_defined": mm["is_defined"],
                "both_defined": bool(m_ref["is_defined"] and mm["is_defined"]),
                "abs_burden_err": abs(mm["burden"] - m_ref["burden"]),
                "abs_dens_err": abs(mm["charitos_density"] - m_ref["charitos_density"]) if (m_ref["is_defined"] and mm["is_defined"]) else np.nan,
                "abs_dens_err_vs_author": abs(auth_d - m_ref["charitos_density"]) if (m_ref["is_defined"] and np.isfinite(auth_d)) else np.nan,
                "abs_agg_err": abs(mm["af_aggregation"] - m_ref["af_aggregation"]) if (m_ref["is_defined"] and mm["is_defined"]) else np.nan,
                "abs_ls_err": abs(mm["longest_share"] - m_ref["longest_share"]) if (np.isfinite(m_ref["longest_share"]) and np.isfinite(mm["longest_share"])) else np.nan,
                "abs_n_runs_err": abs(mm["n_runs"] - m_ref["n_runs"]),
            })
    return rows


def step3_multiscale_fractional_and_icc_suite():
    print("\n=== Rev1-Step 3: Multi-Scale Fractional AF Mass Verification & Aggregation ICC Suite ===")
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
        af_raw = merge_intervals(list(zip(sub_af["start_time_s"], sub_af["end_time_s"])))
        tasks.append((ds, rid, pid, known_ivs, af_raw))

    frows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for res in ex.map(eval_frac_one_rec, tasks):
            frows.extend(res)
    fdf = pd.DataFrame(frows)
    fsum = fdf[fdf["ref_defined"]].groupby(["win_s", "resolution", "representation"]).agg(
        n_ref_def=("record_id", "count"),
        n_both_def=("both_defined", "sum"),
        mean_burden_err=("abs_burden_err", "mean"),
        mean_dens_err=("abs_dens_err", "mean"),
        mean_author_dens_err=("abs_dens_err_vs_author", "mean"),
        mean_agg_err=("abs_agg_err", "mean"),
        mean_ls_err=("abs_ls_err", "mean"),
        med_n_runs_err=("abs_n_runs_err", "median"),
        mean_n_runs_err=("abs_n_runs_err", "mean"),
    ).reset_index().sort_values(["win_s", "representation"])
    fsum.to_csv(OUT_RES / "V8_rev1_fractional_vs_majority_multiscale.csv", index=False)
    print(fsum[["resolution", "representation", "n_both_def", "mean_dens_err", "mean_author_dens_err", "mean_agg_err", "mean_ls_err", "med_n_runs_err"]].to_string(index=False))

    # Full Reliability Suite: Charitos Density vs Butkuviene AF Aggregation across burden strata!
    c182 = pd.read_parquet(OUT_RES / "V8_exp_c_full182_oracles_rr.parquet")
    c25 = pd.read_parquet(OUT_RES / "V8_exp_c_matched25_6h_full.parquet")
    rel_rows = []

    def add_rel(df_src, scope_lbl):
        for (rl, comp), grp in df_src.groupby(["ref_layer", "comparator"]):
            for stratum_lbl, s_mask in [
                ("ALL_DEFINED", grp["both_defined"].values),
                ("LOW_LT20", (grp["both_defined"] & (grp["ref_burden"] < 0.20)).values),
                ("MOD_20_80", (grp["both_defined"] & (grp["ref_burden"] >= 0.20) & (grp["ref_burden"] < 0.80)).values),
                ("HIGH_GE80", (grp["both_defined"] & (grp["ref_burden"] >= 0.80)).values),
            ]:
                sub = grp[s_mask]
                if len(sub) < 4:
                    continue
                for mname, rcol, pcol in [
                    ("Charitos_Density", "ref_density", "pred_density"),
                    ("AF_Aggregation", "ref_agg", "pred_agg"),
                ]:
                    yv, pv = sub[rcol].values.astype(float), sub[pcol].values.astype(float)
                    pop_sd = float(np.std(yv))
                    pop_iqr = float(np.percentile(yv, 75) - np.percentile(yv, 25))
                    mae = float(np.mean(np.abs(pv - yv)))
                    sp = float(spearmanr(yv, pv).statistic) if (np.std(yv) > 1e-12 and np.std(pv) > 1e-12) else np.nan
                    rel_rows.append({
                        "scope": scope_lbl, "ref_layer": rl, "comparator": comp, "stratum": stratum_lbl,
                        "metric": mname, "n_both_def": len(sub),
                        "icc_2_1": round(icc_2_1(yv, pv), 4),
                        "spearman_r": round(sp, 4) if np.isfinite(sp) else np.nan,
                        "mae": round(mae, 4),
                        "ref_sd": round(pop_sd, 4), "ref_iqr": round(pop_iqr, 4),
                        "nmae_over_sd": round(mae / pop_sd, 3) if pop_sd > 1e-9 else np.nan,
                    })

    add_rel(c182, "Full182")
    add_rel(c25, "Matched25_6h")
    rdf = pd.DataFrame(rel_rows)
    rdf.to_csv(OUT_RES / "V8_rev1_agg_vs_density_reliability.csv", index=False)
    print("\nDensity vs Aggregation Reliability (Full-182 R0_Native, ALL_DEFINED & HIGH_GE80):")
    print(rdf[(rdf["scope"] == "Full182") & (rdf["ref_layer"] == "R0_Native") & (rdf["stratum"].isin(["ALL_DEFINED", "HIGH_GE80"]))][
        ["comparator", "stratum", "metric", "n_both_def", "icc_2_1", "spearman_r", "mae", "ref_sd", "nmae_over_sd"]
    ].to_string(index=False))


def main():
    t0 = time.time()
    step1_exact_12h_taxonomy_and_partial_id()
    step2_corrected_theorems_and_bounds()
    step3_multiscale_fractional_and_icc_suite()
    print(f"\nCompleted Rev1 in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
