"""AF-DENSITY V8 Manuscript-Readiness M1:
1. Priority 1: Full 8,213-gap native rhythm taxonomy + raw-ECG adjudication of 2-10s LTAFDB gaps + ECG figures.
2. Priority 2: Unified E01/E02 vs E03 12h/24h denominator reconciliation.
3. Priority 3: Formal mathematical conditioning & perturbation theorem verification + phase diagram.
4. Priority 10 & E06 maturity: E05 free-dispersion/outlier/OOF forecast audit + E06 matured causal propensity & P02 CI.
"""
import math
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wfdb
from scipy.signal import butter, sosfiltfilt
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, merge_intervals

OUT_RES = Path("experiments/af_density_v8/results")
OUT_FIG = Path("experiments/af_density_v8/figures")
OUT_FIG.mkdir(parents=True, exist_ok=True)
V5_MAN = Path("experiments/af_density_continuity_v5/manifests")
V7P_DIR = Path("experiments/af_density_prediction_v7p")
LTAF_DIR = Path("/data/mithunmanivannan/datasets/ltafdb_1.0.0_reference")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")

NON_BEAT = {'+', '"', '~', '|', 's', 't', 'p', 'u', '='}


def run_priority1_adjudication():
    print("=== Priority 1: Full Internal-Gap Taxonomy & Raw-ECG 2-10s LTAFDB Adjudication ===")
    rhy = pd.read_parquet(V5_MAN / "canonical_rhythm_intervals.parquet")
    runs = pd.read_parquet(V5_MAN / "canonical_af_runs.parquet")
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    inv_map = {(r.dataset, str(r.record_id)): r for r in inv.itertuples()}

    # 1. Full known internal gap native rhythm taxonomy across all 3 cohorts
    tax_rows = []
    for ds in ["LTAFDB", "SHDB-AF", "IRIDIA-AF"]:
        r_ds = rhy[rhy["dataset"] == ds]
        af_ds = runs[runs["dataset"] == ds]
        for rid, agrp in af_ds.groupby("record_id"):
            agrp = agrp.sort_values("start_time_s")
            s = agrp["start_time_s"].values
            e = agrp["end_time_s"].values
            sub_r = r_ds[(r_ds["record_id"] == rid) & (r_ds["rhythm_label_harmonized"] != "AF")]
            for i in range(len(s) - 1):
                ga, gb = float(e[i]), float(s[i + 1])
                dur = gb - ga
                if dur <= 1e-9:
                    continue
                ov = sub_r[(sub_r["end_time_s"] > ga + 1e-6) & (sub_r["start_time_s"] < gb - 1e-6)]
                has_unk = bool((~ov["rhythm_known"]).any()) if len(ov) else False
                if has_unk:
                    continue
                lbls = "+".join(sorted(ov["rhythm_label_native"].astype(str).unique())) if len(ov) else "NONE"
                tax_rows.append({"dataset": ds, "record_id": str(rid), "gap_start_s": ga, "gap_end_s": gb, "dur_s": dur, "native_tags": lbls})

    tdf = pd.DataFrame(tax_rows)
    tdf["dur_bin"] = pd.cut(tdf["dur_s"], bins=[0, 10, 30, 60, 120, 360, 1e12],
                            labels=["01_lt10s", "02_10_30s", "03_30_60s", "04_1_2m", "05_2_6m", "06_ge6m"])
    # Simplify native tag group
    def tag_family(t):
        if t in ["VT", "IVR", "B", "T"]:
            return f"VENTRICULAR_{t}"
        if "VT" in t or "IVR" in t or "B" == t:
            return "VENTRICULAR_MIXED"
        if t in ["SBR", "N+SBR"]:
            return "BRADYCARDIA_SBR"
        if t in ["N", "NON_AF"]:
            return "NORMAL_OR_NON_AF"
        if "AFL" in t or "AT" in t or "SVTA" in t or "PAT" in t:
            return "ATRIAL_TACHY_FLUTTER"
        return f"OTHER_{t}"
    tdf["tag_family"] = tdf["native_tags"].apply(tag_family)
    tax_summary = tdf.groupby(["dataset", "dur_bin", "tag_family"], observed=True).size().reset_index(name="n_gaps")
    tax_summary.to_csv(OUT_RES / "V8_m1_all_internal_gaps_taxonomy.csv", index=False)
    print(tax_summary[tax_summary["dur_bin"] == "01_lt10s"].to_string(index=False))

    # 2. Direct Raw-ECG Waveform & Beat-Sequence Adjudication of 50 representative 2-10s LTAFDB gaps
    ltaf_2_10 = tdf[(tdf["dataset"] == "LTAFDB") & (tdf["dur_s"] >= 2.0) & (tdf["dur_s"] <= 10.0)].copy()
    rng = np.random.RandomState(42)
    picks = []
    # Pick 4 from each of worst-5 (15, 25, 33, 62, 74) + 30 stratified across (N, SBR, VT, IVR, B) in other records
    for rid in ["15", "25", "33", "62", "74"]:
        sub = ltaf_2_10[ltaf_2_10["record_id"] == rid]
        if len(sub):
            picks.append(sub.sample(n=min(4, len(sub)), random_state=42))
    other = ltaf_2_10[~ltaf_2_10["record_id"].isin(["15", "25", "33", "62", "74"])]
    for fam in ["NORMAL_OR_NON_AF", "BRADYCARDIA_SBR", "VENTRICULAR_VT", "VENTRICULAR_IVR", "VENTRICULAR_B"]:
        fsub = other[other["tag_family"] == fam]
        if len(fsub):
            picks.append(fsub.sample(n=min(6, len(fsub)), random_state=42))
    adj_cands = pd.concat(picks, ignore_index=True).drop_duplicates(["record_id", "gap_start_s"]).reset_index(drop=True)

    sos_hf = butter(2, [25.0, 55.0], btype="bandpass", fs=128.0, output="sos")
    adj_rows = []
    strip_data = []  # Save 6 representative raw ECG examples for plotting
    wanted_examples = {}

    for rid, grp in adj_cands.groupby("record_id"):
        rec = wfdb.rdrecord(str(LTAF_DIR / rid))
        ann = wfdb.rdann(str(LTAF_DIR / rid), "atr")
        fs = float(rec.fs)
        sig0 = rec.p_signal[:, 0].astype(np.float32)
        sig1 = rec.p_signal[:, 1].astype(np.float32) if rec.p_signal.shape[1] > 1 else sig0
        beat_mask = np.array([s not in NON_BEAT for s in ann.symbol], dtype=bool)
        b_samp = ann.sample[beat_mask]
        b_sym = np.array(ann.symbol)[beat_mask]

        for row in grp.itertuples():
            ga, gb, dur = row.gap_start_s, row.gap_end_s, row.dur_s
            sa, sb = int(round(ga * fs)), int(round(gb * fs))
            pad_s = int(round(6.0 * fs))
            w_lo, w_hi = max(0, sa - pad_s), min(len(sig0), sb + pad_s)
            g_sig = sig0[sa:sb]
            flank_pre = sig0[max(0, sa - pad_s):sa]
            flank_post = sig0[sb:min(len(sig0), sb + pad_s)]

            # Beats strictly inside gap
            in_g = (b_samp >= sa) & (b_samp < sb)
            g_beats = b_samp[in_g]
            g_syms = b_sym[in_g]
            n_beats = len(g_beats)
            n_v = int(np.sum(g_syms == "V"))
            n_a = int(np.sum(g_syms == "A"))
            n_n = int(np.sum(g_syms == "N"))

            rr_gap = np.diff(g_beats) / fs if n_beats >= 2 else np.array([])
            med_rr = float(np.median(rr_gap)) if len(rr_gap) else np.nan
            cvrr_gap = float(np.std(rr_gap) / np.mean(rr_gap)) if len(rr_gap) >= 2 and np.mean(rr_gap) > 0 else np.nan

            # P-wave morphology heuristic in sinus gaps: check 50-220ms before R-peak across N beats (RMS ratio vs TP baseline)
            p_ratios = []
            for bs, bsm in zip(g_beats, g_syms):
                if bsm == "N" and bs - int(0.32 * fs) >= 0 and bs + int(0.05 * fs) < len(sig0):
                    pr_seg = sig0[bs - int(0.22 * fs):bs - int(0.06 * fs)]
                    tp_seg = sig0[bs - int(0.32 * fs):bs - int(0.24 * fs)]
                    pr_std = float(np.std(pr_seg - np.mean(pr_seg)))
                    tp_std = float(np.std(tp_seg - np.mean(tp_seg))) + 1e-5
                    p_ratios.append(pr_std / tp_std)
            med_pr_ratio = float(np.median(p_ratios)) if p_ratios else np.nan

            # Noise / flatness check
            gap_std = float(np.std(g_sig))
            flank_std = float(0.5 * (np.std(flank_pre) + np.std(flank_post))) + 1e-6
            amp_ratio = gap_std / flank_std
            hf_gap = float(np.std(sosfiltfilt(sos_hf, sig0[max(0, sa - 32):min(len(sig0), sb + 32)])[32:-32])) if sb - sa > 16 else np.nan
            hf_flank = float(np.std(sosfiltfilt(sos_hf, flank_pre))) if len(flank_pre) > 32 else np.nan
            hf_ratio = (hf_gap / (hf_flank + 1e-6)) if (np.isfinite(hf_gap) and np.isfinite(hf_flank)) else np.nan

            # Objective physiological adjudication class
            if amp_ratio < 0.25 or not np.isfinite(gap_std):
                adj_cls = "FLATLINE_OR_SIGNAL_LOSS"
            elif np.isfinite(hf_ratio) and hf_ratio > 2.5:
                adj_cls = "HIGH_FREQ_ARTIFACT_BURST"
            elif row.native_tags in ["VT", "IVR", "B", "T"] or n_v >= 2:
                adj_cls = "VENTRICULAR_OVERRIDE_NOT_SINUS_RECOVERY"
            elif row.native_tags == "SBR" and n_beats <= 5:
                adj_cls = "TRANSIENT_BRADY_PAUSE_LE5_BEATS"
            elif n_n >= 3 and (np.isnan(cvrr_gap) or cvrr_gap < 0.12):
                adj_cls = "REGULAR_SINUS_LIKE_INTERRUPTION"
            else:
                adj_cls = "IRREGULAR_OR_AMBIGUOUS_SHORT_TAG"

            adj_rows.append({
                "record_id": rid, "gap_start_s": round(ga, 3), "gap_end_s": round(gb, 3), "dur_s": round(dur, 3),
                "native_tags": row.native_tags, "tag_family": row.tag_family,
                "n_beats": n_beats, "n_N": n_n, "n_V": n_v, "n_A": n_a,
                "med_rr_s": round(med_rr, 3) if np.isfinite(med_rr) else np.nan,
                "cvrr_gap": round(cvrr_gap, 4) if np.isfinite(cvrr_gap) else np.nan,
                "amp_ratio_vs_flank": round(amp_ratio, 3),
                "hf_noise_ratio_vs_flank": round(hf_ratio, 3) if np.isfinite(hf_ratio) else np.nan,
                "pre_qrs_pwave_rms_ratio": round(med_pr_ratio, 3) if np.isfinite(med_pr_ratio) else np.nan,
                "adjudication_class": adj_cls,
            })

            # Pick 6 clean illustrative raw strips for figure!
            key = (rid, row.native_tags)
            if len(wanted_examples) < 6 and (
                (rid == "15" and row.native_tags == "SBR" and "15_SBR" not in wanted_examples) or
                (rid == "25" and row.native_tags == "VT" and "25_VT" not in wanted_examples) or
                (rid == "33" and row.native_tags == "B" and "33_B" not in wanted_examples) or
                (rid == "74" and row.native_tags == "SBR" and "74_SBR" not in wanted_examples) or
                (rid == "62" and row.native_tags == "SBR" and "62_SBR" not in wanted_examples) or
                (row.native_tags == "N" and "ANY_N" not in wanted_examples and amp_ratio >= 0.4)
            ):
                label_k = "ANY_N" if (row.native_tags == "N" and rid not in ["15","25","33","62","74"]) else f"{rid}_{row.native_tags}"
                if label_k not in wanted_examples:
                    wanted_examples[label_k] = True
                    t_vec = (np.arange(w_lo, w_hi) / fs) - ga
                    b_win = (b_samp >= w_lo) & (b_samp < w_hi)
                    strip_data.append({
                        "title": f"LTAFDB {rid} | Native ({row.native_tags} [{dur:.2f}s] -> {adj_cls}",
                        "t": t_vec, "s0": sig0[w_lo:w_hi], "s1": sig1[w_lo:w_hi],
                        "dur": dur, "bt": (b_samp[b_win] / fs) - ga, "bs": b_sym[b_win],
                    })

    adj_df = pd.DataFrame(adj_rows)
    adj_df.to_csv(OUT_RES / "V8_m1_ltafdb_short_gap_ecg_adjudication.csv", index=False)
    print("\nRaw-ECG Adjudication Summary of Sampled 2-10s LTAFDB Gaps (n={}):".format(len(adj_df)))
    print(adj_df.groupby(["tag_family", "adjudication_class"]).size().reset_index(name="n_gaps").to_string(index=False))

    # Plot actual 2-lead Holter ECG strips
    if strip_data:
        fig, axes = plt.subplots(len(strip_data), 1, figsize=(13, 2.4 * len(strip_data)), sharex=False)
        if len(strip_data) == 1:
            axes = [axes]
        for ax, st in zip(axes, strip_data[:6]):
            ax.axvspan(0.0, st["dur"], color="#fde2e2", alpha=0.75, label="Annotated Non-AF Gap")
            ax.plot(st["t"], st["s0"], lw=0.9, color="#1f4e79", label="Lead 0 (mV)")
            ax.plot(st["t"], st["s1"] - np.ptp(st["s0"]) * 0.9, lw=0.75, color="#595959", alpha=0.85, label="Lead 1 (offset)")
            y_top = float(np.max(st["s0"])) * 0.98
            for bt, bs in zip(st["bt"], st["bs"]):
                col = "#c00000" if bs == "V" else "#006633" if bs == "N" else "#7030a0"
                ax.text(bt, y_top, bs, fontsize=7.5, ha="center", va="bottom", color=col, fontweight="bold")
            ax.set_title(st["title"], fontsize=9.5, fontweight="bold", loc="left")
            ax.set_ylabel("mV", fontsize=8)
            ax.set_xlim(-5.5, st["dur"] + 5.5)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel("Time Relative to Non-AF Gap Onset (seconds)", fontsize=9)
        plt.tight_layout()
        fig.savefig(OUT_FIG / "V8_ltafdb_short_gap_ecg_examples.png", dpi=240)
        fig.savefig(OUT_FIG / "V8_ltafdb_short_gap_ecg_examples.pdf")
        plt.close(fig)


def run_priority2_denominators():
    print("\n=== Priority 2: Unified E01/E02 vs E03 12h/24h Denominator Reconciliation ===")
    pfull = pd.read_parquet(OUT_RES / "V8_e01_e02_perturbations_full.parquet")
    cs = pd.read_csv(OUT_RES / "V8_e03_common_support_axes.csv")

    rows = []
    for h_val in [12.0, 24.0]:
        for ds in ["IRIDIA-AF", "LTAFDB", "SHDB-AF"]:
            # Method A (E03 canonical): deduplicate to 1st recording per patient BEFORE horizon/AF/defined filters
            cs_h = cs[(cs["horizon_h"] == h_val) & (cs["dataset"] == ds)]
            # Method B (E01): filter horizon & defined BEFORE deduplicating patient
            e1_h = pfull[(pfull["horizon_h"] == h_val) & (pfull["perturbation"] == "reference") & (pfull["dataset"] == ds)]
            e1_pre_dedup_def = e1_h[e1_h["ref_is_defined"]].sort_values(["patient_id", "record_id"]).drop_duplicates("patient_id")
            diff_pts = sorted(set(e1_pre_dedup_def["patient_id"]) - set(cs_h[cs_h["density"].notna()]["patient_id"]))
            rows.append({
                "horizon_h": h_val, "dataset": ds,
                "e03_1st_rec_pts_eligible": len(cs_h),
                "e03_1st_rec_af_plus": int((cs_h["n_runs"] >= 1).sum()),
                "e03_1st_rec_dens_defined": int(cs_h["density"].notna().sum()),
                "e01_any_rec_dens_defined_pts": len(e1_pre_dedup_def),
                "discrepancy_count": len(diff_pts),
                "discrepant_patient_ids": ";".join(diff_pts),
            })
    ddf = pd.DataFrame(rows)
    ddf.to_csv(OUT_RES / "V8_m1_reconciled_denominators.csv", index=False)
    print(ddf.to_string(index=False))


def run_priority3_theorems():
    print("\n=== Priority 3: Formal Conditioning & Perturbation Theorem Verification ===")
    T = 1000.0
    rows = []
    # Theorem 1 & 2 quantitative sweep across epsilon in [0.002, 0.50] and gap multiplicity K in [1, 2, 10, 50, 100]
    for eps in [0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        B = 1.0 - eps
        # Construction A: Single boundary gap [eps*T, T] -> exact D == 1.0!
        d_boundary = exact_reference_charitos_components([(eps * T, T)], T)["density"]
        # Construction B: K equally spaced micro-gaps of total fraction eps across [0, T] -> equidistant blocks!
        for K in [2, 10, 50, 100]:
            # Block length = (1 - eps)*T / K, gap length = eps*T / K
            blk = (1.0 - eps) * T / K
            step = T / K
            ivs = [(i * step, i * step + blk) for i in range(K)]
            d_frag = exact_reference_charitos_components(ivs, T)["density"]
            # Analytical formula verification:
            # Step function has jumps of (eps*T/K) at every multiple of blk -> F(p) ~ p + eps*(1 - ...), exact 2A/(1-B) ~ 1/K!
            # Wait: exact equidistant periodic blocks give D = 1/K when 1-interval shifted! Let's check exact difference!
            # Theorem 1 perturbation: bridge K-1 gaps (keep only 1 residual gap of size eps*T/K at end)!
            # Remaining AF = [0, T - eps*T/K], burden B_br = 1 - eps/K -> D_br == 1.000!
            ivs_br = [(0.0, T - (eps * T / K))]
            d_br = exact_reference_charitos_components(ivs_br, T)["density"]
            # Theorem 1 full collapse: bridge all K gaps -> ALL_AF (NaN)
            rows.append({
                "epsilon_nonaf": eps, "burden_ref": B, "K_gaps": K,
                "d_single_boundary_gap": round(float(d_boundary), 8),
                "d_uniform_K_gaps": round(float(d_frag), 8),
                "theoretical_1_over_K": round(1.0 / K, 8),
                "err_vs_1_over_K": abs(float(d_frag) - 1.0 / K),
                "burden_after_bridging_Km1": round(1.0 - eps / K, 6),
                "delta_burden_bridging_Km1": round(eps - eps / K, 6),
                "d_after_bridging_Km1": round(float(d_br), 8),
                "delta_density_bridging_Km1": round(float(d_br - d_frag), 8),
                "ratio_delta_D_over_delta_B": round(float(d_br - d_frag) / (eps - eps / K), 3),
            })

    tdf = pd.DataFrame(rows)
    tdf.to_csv(OUT_RES / "V8_m1_conditioning_theorems.csv", index=False)
    print(tdf[tdf["K_gaps"].isin([2, 50])].to_string(index=False))

    # Plot conditioning phase diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for K, col in [(2, "#1f77b4"), (10, "#ff7f0e"), (50, "#2ca02c"), (100, "#d62728")]:
        s = tdf[tdf["K_gaps"] == K]
        ax1.plot(s["burden_ref"], s["ratio_delta_D_over_delta_B"], marker="o", lw=2, label=f"K={K} gaps (bridging K-1 -> ΔD ≈ {1-1/K:.2f})", color=col)
    ax1.set_yscale("log")
    ax1.set_xlabel("Reference AF Burden B = 1 - ε", fontsize=10)
    ax1.set_ylabel("Sensitivity Ratio |ΔD / ΔB| (log scale)", fontsize=10)
    ax1.set_title("A. Amplification of Residual-Gap Bridging Error as B → 1⁻", fontsize=10.5, fontweight="bold")
    ax1.grid(alpha=0.25, which="both")
    ax1.legend(fontsize=8.5)

    # Panel B: Real 12h LTAFDB + SHDB + IRIDIA empirical <60s bridging jump vs burden
    pfull = pd.read_parquet(OUT_RES / "V8_e01_e02_perturbations_full.parquet")
    h12 = pfull[(pfull["horizon_h"] == 12.0) & (pfull["perturbation"] == "bridge_lt_60s") & (pfull["ref_is_defined"])].copy()
    for ds, col, mk in [("IRIDIA-AF", "#4c72b0", "o"), ("SHDB-AF", "#55a868", "s"), ("LTAFDB", "#c44e52", "^")]:
        sd = h12[h12["dataset"] == ds]
        def_sd = sd[sd["pert_is_defined"]]
        und_sd = sd[~sd["pert_is_defined"]]
        ax2.scatter(def_sd["ref_burden"], def_sd["abs_dens_err"], s=28, alpha=0.75, color=col, marker=mk, label=f"{ds} Defined (n={len(def_sd)})")
        if len(und_sd):
            ax2.scatter(und_sd["ref_burden"], np.full(len(und_sd), 0.56), s=55, color=col, marker="x", lw=1.8, label=f"{ds} → ALL_AF Undef (n={len(und_sd)})")
    ax2.axvline(0.80, color="black", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("Reference AF Burden B (12h Known Rhythm)", fontsize=10)
    ax2.set_ylabel("|ΔDensity| (x = Undefined ALL_AF)", fontsize=10)
    ax2.set_title("B. Empirical <60s Bridging Error & Density Loss Across Cohorts", fontsize=10.5, fontweight="bold")
    ax2.set_ylim(-0.02, 0.62)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_FIG / "V8_conditioning_phase_diagram.png", dpi=240)
    fig.savefig(OUT_FIG / "V8_conditioning_phase_diagram.pdf")
    plt.close(fig)


def run_priority10_e05_e06_audit():
    print("\n=== Priority 10 & E06 Maturity Audit ===")
    df = pd.read_csv(OUT_RES / "V8_e05_prospective_6h_6h_cohort.csv")
    sub = df[df["hist_density_defined"] == 1].dropna(subset=["hist_longest_share", "hist_lam_on_risk_h"]).copy().reset_index(drop=True)
    # Create winsorized (95th pct) & log1p rate predictors + reproducible 5-fold stratified patient splits
    sub["log_hist_lam_on"] = np.log1p(sub["hist_lam_on_risk_h"].clip(lower=0.0))
    cap = float(np.percentile(sub["hist_lam_on_risk_h"], 95))
    sub["wins_hist_lam_on"] = sub["hist_lam_on_risk_h"].clip(upper=cap)
    for col in ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "hist_lam_on_risk_h", "log_hist_lam_on", "wins_hist_lam_on"]:
        sub[f"z_{col}"] = (sub[col] - sub[col].mean()) / (sub[col].std() + 1e-9)

    top5_ids = set(sub.sort_values("fut_n_onsets", ascending=False).head(5)["patient_id"])
    no_top5 = sub[~sub["patient_id"].isin(top5_ids)].copy().reset_index(drop=True)
    for col in ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"]:
        no_top5[f"z_{col}"] = (no_top5[col] - no_top5[col].mean()) / (no_top5[col].std() + 1e-9)

    # 1. Free-dispersion MLE Negative Binomial (Discrete NegativeBinomial) across Full, No-Top-5, and Log1p-Rate!
    audit_rows = []
    for pop_lbl, pdf, rate_term in [
        ("Full_Defined_Uncond_FreeAlpha", sub, "z_hist_lam_on_risk_h"),
        ("Full_Defined_Uncond_Log1pRate_FreeAlpha", sub, "z_log_hist_lam_on"),
        ("LeaveTop5Out_Uncond_FreeAlpha", no_top5, "z_log_hist_lam_on"),
    ]:
        models = {
            "M1_Burden": ["C(dataset)", "z_hist_burden"],
            "M2_Burden_Density": ["C(dataset)", "z_hist_burden", "z_hist_density"],
            "M3_M2_Plus_Continuity": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share"],
            "M4_M3_Plus_RunCount": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share", "z_hist_n_runs"],
            "M5_M4_Plus_HistRate": ["C(dataset)", "z_hist_burden", "z_hist_density", "z_hist_longest_share", "z_hist_n_runs", rate_term],
        }
        prev_ll = None
        for mname, terms in models.items():
            try:
                res = smf.negativebinomial("fut_n_onsets ~ " + " + ".join(terms), data=pdf).fit(disp=False, maxiter=200)
                ll = float(res.llf)
                alpha_mle = float(res.params.get("alpha", np.nan))
                audit_rows.append({
                    "population": pop_lbl, "model": mname, "n_pts": len(pdf),
                    "log_lik": round(ll, 3), "aic": round(float(res.aic), 2),
                    "alpha_mle": round(alpha_mle, 3),
                    "delta_ll": round(ll - prev_ll, 3) if prev_ll is not None else np.nan,
                    "coef_dens": round(float(res.params.get("z_hist_density", np.nan)), 4),
                    "p_dens": round(float(res.pvalues.get("z_hist_density", np.nan)), 4),
                    "coef_cont": round(float(res.params.get("z_hist_longest_share", np.nan)), 4),
                    "p_cont": round(float(res.pvalues.get("z_hist_longest_share", np.nan)), 4),
                    "coef_rate": round(float(res.params.get(rate_term, np.nan)), 4),
                    "p_rate": round(float(res.pvalues.get(rate_term, np.nan)), 4),
                })
                prev_ll = ll
            except Exception as e:
                audit_rows.append({"population": pop_lbl, "model": mname, "n_pts": len(pdf), "error": str(e)})

    adf = pd.DataFrame(audit_rows)
    adf.to_csv(OUT_RES / "V8_m1_e05_statistical_audit.csv", index=False)
    print(adf.to_string(index=False))

    # 2. Strict Out-of-Sample 5-Fold Unconditional 6h -> 6h Forecast (NO future offset!)
    rng = np.random.RandomState(42)
    sub["fold"] = rng.randint(0, 5, size=len(sub))
    oof_specs = {
        "M1_Burden": ["hist_burden"],
        "M2_Burden_Density": ["hist_burden", "hist_density"],
        "M3_M2_Plus_Continuity": ["hist_burden", "hist_density", "hist_longest_share"],
        "M4_M3_Plus_RunCount": ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs"],
        "M5_Full_LogRate": ["hist_burden", "hist_density", "hist_longest_share", "hist_n_runs", "log_hist_lam_on"],
    }
    ds_dummies = pd.get_dummies(sub["dataset"], drop_first=True, dtype=float)
    X_ds = ds_dummies.values
    oof_rows = []
    for mname, cols in oof_specs.items():
        X = np.hstack([X_ds, sub[cols].values.astype(float)])
        y_cnt = sub["fut_n_onsets"].values.astype(float)
        y_prol = sub["fut_prolonged"].values.astype(int)
        pred_cnt = np.zeros(len(sub))
        pred_prol = np.zeros(len(sub))
        for f in range(5):
            tr, va = sub["fold"].values != f, sub["fold"].values == f
            sc = StandardScaler()
            Xtr = sm.add_constant(sc.fit_transform(X[tr]), has_constant="add")
            Xva = sm.add_constant(sc.transform(X[va]), has_constant="add")
            # Poisson GLM + Ridge/Clipped log-link for stable OOF onset count prediction
            g = sm.GLM(y_cnt[tr], Xtr, family=sm.families.Poisson()).fit_regularized(alpha=0.05, L1_wt=0.0)
            pred_cnt[va] = np.clip(np.exp(Xva @ g.params), 0.0, 1000.0)
            clf = LogisticRegression(C=1.0, max_iter=500).fit(sc.transform(X[tr]), y_prol[tr])
            pred_prol[va] = clf.predict_proba(sc.transform(X[va]))[:, 1]
        # Poisson deviance + Spearman correlation + prolonged AUROC
        y_safe = np.maximum(y_cnt, 1e-9)
        p_safe = np.maximum(pred_cnt, 1e-9)
        pois_dev = float(2.0 * np.mean(np.where(y_cnt > 0, y_cnt * np.log(y_safe / p_safe), 0.0) - (y_cnt - pred_cnt)))
        mae_cnt = float(mean_absolute_error(y_cnt, pred_cnt))
        sp_r = float(pd.Series(y_cnt).corr(pd.Series(pred_cnt), method="spearman"))
        auc_p = float(roc_auc_score(y_prol, pred_prol))
        oof_rows.append({
            "model": mname, "n_pts": len(sub), "oof_poisson_deviance": round(pois_dev, 3),
            "oof_count_mae": round(mae_cnt, 3), "oof_spearman_r": round(sp_r, 4),
            "oof_fut_prolonged_auroc": round(auc_p, 4),
        })
    oof_df = pd.DataFrame(oof_rows)
    oof_df.to_csv(OUT_RES / "V8_m1_e05_oof_forecast.csv", index=False)
    print("\nStrict Unconditional 5-Fold OOF 6h -> 6h Forecast (No Future Offset):")
    print(oof_df.to_string(index=False))

    # 3. E06 Strict Window-Maturity Causal Patient Propensity Audit + P02 ValidRR ECGFounder Bootstrap CI
    p1 = pd.read_parquet(V7P_DIR / "P01_landmark_manifest.parquet")
    p3 = pd.read_parquet(V7P_DIR / "P03_survival_manifest.parquet")
    inv = pd.read_parquet(OUT_RES / "V8_stage0_sample_inventory.parquet")
    unann_shdb = set(inv[(inv["dataset"] == "SHDB-AF") & (~inv["is_annotated"])]["record_id"].astype(str))
    p1_mask = (~((p1["dataset"] == "SHDB-AF") & (p1["record_id"].astype(str).isin(unann_shdb)))) & (
        ~((p1["dataset"] == "LTAFDB") & (p1["current_rhythm"] == "UNKNOWN"))
    )
    p1c = p1[p1_mask].copy().reset_index(drop=True)
    # P01 outcome matures at event_time_s + 1800s if positive, else landmark_time_s + 1800s!
    p1c["maturity_time_s"] = np.where(p1c["target_sustained_30m"].astype(bool), p1c["event_time_s"] + 1800.0, p1c["landmark_time_s"] + 1800.0)
    # P03 outcome matures at landmark_time_s + min(900s, time_to_event_s)
    p3["maturity_time_s"] = p3["landmark_time_s"] + np.minimum(900.0, p3["time_to_event_s"])

    mat_rows = []
    for tname, mdf, ycol in [("P01_CleanAnnotated", p1c, "target_sustained_30m"), ("P03_Survival15m", p3, "target_terminate_in_15m")]:
        mdf = mdf.sort_values(["patient_id", "record_id", "landmark_time_s"]).copy()
        y_true = mdf[ycol].astype(int).values
        matured_prop = np.zeros(len(mdf))
        for f in mdf["fold"].unique():
            tr_mean = float(y_true[mdf["fold"].values != f].mean())
            for (pt, rid), grp in mdf[mdf["fold"].values == f].groupby(["patient_id", "record_id"]):
                lt = grp["landmark_time_s"].values
                mt = grp["maturity_time_s"].values
                ys = grp[ycol].astype(int).values
                vals = []
                for i in range(len(lt)):
                    # Only prior landmarks in the same patient/record whose outcome matured BEFORE lt[i]!
                    m_past = (mt[:i] <= lt[i])
                    n_m = int(m_past.sum())
                    c_m = float(ys[:i][m_past].sum())
                    vals.append((c_m + 2.0 * tr_mean) / (n_m + 2.0))
                matured_prop[grp.index] = vals
        # Compare causal-matured AUROC!
        auc_mat = float(roc_auc_score(y_true, matured_prop))
        mat_rows.append({"task": tname, "n_rows": len(mdf), "strict_matured_propensity_auroc": round(auc_mat, 4)})
    mat_df = pd.DataFrame(mat_rows)
    mat_df.to_csv(OUT_RES / "V8_m1_e06_causal_maturity_audit.csv", index=False)
    print("\nE06 Strict Outcome-Maturity Causal Propensity AUROC:")
    print(mat_df.to_string(index=False))

    # P02 ValidRR ECGFounder equal-patient B=2000 bootstrap CI (was 0.5029 -> 0.5548 logistic; 0.4740 -> 0.5500 lgbm)
    p2 = pd.read_parquet(V7P_DIR / "P02_episode_persistence_manifest.parquet")
    f4 = pd.read_parquet(V7P_DIR / "embeddings/P02_f4_rr_features.parquet")
    ef = np.load(V7P_DIR / "embeddings/P02_ecgfounder_10s.npz", allow_pickle=True)["embeddings"]
    keep = f4["is_rr_valid"].astype(bool).values & np.all(np.isfinite(ef), axis=1)
    sub2 = p2[keep].reset_index(drop=True)
    rr_cols = ["mean_hr_bpm", "std_hr_bpm", "rmssd_ms", "cvrr", "pnn50", "num_beats"]
    sub2_rr = f4.loc[keep, rr_cols].values.astype(float)
    sub2_ef = ef[keep].astype(np.float32)
    f_s3a = [c for c in [
        "time_since_rec_start_s", "elapsed_af_duration_s", "last_af_duration_s", "time_since_last_af_s", "has_prior_af",
        "af_burden_15m", "af_burden_30m", "af_burden_1h", "af_burden_3h", "af_burden_6h",
        "episode_count_15m", "episode_count_30m", "episode_count_1h", "episode_count_3h", "episode_count_6h",
        "longest_af_run_15m", "longest_af_run_30m", "longest_af_run_1h", "longest_af_run_3h", "longest_af_run_6h",
        "brief_run_count_15m", "brief_run_count_30m", "brief_run_count_1h", "brief_run_count_3h", "brief_run_count_6h",
        "charitos_density_15m", "charitos_density_30m", "charitos_density_1h", "charitos_density_3h", "charitos_density_6h",
        "is_density_defined_3h",
    ] if c in sub2.columns]
    X_h = sub2[f_s3a].values.astype(float)
    y2 = sub2["target_persists_30m"].astype(int).values
    folds2 = sub2["fold"].values
    pts2 = sub2["patient_id"].values
    p2_eq_rows = []
    for mk in ["logistic", "lgbm"]:
        ohr, ohrf = np.zeros(len(y2)), np.zeros(len(y2))
        for f in sorted(np.unique(folds2)):
            tr, va = folds2 != f, folds2 == f
            imp = SimpleImputer(strategy="median"); sc = StandardScaler()
            Xtr = sc.fit_transform(imp.fit_transform(np.hstack([X_h[tr], sub2_rr[tr]])))
            Xva = sc.transform(imp.transform(np.hstack([X_h[va], sub2_rr[va]])))
            clf = LogisticRegression(C=1.0, max_iter=500) if mk == "logistic" else LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1)
            ohr[va] = clf.fit(Xtr, y2[tr]).predict_proba(Xva)[:, 1]
            sc_f = StandardScaler()
            pca = PCA(n_components=32, random_state=42)
            Ztr = pca.fit_transform(sc_f.fit_transform(sub2_ef[tr]))
            Zva = pca.transform(sc_f.transform(sub2_ef[va]))
            Xtr2 = sc.fit_transform(imp.fit_transform(np.hstack([X_h[tr], sub2_rr[tr], Ztr])))
            Xva2 = sc.transform(imp.transform(np.hstack([X_h[va], sub2_rr[va], Zva])))
            clf2 = LogisticRegression(C=1.0, max_iter=500) if mk == "logistic" else LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=1, verbose=-1)
            ohrf[va] = clf2.fit(Xtr2, y2[tr]).predict_proba(Xva2)[:, 1]
        diffs = np.array([roc_auc_score(y2[pts2 == p], ohrf[pts2 == p]) - roc_auc_score(y2[pts2 == p], ohr[pts2 == p])
                          for p in np.unique(pts2) if len(np.unique(y2[pts2 == p])) == 2])
        rng2 = np.random.RandomState(42)
        bs = np.array([diffs[rng2.choice(len(diffs), size=len(diffs), replace=True)].mean() for _ in range(2000)])
        plo = (np.sum(bs <= 0) + 1) / 2001.0; phi = (np.sum(bs >= 0) + 1) / 2001.0
        p2_eq_rows.append({
            "model": mk, "n_eq_pts": len(diffs), "eq_pt_delta": round(float(diffs.mean()), 4),
            "ci95_lo": round(float(np.percentile(bs, 2.5)), 4), "ci95_hi": round(float(np.percentile(bs, 97.5)), 4),
            "p_val": round(float(min(1.0, 2 * min(plo, phi))), 4),
        })
    p2_eq_df = pd.DataFrame(p2_eq_rows)
    p2_eq_df.to_csv(OUT_RES / "V8_m1_p02_eqpt_ecgf_ci.csv", index=False)
    print("\nP02 ValidRR ECGFounder Equal-Patient Paired Bootstrap CI:")
    print(p2_eq_df.to_string(index=False))


if __name__ == "__main__":
    run_priority1_adjudication()
    run_priority2_denominators()
    run_priority3_theorems()
    run_priority10_e05_e06_audit()
