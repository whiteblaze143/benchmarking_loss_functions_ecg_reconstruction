"""V8-E01/E02 Synthetic & Author Reference Fixtures + Residual Gap Mechanism Analysis."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/af_density_v8/src")
from v8_stage0_and_e01_e02 import exact_reference_charitos_components, compute_continuous_density

OUT_RES = Path("experiments/af_density_v8/results")
IRIDIA_META = Path("/data/mithunmanivannan/datasets/iridia_af_8405941_v1.0.1")


def charitos_discrete_bins(bins: np.ndarray) -> float:
    """Reference discrete-bin Charitos density matching author R implementation (on fractional bin burdens)."""
    T = float(len(bins))
    M = float(np.sum(bins))
    B = M / T
    if B <= 0.0:
        return 0.0
    if B >= 1.0 - 1e-12:
        return np.inf
    # For each contiguous block of non-zero? No, Charitos discrete algorithm sorts bins descending by burden b_i:
    # Actually let's check author R definition: F(p) on sorted blocks or minimal contiguous windows!
    # In Charitos et al. 2012 (Eq. 2): bins sorted descending! Wait: let's verify against author_r_results.tsv directly!
    s = np.sort(bins)[::-1]
    cum_m = np.cumsum(s) / M
    # Trapezoidal integration of F(p) - p over 1/T time steps
    p = cum_m
    F = np.arange(1, len(s) + 1) / T
    # Linear interpolation / trapezoidal area between curve (F, p)
    F0 = np.concatenate([[0.0], F])
    p0 = np.concatenate([[0.0], p])
    area = 0.0
    for k in range(len(s)):
        d_prev = abs(F0[k] - p0[k])
        d_curr = abs(F0[k + 1] - p0[k + 1])
        # Note: original discrete integration in Charitos 2012 uses sum |F(p_i) - p_i| * dp or unit-step
        # Let's compute both exact trapezoidal on (p, F) and compare!
        area += 0.5 * ((p0[k + 1] - F0[k + 1]) + (p0[k] - F0[k])) * (p0[k + 1] - p0[k])
    raw_d = 2.0 * abs(area) / (1.0 - B)
    return raw_d


def verify_synthetic_continuous():
    rows = []
    T = 1000.0
    # 1. Zero AF
    r0 = exact_reference_charitos_components([], T)
    rows.append({"fixture": "zero_af", "burden": r0["burden"], "density": r0["density"],
                 "expected_density": np.nan, "reason": r0["undef_reason"], "abs_diff": 0.0 if not r0["is_defined"] else 999.0})
    # 2. All AF
    r1 = exact_reference_charitos_components([(0.0, T)], T)
    rows.append({"fixture": "all_af", "burden": r1["burden"], "density": r1["density"],
                 "expected_density": np.nan, "reason": r1["undef_reason"], "abs_diff": 0.0 if not r1["is_defined"] else 999.0})
    # 3. Single contiguous block anywhere (middle [200, 400], B=0.2) -> analytical density == 1.0 exact!
    rs = exact_reference_charitos_components([(200.0, 400.0)], T)
    rows.append({"fixture": "single_block_B0.2", "burden": rs["burden"], "density": rs["density"],
                 "expected_density": 1.0, "reason": rs["undef_reason"], "abs_diff": abs(rs["density"] - 1.0)})
    # 4. Endpoints [0, 250] and [750, 1000], T=1000, B=0.5, M=500.
    # For m in [0, 250]: min_gap = 0 -> F = m/T = 0.5*p.
    # For m in [250, 500]: min_gap = 500 -> F = (m+500)/T = 0.5 + 0.5*p.
    # On [0, 0.5]: |0.5p - p| = 0.5p -> int_0^0.5 0.5p dp = 0.5 * 0.25 / 2 = 0.0625.
    # On [0.5, 1]: |0.5 + 0.5p - p| = 0.5(1-p) -> int_0.5^1 0.5(1-p) dp = 0.0625.
    # Total area A = 0.125 -> density = 2*0.125 / (1 - 0.5) = 0.25 / 0.5 = 0.500000!
    re = exact_reference_charitos_components([(0.0, 250.0), (750.0, 1000.0)], T)
    rows.append({"fixture": "two_endpoints_B0.5", "burden": re["burden"], "density": re["density"],
                 "expected_density": 0.5, "reason": re["undef_reason"], "abs_diff": abs(re["density"] - 0.5)})
    # 5. High-burden single boundary gap [20, 1000] (B=0.98) -> analytical 2A/(1-B) == 1.0!
    rh1 = exact_reference_charitos_components([(20.0, 1000.0)], T)
    rows.append({"fixture": "high_burden_boundary_gap_B0.98", "burden": rh1["burden"], "density": rh1["density"],
                 "expected_density": 1.0, "reason": rh1["undef_reason"], "abs_diff": abs(rh1["density"] - 1.0)})
    # 6. High-burden equidistant 50 tiny gaps of 0.4s across B=0.98 (uniform dispersion -> near minimum density!)
    # Each episode has length (980/50)=19.6s followed by 0.4s gap -> nearly uniform!
    ivs_u = [(i * 20.0, i * 20.0 + 19.6) for i in range(50)]
    rhu = exact_reference_charitos_components(ivs_u, T)
    d_dc = compute_continuous_density(ivs_u, T)
    rows.append({"fixture": "high_burden_uniform_50gaps_B0.98", "burden": rhu["burden"], "density": rhu["density"],
                 "expected_density": d_dc, "reason": rhu["undef_reason"], "abs_diff": abs(rhu["density"] - d_dc)})
    # 7. Same high-burden fixture AFTER bridging gaps < 1s -> single block [0, 999.6], density jumps from ~0.04 to 1.000!
    rhu_br = exact_reference_charitos_components([(0.0, 999.6)], T)
    rows.append({"fixture": "high_burden_after_bridging_to_1block", "burden": rhu_br["burden"], "density": rhu_br["density"],
                 "expected_density": 1.0, "reason": rhu_br["undef_reason"], "abs_diff": abs(rhu_br["density"] - 1.0)})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_RES / "V8_e01_synthetic_fixtures.csv", index=False)
    print("=== V8-E01 Continuous Synthetic Fixture Verification ===")
    print(df.to_string(index=False))


def verify_author_r_discrete():
    fix = pd.read_csv(IRIDIA_META / "author_r_fixtures.tsv", sep="\t")
    res = pd.read_csv(IRIDIA_META / "author_r_results.tsv", sep="\t")
    m = fix.merge(res, on="id")
    diffs = []
    for row in m.itertuples():
        bins = np.array([float(x) for x in str(row.data).split(";")])
        # Binary-compatible cases (all values in {0, 1}) -> compare with continuous exact!
        if set(np.unique(bins)).issubset({0.0, 1.0}):
            ivs = []
            in_af = False
            s = 0.0
            for idx, v in enumerate(bins):
                if v == 1.0 and not in_af:
                    s = float(idx)
                    in_af = True
                elif v == 0.0 and in_af:
                    ivs.append((s, float(idx)))
                    in_af = False
            if in_af:
                ivs.append((s, float(len(bins))))
            c = exact_reference_charitos_components(ivs, float(len(bins)))
            diffs.append({"id": row.id, "author_density": row.density, "our_cont": c["density"], "undef": c["undef_reason"]})
    print("\n=== Binary Author R Fixtures vs Exact Continuous ===")
    print(pd.DataFrame(diffs).to_string(index=False))


def analyze_e02_mechanism():
    pfull = pd.read_parquet(OUT_RES / "V8_e01_e02_perturbations_full.parquet")
    # 12h high-burden (>=80%) analysis across LTAFDB, SHDB-AF, IRIDIA-AF
    h12_ref = pfull[(pfull["horizon_h"] == 12.0) & (pfull["perturbation"] == "reference") & (pfull["ref_is_defined"])].copy()
    hi = h12_ref[h12_ref["ref_burden"] >= 0.80]
    print(f"\n=== V8-E02 Residual Gap Mechanism (12h horizon, high-burden >=80%, n_rec={len(hi)}) ===")
    print(hi.groupby("dataset").agg(
        n_records=("record_id", "count"),
        median_non_af_s=("total_non_af_s", "median"),
        median_n_gaps=("n_residual_gaps", "median"),
        median_gap_s=("median_gap_s", "median"),
        median_frac_lt60s=("frac_nonaf_in_lt60s_gaps", "median"),
        n_competing_afl_at=("has_competing_afl_at", "sum"),
    ).reset_index().to_string(index=False))

    # Compare all bridging thresholds across burden >= 0.80 vs < 0.80 at 12h
    h12_all = pfull[(pfull["horizon_h"] == 12.0) & (pfull["ref_is_defined"])].copy()
    h12_all["is_high_burden"] = h12_all["ref_burden"] >= 0.80
    mech = h12_all.groupby(["is_high_burden", "dataset", "perturbation"]).agg(
        n_ref=("record_id", "count"),
        mean_abs_err=("abs_dens_err", "mean"),
        median_abs_err=("abs_dens_err", "median"),
        max_abs_err=("abs_dens_err", "max"),
        n_post_undef=("pert_is_defined", lambda s: int((~s).sum())),
        n_post_single_run=("pert_n_runs", lambda s: int((s == 1).sum())),
        n_post_exact_one=("pert_density", lambda s: int((np.abs(s - 1.0) < 1e-6).sum())),
    ).reset_index()
    mech.to_csv(OUT_RES / "V8_e02_gap_mechanism_summary.csv", index=False)
    print("\nHigh-Burden (>=80%) vs Lower-Burden (<80%) Bridging Error & Single-Run Saturation (12h):")
    print(mech[mech["perturbation"].isin(["bridge_lt_10s", "bridge_lt_30s", "bridge_lt_60s", "bridge_lt_120s"])].to_string(index=False))


if __name__ == "__main__":
    verify_synthetic_continuous()
    verify_author_r_discrete()
    analyze_e02_mechanism()
