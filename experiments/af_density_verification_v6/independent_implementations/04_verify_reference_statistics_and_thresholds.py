"""Independent Reference Statistics & Threshold Sensitivity Audit (AF-DENSITY V6).

Recomputes and audits:
1. Burden and Density distributions across IRIDIA, SHDB, LTAFDB (All AF+ vs Multi-run).
2. Continuous vs Binned density discrepancy statistics (median 0.001358, mean 0.015541, 95th percentile 0.051600, max 0.571650).
3. Duration threshold sensitivity (native, 30s, 2m, 6m):
   - Verifies run attrition vs cumulative AF duration preservation.
   - Audits LTAFDB micro-burst filtering (63.9% of runs <30s, 98.89% burden preserved).
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np

def audit_reference_distributions():
    ref_path = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/three_dataset_reference_af_density_manifest.csv")
    df = pd.read_csv(ref_path)
    
    # Filter AF-positive
    df_af = df[df["has_af"] == True].copy()
    
    stats_by_cohort = {}
    for ds in ["IRIDIA-AF", "SHDB-AF", "LTAFDB"]:
        sub = df_af[df_af["dataset"] == ds]
        sub_multi = sub[sub["is_multi_episode"] == True]
        
        stats_by_cohort[ds] = {
            "total_af_positive_recordings": int(len(sub)),
            "multi_run_recordings": int(len(sub_multi)),
            "single_run_recordings": int(len(sub) - len(sub_multi)),
            "multi_run_fraction_pct": float(round(len(sub_multi) / len(sub) * 100.0, 1)),
            "median_burden": float(round(sub["af_burden"].median(), 4)),
            "iqr_burden": float(round(sub["af_burden"].quantile(0.75) - sub["af_burden"].quantile(0.25), 4)),
            "median_density_all_af": float(round(sub["charitos_density_continuous"].median(), 4)),
            "median_density_multi_run": float(round(sub_multi["charitos_density_continuous"].median(), 4)),
            "median_binned_density_all_af": float(round(sub["charitos_density_binned_30s"].median(), 4)),
            "median_binned_density_multi_run": float(round(sub_multi["charitos_density_binned_30s"].median(), 4)),
        }
        
    # Discrepancy analysis
    valid_diff = df_af.dropna(subset=["charitos_density_continuous", "charitos_density_binned_30s"]).copy()
    valid_diff["abs_diff"] = (valid_diff["charitos_density_continuous"] - valid_diff["charitos_density_binned_30s"]).abs()
    
    med_diff = float(valid_diff["abs_diff"].median())
    mean_diff = float(valid_diff["abs_diff"].mean())
    p95_diff = float(valid_diff["abs_diff"].quantile(0.95))
    max_diff = float(valid_diff["abs_diff"].max())
    max_rec = valid_diff.loc[valid_diff["abs_diff"].idxmax()]
    
    discrepancy_stats = {
        "median_absolute_discrepancy": round(med_diff, 6),
        "mean_absolute_discrepancy": round(mean_diff, 6),
        "p95_absolute_discrepancy": round(p95_diff, 6),
        "max_absolute_discrepancy": round(max_diff, 6),
        "max_discrepancy_record": str(max_rec["record_id"]),
        "max_discrepancy_dataset": str(max_rec["dataset"]),
        "max_discrepancy_burden": float(max_rec["af_burden"]),
        "max_discrepancy_continuous_d": float(max_rec["charitos_density_continuous"]),
        "max_discrepancy_binned_d": float(max_rec["charitos_density_binned_30s"]),
        "discrepancy_target_match": (
            abs(med_diff - 0.001358) < 1e-4 and
            abs(mean_diff - 0.015541) < 1e-4 and
            abs(p95_diff - 0.051600) < 1e-4 and
            abs(max_diff - 0.571650) < 1e-4
        )
    }
    
    return stats_by_cohort, discrepancy_stats

def audit_threshold_sensitivity():
    sens_path = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/reports/three_dataset_harmonized_sensitivity_manifest.csv")
    df_sens = pd.read_csv(sens_path)
    
    thresholds = ["native", "30s", "2min", "6min"]
    res = {}
    
    for t in thresholds:
        sub = df_sens[df_sens["harmonization_threshold"] == t]
        tot_runs = int(sub["episode_count"].sum())
        tot_af_sec = float(sub["total_af_seconds"].sum())
        res[t] = {
            "recordings_count": len(sub),
            "total_af_runs": tot_runs,
            "total_af_hours": float(round(tot_af_sec / 3600.0, 2)),
        }
        
    # Specifically check LTAFDB native vs 30s
    ltaf_nat = df_sens[(df_sens["harmonization_threshold"] == "native") & (df_sens["dataset"] == "LTAFDB")]
    ltaf_30s = df_sens[(df_sens["harmonization_threshold"] == "30s") & (df_sens["dataset"] == "LTAFDB")]
    
    ltaf_nat_runs = int(ltaf_nat["episode_count"].sum())
    ltaf_30s_runs = int(ltaf_30s["episode_count"].sum())
    ltaf_nat_dur = float(ltaf_nat["total_af_seconds"].sum())
    ltaf_30s_dur = float(ltaf_30s["total_af_seconds"].sum())
    
    runs_excluded_pct = (ltaf_nat_runs - ltaf_30s_runs) / ltaf_nat_runs * 100.0
    dur_preserved_pct = ltaf_30s_dur / ltaf_nat_dur * 100.0
    
    res["LTAFDB_30s_Filtering"] = {
        "native_runs": ltaf_nat_runs,
        "runs_ge_30s": ltaf_30s_runs,
        "runs_excluded_count": ltaf_nat_runs - ltaf_30s_runs,
        "runs_excluded_pct": float(round(runs_excluded_pct, 2)),
        "native_af_hours": float(round(ltaf_nat_dur / 3600.0, 2)),
        "af_hours_ge_30s": float(round(ltaf_30s_dur / 3600.0, 2)),
        "duration_preserved_pct": float(round(dur_preserved_pct, 2)),
        "audit_target_match": (ltaf_nat_runs == 7358 and round(runs_excluded_pct, 1) == 63.9 and round(dur_preserved_pct, 2) == 98.89)
    }
    
    return res

def main():
    out_dir = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/experiments/af_density_verification_v6/recomputed_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PHASE 9: REFERENCE STATISTICS & THRESHOLD SENSITIVITY AUDIT")
    print("=" * 80)
    
    cohort_stats, disc_stats = audit_reference_distributions()
    sens_stats = audit_threshold_sensitivity()
    
    print("Cohort Density Profiles:")
    for ds, st in cohort_stats.items():
        print(f"  {ds:10s} (N={st['total_af_positive_recordings']}): Burden Med={st['median_burden']:.3f} | Density All={st['median_density_all_af']:.3f}, Multi={st['median_density_multi_run']:.3f} | Multi-run={st['multi_run_fraction_pct']}%")
        
    print("\nContinuous vs Binned Discrepancy:")
    print(f"  Median: {disc_stats['median_absolute_discrepancy']:.6f} (target 0.001358)")
    print(f"  Mean:   {disc_stats['mean_absolute_discrepancy']:.6f} (target 0.015541)")
    print(f"  P95:    {disc_stats['p95_absolute_discrepancy']:.6f} (target 0.051600)")
    print(f"  Max:    {disc_stats['max_absolute_discrepancy']:.6f} (target 0.571650, record {disc_stats['max_discrepancy_record']} in {disc_stats['max_discrepancy_dataset']})")
    print(f"  Discrepancy Match: {'PASS' if disc_stats['discrepancy_target_match'] else 'FAIL'}")
    
    print("\nDuration Threshold Sensitivity (LTAFDB 30s):")
    ltaf_filt = sens_stats["LTAFDB_30s_Filtering"]
    print(f"  Native runs: {ltaf_filt['native_runs']} -> >=30s runs: {ltaf_filt['runs_ge_30s']} ({ltaf_filt['runs_excluded_pct']}% excluded)")
    print(f"  AF duration preserved: {ltaf_filt['duration_preserved_pct']}% (Native {ltaf_filt['native_af_hours']}h -> {ltaf_filt['af_hours_ge_30s']}h)")
    print(f"  Sensitivity Match: {'PASS' if ltaf_filt['audit_target_match'] else 'FAIL'}")
    
    out_data = {
        "cohort_distributions": cohort_stats,
        "continuous_vs_binned_discrepancy": disc_stats,
        "threshold_sensitivity": sens_stats
    }
    
    out_file = out_dir / "independent_reference_statistics_audit.json"
    with open(out_file, "w") as fp:
        json.dump(out_data, fp, indent=2)
    print(f"\nSaved reference statistics audit to: {out_file}")

if __name__ == "__main__":
    main()
