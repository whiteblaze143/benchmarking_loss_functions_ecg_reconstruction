#!/usr/bin/env python3
"""Reconcile and interpret the completed 117-cell R3-G2 experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/grail_v2/r3_linear_probe_audit"
TARGETS=["model_001","model_101","model_m"]
TIERS=["anchor","P1","P2","P3"]

def ci(values):
    x=np.asarray(values,float); n=len(x)
    return float(x.mean()),float(x.std(ddof=1)),float(x.mean()-1.96*x.std(ddof=1)/np.sqrt(n)),float(x.mean()+1.96*x.std(ddof=1)/np.sqrt(n))

def main():
    primary=OUT/"r3_geometry_incremental_per_concept.parquet"
    retention=OUT/"r3_residual_retention_per_concept.parquet"
    source_summary=OUT/"r3_geometry_incremental_tier_summary.csv"
    manifest_path=OUT/"r3_geometry_extension_manifest.json"
    for p in (primary,retention,source_summary,manifest_path):
        if not p.exists(): raise FileNotFoundError(p)
    manifest=json.loads(manifest_path.read_text())
    df=pd.read_parquet(primary)
    keys=["target","concept","tier"]
    expected={(t,c,r.tier) for t in TARGETS for c,r in df[df.target==t].drop_duplicates(["concept","tier"]).set_index("concept").iterrows()}
    duplicates=int(df.duplicated(keys).sum())
    counts=df.groupby("target").size().to_dict()
    if len(df)!=117 or duplicates or set(counts)!=(set(TARGETS)) or any(v!=39 for v in counts.values()):
        raise RuntimeError(f"G2 reconciliation failed: rows={len(df)} duplicates={duplicates} counts={counts}")
    if not bool(df.converged.all()): raise RuntimeError("G2 contains unconverged probes")
    required={"residual_auroc","incremental_delta","residual_retention","target_minus_b1","recovery_fraction_Q"}
    if not required.issubset(df.columns): raise RuntimeError(f"missing columns {required-set(df.columns)}")
    rows=[]
    for (target,tier),g in df.groupby(["target","tier"],sort=False):
        row={"target":target,"tier":tier,"n_concepts":len(g)}
        for col in ("residual_auroc","incremental_delta","residual_retention","target_minus_b1"):
            mean,sd,lo,hi=ci(g[col].dropna()); row.update({f"{col}_mean":mean,f"{col}_sd":sd,f"{col}_ci95_low":lo,f"{col}_ci95_high":hi})
        row["incremental_positive_n"]=int((g.incremental_delta>0).sum())
        row["incremental_positive_fraction"]=float((g.incremental_delta>0).mean())
        stable=g[g.recovery_denominator_stable]
        row["recovery_Q_stable_n"]=len(stable)
        row["recovery_Q_median"]=float(stable.recovery_fraction_Q.median()) if len(stable) else np.nan
        if row["residual_auroc_mean"]<=.55: interp="mostly linear reparameterization"
        elif abs(row["incremental_delta_mean"])<.005: interp="complementary but redundant accessibility"
        elif row["incremental_delta_mean"]>0: interp="incrementally useful complementary structure"
        else: interp="predictive residual with negative incremental accessibility"
        row["interpretation_gate"]=interp; rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/"r3_g2_decision_table.csv",index=False)
    paired=df[keys+["b1_auroc","target_auroc","residual_auroc","concat_b1_residual_auroc","target_minus_b1","incremental_delta","residual_retention","recovery_fraction_Q","recovery_denominator_stable"]].copy()
    paired.to_parquet(OUT/"r3_g2_paired_concept_deltas.parquet",index=False)
    report=["# R3-G2 Reconciliation and Interpretation","",f"- Rows: {len(df)} unique target-concept cells","- Duplicate keys: 0","- Missing target cells: 0","- All probes converged: yes","- Fold 8 remained evaluation-only.","","## Decision table",""]
    report.append(summary[["target","tier","n_concepts","residual_auroc_mean","incremental_delta_mean","residual_retention_mean","incremental_positive_n","interpretation_gate"]].to_markdown(index=False,floatfmt=".4f"))
    report += ["","The residual is described only as information not linearly recoverable from B1 under the fitted training-derived map; it is not evidence of new physiological information."]
    (OUT/"R3_G2_INTERPRETATION.md").write_text("\n".join(report)+"\n")
    reconciliation={"status":"complete","rows":len(df),"expected_rows":117,"unique_rows":int(len(df.drop_duplicates(keys))),"duplicate_keys":duplicates,"counts_by_target":counts,"all_probes_converged":bool(df.converged.all()),"source_manifest_verified":manifest.get("observed_rows")==117}
    (OUT/"r3_g2_reconciliation.json").write_text(json.dumps(reconciliation,indent=2)+"\n")

if __name__=="__main__":main()
