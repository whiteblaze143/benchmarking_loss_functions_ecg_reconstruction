#!/usr/bin/env python3
"""Analyze compact fixed-region RDB results with rhythm-tail robustness."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_ecgaim_ludb_oracle import (
    BINARY_FACTORS,
    EXPECTED_ECGAIM_GRID,
    add_sensitivity_order,
    atomic_csv,
    atomic_json,
    decode_mask,
    finite,
    pareto_flags,
    parse_validation_log,
    queue_counts,
)


PRIMARY_ROLE = "primary_missing_precordial"
ENDPOINTS: tuple[tuple[str, int], ...] = (
    ("signal_pearson_p05", 1),
    ("signal_mse_p95", -1),
    ("boundary_voltage_abs_error_record_p95_mv", -1),
    ("qrs_window_rmse_record_p95_mv", -1),
    ("qrs_absolute_area_error_record_p95_mv_ms", -1),
    ("t_window_rmse_record_p95_mv", -1),
    ("t_absolute_area_error_record_p95_mv_ms", -1),
    ("st_j_abs_error_record_p95_mv", -1),
)


def load_models(connection: sqlite3.Connection, log_dir: Path) -> list[dict[str, Any]]:
    output = []
    for db_row in connection.execute("SELECT * FROM evaluations WHERE status='complete' ORDER BY factorial_mask"):
        row = dict(db_row); row.update(json.loads(row.pop("primary_summary_json")))
        row.update(decode_mask(row["factorial_mask"]))
        row.update(parse_validation_log(log_dir / f"ecg_aim_f_{row['factorial_mask']}_s42.log"))
        output.append(row)
    return output


def record_panel(connection: sqlite3.Connection, evaluation_id: int) -> dict[str, dict[str, float | str]]:
    result: dict[str, dict[str, float | str]] = {}
    for row in connection.execute(
        "SELECT * FROM record_role_signal_metrics WHERE evaluation_id=? AND lead_role=?",
        (evaluation_id, PRIMARY_ROLE),
    ):
        result[row["record_id"]] = {
            "canonical_rhythm": row["canonical_rhythm"],
            "signal_pearson": row["pearson_mean"], "signal_mse": row["mse_mean"],
        }
    for row in connection.execute(
        "SELECT * FROM record_role_wave_metrics WHERE evaluation_id=? AND lead_role=?",
        (evaluation_id, PRIMARY_ROLE),
    ):
        item = result[row["record_id"]]; wave = row["wave"].lower()
        item[f"{wave}_rmse"] = row["window_rmse_p95_mv"]
        item[f"{wave}_area"] = row["absolute_area_abs_error_p95_mv_ms"]
        item[f"{wave}_boundary"] = max(row["onset_abs_error_p95_mv"], row["offset_abs_error_p95_mv"])
    for row in connection.execute(
        "SELECT * FROM record_role_st_metrics WHERE evaluation_id=? AND lead_role=?",
        (evaluation_id, PRIMARY_ROLE),
    ):
        result[row["record_id"]]["st_j"] = row["j_abs_error_p95_mv"]
    return result


def aggregate_panel(records: list[dict[str, Any]]) -> dict[str, Any]:
    def q(key: str, probability: float) -> float | None:
        values = np.asarray([r[key] for r in records if finite(r.get(key))], dtype=float)
        return float(np.quantile(values, probability)) if len(values) else None
    boundaries = [max(v for k, v in row.items() if k.endswith("_boundary") and finite(v))
                  for row in records if any(k.endswith("_boundary") and finite(v) for k, v in row.items())]
    return {
        "n_records": len(records), "signal_pearson_p05": q("signal_pearson", .05),
        "signal_mse_p95": q("signal_mse", .95),
        "boundary_voltage_abs_error_record_p95_mv": float(np.quantile(boundaries, .95)) if boundaries else None,
        "qrs_window_rmse_record_p95_mv": q("qrs_rmse", .95),
        "qrs_absolute_area_error_record_p95_mv_ms": q("qrs_area", .95),
        "t_window_rmse_record_p95_mv": q("t_rmse", .95),
        "t_absolute_area_error_record_p95_mv_ms": q("t_area", .95),
        "st_j_abs_error_record_p95_mv": q("st_j", .95),
    }


def rhythm_rows(connection: sqlite3.Connection, models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for model in models:
        records = record_panel(connection, model["evaluation_id"])
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records.values(): groups[str(row["canonical_rhythm"])].append(row)
        for rhythm, values in sorted(groups.items()):
            output.append({"model_id": model["model_id"], "factorial_mask": model["factorial_mask"],
                           "canonical_rhythm": rhythm, **aggregate_panel(values)})
    return output


def factorial_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_mask = {row["factorial_mask"]: row for row in rows}; values = defaultdict(list)
    def add(kind: str, contrast: str, endpoint: str, value: float) -> None: values[(kind,contrast,endpoint)].append(value)
    for factor,index in BINARY_FACTORS.items():
        for mask,off in by_mask.items():
            if mask[index] != "0": continue
            on=by_mask.get(mask[:index]+"1"+mask[index+1:])
            if not on: continue
            for endpoint,direction in ENDPOINTS:
                if finite(off.get(endpoint)) and finite(on.get(endpoint)):
                    add("binary_main",factor,endpoint,direction*(float(on[endpoint])-float(off[endpoint])))
    for kernel in "1234":
        for mask,off in by_mask.items():
            if mask[6] != "0": continue
            on=by_mask.get(mask[:6]+kernel)
            if not on: continue
            for endpoint,direction in ENDPOINTS:
                if finite(off.get(endpoint)) and finite(on.get(endpoint)):
                    add("mmd_vs_none",f"kernel_{kernel}_vs_0",endpoint,direction*(float(on[endpoint])-float(off[endpoint])))
    factors=list(BINARY_FACTORS.items())
    for pos,(first,i) in enumerate(factors):
        for second,j in factors[pos+1:]:
            for mask,r00 in by_mask.items():
                if mask[i]!="0" or mask[j]!="0": continue
                r10=by_mask.get(mask[:i]+"1"+mask[i+1:]); r01=by_mask.get(mask[:j]+"1"+mask[j+1:])
                mask11=(mask[:i]+"1"+mask[i+1:]); mask11=mask11[:j]+"1"+mask11[j+1:]
                r11=by_mask.get(mask11)
                if not all((r10,r01,r11)): continue
                for endpoint,direction in ENDPOINTS:
                    if all(finite(r.get(endpoint)) for r in (r00,r10,r01,r11)):
                        add("binary_interaction",f"{first}_x_{second}",endpoint,direction*(float(r11[endpoint])-float(r10[endpoint])-float(r01[endpoint])+float(r00[endpoint])))
    return [{"contrast_type":k[0],"contrast":k[1],"endpoint":k[2],"matched_cell_sets":len(v),
             "mean_improvement":float(np.mean(v)),"median_improvement":float(np.median(v)),
             "minimum_improvement":float(np.min(v)),"maximum_improvement":float(np.max(v)),
             "orientation":"positive favors enabled factor; interactions are difference-in-differences"}
            for k,v in sorted(values.items())]


def bootstrap(connection: sqlite3.Connection, models: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    baseline = next((m for m in models if m["factorial_mask"]=="1000000"), None)
    if not baseline: return []
    base=record_panel(connection,baseline["evaluation_id"]); ids=sorted(base); rng=np.random.default_rng(seed); output=[]

    endpoint_fields = {
        "signal_pearson_p05": ("signal_pearson", .05),
        "signal_mse_p95": ("signal_mse", .95),
        "qrs_window_rmse_record_p95_mv": ("qrs_rmse", .95),
        "qrs_absolute_area_error_record_p95_mv_ms": ("qrs_area", .95),
        "t_window_rmse_record_p95_mv": ("t_rmse", .95),
        "t_absolute_area_error_record_p95_mv_ms": ("t_area", .95),
        "st_j_abs_error_record_p95_mv": ("st_j", .95),
    }

    def values(panel: dict[str, dict[str, Any]], endpoint: str) -> np.ndarray:
        if endpoint == "boundary_voltage_abs_error_record_p95_mv":
            return np.asarray([
                max((float(v) for k,v in panel[record_id].items() if k.endswith("_boundary") and finite(v)), default=math.nan)
                for record_id in ids
            ])
        field,_ = endpoint_fields[endpoint]
        return np.asarray([panel[record_id].get(field, math.nan) for record_id in ids], dtype=float)

    for model in models:
        if model is baseline: continue
        current=record_panel(connection,model["evaluation_id"])
        if set(current)!=set(base): raise RuntimeError("paired record coverage mismatch")
        for endpoint,direction in ENDPOINTS:
            base_values=values(base,endpoint); current_values=values(current,endpoint)
            paired=np.isfinite(base_values)&np.isfinite(current_values)
            base_values=base_values[paired]; current_values=current_values[paired]
            if not len(base_values): continue
            probability=.95 if endpoint=="boundary_voltage_abs_error_record_p95_mv" else endpoint_fields[endpoint][1]
            indices=rng.integers(0,len(base_values),size=(n,len(base_values)))
            estimate=direction*(float(np.quantile(current_values,probability))-float(np.quantile(base_values,probability)))
            samples=direction*(np.quantile(current_values[indices],probability,axis=1)-np.quantile(base_values[indices],probability,axis=1))
            output.append({"baseline_mask":"1000000","factorial_mask":model["factorial_mask"],"endpoint":endpoint,
                           "records":len(base_values),"improvement_estimate":estimate,"bootstrap_ci_low":float(np.quantile(samples,.025)),
                           "bootstrap_ci_high":float(np.quantile(samples,.975)),"bootstrap_probability_improvement":float(np.mean(np.asarray(samples)>0)),
                           "bootstrap_resamples":n,"orientation":"positive favors factorial_mask"})
    return output


def parser() -> argparse.ArgumentParser:
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db",type=Path,default=ROOT/"results/ecgaim_rdb_oracle/ecgaim_rdb_oracle.sqlite")
    result.add_argument("--queue",type=Path,default=ROOT/"refine-logs/queue_3arch/queue_state.json")
    result.add_argument("--log-dir",type=Path,default=ROOT/"refine-logs/queue_3arch/jobs")
    result.add_argument("--output-dir",type=Path,default=ROOT/"results/ecgaim_rdb_oracle/analysis")
    result.add_argument("--bootstrap-resamples",type=int,default=2000); return result


def main() -> None:
    args=parser().parse_args(); connection=sqlite3.connect(f"file:{args.db}?mode=ro",uri=True); connection.row_factory=sqlite3.Row
    models=load_models(connection,args.log_dir)
    if not models: raise RuntimeError("no complete RDB evaluations")
    flags=pareto_flags(models,ENDPOINTS)
    for row,flag in zip(models,flags): row["pareto_nondominated"]=int(flag)
    add_sensitivity_order(models,ENDPOINTS)
    rhythms=rhythm_rows(connection,models); contrasts=factorial_contrasts(models)
    paired=bootstrap(connection,models,args.bootstrap_resamples,20260822)
    counts=queue_counts(args.queue); complete=len(models)==EXPECTED_ECGAIM_GRID and counts.get("queue_ecgaim_completed",0)==EXPECTED_ECGAIM_GRID
    status={**counts,"expected_ecgaim_grid":EXPECTED_ECGAIM_GRID,"rdb_evaluations_complete":len(models),
            "grid_complete":complete,"ranking_status":"final" if complete else "provisional_do_not_select_winner",
            "pareto_models":[r["factorial_mask"] for r in models if r["pareto_nondominated"]],
            "minimax_sensitivity_leader":min(models,key=lambda r:r["minimax_sensitivity_order"])["factorial_mask"],
            "single_seed_warning":"All cells use seed 42; confirm finalists with additional seeds.",
            "segmentation_warning":"Oracle fidelity does not measure predicted timing or Dice/F1.",
            "rhythm_label_warning":"Canonical labels are mapping spreadsheet Chapman codes; released VT filenames map to SVT."}
    atomic_csv(args.output_dir/"model_comparison.csv",models); atomic_csv(args.output_dir/"pareto_front.csv",[r for r in models if r["pareto_nondominated"]])
    atomic_csv(args.output_dir/"rhythm_primary_tails.csv",rhythms); atomic_csv(args.output_dir/"factorial_contrasts.csv",contrasts)
    atomic_csv(args.output_dir/"paired_bootstrap_vs_1000000.csv",paired); atomic_json(args.output_dir/"analysis_status.json",status)
    print(json.dumps(status,indent=2,sort_keys=True))


if __name__=="__main__": main()
