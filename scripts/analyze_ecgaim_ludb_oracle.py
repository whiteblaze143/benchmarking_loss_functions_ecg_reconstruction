#!/usr/bin/env python3
"""Analyze the fixed-clock LUDB oracle evaluation without changing its DB.

The primary result is a Pareto front over record-level tail metrics.  A
minimax rank is emitted only as a transparent, equal-endpoint sensitivity
ordering; it is never substituted for the raw panel.  Validation composite
loss is reported for audit but excluded from cross-mask ranking because each
mask optimizes a different composite objective.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_ROLE = "primary_missing_precordial"
EXPECTED_ECGAIM_GRID = 160  # five binary factors and a five-level MMD factor
LOG_PATTERN = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s+\|\s+Val Loss:\s+(?P<loss>[-+0-9.eE]+)"
    r"\s+\|\s+Val Missing Pearson:\s+(?P<pearson>[-+0-9.eE]+)"
)

# direction=1 means larger is better; -1 means smaller is better.
ORACLE_ENDPOINTS: tuple[tuple[str, int], ...] = (
    ("signal_pearson_p05", 1),
    ("signal_mse_p95", -1),
    ("event_abs_error_record_p95_mv", -1),
    ("qrs_window_rmse_record_p95_mv", -1),
    ("qrs_absolute_area_error_record_p95_mv_ms", -1),
    ("t_window_rmse_record_p95_mv", -1),
    ("t_absolute_area_error_record_p95_mv_ms", -1),
    ("st_j_abs_error_record_p95_mv", -1),
)
VALIDATION_ENDPOINTS: tuple[tuple[str, int], ...] = (
    ("validation_missing_pearson_best", 1),
)
ANALYSIS_ENDPOINTS = ORACLE_ENDPOINTS + VALIDATION_ENDPOINTS

SUBGROUP_METRICS: tuple[tuple[str, float], ...] = (
    ("signal_pearson_mean", 0.05),
    ("signal_mse_mean", 0.95),
    ("event_abs_error_p95_mv", 0.95),
    ("qrs_window_rmse_mean_mv", 0.95),
    ("qrs_area_abs_error_mean_mv_ms", 0.95),
    ("t_window_rmse_mean_mv", 0.95),
    ("t_area_abs_error_mean_mv_ms", 0.95),
    ("st_j_abs_error_mean_mv", 0.95),
)
RECORD_ENDPOINTS: tuple[tuple[str, str, float, int], ...] = (
    ("signal_pearson_p05", "signal_pearson_mean", 0.05, 1),
    ("signal_mse_p95", "signal_mse_mean", 0.95, -1),
    ("event_abs_error_record_p95_mv", "event_abs_error_p95_mv", 0.95, -1),
    ("qrs_window_rmse_record_p95_mv", "qrs_window_rmse_mean_mv", 0.95, -1),
    ("qrs_absolute_area_error_record_p95_mv_ms", "qrs_area_abs_error_mean_mv_ms", 0.95, -1),
    ("t_window_rmse_record_p95_mv", "t_window_rmse_mean_mv", 0.95, -1),
    ("t_absolute_area_error_record_p95_mv_ms", "t_area_abs_error_mean_mv_ms", 0.95, -1),
    ("st_j_abs_error_record_p95_mv", "st_j_abs_error_mean_mv", 0.95, -1),
)
BINARY_FACTORS = {
    "correlation": 1, "derivative": 2, "vcg": 3,
    "energy_distance": 4, "lead_consistency": 5,
}


def decode_mask(mask: str) -> dict[str, Any]:
    if len(mask) != 7 or any(ch not in "01" for ch in mask[:6]) or mask[6] not in "01234":
        raise ValueError(f"invalid seven-factor mask: {mask!r}")
    return {
        "mse": int(mask[0]), "correlation": int(mask[1]),
        "derivative": int(mask[2]), "vcg": int(mask[3]),
        "energy_distance": int(mask[4]), "lead_consistency": int(mask[5]),
        "mmd_kernel": int(mask[6]),
    }


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def parse_validation_log(path: Path) -> dict[str, Any]:
    epochs = [
        {"epoch": int(m.group("epoch")), "loss": float(m.group("loss")),
         "pearson": float(m.group("pearson"))}
        for m in LOG_PATTERN.finditer(path.read_text(errors="replace"))
    ] if path.is_file() else []
    if not epochs:
        return {
            "validation_epochs_found": 0, "validation_best_epoch": None,
            "validation_missing_pearson_best": None,
            "validation_composite_at_selected_epoch": None,
            "validation_composite_min": None,
        }
    # Matches the training checkpoint selector: highest validation Pearson.
    selected = max(enumerate(epochs), key=lambda item: (item[1]["pearson"], item[0]))[1]
    return {
        "validation_epochs_found": len(epochs),
        "validation_best_epoch": selected["epoch"],
        "validation_missing_pearson_best": selected["pearson"],
        "validation_composite_at_selected_epoch": selected["loss"],
        "validation_composite_min": min(item["loss"] for item in epochs),
    }


def dominates(left: dict[str, Any], right: dict[str, Any], endpoints: Iterable[tuple[str, int]]) -> bool:
    comparable = [(name, direction) for name, direction in endpoints if finite(left.get(name)) and finite(right.get(name))]
    if not comparable:
        return False
    weak = all(direction * float(left[name]) >= direction * float(right[name]) for name, direction in comparable)
    strict = any(direction * float(left[name]) > direction * float(right[name]) for name, direction in comparable)
    return weak and strict


def pareto_flags(rows: list[dict[str, Any]], endpoints: Iterable[tuple[str, int]]) -> list[bool]:
    return [not any(i != j and dominates(other, row, endpoints) for j, other in enumerate(rows))
            for i, row in enumerate(rows)]


def average_ranks(values: list[float], larger_better: bool) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=larger_better)
    ranks = [math.nan] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def add_sensitivity_order(rows: list[dict[str, Any]], endpoints: tuple[tuple[str, int], ...]) -> None:
    rank_columns: list[list[float]] = []
    used: list[str] = []
    for name, direction in endpoints:
        if not rows or not all(finite(row.get(name)) for row in rows):
            continue
        rank_columns.append(average_ranks([float(row[name]) for row in rows], direction == 1))
        used.append(name)
    for index, row in enumerate(rows):
        ranks = [column[index] for column in rank_columns]
        row["ranked_endpoints"] = ";".join(used)
        row["minimax_worst_endpoint_rank"] = max(ranks) if ranks else None
        row["mean_endpoint_rank"] = float(np.mean(ranks)) if ranks else None
    ordering = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["minimax_worst_endpoint_rank"] if rows[i]["minimax_worst_endpoint_rank"] is not None else math.inf,
            rows[i]["mean_endpoint_rank"] if rows[i]["mean_endpoint_rank"] is not None else math.inf,
            rows[i]["factorial_mask"],
        ),
    )
    for place, index in enumerate(ordering, 1):
        rows[index]["minimax_sensitivity_order"] = place


def factorial_contrasts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Descriptive matched-cell effects in improvement-positive orientation."""
    by_mask = {row["factorial_mask"]: row for row in rows}
    values: dict[tuple[str, str, str], list[float]] = {}

    def add(kind: str, contrast: str, endpoint: str, value: float) -> None:
        values.setdefault((kind, contrast, endpoint), []).append(value)

    for factor, index in BINARY_FACTORS.items():
        for mask, off in by_mask.items():
            if mask[index] != "0":
                continue
            on_mask = mask[:index] + "1" + mask[index + 1:]
            on = by_mask.get(on_mask)
            if not on:
                continue
            for endpoint, direction in ANALYSIS_ENDPOINTS:
                if finite(off.get(endpoint)) and finite(on.get(endpoint)):
                    add("binary_main", factor, endpoint,
                        direction * (float(on[endpoint]) - float(off[endpoint])))

    for kernel in "1234":
        for mask, off in by_mask.items():
            if mask[6] != "0":
                continue
            on = by_mask.get(mask[:6] + kernel)
            if not on:
                continue
            for endpoint, direction in ANALYSIS_ENDPOINTS:
                if finite(off.get(endpoint)) and finite(on.get(endpoint)):
                    add("mmd_vs_none", f"kernel_{kernel}_vs_0", endpoint,
                        direction * (float(on[endpoint]) - float(off[endpoint])))

    factors = list(BINARY_FACTORS.items())
    for first_position, (first, i) in enumerate(factors):
        for second, j in factors[first_position + 1:]:
            for mask, row00 in by_mask.items():
                if mask[i] != "0" or mask[j] != "0":
                    continue
                mask10 = mask[:i] + "1" + mask[i + 1:]
                mask01 = mask[:j] + "1" + mask[j + 1:]
                mask11 = mask10[:j] + "1" + mask10[j + 1:]
                row10, row01, row11 = by_mask.get(mask10), by_mask.get(mask01), by_mask.get(mask11)
                if not all((row10, row01, row11)):
                    continue
                for endpoint, direction in ANALYSIS_ENDPOINTS:
                    group = (row00, row10, row01, row11)
                    if all(finite(row.get(endpoint)) for row in group):
                        interaction = (
                            float(row11[endpoint]) - float(row10[endpoint])
                            - float(row01[endpoint]) + float(row00[endpoint])
                        )
                        add("binary_interaction", f"{first}_x_{second}", endpoint,
                            direction * interaction)

    output: list[dict[str, Any]] = []
    for (kind, contrast, endpoint), samples in sorted(values.items()):
        array = np.asarray(samples, dtype=float)
        output.append({
            "contrast_type": kind, "contrast": contrast, "endpoint": endpoint,
            "matched_cell_sets": len(samples), "mean_improvement": float(array.mean()),
            "median_improvement": float(np.median(array)),
            "minimum_improvement": float(array.min()), "maximum_improvement": float(array.max()),
            "orientation": "positive favors enabled factor/first-named MMD kernel; interaction is difference-in-differences",
        })
    return output


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def queue_counts(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text())
    jobs = payload if isinstance(payload, list) else payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise ValueError("queue jobs must be a list")
    selected = [job for job in jobs if str(job.get("id", "")).startswith("ecg_aim_f_")]
    result = {"queue_ecgaim_total": len(selected)}
    for status in sorted({str(job.get("status")) for job in selected}):
        result[f"queue_ecgaim_{status}"] = sum(str(job.get("status")) == status for job in selected)
    return result


def load_models(connection: sqlite3.Connection, log_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for db_row in connection.execute(
        "SELECT * FROM evaluations WHERE status='complete' ORDER BY factorial_mask"
    ):
        row = dict(db_row)
        row.update(json.loads(row.pop("primary_summary_json")))
        row.update(decode_mask(row["factorial_mask"]))
        run_name = row["model_id"].removeprefix("factorial_").removesuffix("_s42")
        # factorial_ecg_aim_MASK_s42 -> ecg_aim_f_MASK_s42
        log_name = f"ecg_aim_f_{row['factorial_mask']}_s42.log"
        row.update(parse_validation_log(log_dir / log_name))
        rows.append(row)
    return rows


def subgroup_rows(connection: sqlite3.Connection, minimum_records: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    subgroups = connection.execute(
        """SELECT subgroup,count(*) n_records FROM record_subgroups
        WHERE subgroup NOT LIKE 'diagnosis_exact:%'
        GROUP BY subgroup HAVING count(*)>=? ORDER BY subgroup""", (minimum_records,)
    ).fetchall()
    evaluations = connection.execute(
        "SELECT evaluation_id,model_id,factorial_mask FROM evaluations WHERE status='complete' ORDER BY factorial_mask"
    ).fetchall()
    for evaluation in evaluations:
        for subgroup in subgroups:
            records = connection.execute(
                """SELECT r.* FROM record_role_metrics r JOIN record_subgroups s USING(record_id)
                WHERE r.evaluation_id=? AND r.lead_role=? AND s.subgroup=?""",
                (evaluation["evaluation_id"], PRIMARY_ROLE, subgroup["subgroup"]),
            ).fetchall()
            if len(records) != subgroup["n_records"]:
                raise RuntimeError(f"incomplete subgroup coverage for {evaluation['model_id']} {subgroup['subgroup']}")
            item: dict[str, Any] = {
                "model_id": evaluation["model_id"], "factorial_mask": evaluation["factorial_mask"],
                "subgroup": subgroup["subgroup"], "n_records": len(records),
            }
            for metric, quantile in SUBGROUP_METRICS:
                values = np.asarray([row[metric] for row in records if finite(row[metric])], dtype=float)
                item[f"{metric}_{'p05' if quantile == 0.05 else 'p95'}"] = (
                    float(np.quantile(values, quantile)) if values.size else None
                )
            result.append(item)
    return result


def paired_bootstrap_vs_baseline(
    connection: sqlite3.Connection,
    models: list[dict[str, Any]],
    baseline_mask: str = "1000000",
    n_resamples: int = 2000,
    seed: int = 20260822,
) -> list[dict[str, Any]]:
    """Paired record bootstrap of tail-statistic improvements over baseline."""
    ids = {row["factorial_mask"]: row["evaluation_id"] for row in models}
    baseline_id = ids.get(baseline_mask)
    if baseline_id is None:
        return []

    def record_values(evaluation_id: int) -> dict[str, dict[str, float]]:
        return {
            row["record_id"]: dict(row)
            for row in connection.execute(
                "SELECT * FROM record_role_metrics WHERE evaluation_id=? AND lead_role=? ORDER BY record_id",
                (evaluation_id, PRIMARY_ROLE),
            )
        }

    baseline = record_values(baseline_id)
    record_ids = sorted(baseline, key=lambda value: int(value) if value.isdigit() else value)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(record_ids), size=(n_resamples, len(record_ids)))
    output: list[dict[str, Any]] = []
    for model in models:
        if model["factorial_mask"] == baseline_mask:
            continue
        current = record_values(model["evaluation_id"])
        if set(current) != set(baseline):
            raise RuntimeError(f"paired record coverage mismatch for {model['factorial_mask']}")
        for endpoint, record_metric, quantile, direction in RECORD_ENDPOINTS:
            base = np.asarray([baseline[rid][record_metric] for rid in record_ids], dtype=float)
            test = np.asarray([current[rid][record_metric] for rid in record_ids], dtype=float)
            if not np.isfinite(base).all() or not np.isfinite(test).all():
                continue
            estimate = direction * (float(np.quantile(test, quantile)) - float(np.quantile(base, quantile)))
            samples = direction * (
                np.quantile(test[indices], quantile, axis=1)
                - np.quantile(base[indices], quantile, axis=1)
            )
            output.append({
                "baseline_mask": baseline_mask, "factorial_mask": model["factorial_mask"],
                "endpoint": endpoint, "records": len(record_ids), "quantile": quantile,
                "improvement_estimate": estimate,
                "bootstrap_ci_low": float(np.quantile(samples, 0.025)),
                "bootstrap_ci_high": float(np.quantile(samples, 0.975)),
                "bootstrap_probability_improvement": float(np.mean(samples > 0.0)),
                "bootstrap_resamples": n_resamples,
                "orientation": "positive favors factorial_mask over baseline_mask",
            })
    return output


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--db", type=Path, default=ROOT / "results/ecgaim_ludb_oracle/ecgaim_ludb_oracle.sqlite")
    result.add_argument("--queue", type=Path, default=ROOT / "refine-logs/queue_3arch/queue_state.json")
    result.add_argument("--log-dir", type=Path, default=ROOT / "refine-logs/queue_3arch/jobs")
    result.add_argument("--output-dir", type=Path, default=ROOT / "results/ecgaim_ludb_oracle/analysis")
    result.add_argument("--minimum-subgroup-records", type=int, default=10)
    return result


def main() -> None:
    args = parser().parse_args()
    connection = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    models = load_models(connection, args.log_dir)
    if not models:
        raise RuntimeError("no complete oracle evaluations")

    flags = pareto_flags(models, ORACLE_ENDPOINTS)
    augmented_flags = pareto_flags(models, ANALYSIS_ENDPOINTS)
    baseline = next((row for row in models if row["factorial_mask"] == "1000000"), None)
    for row, flag, augmented_flag in zip(models, flags, augmented_flags):
        row["pareto_nondominated"] = int(flag)
        row["validation_augmented_pareto_nondominated"] = int(augmented_flag)
        if baseline:
            for name, direction in ANALYSIS_ENDPOINTS:
                if finite(row.get(name)) and finite(baseline.get(name)):
                    row[f"{name}_improvement_vs_1000000"] = direction * (float(row[name]) - float(baseline[name]))
    add_sensitivity_order(models, ORACLE_ENDPOINTS)
    models.sort(key=lambda row: row["factorial_mask"])

    subgroup = subgroup_rows(connection, args.minimum_subgroup_records)
    contrasts = factorial_contrasts(models)
    bootstrap = paired_bootstrap_vs_baseline(connection, models)
    counts = queue_counts(args.queue)
    complete = len(models) == EXPECTED_ECGAIM_GRID and counts.get("queue_ecgaim_completed", 0) == EXPECTED_ECGAIM_GRID
    status = {
        **counts,
        "expected_ecgaim_grid": EXPECTED_ECGAIM_GRID,
        "oracle_evaluations_complete": len(models),
        "grid_complete": complete,
        "ranking_status": "final" if complete else "provisional_do_not_select_winner",
        "pareto_models": [row["factorial_mask"] for row in models if row["pareto_nondominated"]],
        "validation_augmented_pareto_models": [
            row["factorial_mask"] for row in models
            if row["validation_augmented_pareto_nondominated"]
        ],
        "minimax_sensitivity_leader": min(models, key=lambda row: row["minimax_sensitivity_order"])["factorial_mask"],
        "factorial_contrast_rows": len(contrasts),
        "paired_bootstrap_rows": len(bootstrap),
        "validation_composite_warning": (
            "Reported for within-run audit only; it is not cross-mask comparable and is excluded from Pareto/rank endpoints."
        ),
        "validation_endpoint_role": (
            "Validation Pearson is supporting evidence only; primary Pareto and minimax selection use LUDB oracle endpoints."
        ),
        "single_seed_warning": "All current ECG-AIM cells use seed 42; winner uncertainty requires confirmatory seeds.",
    }
    atomic_csv(args.output_dir / "model_comparison.csv", models)
    atomic_csv(args.output_dir / "pareto_front.csv", [row for row in models if row["pareto_nondominated"]])
    atomic_csv(args.output_dir / "diagnostic_subgroup_tails.csv", subgroup)
    atomic_csv(args.output_dir / "factorial_contrasts.csv", contrasts)
    atomic_csv(args.output_dir / "paired_bootstrap_vs_1000000.csv", bootstrap)
    atomic_json(args.output_dir / "analysis_status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
