#!/home/mithunmanivannan/.venv/bin/python3
"""Validate and aggregate generation-bound temporal morphology artifacts.

Only artifacts matching the current evaluator bytes and approved checkpoint/data
contract are accepted. Outputs are operational diagnostics until the eligible
model grid is complete; this script does not estimate factorial effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/factorial_v4/temporal_mmd_generation_bound"
DEFAULT_AUDIT = ROOT / "results/checkpoint_store/compatibility_audit.json"
DEFAULT_EVALUATOR = ROOT / "scripts/evaluate_temporal_mmd.py"
DEFAULT_TRAINING_SUMMARY = (
    ROOT / "results/factorial_mixed_level/training_diagnostics/summary.json"
)

IDENTITY_COLUMNS = (
    "model_id",
    "checkpoint_sha256",
    "contract_id",
    "source_bundle_sha256",
    "test_content_root_sha256",
    "evaluation_code_sha256",
    "target_feature_cache_sha256",
)
REQUIRED_COLUMNS = set(IDENTITY_COLUMNS) | {
    "dataset",
    "sample_rate_hz",
    "detector",
    "feature_pairing_rule",
    "pairing_tolerance_ms",
    "observed_lead_indices",
    "lead",
    "evaluated_lead_index",
    "clinical_feature",
    "n_beats",
    "n_records_total",
    "n_records_real_detected",
    "n_records_recon_detected",
    "n_records_paired",
    "record_pair_coverage",
    "n_real_beats_detected",
    "n_recon_beats_detected",
    "n_paired_beats_before_finite_filter",
    "n_unmatched_real_events",
    "n_unmatched_recon_events",
    "median_abs_pairing_error_ms",
    "p95_abs_pairing_error_ms",
    "mean_real",
    "mean_recon",
    "variance_ratio",
    "ba_robust_slope",
    "distribution_rbf_mmd2",
    "distribution_rbf_bandwidth",
    "distribution_mmd_real_samples",
    "distribution_mmd_recon_samples",
    "distribution_mmd_estimator",
    "distribution_mmd_bandwidth_rule",
    "distribution_mmd_sketch_rule",
    "distribution_mmd_sample_cap",
}
FINITE_COLUMNS = (
    "record_pair_coverage",
    "pairing_tolerance_ms",
    "median_abs_pairing_error_ms",
    "p95_abs_pairing_error_ms",
    "mean_real",
    "mean_recon",
    "variance_ratio",
    "ba_robust_slope",
    "distribution_rbf_mmd2",
    "distribution_rbf_bandwidth",
)
PER_RECORD_REQUIRED_COLUMNS = {
    "model_id", "checkpoint_sha256", "evaluation_code_sha256",
    "target_feature_cache_sha256", "dataset", "record_id", "lead",
    "clinical_feature", "detector_state", "target_detected",
    "reconstruction_detected", "n_real_events", "n_recon_events",
    "n_paired_events", "n_finite_paired_events", "n_unmatched_real_events",
    "n_unmatched_recon_events", "paired_sum_real", "paired_sum_recon",
    "paired_sum_sq_real", "paired_sum_sq_recon", "paired_mean_real",
    "paired_mean_recon", "paired_mean_difference", "paired_mae",
    "median_abs_pairing_error_ms", "p95_abs_pairing_error_ms",
}
SENSITIVITY_REQUIRED_COLUMNS = {
    "model_id", "checkpoint_sha256", "evaluation_code_sha256",
    "target_feature_cache_sha256", "dataset", "lead", "clinical_feature",
    "pairing_tolerance_ms", "n_records_total", "n_records_paired",
    "record_pair_coverage", "n_real_events", "n_recon_events",
    "n_paired_events", "n_finite_paired_events", "n_unmatched_real_events",
    "n_unmatched_recon_events", "mean_real", "mean_recon", "variance_ratio",
}


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def singleton(frame: pd.DataFrame, column: str):
    values = frame[column].drop_duplicates().tolist()
    if len(values) != 1:
        raise ValueError(f"{column} has {len(values)} values, expected exactly one")
    return values[0]


def build_controlled_kernel_pareto(
    feature_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build within-feature Pareto and pairwise-dominance diagnostics.

    Only complete seed/binary-prefix blocks containing exactly one model at
    each kernel level 0--4 are admitted. Metrics are never pooled across
    clinical features because the adaptive MMD bandwidth is feature-specific.
    """
    columns = [
        "seed", "binary_prefix", "lead", "clinical_feature", "model_id",
        "mmd_kernel", "absolute_bias", "abs_log_variance_deviation",
        "distribution_rbf_mmd2", "record_pair_coverage", "pareto_nondominated",
        "dominated_by_count", "dominated_by_model_ids",
    ]
    dominance_columns = [
        "seed", "binary_prefix", "dominator_model_id", "dominator_kernel",
        "dominated_model_id", "dominated_kernel", "complete_block_lead_feature_rows",
        "rows_dominated",
    ]
    if feature_frame.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=dominance_columns)

    frame = feature_frame.copy()
    frame["mask_text"] = frame.model_mask.astype(str).str.zfill(7)
    frame["binary_prefix"] = frame.mask_text.str[:6]
    if (frame.variance_ratio <= 0).any():
        raise ValueError("variance ratio must be positive for Pareto diagnostics")
    frame["absolute_bias"] = (frame.mean_recon - frame.mean_real).abs()
    frame["abs_log_variance_deviation"] = np.log(frame.variance_ratio).abs()

    pareto_rows = []
    dominance_events = []
    metric_columns = [
        "absolute_bias", "abs_log_variance_deviation",
        "distribution_rbf_mmd2", "coverage_shortfall",
    ]
    for (seed, prefix), block in frame.groupby(["seed", "binary_prefix"], sort=True):
        identities = block[["model_id", "mmd_kernel"]].drop_duplicates()
        if set(identities.mmd_kernel.astype(int)) != set(range(5)) or len(identities) != 5:
            continue
        if identities.mmd_kernel.duplicated().any():
            raise ValueError(f"duplicate kernel identity for seed={seed}, prefix={prefix}")
        for (lead, feature), endpoint in block.groupby(
            ["lead", "clinical_feature"], sort=True
        ):
            endpoint = endpoint.copy()
            if len(endpoint) != 5 or endpoint.model_id.nunique() != 5:
                raise ValueError(
                    f"complete kernel block has duplicate/missing endpoint row: "
                    f"seed={seed}, prefix={prefix}, lead={lead}, feature={feature}"
                )
            endpoint["coverage_shortfall"] = 1 - endpoint.record_pair_coverage
            endpoint = endpoint.sort_values("mmd_kernel")
            values = endpoint[metric_columns].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("non-finite controlled-kernel Pareto metric")
            model_ids = endpoint.model_id.tolist()
            kernels = endpoint.mmd_kernel.astype(int).tolist()
            dominated_by = [[] for _ in model_ids]
            for i, (dominator_id, dominator_kernel) in enumerate(zip(model_ids, kernels)):
                for j, (dominated_id, dominated_kernel) in enumerate(zip(model_ids, kernels)):
                    if i == j:
                        continue
                    dominates = bool(
                        np.less_equal(values[i], values[j]).all()
                        and np.less(values[i], values[j]).any()
                    )
                    if dominates:
                        dominated_by[j].append(dominator_id)
                        dominance_events.append({
                            "seed": int(seed),
                            "binary_prefix": prefix,
                            "dominator_model_id": dominator_id,
                            "dominator_kernel": int(dominator_kernel),
                            "dominated_model_id": dominated_id,
                            "dominated_kernel": int(dominated_kernel),
                            "lead": lead,
                            "clinical_feature": feature,
                        })
            for row, dominators in zip(endpoint.itertuples(index=False), dominated_by):
                pareto_rows.append({
                    "seed": int(seed),
                    "binary_prefix": prefix,
                    "lead": lead,
                    "clinical_feature": feature,
                    "model_id": row.model_id,
                    "mmd_kernel": int(row.mmd_kernel),
                    "absolute_bias": float(row.absolute_bias),
                    "abs_log_variance_deviation": float(row.abs_log_variance_deviation),
                    "distribution_rbf_mmd2": float(row.distribution_rbf_mmd2),
                    "record_pair_coverage": float(row.record_pair_coverage),
                    "pareto_nondominated": not dominators,
                    "dominated_by_count": len(dominators),
                    "dominated_by_model_ids": ";".join(sorted(dominators)),
                })

    pareto = pd.DataFrame(pareto_rows, columns=columns)
    events = pd.DataFrame(dominance_events)
    if events.empty:
        dominance = pd.DataFrame(columns=dominance_columns)
    else:
        dominance = (
            events.groupby([
                "seed", "binary_prefix", "dominator_model_id", "dominator_kernel",
                "dominated_model_id", "dominated_kernel",
            ], as_index=False)
            .agg(rows_dominated=("clinical_feature", "size"))
            .sort_values([
                "seed", "binary_prefix", "rows_dominated",
                "dominator_kernel", "dominated_kernel",
            ], ascending=[True, True, False, True, True])
        )
        denominators = (
            pareto[["seed", "binary_prefix", "lead", "clinical_feature"]]
            .drop_duplicates()
            .groupby(["seed", "binary_prefix"], as_index=False)
            .size()
            .rename(columns={"size": "complete_block_lead_feature_rows"})
        )
        dominance = dominance.merge(
            denominators, on=["seed", "binary_prefix"], validate="many_to_one"
        )[dominance_columns]
    return pareto, dominance


def compute_pipeline_throughput(
    *, training_median_minutes: float, evaluation_durations_seconds: pd.Series,
    eligible_models: int, accepted_models: int,
) -> dict:
    """Estimate whether one serial evaluator can keep pace with serial training."""
    evaluation_minutes = pd.to_numeric(
        evaluation_durations_seconds, errors="coerce"
    ).dropna() / 60
    if training_median_minutes <= 0 or evaluation_minutes.empty:
        raise ValueError("positive training and evaluation durations are required")
    backlog = eligible_models - accepted_models
    if backlog < 0:
        raise ValueError("accepted model count exceeds eligible model count")
    evaluation_median = float(evaluation_minutes.median())
    training_rate = 60 / float(training_median_minutes)
    evaluation_rate = 60 / evaluation_median
    net_drain_rate = evaluation_rate - training_rate
    keeps_pace = net_drain_rate > 0
    catchup_hours = backlog / net_drain_rate if keeps_pace else None
    return {
        "status": "observed_median_rate_estimate",
        "training_median_minutes_per_model": float(training_median_minutes),
        "evaluation_duration_samples": len(evaluation_minutes),
        "evaluation_minimum_minutes_per_model": float(evaluation_minutes.min()),
        "evaluation_median_minutes_per_model": evaluation_median,
        "evaluation_maximum_minutes_per_model": float(evaluation_minutes.max()),
        "eligible_models": int(eligible_models),
        "accepted_models": int(accepted_models),
        "backlog_models": int(backlog),
        "training_models_per_hour": training_rate,
        "evaluation_models_per_hour": evaluation_rate,
        "net_backlog_drain_models_per_hour": net_drain_rate,
        "evaluator_keeps_pace_at_observed_medians": keeps_pace,
        "estimated_backlog_catchup_hours": catchup_hours,
        "limitations": [
            "training and evaluation remain serial, uninterrupted, and concurrent",
            "future model mixes preserve the observed median service times",
            "the estimate ignores code-generation resets and hardware contention changes",
            "the catch-up value is an operational projection, not a confidence interval",
        ],
    }


def validate_artifact(
    metadata_path: Path,
    eligible: dict,
    contract: dict,
    evaluator_sha256: str,
    target_cache_sha256: str,
):
    metadata = json.loads(metadata_path.read_text())
    model_id = metadata.get("model_id")
    if model_id not in eligible:
        raise ValueError("model is not current-contract compatible")
    identity = eligible[model_id]
    if metadata.get("evaluation_code_sha256") != evaluator_sha256:
        raise ValueError("evaluator SHA does not match current evaluator bytes")
    if metadata.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
        raise ValueError("checkpoint SHA does not match compatibility audit")
    if metadata.get("contract_id") != contract["contract_id"]:
        raise ValueError("training contract ID mismatch")
    if metadata.get("source_bundle_sha256") != identity["source_bundle_sha256"]:
        raise ValueError("source-bundle SHA mismatch")
    expected_root = contract["split_content_roots"]["test"]["content_root_sha256"]
    if metadata.get("test_content_root_sha256") != expected_root:
        raise ValueError("test byte-content root mismatch")
    if metadata.get("target_feature_cache_sha256") != [target_cache_sha256]:
        raise ValueError("target-feature cache SHA mismatch")
    if metadata.get("reconstruction_batch_failures") != 0:
        raise ValueError("artifact reports reconstruction batch failures")
    if metadata.get("feature_pairing_rule") != (
        "within_record_monotonic_max_cardinality_min_abs_time"
    ):
        raise ValueError("artifact uses the wrong feature pairing rule")
    if metadata.get("pairing_tolerance_ms") != 100.0:
        raise ValueError("artifact uses the wrong pairing tolerance")
    if metadata.get("pairing_sensitivity_ms") != [25.0, 50.0, 75.0, 100.0, 150.0]:
        raise ValueError("artifact uses the wrong pairing-sensitivity grid")
    if metadata.get("distribution_mmd_estimator") != "biased_squared_rbf_mmd":
        raise ValueError("artifact uses the wrong distribution MMD estimator")
    if metadata.get("distribution_mmd_bandwidth_rule") != (
        "median_nonzero_pairwise_absolute_distance"
    ):
        raise ValueError("artifact uses the wrong distribution MMD bandwidth rule")
    if metadata.get("distribution_mmd_sketch_rule") != (
        "sorted_evenly_spaced_quantile_indices"
    ):
        raise ValueError("artifact uses the wrong distribution MMD sketch rule")
    if metadata.get("distribution_mmd_sample_cap") != 512:
        raise ValueError("artifact uses the wrong distribution MMD sample cap")

    csv_name = metadata.get("csv")
    if not csv_name or Path(csv_name).name != csv_name:
        raise ValueError("unsafe or missing CSV filename")
    csv_path = metadata_path.parent / csv_name
    if not csv_path.is_file():
        raise ValueError("CSV is missing")
    if not metadata.get("csv_sha256"):
        raise ValueError("CSV digest is absent")
    if sha256_file(csv_path) != metadata["csv_sha256"]:
        raise ValueError("CSV digest mismatch")

    frame = pd.read_csv(csv_path)
    if len(frame) != metadata.get("rows"):
        raise ValueError("CSV row count differs from metadata")
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"required CSV columns missing: {missing}")
    for column in IDENTITY_COLUMNS:
        observed = str(singleton(frame, column))
        expected = {
            "model_id": model_id,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "contract_id": contract["contract_id"],
            "source_bundle_sha256": identity["source_bundle_sha256"],
            "test_content_root_sha256": expected_root,
            "evaluation_code_sha256": evaluator_sha256,
            "target_feature_cache_sha256": target_cache_sha256,
        }[column]
        if observed != str(expected):
            raise ValueError(f"CSV {column} mismatch")
    expected_records = metadata["datasets"]["ptb_xl"]["records_expected"]
    if not frame["n_records_total"].eq(expected_records).all():
        raise ValueError("one or more feature rows omit expected test records")
    if not frame["dataset"].eq("ptb_xl").all():
        raise ValueError("unexpected dataset in current contract")
    numeric = frame[list(FINITE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("non-finite diagnostic metric")
    if not frame["record_pair_coverage"].between(0, 1, inclusive="both").all():
        raise ValueError("record-pair coverage outside [0,1]")
    if not frame["feature_pairing_rule"].eq(
        "within_record_monotonic_max_cardinality_min_abs_time"
    ).all():
        raise ValueError("CSV uses the wrong feature pairing rule")
    if not frame["pairing_tolerance_ms"].eq(100.0).all():
        raise ValueError("CSV uses the wrong pairing tolerance")
    if not frame["distribution_mmd_estimator"].eq("biased_squared_rbf_mmd").all():
        raise ValueError("CSV uses the wrong distribution MMD estimator")
    if not frame["distribution_mmd_bandwidth_rule"].eq(
        "median_nonzero_pairwise_absolute_distance"
    ).all():
        raise ValueError("CSV uses the wrong distribution MMD bandwidth rule")
    if not frame["distribution_mmd_sketch_rule"].eq(
        "sorted_evenly_spaced_quantile_indices"
    ).all():
        raise ValueError("CSV uses the wrong distribution MMD sketch rule")
    if not frame["distribution_mmd_sample_cap"].eq(512).all():
        raise ValueError("CSV uses the wrong distribution MMD sample cap")
    if (frame["distribution_rbf_mmd2"] < 0).any():
        raise ValueError("negative biased distribution MMD squared")
    for column in ("distribution_mmd_real_samples", "distribution_mmd_recon_samples"):
        if (frame[column] < 2).any() or (frame[column] > 512).any():
            raise ValueError(f"{column} outside [2, 512]")
        if (frame[column] > frame["n_beats"]).any():
            raise ValueError(f"{column} exceeds finite paired beats")
    if (frame[["n_unmatched_real_events", "n_unmatched_recon_events"]] < 0).any().any():
        raise ValueError("negative unmatched-event count")
    if not (
        frame["n_paired_beats_before_finite_filter"]
        + frame["n_unmatched_real_events"]
    ).eq(frame["n_real_beats_detected"]).all():
        raise ValueError("real event accounting does not close")
    if not (
        frame["n_paired_beats_before_finite_filter"]
        + frame["n_unmatched_recon_events"]
    ).eq(frame["n_recon_beats_detected"]).all():
        raise ValueError("reconstructed event accounting does not close")
    if (frame["n_beats"] > frame["n_paired_beats_before_finite_filter"]).any():
        raise ValueError("finite paired beats exceed pre-filter paired beats")
    if (frame["median_abs_pairing_error_ms"] > frame["p95_abs_pairing_error_ms"]).any():
        raise ValueError("median pairing error exceeds p95")
    if (
        frame["p95_abs_pairing_error_ms"]
        > frame["pairing_tolerance_ms"] + 1e-9
    ).any():
        raise ValueError("pairing error exceeds tolerance")
    for paired, total in (
        ("n_records_paired", "n_records_total"),
        ("n_records_real_detected", "n_records_total"),
        ("n_records_recon_detected", "n_records_total"),
    ):
        if (frame[paired] > frame[total]).any():
            raise ValueError(f"{paired} exceeds {total}")

    per_record_name = metadata.get("per_record_parquet")
    if not per_record_name or Path(per_record_name).name != per_record_name:
        raise ValueError("unsafe or missing per-record Parquet filename")
    per_record_path = metadata_path.parent / per_record_name
    if not per_record_path.is_file():
        raise ValueError("per-record Parquet is missing")
    if sha256_file(per_record_path) != metadata.get("per_record_parquet_sha256"):
        raise ValueError("per-record Parquet digest mismatch")
    per_record = pd.read_parquet(per_record_path)
    if len(per_record) != metadata.get("per_record_rows"):
        raise ValueError("per-record row count differs from metadata")
    missing_per_record = sorted(PER_RECORD_REQUIRED_COLUMNS - set(per_record.columns))
    if missing_per_record:
        raise ValueError(f"required per-record columns missing: {missing_per_record}")
    expected_per_record_rows = expected_records * 2 * 6
    if len(per_record) != expected_per_record_rows:
        raise ValueError("per-record ledger does not preserve the full denominator")
    if per_record.duplicated(["record_id", "lead", "clinical_feature"]).any():
        raise ValueError("duplicate per-record lead-feature identity")
    if per_record["record_id"].nunique() != expected_records:
        raise ValueError("per-record ledger has the wrong record count")
    if set(per_record["lead"]) != {"V3", "V6"}:
        raise ValueError("per-record ledger has the wrong lead set")
    if set(per_record["clinical_feature"]) != {
        "P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp", "QT_Interval_ms"
    }:
        raise ValueError("per-record ledger has the wrong feature set")
    per_record_expected_identity = {
        "model_id": model_id,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "evaluation_code_sha256": evaluator_sha256,
        "target_feature_cache_sha256": target_cache_sha256,
        "dataset": "ptb_xl",
    }
    for column, expected_value in per_record_expected_identity.items():
        if str(singleton(per_record, column)) != str(expected_value):
            raise ValueError(f"per-record {column} mismatch")
    count_columns = [
        "n_real_events", "n_recon_events", "n_paired_events",
        "n_finite_paired_events", "n_unmatched_real_events",
        "n_unmatched_recon_events",
    ]
    counts = per_record[count_columns].to_numpy(dtype=float)
    if (counts < 0).any() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError("per-record event counts must be nonnegative integers")
    if not (per_record.n_paired_events + per_record.n_unmatched_real_events).eq(
        per_record.n_real_events
    ).all():
        raise ValueError("per-record real event accounting does not close")
    if not (per_record.n_paired_events + per_record.n_unmatched_recon_events).eq(
        per_record.n_recon_events
    ).all():
        raise ValueError("per-record reconstruction event accounting does not close")
    if (per_record.n_finite_paired_events > per_record.n_paired_events).any():
        raise ValueError("per-record finite pairs exceed paired events")
    if not per_record.target_detected.eq(per_record.n_real_events > 0).all():
        raise ValueError("per-record target detector flag mismatch")
    if not per_record.reconstruction_detected.eq(per_record.n_recon_events > 0).all():
        raise ValueError("per-record reconstruction detector flag mismatch")
    expected_state = np.select(
        [
            per_record.target_detected & per_record.reconstruction_detected,
            per_record.target_detected,
            per_record.reconstruction_detected,
        ],
        ["both_detected", "target_only", "reconstruction_only"],
        default="neither_detected",
    )
    if not np.array_equal(per_record.detector_state.to_numpy(), expected_state):
        raise ValueError("per-record detector state mismatch")
    finite_rows = per_record.n_finite_paired_events > 0
    finite_metrics = [
        "paired_mean_real", "paired_mean_recon", "paired_mean_difference",
        "paired_mae", "median_abs_pairing_error_ms", "p95_abs_pairing_error_ms",
    ]
    if not np.isfinite(per_record.loc[finite_rows, finite_metrics].to_numpy(float)).all():
        raise ValueError("paired per-record metrics are non-finite")
    if per_record.loc[~finite_rows, finite_metrics].notna().any().any():
        raise ValueError("unpaired per-record rows contain paired metrics")

    for _, aggregate in frame.iterrows():
        rows = per_record[
            per_record.lead.eq(aggregate.lead)
            & per_record.clinical_feature.eq(aggregate.clinical_feature)
        ]
        checks = {
            "n_records_total": len(rows),
            "n_records_real_detected": int(rows.target_detected.sum()),
            "n_records_recon_detected": int(rows.reconstruction_detected.sum()),
            "n_records_paired": int((rows.n_paired_events > 0).sum()),
            "n_real_beats_detected": int(rows.n_real_events.sum()),
            "n_recon_beats_detected": int(rows.n_recon_events.sum()),
            "n_paired_beats_before_finite_filter": int(rows.n_paired_events.sum()),
            "n_unmatched_real_events": int(rows.n_unmatched_real_events.sum()),
            "n_unmatched_recon_events": int(rows.n_unmatched_recon_events.sum()),
            "n_beats": int(rows.n_finite_paired_events.sum()),
        }
        for column, observed in checks.items():
            if int(aggregate[column]) != observed:
                raise ValueError(f"aggregate {column} does not match per-record ledger")
        n = checks["n_beats"]
        sum_real = float(rows.paired_sum_real.sum())
        sum_recon = float(rows.paired_sum_recon.sum())
        if not np.isclose(float(aggregate.mean_real), sum_real / n, rtol=1e-10, atol=1e-12):
            raise ValueError("aggregate target mean does not match per-record ledger")
        if not np.isclose(float(aggregate.mean_recon), sum_recon / n, rtol=1e-10, atol=1e-12):
            raise ValueError("aggregate reconstruction mean does not match per-record ledger")
        real_variance = (float(rows.paired_sum_sq_real.sum()) - sum_real ** 2 / n) / (n - 1)
        recon_variance = (float(rows.paired_sum_sq_recon.sum()) - sum_recon ** 2 / n) / (n - 1)
        variance_ratio = recon_variance / max(real_variance, 1e-12)
        if not np.isclose(float(aggregate.variance_ratio), variance_ratio, rtol=1e-9, atol=1e-11):
            raise ValueError("aggregate variance ratio does not match per-record ledger")

    sensitivity_name = metadata.get("tolerance_sensitivity_csv")
    if not sensitivity_name or Path(sensitivity_name).name != sensitivity_name:
        raise ValueError("unsafe or missing tolerance-sensitivity CSV filename")
    sensitivity_path = metadata_path.parent / sensitivity_name
    if not sensitivity_path.is_file():
        raise ValueError("tolerance-sensitivity CSV is missing")
    if sha256_file(sensitivity_path) != metadata.get("tolerance_sensitivity_csv_sha256"):
        raise ValueError("tolerance-sensitivity CSV digest mismatch")
    sensitivity = pd.read_csv(sensitivity_path)
    if len(sensitivity) != metadata.get("tolerance_sensitivity_rows"):
        raise ValueError("tolerance-sensitivity row count differs from metadata")
    missing_sensitivity = sorted(
        SENSITIVITY_REQUIRED_COLUMNS - set(sensitivity.columns)
    )
    if missing_sensitivity:
        raise ValueError(
            f"required tolerance-sensitivity columns missing: {missing_sensitivity}"
        )
    if len(sensitivity) != 2 * 6 * 5:
        raise ValueError("tolerance-sensitivity grid does not have 60 rows")
    if sensitivity.duplicated(
        ["lead", "clinical_feature", "pairing_tolerance_ms"]
    ).any():
        raise ValueError("duplicate tolerance-sensitivity identity")
    if set(sensitivity.pairing_tolerance_ms) != {25.0, 50.0, 75.0, 100.0, 150.0}:
        raise ValueError("tolerance-sensitivity grid mismatch")
    for column, expected_value in per_record_expected_identity.items():
        if str(singleton(sensitivity, column)) != str(expected_value):
            raise ValueError(f"tolerance-sensitivity {column} mismatch")
    sensitivity_numeric = sensitivity[[
        "record_pair_coverage", "mean_real", "mean_recon", "variance_ratio"
    ]].to_numpy(float)
    if not np.isfinite(sensitivity_numeric).all():
        raise ValueError("non-finite tolerance-sensitivity metric")
    if not sensitivity.n_records_total.eq(expected_records).all():
        raise ValueError("tolerance-sensitivity denominator mismatch")
    sensitivity_count_columns = [
        "n_records_total", "n_records_paired", "n_real_events",
        "n_recon_events", "n_paired_events", "n_finite_paired_events",
        "n_unmatched_real_events", "n_unmatched_recon_events",
    ]
    sensitivity_counts = sensitivity[sensitivity_count_columns].to_numpy(float)
    if (sensitivity_counts < 0).any() or not np.equal(
        sensitivity_counts, np.floor(sensitivity_counts)
    ).all():
        raise ValueError("tolerance-sensitivity counts must be nonnegative integers")
    if (sensitivity.n_records_paired > sensitivity.n_records_total).any():
        raise ValueError("tolerance-sensitivity paired records exceed denominator")
    if (sensitivity.n_finite_paired_events > sensitivity.n_paired_events).any():
        raise ValueError("tolerance-sensitivity finite pairs exceed paired events")
    if not np.allclose(
        sensitivity.record_pair_coverage,
        sensitivity.n_records_paired / sensitivity.n_records_total,
        rtol=0, atol=1e-12,
    ):
        raise ValueError("tolerance-sensitivity coverage does not match counts")
    if not (sensitivity.n_paired_events + sensitivity.n_unmatched_real_events).eq(
        sensitivity.n_real_events
    ).all():
        raise ValueError("tolerance-sensitivity real accounting does not close")
    if not (sensitivity.n_paired_events + sensitivity.n_unmatched_recon_events).eq(
        sensitivity.n_recon_events
    ).all():
        raise ValueError("tolerance-sensitivity reconstruction accounting does not close")
    for _, group in sensitivity.groupby(["lead", "clinical_feature"]):
        ordered = group.sort_values("pairing_tolerance_ms")
        if (ordered.n_paired_events.diff().dropna() < 0).any():
            raise ValueError("paired events decrease as tolerance increases")
        if (ordered.n_finite_paired_events.diff().dropna() < 0).any():
            raise ValueError("finite paired events decrease as tolerance increases")
        if (ordered.n_records_paired.diff().dropna() < 0).any():
            raise ValueError("paired records decrease as tolerance increases")
        if (ordered.n_unmatched_real_events.diff().dropna() > 0).any():
            raise ValueError("unmatched real events increase as tolerance increases")
        if (ordered.n_unmatched_recon_events.diff().dropna() > 0).any():
            raise ValueError("unmatched reconstruction events increase as tolerance increases")
    primary_sensitivity = sensitivity.loc[
        sensitivity.pairing_tolerance_ms.eq(100.0)
    ]
    primary = frame.merge(
        primary_sensitivity,
        on=["model_id", "checkpoint_sha256", "evaluation_code_sha256",
            "target_feature_cache_sha256", "dataset", "lead", "clinical_feature"],
        suffixes=("_aggregate", "_sensitivity"),
        validate="one_to_one",
    )
    if len(primary) != len(frame):
        raise ValueError("100 ms sensitivity slice does not cover aggregate rows")
    primary_pairs = {
        "n_records_total": "n_records_total",
        "n_records_paired": "n_records_paired",
        "n_real_beats_detected": "n_real_events",
        "n_recon_beats_detected": "n_recon_events",
        "n_paired_beats_before_finite_filter": "n_paired_events",
        "n_beats": "n_finite_paired_events",
        "n_unmatched_real_events": "n_unmatched_real_events",
        "n_unmatched_recon_events": "n_unmatched_recon_events",
    }
    for aggregate_column, sensitivity_column in primary_pairs.items():
        aggregate_key = (
            f"{aggregate_column}_aggregate"
            if aggregate_column == sensitivity_column else aggregate_column
        )
        sensitivity_key = (
            f"{sensitivity_column}_sensitivity"
            if aggregate_column == sensitivity_column else sensitivity_column
        )
        if not primary[aggregate_key].eq(primary[sensitivity_key]).all():
            raise ValueError(
                f"100 ms sensitivity {sensitivity_column} does not match aggregate"
            )
    for column in ("record_pair_coverage", "mean_real", "mean_recon", "variance_ratio"):
        if not np.allclose(
            primary[f"{column}_aggregate"], primary[f"{column}_sensitivity"],
            rtol=1e-9, atol=1e-11,
        ):
            raise ValueError(f"100 ms sensitivity {column} does not match aggregate")
    return metadata, frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument(
        "--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    audit = json.loads(args.audit.resolve().read_text())
    contract = audit["contract"]
    eligible = {
        item["model_id"]: item for item in audit["models"] if item.get("compatible")
    }
    evaluator_sha256 = sha256_file(args.evaluator.resolve())
    target_metadata_path = input_dir / "_target_ptb_xl_features.json"
    target_metadata = json.loads(target_metadata_path.read_text())
    target_cache_sha256 = target_metadata["parquet_sha256"]

    accepted_metadata = []
    accepted_frames = []
    exclusions = []
    for metadata_path in sorted(input_dir.glob("f_*.json")):
        try:
            metadata, frame = validate_artifact(
                metadata_path,
                eligible,
                contract,
                evaluator_sha256,
                target_cache_sha256,
            )
        except Exception as error:
            exclusions.append({
                "artifact": metadata_path.name,
                "reason": str(error),
            })
            continue
        accepted_metadata.append(metadata)
        accepted_frames.append(frame)

    artifact_columns = [
        "model_id", "checkpoint_sha256", "checkpoint_size_bytes", "contract_id",
        "source_bundle_sha256", "test_content_root_sha256", "evaluation_code_sha256",
        "feature_pairing_rule", "pairing_tolerance_ms",
        "distribution_mmd_estimator", "distribution_mmd_bandwidth_rule",
        "distribution_mmd_sketch_rule", "distribution_mmd_sample_cap",
        "per_record_rows", "per_record_parquet", "per_record_parquet_sha256",
        "tolerance_sensitivity_rows", "tolerance_sensitivity_csv",
        "tolerance_sensitivity_csv_sha256",
        "rows", "reconstruction_batch_failures", "started_at", "completed_at",
        "duration_seconds", "csv", "csv_sha256",
    ]
    artifact_frame = pd.DataFrame(accepted_metadata, columns=artifact_columns)
    feature_frame = (
        pd.concat(accepted_frames, ignore_index=True)
        if accepted_frames else pd.DataFrame(columns=sorted(REQUIRED_COLUMNS))
    )
    exclusion_frame = pd.DataFrame(exclusions, columns=["artifact", "reason"])
    pareto_frame, dominance_frame = build_controlled_kernel_pareto(feature_frame)
    training_summary_path = args.training_summary.resolve()
    # Bind the temporal generation to the exact diagnostic bytes used for its
    # throughput calculation. The live diagnostics watcher may atomically
    # refresh summary.json immediately afterward, so hashing that mutable path
    # in the book creates a time-of-check/time-of-use race.
    training_summary_bytes = training_summary_path.read_bytes()
    training_summary = json.loads(training_summary_bytes)
    training_snapshot_name = "training_diagnostic_summary.snapshot.json"
    atomic_bytes(input_dir / training_snapshot_name, training_summary_bytes)
    if training_summary.get("compatible_models") == len(eligible):
        pipeline_throughput = compute_pipeline_throughput(
            training_median_minutes=training_summary["duration_minutes"]["median"],
            evaluation_durations_seconds=artifact_frame.duration_seconds,
            eligible_models=len(eligible),
            accepted_models=len(artifact_frame),
        )
    else:
        pipeline_throughput = {
            "status": "stale_training_summary",
            "training_summary_compatible_models": training_summary.get(
                "compatible_models"
            ),
            "eligible_models": len(eligible),
        }

    atomic_csv(input_dir / "accepted_model_artifacts.csv", artifact_frame)
    atomic_csv(input_dir / "accepted_feature_summary.csv", feature_frame)
    atomic_csv(input_dir / "excluded_artifacts.csv", exclusion_frame)
    atomic_csv(input_dir / "controlled_kernel_feature_pareto.csv", pareto_frame)
    atomic_csv(input_dir / "controlled_kernel_pairwise_dominance.csv", dominance_frame)
    output_names = {
        "models": "accepted_model_artifacts.csv",
        "features": "accepted_feature_summary.csv",
        "exclusions": "excluded_artifacts.csv",
        "controlled_kernel_pareto": "controlled_kernel_feature_pareto.csv",
        "controlled_kernel_dominance": "controlled_kernel_pairwise_dominance.csv",
        "training_diagnostic_snapshot": training_snapshot_name,
    }
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial_operational_diagnostic",
        "contract_id": contract["contract_id"],
        "evaluation_code_sha256": evaluator_sha256,
        "builder_code_sha256": sha256_file(Path(__file__).resolve()),
        "target_feature_cache_sha256": target_cache_sha256,
        "training_diagnostic_summary_sha256": hashlib.sha256(
            training_summary_bytes
        ).hexdigest(),
        "eligible_compatible_models": len(eligible),
        "accepted_models": len(artifact_frame),
        "accepted_feature_rows": len(feature_frame),
        "excluded_artifacts": len(exclusion_frame),
        "complete_for_factorial_inference": len(artifact_frame) == 480,
        "complete_controlled_kernel_blocks": int(
            pareto_frame[["seed", "binary_prefix"]].drop_duplicates().shape[0]
        ) if not pareto_frame.empty else 0,
        "controlled_kernel_pareto_rows": len(pareto_frame),
        "pipeline_throughput": pipeline_throughput,
        "outputs": output_names,
        "output_sha256": {
            key: sha256_file(input_dir / filename)
            for key, filename in output_names.items()
        },
    }
    atomic_text(input_dir / "accepted_summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
