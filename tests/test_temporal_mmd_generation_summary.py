from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_temporal_mmd_generation_summary import (
    atomic_bytes,
    REQUIRED_COLUMNS,
    build_controlled_kernel_pareto,
    compute_pipeline_throughput,
    validate_artifact,
)


def test_atomic_bytes_publishes_exact_snapshot(tmp_path):
    destination = tmp_path / "training-summary.snapshot.json"
    payload = b'{"compatible_models":20}\n'
    atomic_bytes(destination, payload)
    assert destination.read_bytes() == payload
    assert not list(tmp_path.glob(".*.tmp"))


def fixture_artifact(tmp_path):
    model_id = "f_1000000_s42"
    checkpoint_sha = "a" * 64
    source_sha = "b" * 64
    evaluator_sha = "c" * 64
    cache_sha = "d" * 64
    test_root = "e" * 64
    contract = {
        "contract_id": "contract-v1",
        "split_content_roots": {
            "test": {"content_root_sha256": test_root},
        },
    }
    eligible = {
        model_id: {
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha,
            "source_bundle_sha256": source_sha,
        }
    }
    row = {column: 0 for column in REQUIRED_COLUMNS}
    row.update({
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha,
        "contract_id": contract["contract_id"],
        "source_bundle_sha256": source_sha,
        "test_content_root_sha256": test_root,
        "evaluation_code_sha256": evaluator_sha,
        "target_feature_cache_sha256": cache_sha,
        "dataset": "ptb_xl",
        "sample_rate_hz": 500,
        "detector": "detector",
        "feature_pairing_rule": "within_record_monotonic_max_cardinality_min_abs_time",
        "pairing_tolerance_ms": 100.0,
        "observed_lead_indices": "0,1,7",
        "lead": "V3",
        "evaluated_lead_index": 8,
        "clinical_feature": "R_Amp",
        "n_beats": 10,
        "n_records_total": 2,
        "n_records_real_detected": 2,
        "n_records_recon_detected": 2,
        "n_records_paired": 2,
        "record_pair_coverage": 1.0,
        "n_real_beats_detected": 10,
        "n_recon_beats_detected": 10,
        "n_paired_beats_before_finite_filter": 10,
        "n_unmatched_real_events": 0,
        "n_unmatched_recon_events": 0,
        "median_abs_pairing_error_ms": 8.0,
        "p95_abs_pairing_error_ms": 20.0,
        "mean_real": 4.5,
        "mean_recon": 9.0,
        "variance_ratio": 4.0,
        "ba_robust_slope": -0.1,
        "distribution_rbf_mmd2": 0.03,
        "distribution_rbf_bandwidth": 0.2,
        "distribution_mmd_real_samples": 10,
        "distribution_mmd_recon_samples": 10,
        "distribution_mmd_estimator": "biased_squared_rbf_mmd",
        "distribution_mmd_bandwidth_rule": "median_nonzero_pairwise_absolute_distance",
        "distribution_mmd_sketch_rule": "sorted_evenly_spaced_quantile_indices",
        "distribution_mmd_sample_cap": 512,
    })
    csv_path = tmp_path / f"{model_id}.csv"
    pd.DataFrame([row]).to_csv(csv_path, index=False)
    per_record_rows = []
    features = ["P_Amp", "Q_Amp", "R_Amp", "S_Amp", "T_Amp", "QT_Interval_ms"]
    for record_id in ("1", "2"):
        for lead in ("V3", "V6"):
            for feature in features:
                active = lead == "V3" and feature == "R_Amp"
                real_values = (
                    [0, 1, 2, 3, 4] if record_id == "1"
                    else [5, 6, 7, 8, 9]
                ) if active else []
                recon_values = [2 * value for value in real_values]
                per_record_rows.append({
                    "model_id": model_id,
                    "checkpoint_sha256": checkpoint_sha,
                    "evaluation_code_sha256": evaluator_sha,
                    "target_feature_cache_sha256": cache_sha,
                    "dataset": "ptb_xl",
                    "record_id": record_id,
                    "lead": lead,
                    "clinical_feature": feature,
                    "detector_state": "both_detected" if active else "neither_detected",
                    "target_detected": active,
                    "reconstruction_detected": active,
                    "n_real_events": 5 if active else 0,
                    "n_recon_events": 5 if active else 0,
                    "n_paired_events": 5 if active else 0,
                    "n_finite_paired_events": 5 if active else 0,
                    "n_unmatched_real_events": 0,
                    "n_unmatched_recon_events": 0,
                    "paired_sum_real": sum(real_values),
                    "paired_sum_recon": sum(recon_values),
                    "paired_sum_sq_real": sum(value ** 2 for value in real_values),
                    "paired_sum_sq_recon": sum(value ** 2 for value in recon_values),
                    "paired_mean_real": sum(real_values) / 5 if active else float("nan"),
                    "paired_mean_recon": sum(recon_values) / 5 if active else float("nan"),
                    "paired_mean_difference": sum(real_values) / 5 if active else float("nan"),
                    "paired_mae": sum(real_values) / 5 if active else float("nan"),
                    "median_abs_pairing_error_ms": 8.0 if active else float("nan"),
                    "p95_abs_pairing_error_ms": 8.0 if active else float("nan"),
                })
    per_record_path = tmp_path / f"{model_id}_per_record.parquet"
    pd.DataFrame(per_record_rows).to_parquet(per_record_path, index=False)
    sensitivity_rows = []
    for tolerance_ms in (25.0, 50.0, 75.0, 100.0, 150.0):
        for lead in ("V3", "V6"):
            for feature in features:
                sensitivity_rows.append({
                    "model_id": model_id,
                    "checkpoint_sha256": checkpoint_sha,
                    "evaluation_code_sha256": evaluator_sha,
                    "target_feature_cache_sha256": cache_sha,
                    "dataset": "ptb_xl",
                    "lead": lead,
                    "clinical_feature": feature,
                    "pairing_tolerance_ms": tolerance_ms,
                    "n_records_total": 2,
                    "n_records_paired": 2,
                    "record_pair_coverage": 1.0,
                    "n_real_events": 10,
                    "n_recon_events": 10,
                    "n_paired_events": 10,
                    "n_finite_paired_events": 10,
                    "n_unmatched_real_events": 0,
                    "n_unmatched_recon_events": 0,
                    "mean_real": 4.5,
                    "mean_recon": 9.0,
                    "variance_ratio": 4.0,
                })
    sensitivity_path = tmp_path / f"{model_id}_tolerance_sensitivity.csv"
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    metadata = {
        "model_id": model_id,
        "rows": 1,
        "checkpoint_sha256": checkpoint_sha,
        "contract_id": contract["contract_id"],
        "source_bundle_sha256": source_sha,
        "test_content_root_sha256": test_root,
        "evaluation_code_sha256": evaluator_sha,
        "target_feature_cache_sha256": [cache_sha],
        "reconstruction_batch_failures": 0,
        "feature_pairing_rule": "within_record_monotonic_max_cardinality_min_abs_time",
        "pairing_tolerance_ms": 100.0,
        "pairing_sensitivity_ms": [25.0, 50.0, 75.0, 100.0, 150.0],
        "distribution_mmd_estimator": "biased_squared_rbf_mmd",
        "distribution_mmd_bandwidth_rule": "median_nonzero_pairwise_absolute_distance",
        "distribution_mmd_sketch_rule": "sorted_evenly_spaced_quantile_indices",
        "distribution_mmd_sample_cap": 512,
        "datasets": {"ptb_xl": {"records_expected": 2}},
        "per_record_rows": len(per_record_rows),
        "per_record_parquet": per_record_path.name,
        "per_record_parquet_sha256": hashlib.sha256(per_record_path.read_bytes()).hexdigest(),
        "tolerance_sensitivity_rows": len(sensitivity_rows),
        "tolerance_sensitivity_csv": sensitivity_path.name,
        "tolerance_sensitivity_csv_sha256": hashlib.sha256(sensitivity_path.read_bytes()).hexdigest(),
        "csv": csv_path.name,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    }
    metadata_path = tmp_path / f"{model_id}.json"
    metadata_path.write_text(json.dumps(metadata))
    return metadata_path, eligible, contract, evaluator_sha, cache_sha, csv_path


def test_generation_summary_accepts_fully_bound_artifact(tmp_path):
    args = fixture_artifact(tmp_path)
    metadata, frame = validate_artifact(*args[:-1])
    assert metadata["model_id"] == "f_1000000_s42"
    assert len(frame) == 1


def test_generation_summary_rejects_csv_mutation(tmp_path):
    args = fixture_artifact(tmp_path)
    args[-1].write_text(args[-1].read_text() + "\n")
    with pytest.raises(ValueError, match="CSV digest mismatch"):
        validate_artifact(*args[:-1])


def test_generation_summary_rejects_event_accounting_mismatch(tmp_path):
    args = fixture_artifact(tmp_path)
    frame = pd.read_csv(args[-1])
    frame.loc[0, "n_unmatched_real_events"] = 1
    frame.to_csv(args[-1], index=False)
    metadata = json.loads(args[0].read_text())
    metadata["csv_sha256"] = hashlib.sha256(args[-1].read_bytes()).hexdigest()
    args[0].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="real event accounting"):
        validate_artifact(*args[:-1])


def test_generation_summary_rejects_per_record_accounting_mismatch(tmp_path):
    args = fixture_artifact(tmp_path)
    metadata = json.loads(args[0].read_text())
    per_record_path = tmp_path / metadata["per_record_parquet"]
    per_record = pd.read_parquet(per_record_path)
    per_record.loc[0, "n_unmatched_real_events"] = 1
    per_record.to_parquet(per_record_path, index=False)
    metadata["per_record_parquet_sha256"] = hashlib.sha256(
        per_record_path.read_bytes()
    ).hexdigest()
    args[0].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="per-record real event accounting"):
        validate_artifact(*args[:-1])


def test_generation_summary_rejects_nonmonotonic_tolerance_matches(tmp_path):
    args = fixture_artifact(tmp_path)
    metadata = json.loads(args[0].read_text())
    sensitivity_path = tmp_path / metadata["tolerance_sensitivity_csv"]
    sensitivity = pd.read_csv(sensitivity_path)
    row = (
        sensitivity.lead.eq("V3")
        & sensitivity.clinical_feature.eq("R_Amp")
        & sensitivity.pairing_tolerance_ms.eq(50.0)
    )
    sensitivity.loc[row, "n_paired_events"] = 9
    sensitivity.loc[row, "n_finite_paired_events"] = 9
    sensitivity.loc[row, "n_unmatched_real_events"] = 1
    sensitivity.loc[row, "n_unmatched_recon_events"] = 1
    sensitivity.to_csv(sensitivity_path, index=False)
    metadata["tolerance_sensitivity_csv_sha256"] = hashlib.sha256(
        sensitivity_path.read_bytes()
    ).hexdigest()
    args[0].write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="paired events decrease"):
        validate_artifact(*args[:-1])


def test_controlled_kernel_pareto_is_within_feature_and_requires_complete_block():
    rows = []
    for feature in ("R_Amp", "QT_Interval_ms"):
        for kernel in range(5):
            # Kernel 1 dominates kernel 0 for R amplitude, while kernel 0
            # dominates kernel 1 for QT. This must not be pooled away.
            if feature == "R_Amp":
                score = {0: 1.0, 1: 0.5}.get(kernel, 1.2 + kernel / 10)
            else:
                score = {0: 0.4, 1: 0.8}.get(kernel, 1.2 + kernel / 10)
            rows.append({
                "model_id": f"f_100000{kernel}_s42",
                "model_mask": f"100000{kernel}",
                "seed": 42,
                "mmd_kernel": kernel,
                "lead": "V3",
                "clinical_feature": feature,
                "mean_real": 0.0,
                "mean_recon": score,
                "variance_ratio": float(np.exp(score)),
                "distribution_rbf_mmd2": score,
                "record_pair_coverage": 1.0 - score / 10,
            })
    frame = pd.DataFrame(rows)
    pareto, dominance = build_controlled_kernel_pareto(frame)
    assert len(pareto) == 10
    r0 = pareto[
        pareto.clinical_feature.eq("R_Amp") & pareto.mmd_kernel.eq(0)
    ].iloc[0]
    q1 = pareto[
        pareto.clinical_feature.eq("QT_Interval_ms") & pareto.mmd_kernel.eq(1)
    ].iloc[0]
    assert r0.dominated_by_model_ids == "f_1000001_s42"
    assert q1.dominated_by_model_ids == "f_1000000_s42"
    assert (
        dominance.dominator_model_id.eq("f_1000001_s42")
        & dominance.dominated_model_id.eq("f_1000000_s42")
    ).any()


def test_controlled_kernel_pareto_skips_incomplete_kernel_block():
    frame = pd.DataFrame([
        {
            "model_id": f"f_100000{kernel}_s42", "model_mask": f"100000{kernel}",
            "seed": 42, "mmd_kernel": kernel, "lead": "V3",
            "clinical_feature": "R_Amp", "mean_real": 0.0,
            "mean_recon": 0.1, "variance_ratio": 1.0,
            "distribution_rbf_mmd2": 0.1, "record_pair_coverage": 0.9,
        }
        for kernel in range(4)
    ])
    pareto, dominance = build_controlled_kernel_pareto(frame)
    assert pareto.empty
    assert dominance.empty


def test_pipeline_throughput_reports_positive_backlog_drain():
    result = compute_pipeline_throughput(
        training_median_minutes=20.0,
        evaluation_durations_seconds=pd.Series([900.0, 960.0, 1020.0]),
        eligible_models=20,
        accepted_models=5,
    )
    assert result["evaluation_median_minutes_per_model"] == 16.0
    assert result["evaluator_keeps_pace_at_observed_medians"] is True
    assert result["backlog_models"] == 15
    assert result["estimated_backlog_catchup_hours"] == pytest.approx(20.0)


def test_pipeline_throughput_reports_no_catchup_when_evaluator_is_slower():
    result = compute_pipeline_throughput(
        training_median_minutes=15.0,
        evaluation_durations_seconds=pd.Series([1200.0]),
        eligible_models=10,
        accepted_models=4,
    )
    assert result["evaluator_keeps_pace_at_observed_medians"] is False
    assert result["estimated_backlog_catchup_hours"] is None
