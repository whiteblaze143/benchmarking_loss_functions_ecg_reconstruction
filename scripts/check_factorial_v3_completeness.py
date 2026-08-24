#!/usr/bin/env python3
"""Strict, data-bound completion gate for factorial_v3_clinical."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    "ptbxl_ecgfounder_task_rank",
    "ptbxl_five_superclasses",
    "echonext_shd_tasks",
    "echonext_shd_stress",
    "smartwatch_radar",
    "smartwatch_protocol_accuracy",
    "smartwatch_task_rank",
    "echonext_shd_calibration",
]
EXPECTED_WATCH_COUNTS = {
    "applewatch_serie8": 180,
    "fitbitsense2": 180,
    "samsunggalaxy6": 179,
    "withingsscanwatch": 180,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation_dir",
        type=Path,
        default=None,
        help="Directory containing echonext_results.json and smartwatch_results.json.",
    )
    args = parser.parse_args()
    root = args.result_dir if args.result_dir.is_absolute() else PROJECT_ROOT / args.result_dir
    evaluation_root = (
        root
        if args.evaluation_dir is None
        else (
            args.evaluation_dir
            if args.evaluation_dir.is_absolute()
            else PROJECT_ROOT / args.evaluation_dir
        )
    )
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    preflight_path = root / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    add("preflight_pass", preflight.get("status") == "PASS", preflight.get("checks"))

    ptb_path = root / "ptbxl_task_metrics.parquet"
    ptb = pd.read_parquet(ptb_path)
    add("ptbxl_task_rows_14640", len(ptb) == 14640, len(ptb))
    add(
        "ptbxl_tasks_150_plus_5",
        (
            ptb.loc[ptb.classifier == "ECGFounder_150", "task_name"].nunique() == 150
            and ptb.loc[
                ptb.classifier == "PTBXL_five_superclass", "task_name"
            ].nunique()
            == 5
        ),
        ptb.groupby("classifier").task_name.nunique().to_dict(),
    )
    required_task_fields = {
        "support_positive", "support_negative", "threshold", "auroc",
        "average_precision", "f1", "precision", "recall", "specificity",
        "brier", "ece", "tp", "tn", "fp", "fn", "status",
    }
    add(
        "ptbxl_extensive_schema",
        required_task_fields.issubset(ptb.columns),
        sorted(required_task_fields - set(ptb.columns)),
    )

    echo_path = evaluation_root / "echonext_results.json"
    echo = json.loads(echo_path.read_text())
    echo_models = echo.get("models", {})
    add("echonext_models_48", len(echo_models) == 48, len(echo_models))
    add(
        "echonext_reference_12_tasks",
        echo.get("shd_reference", {}).get("clinical", {}).get("n_tasks") == 12,
        echo.get("shd_reference", {}).get("clinical", {}).get("n_tasks"),
    )
    echo_cells_ok = True
    echo_row_details: dict[str, int] = {}
    for model_id, model in echo_models.items():
        conditions = model.get("robustness", {}).get("conditions", {})
        echo_cells_ok &= model.get("signal", {}).get("n_samples") == 5442
        echo_cells_ok &= model.get("shd_clinical", {}).get("n_tasks") == 12
        echo_cells_ok &= len(conditions) == 17
        echo_cells_ok &= all(
            condition.get("n_samples") == 2198
            and condition.get("shd_clinical", {}).get("n_tasks") == 12
            for condition in conditions.values()
        )
        parquet_path = PROJECT_ROOT / model.get("shd_per_record_parquet", "__missing__")
        if parquet_path.is_file():
            echo_row_details[model_id] = parquet_rows(parquet_path)
            echo_cells_ok &= echo_row_details[model_id] == 5442 + 17 * 2198
        else:
            echo_cells_ok = False
    add(
        "echonext_complete_clean_stress_per_task",
        echo_cells_ok and len(echo_row_details) == 48,
        {
            "models_with_parquet": len(echo_row_details),
            "unique_row_counts": sorted(set(echo_row_details.values())),
        },
    )

    watch_path = evaluation_root / "smartwatch_results.json"
    watch = json.loads(watch_path.read_text())
    watch_models = watch.get("models", {})
    add("smartwatch_models_48", len(watch_models) == 48, len(watch_models))
    watch_ok = True
    for devices in watch_models.values():
        watch_ok &= set(devices) == set(EXPECTED_WATCH_COUNTS)
        for device, expected in EXPECTED_WATCH_COUNTS.items():
            result = devices.get(device, {})
            watch_ok &= result.get("n_paired_records") == expected
            watch_ok &= (
                result.get("signal_trained_target_nine", {})
                .get("missing_leads", {})
                .get("names", [])
                == ["III", "aVR", "aVL", "aVF", "V1", "V3", "V4", "V5", "V6"]
            )
            watch_ok &= (
                result.get("signal_wearable_missing_eleven", {})
                .get("missing_leads", {})
                .get("names", [])
                == [
                    "I", "III", "aVR", "aVL", "aVF", "V1",
                    "V2", "V3", "V4", "V5", "V6",
                ]
            )
            watch_ok &= result.get("reconstruction_passthrough_leads") == ["II"]
            watch_ok &= (
                result.get("ecgfounder_probability_fidelity", {}).get("n_tasks")
                == 150
            )
            parquet_path = PROJECT_ROOT / result.get("per_record_parquet", "__missing__")
            watch_ok &= parquet_path.is_file() and parquet_rows(parquet_path) == expected
    add(
        "smartwatch_exact_pairs_and_150_tasks",
        watch_ok,
        EXPECTED_WATCH_COUNTS,
    )
    add(
        "smartwatch_taxonomy_explicit",
        (
            watch.get("protocol", {}).get("evaluation_taxonomy", {}).get(
                "human_diagnostic_ground_truth"
            )
            == "unavailable"
            and set(watch.get("protocol", {}).get("pairing_audit", {}))
            == set(EXPECTED_WATCH_COUNTS)
            and all(
                audit.get("pairing_key") == "casefolded_relative_posix_path"
                and audit.get("paired_records") == EXPECTED_WATCH_COUNTS[device]
                for device, audit in watch.get("protocol", {})
                .get("pairing_audit", {})
                .items()
            )
        ),
        {
            "taxonomy": watch.get("protocol", {}).get("evaluation_taxonomy"),
            "pairing_audit": watch.get("protocol", {}).get("pairing_audit"),
        },
    )

    classification = pd.read_parquet(root / "classification_task_metrics.parquet")
    fidelity = pd.read_parquet(root / "smartwatch_task_fidelity.parquet")
    protocol = pd.read_parquet(root / "smartwatch_protocol_metrics.parquet")
    add(
        "classification_database_rows_25020",
        len(classification) == 25020,
        len(classification),
    )
    add(
        "smartwatch_fidelity_rows_28800",
        len(fidelity) == 28800,
        len(fidelity),
    )
    add(
        "smartwatch_protocol_rows_16",
        len(protocol) == 16,
        len(protocol),
    )

    figure_dir = root / "figures"
    figure_ok = True
    figure_detail: dict[str, Any] = {}
    for name in FIGURES:
        provenance_path = figure_dir / f"{name}.provenance.json"
        outputs_ok = provenance_path.is_file()
        detail: dict[str, Any] = {}
        if outputs_ok:
            provenance = json.loads(provenance_path.read_text())
            script_path = Path(provenance.get("script", "__missing__"))
            helper_path = PROJECT_ROOT / "scripts/figures/clinical_common.py"
            script_matches = (
                script_path.is_file()
                and provenance.get("script_sha256") == sha256(script_path)
            )
            helper_matches = (
                helper_path.is_file()
                and provenance.get("common_helper_sha256") == sha256(helper_path)
            )
            detail["generator_script_hash"] = script_matches
            detail["common_helper_hash"] = helper_matches
            outputs_ok &= script_matches and helper_matches
            for suffix in ("pdf", "svg", "png"):
                output_path = Path(provenance["outputs"][suffix]["path"])
                matches = (
                    output_path.is_file()
                    and sha256(output_path)
                    == provenance["outputs"][suffix]["sha256"]
                )
                detail[suffix] = matches
                outputs_ok &= matches
            for path, expected_hash in provenance["inputs"].items():
                source = Path(path)
                matches = source.is_file() and sha256(source) == expected_hash
                detail[f"input:{path}"] = matches
                outputs_ok &= matches
        figure_detail[name] = detail
        figure_ok &= outputs_ok
    add("eight_hashed_figure_triplets", figure_ok, figure_detail)
    review_path = root / "FIGURE_REVIEW.json"
    review = json.loads(review_path.read_text())
    review_figures = review.get("figures", {})
    review_ok = (
        set(review_figures) == set(FIGURES)
        and all(
            entry.get("data_hash_gate") == "PASS"
            and entry.get("visual_review") == "PASS"
            and (figure_dir / f"{name}.png").is_file()
            and entry.get("png_sha256")
            == sha256(figure_dir / f"{name}.png")
            for name, entry in review_figures.items()
        )
    )
    add(
        "eight_figures_visually_reviewed",
        review_ok,
        {
            name: {
                "data_hash_gate": review_figures.get(name, {}).get(
                    "data_hash_gate"
                ),
                "visual_review": review_figures.get(name, {}).get(
                    "visual_review"
                ),
                "png_sha256": review_figures.get(name, {}).get("png_sha256"),
            }
            for name in FIGURES
        },
    )

    withdrawn = [
        PROJECT_ROOT / "results/factorial_v2/all_48_models_3datasets_benchmark.csv",
        PROJECT_ROOT / "paper_figures/figure_6_per_lead.png",
        PROJECT_ROOT / "poster_figures/figure_6_per_lead.png",
        PROJECT_ROOT / "scripts/generate_paper_figures.py",
        PROJECT_ROOT / "scripts/evaluate_clinical_f1.py",
        PROJECT_ROOT / "scripts/evaluate_ece.py",
    ]
    add(
        "withdrawn_invalid_artifacts_absent",
        not any(path.exists() for path in withdrawn),
        [str(path) for path in withdrawn if path.exists()],
    )

    all_passed = all(check["passed"] for check in checks.values())
    report = {
        "schema_version": 1,
        "status": "PASS" if all_passed else "FAIL",
        "checks_passed": sum(check["passed"] for check in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
    }
    json_path = root / "completeness.json"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    lines = [
        "# Factorial v3 Clinical Completeness",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Checks: {report['checks_passed']}/{report['checks_total']}",
        "",
    ]
    for name, check in checks.items():
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} — `{name}`")
    (root / "COMPLETENESS_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
