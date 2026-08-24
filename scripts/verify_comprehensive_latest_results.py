#!/usr/bin/env python3
"""Fail-closed verification for the latest comprehensive 48-model package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    return json.loads(
        path.read_text(),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path}: non-strict JSON constant {token}")
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--package",
        type=Path,
        default=ROOT / "results/comprehensive_latest_48_models",
    )
    args = parser.parse_args()
    package = args.package.resolve()
    core = strict_json(package / "ptbxl/full_clean_clinical_robustness.json")
    echo = strict_json(package / "echonext/echonext_results.json")
    watch = strict_json(package / "smartwatch/four_device_results.json")
    sunny = strict_json(package / "sunnybrook/sunnybrook_results.json")
    quality = strict_json(
        package / "ptbxl_signal_quality/signal_quality_results.json"
    )
    statistics = strict_json(package / "raw/factorial_v4_2x4/statistics.json")
    master = pd.read_parquet(package / "all_48_models_master.parquet")
    ids = set(core["models"])
    expected_conditions = {
        "gaussian_24db",
        "gaussian_12db",
        "gaussian_6db",
        "gaussian_0db",
        "baseline_drift",
        "nstdb_bw_24db",
        "nstdb_bw_12db",
        "nstdb_bw_6db",
        "nstdb_bw_0db",
        "nstdb_em_24db",
        "nstdb_em_12db",
        "nstdb_em_6db",
        "nstdb_em_0db",
        "nstdb_ma_24db",
        "nstdb_ma_12db",
        "nstdb_ma_6db",
        "nstdb_ma_0db",
    }
    checks: dict[str, bool] = {
        "master_48_rows": len(master) == 48 and set(master["model_id"]) == ids,
        "factorial_3_by_16": (
            master.groupby("family").size().to_dict()
            == {"ecg_aim": 16, "multiscale_vae": 16, "unet": 16}
            and master["factorial_mask"].nunique() == 16
        ),
        "ptbxl_48_models": len(ids) == 48,
        "ptbxl_clean_2198_each": all(
            model["signal"]["n_samples"] == 2198 for model in core["models"].values()
        ),
        "ptbxl_ecgfounder_150_tasks_each": all(
            model["clinical"]["n_tasks"] == 150 for model in core["models"].values()
        ),
        "ptbxl_superclass_5_tasks_each": all(
            model["paper_parity_clinical"]["n_tasks"] == 5
            for model in core["models"].values()
        ),
        "ptbxl_17_noise_conditions_2198_each": all(
            set(model["robustness"]["conditions"]) == expected_conditions
            and model["robustness"]["n_samples"] == 2198
            for model in core["models"].values()
        ),
        "echonext_48_models": set(echo["models"]) == ids,
        "echonext_5442_clean_each": all(
            model["signal"]["n_samples"] == 5442 for model in echo["models"].values()
        ),
        "smartwatch_48_models_4_devices": (
            set(watch["models"]) == ids
            and all(len(devices) == 4 for devices in watch["models"].values())
        ),
        "sunnybrook_48_models": set(sunny["models"]) == ids,
        "sunnybrook_20_records_each": (
            sunny["completeness"]["status"] == "complete"
            and all(
                count == 20
                for count in sunny["completeness"]["completed_records_per_model"].values()
            )
        ),
        "sunnybrook_observed_leads_exact": all(
            model["observed_lead_max_abs_error"] == 0
            for model in sunny["models"].values()
        ),
        "sunnybrook_ground_truth_qualified": (
            "non-adjudicated proxy"
            in sunny["protocol"]["classification_ground_truth"]
            and sunny["protocol"]["ecgfounder_ground_truth"].startswith("unavailable")
        ),
        "signal_quality_48_models": set(quality["models"]) == ids,
        "signal_quality_2198_each": (
            quality["completeness"]["status"] == "complete"
            and all(model["n_records"] == 2198 for model in quality["models"].values())
        ),
        "signal_quality_test_isolation": (
            quality["protocol"]["test_used_for_training_or_selection"] is False
        ),
        "signal_quality_observed_leads_exact": all(
            model["observed_lead_max_abs_error"] == 0
            for model in quality["models"].values()
        ),
        "signal_quality_columns_in_master": all(
            column in master.columns
            for column in (
                "ptbxl_signal_quality_macro_auroc",
                "ptbxl_signal_quality_any_artifact_auroc",
                "ptbxl_signal_quality_probability_pearson",
            )
        ),
        "noninferiority_summary_matches_rows": (
            statistics["ecgfounder_noninferiority_summary"]["tested"]
            == len(statistics["ecgfounder_noninferiority"])
            and statistics["ecgfounder_noninferiority_summary"]["passed"]
            == sum(
                row["noninferior"]
                for row in statistics["ecgfounder_noninferiority"]
            )
            and "27/48"
            in statistics["ecgfounder_noninferiority_summary"]["interpretation"]
        ),
    }

    manifest = strict_json(package / "MANIFEST.json")
    manifest_checks = []
    for item in manifest["files"]:
        path = package / item["path"]
        manifest_checks.append(
            path.is_file()
            and path.stat().st_size == item["bytes"]
            and sha256(path) == item["sha256"]
        )
    checks["manifest_files_exist_and_hash"] = all(manifest_checks)
    checks["long_form_tables_present"] = all(
        (package / "tables" / f"{name}.{suffix}").is_file()
        for name in (
            "ptbxl_clean_per_lead",
            "ecgfounder_150_per_task",
            "ptbxl_five_superclass_per_class",
            "ptbxl_noise_stress_17_conditions",
            "smartwatch_four_device_summary",
            "sunnybrook_per_lead",
            "sunnybrook_proxy_superclass_per_class",
            "sunnybrook_ecgfounder_150_probability_fidelity",
            "echonext_shd_per_task",
            "ptbxl_signal_quality_per_class",
        )
        for suffix in ("csv", "parquet")
    )
    status = "complete" if all(checks.values()) else "incomplete"
    report = {
        "schema_version": 1,
        "status": status,
        "passed": sum(checks.values()),
        "total": len(checks),
        "checks": checks,
    }
    (package / "VERIFICATION.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    lines = [
        "# Completeness Report",
        "",
        f"**Status: {status.upper()} — {report['passed']}/{report['total']} checks passed**",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — {name}" for name, passed in checks.items()
    )
    (package / "COMPLETENESS_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
