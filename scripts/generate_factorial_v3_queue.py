#!/usr/bin/env python3
"""Generate the dependency-gated factorial_v3_clinical queue manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/mithunmanivannan/.venv/bin/python"
ENV = f"PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. {PYTHON}"


def job(identifier: str, label: str, command: str, output: str | None = None):
    result = {"id": identifier, "label": label, "cmd": command}
    if output is not None:
        result["expected_output"] = output
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiment_queue/factorial_v3_clinical/manifest.json"),
    )
    parser.add_argument(
        "--registry",
        default="experiment_queue/factorial_v2/model_registry.json",
    )
    parser.add_argument(
        "--source-results",
        default="results/factorial_v2",
        help="Directory containing comprehensive_results.json.",
    )
    parser.add_argument(
        "--evaluation-results",
        default=None,
        help="Directory containing EchoNext/smartwatch JSON; defaults to result-dir.",
    )
    parser.add_argument(
        "--result-dir",
        default="results/factorial_v3_clinical",
    )
    parser.add_argument(
        "--project",
        default="ecg_factorial_v3_clinical",
    )
    parser.add_argument(
        "--reuse-external",
        action="store_true",
        help="Reuse completed EchoNext/smartwatch outputs instead of rerunning inference.",
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    result = args.result_dir
    registry = args.registry
    source_results = args.source_results
    evaluation_results = args.evaluation_results or result
    selection = f"{source_results}/selected_masks.json"
    database = f"{result}/classification_task_metrics.parquet"
    figure_dir = f"{result}/figures"
    phases = [
        {
            "name": "integrity_tests",
            "depends_on": [],
            "jobs": [
                job(
                    "v3_integrity_tests",
                    "Clinical metrics, EchoNext, smartwatch, factorial, and queue regressions",
                    (
                        f"{ENV} -m pytest -p no:cacheprovider -q "
                        "tests/test_smartwatch_protocol.py "
                        "tests/test_clinical_metrics.py "
                        "tests/test_echonext_classifier.py "
                        "tests/test_echonext_streaming.py "
                        "tests/test_factorial_losses.py "
                        "tests/test_factorial_analysis.py "
                        "tests/test_clinical_queue_generator.py "
                        "tests/test_clinical_figures.py "
                        "tests/test_queue_manager.py"
                    ),
                )
            ],
        },
        {
            "name": "preflight",
            "depends_on": ["integrity_tests"],
            "jobs": [
                job(
                    "v3_preflight",
                    "Frozen registry, official EchoNext, watch header, NSTDB, and disk gates",
                    f"{ENV} scripts/preflight_factorial_v3.py "
                    f"--registry {registry} "
                    f"--comprehensive {source_results}/comprehensive_results.json "
                    f"--output {result}/preflight.json",
                    f"{result}/preflight.json",
                )
            ],
        },
        {
            "name": "ptbxl_task_backfill",
            "depends_on": ["preflight"],
            "jobs": [
                job(
                    "v3_ptbxl_task_backfill",
                    "Backfill 150 ECGFounder and five superclass extensive task metrics",
                    f"{ENV} scripts/backfill_ptbxl_clinical_metrics.py "
                    f"--comprehensive {source_results}/comprehensive_results.json "
                    f"--registry {registry} "
                    f"--output {result}/ptbxl_clinical_metrics.json "
                    f"--database {result}/ptbxl_task_metrics.parquet",
                    f"{result}/ptbxl_clinical_metrics.json",
                )
            ],
        },
    ]
    database_dependency = "ptbxl_task_backfill"
    if not args.reuse_external:
        phases.extend([
            {
                "name": "echonext_shd",
                "depends_on": ["ptbxl_task_backfill"],
                "jobs": [
                    job(
                        "v3_echonext_shd",
                        "Official 12-task EchoNext SHD clean plus 17-condition stress evaluation",
                        f"{ENV} scripts/evaluate_echonext.py "
                        f"--registry {registry} --data_dir data/echonext "
                        f"--output {evaluation_results}/echonext_results.json "
                        "--batch_size 32 --morphology_samples 2198 "
                        "--robustness_samples 2198 --robustness_condition_batch 9 "
                        "--skip_ecgfounder --resume --device cuda",
                        f"{evaluation_results}/echonext_results.json",
                    )
                ],
            },
            {
                "name": "smartwatch",
                "depends_on": ["echonext_shd"],
                "jobs": [
                    job(
                        "v3_smartwatch",
                        "Lead-II, 500-Hz aligned simulator/Philips and 150-task fidelity evaluation",
                        f"{ENV} scripts/evaluate_smartwatch.py "
                        f"--registry {registry} "
                        "--data_dir data/physionet.org/files/ecg-capable-smartwatches/1.0.0 "
                        f"--output {evaluation_results}/smartwatch_results.json "
                        "--batch_size 16 --robustness_samples 0 --device cuda",
                        f"{evaluation_results}/smartwatch_results.json",
                    )
                ],
            },
        ])
        database_dependency = "smartwatch"
    phases.extend([
        {
            "name": "clinical_databases",
            "depends_on": [database_dependency],
            "jobs": [
                job(
                    "v3_clinical_databases",
                    "Normalize real-GT, stress, simulator, and proxy task databases",
                    f"{ENV} scripts/build_clinical_metrics_database.py "
                    f"--ptbxl {result}/ptbxl_task_metrics.parquet "
                    f"--echonext {evaluation_results}/echonext_results.json "
                    f"--smartwatch {evaluation_results}/smartwatch_results.json "
                    f"--registry {registry} --output_dir {result}",
                    f"{result}/clinical_database_manifest.json",
                )
            ],
        },
        {
            "name": "poster_figures",
            "depends_on": ["clinical_databases"],
            "jobs": [
                job(
                    "fig_ptbxl_task_rank",
                    "ECGFounder task-rank and prevalence retention",
                    f"{ENV} scripts/figures/figure_ptbxl_task_rank.py "
                    f"--database {database} --selection {selection} "
                    f"--output {figure_dir}/ptbxl_ecgfounder_task_rank",
                    f"{figure_dir}/ptbxl_ecgfounder_task_rank.provenance.json",
                ),
                job(
                    "fig_ptbxl_superclasses",
                    "Five-superclass heatmap",
                    f"{ENV} scripts/figures/figure_ptbxl_superclasses.py "
                    f"--database {database} --selection {selection} "
                    f"--output {figure_dir}/ptbxl_five_superclasses",
                    f"{figure_dir}/ptbxl_five_superclasses.provenance.json",
                ),
                job(
                    "fig_echonext_shd",
                    "EchoNext 12-task SHD comparison",
                    f"{ENV} scripts/figures/figure_echonext_shd.py "
                    f"--database {database} --selection {selection} "
                    f"--output {figure_dir}/echonext_shd_tasks",
                    f"{figure_dir}/echonext_shd_tasks.provenance.json",
                ),
                job(
                    "fig_echonext_stress",
                    "EchoNext SHD degradation curves",
                    f"{ENV} scripts/figures/figure_echonext_stress.py "
                    f"--database {database} --selection {selection} "
                    f"--output {figure_dir}/echonext_shd_stress",
                    f"{figure_dir}/echonext_shd_stress.provenance.json",
                ),
                job(
                    "fig_smartwatch_radar",
                    "Direction-corrected smartwatch profile radar",
                    f"{ENV} scripts/figures/figure_smartwatch_radar.py "
                    f"--results {evaluation_results}/smartwatch_results.json "
                    f"--selection {selection} "
                    f"--output {figure_dir}/smartwatch_radar",
                    f"{figure_dir}/smartwatch_radar.provenance.json",
                ),
                job(
                    "fig_smartwatch_protocol",
                    "Calibrated smartwatch simulator accuracy",
                    f"{ENV} scripts/figures/figure_smartwatch_protocol.py "
                    f"--database {result}/smartwatch_protocol_metrics.parquet "
                    f"--output {figure_dir}/smartwatch_protocol_accuracy",
                    f"{figure_dir}/smartwatch_protocol_accuracy.provenance.json",
                ),
                job(
                    "fig_smartwatch_task_rank",
                    "Smartwatch 150-task probability-fidelity rank",
                    f"{ENV} scripts/figures/figure_smartwatch_task_rank.py "
                    f"--database {result}/smartwatch_task_fidelity.parquet "
                    f"--selection {selection} "
                    f"--output {figure_dir}/smartwatch_task_rank",
                    f"{figure_dir}/smartwatch_task_rank.provenance.json",
                ),
                job(
                    "fig_echonext_calibration",
                    "EchoNext composite SHD reliability",
                    f"{ENV} scripts/figures/figure_echonext_calibration.py "
                    f"--results {evaluation_results}/echonext_results.json "
                    f"--selection {selection} "
                    f"--output {figure_dir}/echonext_shd_calibration",
                    f"{figure_dir}/echonext_shd_calibration.provenance.json",
                ),
            ],
        },
        {
            "name": "completeness",
            "depends_on": ["poster_figures"],
            "jobs": [
                job(
                    "v3_completeness",
                    "Strict per-task, per-record, taxonomy, and figure-hash gate",
                    f"{ENV} scripts/check_factorial_v3_completeness.py "
                    f"--result_dir {result} --evaluation_dir {evaluation_results}",
                    f"{result}/completeness.json",
                )
            ],
        },
    ])
    manifest = {
        "project": args.project,
        "cwd": str(PROJECT_ROOT),
        "ssh": "localhost",
        "conda": "none",
        "default_cmd": "echo no-default",
        "preconditions": [],
        "gpus": [0],
        "max_parallel": 1,
        "gpu_free_threshold_mib": 500,
        "oom_retry": {"delay": 120, "max_attempts": 3},
        "phases": phases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print(
        json.dumps(
            {
                "manifest": str(output),
                "phases": len(phases),
                "jobs": sum(len(phase["jobs"]) for phase in phases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
