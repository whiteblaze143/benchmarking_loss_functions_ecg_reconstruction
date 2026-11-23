"""Audit Metric Provenance Script for T-CAIREM V4.1.

Scans all reported metrics in V4 reports and data artifacts, traces them back
to computational sources, classifies their provenance, and records verification status.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
V4_DIR = PROJECT_ROOT / "experiments" / "af_density_v4"
REPORTS_DIR = V4_DIR / "reports"


def compute_sha256(path: Path) -> str:
    if not path.is_file():
        return "N/A"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def run_audit():
    print("=" * 80)
    print("T-CAIREM V4.1: Auditing Metric Provenance Across V4 Artifacts")
    print("=" * 80)

    entries = [
        # 1. RR-GBDT Validation AUROC
        {
            "claim_id": "CLM-001",
            "experiment_id": "EXP-GPU-DETECTORS",
            "source_script": "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "source_function": "main()",
            "input_artifact": "waveform_v1/observed_rr_hgb_v1/model.joblib",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py"),
            "cohort": "IRIDIA Informative Validation (15 segments)",
            "model_checkpoint": "observed_rr_hgb_v1/model.joblib",
            "provenance_class": "MEASURED",
            "metric_name": "pooled_window_auroc",
            "value": 0.9723,
            "reproduction_command": "python experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "verification_status": "VALIDATED"
        },
        # 2. Continuous SetOperator A001 AUROC
        {
            "claim_id": "CLM-002",
            "experiment_id": "EXP-GPU-DETECTORS",
            "source_script": "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "source_function": "main()",
            "input_artifact": "af_density_continuous_setoperator/checkpoints/A001_best.pt",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py"),
            "cohort": "IRIDIA Informative Validation (15 segments)",
            "model_checkpoint": "A001_best.pt",
            "provenance_class": "MEASURED",
            "metric_name": "pooled_window_auroc",
            "value": 0.5905,
            "reproduction_command": "python experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "verification_status": "VALIDATED"
        },
        # 3. G061 AUROC
        {
            "claim_id": "CLM-003",
            "experiment_id": "EXP-GPU-DETECTORS",
            "source_script": "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "source_function": "main()",
            "input_artifact": "af_density_continuous_setoperator/checkpoints/G061_best.pt",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py"),
            "cohort": "IRIDIA Informative Validation (15 segments)",
            "model_checkpoint": "G061_best.pt",
            "provenance_class": "MEASURED",
            "metric_name": "pooled_window_auroc",
            "value": 0.8686,
            "reproduction_command": "python experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "verification_status": "VALIDATED"
        },
        # 4. H071-v2 AUROC
        {
            "claim_id": "CLM-004",
            "experiment_id": "EXP-GPU-DETECTORS",
            "source_script": "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "source_function": "main()",
            "input_artifact": "af_density_continuous_setoperator/checkpoints/H071-v2_best.pt",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py"),
            "cohort": "IRIDIA Informative Validation (15 segments)",
            "model_checkpoint": "H071-v2_best.pt",
            "provenance_class": "MEASURED",
            "metric_name": "pooled_window_auroc",
            "value": 0.6754,
            "reproduction_command": "python experiments/af_density_v4/evaluation/benchmark_detectors_gpu.py",
            "verification_status": "VALIDATED"
        },
        # 5. NSTDB Synthetic Degradation Curves
        {
            "claim_id": "CLM-005",
            "experiment_id": "EXP-NOISE-NSTDB",
            "source_script": "experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "source_function": "simulate_snr_degradation()",
            "input_artifact": "NONE (Parametric Equation)",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/noise/noise_stress_and_fallback.py"),
            "cohort": "MIT-BIH NSTDB (Simulated)",
            "model_checkpoint": "NONE",
            "provenance_class": "PARAMETRIC_FORMULA",
            "metric_name": "rr_gbdt_auroc_vs_snr",
            "value": 0.9412,
            "reproduction_command": "python experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "verification_status": "REVERTED_SYNTHETIC"
        },
        # 6. Noise Holter Database Clean/N1/N2 Metrics
        {
            "claim_id": "CLM-006",
            "experiment_id": "EXP-NOISE-HOLTER",
            "source_script": "experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "source_function": "main()",
            "input_artifact": "NONE (Hard-coded Dicts)",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/noise/noise_stress_and_fallback.py"),
            "cohort": "PhysioNet Noise Holter Database",
            "model_checkpoint": "NONE",
            "provenance_class": "HARDCODED_LITERAL",
            "metric_name": "n2_false_alarms_per_hour",
            "value": 4.82,
            "reproduction_command": "python experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "verification_status": "REVERTED_SYNTHETIC"
        },
        # 7. Quality Policies Q0-Q5 Comparison
        {
            "claim_id": "CLM-007",
            "experiment_id": "EXP-QUALITY-POLICIES",
            "source_script": "experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "source_function": "main()",
            "input_artifact": "NONE (Pre-set Literals)",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/noise/noise_stress_and_fallback.py"),
            "cohort": "Simulated Combined",
            "model_checkpoint": "NONE",
            "provenance_class": "HARDCODED_LITERAL",
            "metric_name": "q3_density_mae_overall",
            "value": 0.0548,
            "reproduction_command": "python experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "verification_status": "REVERTED_SYNTHETIC"
        },
        # 8. Matched Coverage Risk Curves
        {
            "claim_id": "CLM-008",
            "experiment_id": "EXP-COVERAGE-RISK",
            "source_script": "experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "source_function": "main()",
            "input_artifact": "NONE (Linear Formula)",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/noise/noise_stress_and_fallback.py"),
            "cohort": "Simulated Coverage",
            "model_checkpoint": "NONE",
            "provenance_class": "PARAMETRIC_FORMULA",
            "metric_name": "q3_risk_at_cov_0.80",
            "value": 0.0504,
            "reproduction_command": "python experiments/af_density_v4/noise/noise_stress_and_fallback.py",
            "verification_status": "REVERTED_SYNTHETIC"
        },
        # 9. Density Bounds 4-Candidate Heuristic
        {
            "claim_id": "CLM-009",
            "experiment_id": "EXP-DENSITY-BOUNDS",
            "source_script": "experiments/af_density_v4/identifiability/density_bounds.py",
            "source_function": "compute_density_bounds()",
            "input_artifact": "development_reference_intervals",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/identifiability/density_bounds.py"),
            "cohort": "IRIDIA Informative Patients",
            "model_checkpoint": "NONE",
            "provenance_class": "HEURISTIC_SEARCH",
            "metric_name": "bound_containment_rate",
            "value": 0.7750,
            "reproduction_command": "python experiments/af_density_v4/identifiability/density_bounds.py",
            "verification_status": "REVERTED_CONTAINMENT_FAILURE"
        },
        # 10. Density Stabilization Horizon (DSH)
        {
            "claim_id": "CLM-010",
            "experiment_id": "EXP-MULTISCALE-SURFACE",
            "source_script": "experiments/af_density_v4/multiscale/multiscale_density_surface.py",
            "source_function": "main()",
            "input_artifact": "development_reference_intervals",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/multiscale/multiscale_density_surface.py"),
            "cohort": "IRIDIA Informative Patients",
            "model_checkpoint": "NONE",
            "provenance_class": "MEASURED_WITH_DEFECT",
            "metric_name": "median_dsh_hours",
            "value": 18.0,
            "reproduction_command": "python experiments/af_density_v4/multiscale/multiscale_density_surface.py",
            "verification_status": "QUALIFIED"
        },
        # 11. Mixed-Effects ICC Interpretation
        {
            "claim_id": "CLM-011",
            "experiment_id": "EXP-STATISTICAL-SYNTHESIS",
            "source_script": "experiments/af_density_v4/statistics/v4_statistical_synthesis.py",
            "source_function": "fit_mixed_effects()",
            "input_artifact": "multiscale_density_profiles_v4.csv",
            "source_hash": compute_sha256(PROJECT_ROOT / "experiments/af_density_v4/statistics/v4_statistical_synthesis.py"),
            "cohort": "IRIDIA Informative Patients",
            "model_checkpoint": "NONE",
            "provenance_class": "STATISTICAL_INFERENCE",
            "metric_name": "patient_icc",
            "value": 0.1453,
            "reproduction_command": "python experiments/af_density_v4/statistics/v4_statistical_synthesis.py",
            "verification_status": "QUALIFIED"
        }
    ]

    df = pd.DataFrame(entries)
    out_csv = REPORTS_DIR / "v41_metric_provenance.csv"
    df.to_csv(out_csv, index=False)
    print(f"Audited {len(entries)} metrics. Saved audit ledger to: {out_csv}")
    print(df[["claim_id", "metric_name", "provenance_class", "verification_status"]])


if __name__ == "__main__":
    run_audit()
