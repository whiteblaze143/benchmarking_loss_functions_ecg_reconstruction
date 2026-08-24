#!/usr/bin/env python3
"""Export task-specific result files and a completeness-gated final report."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def json_safe(value: Any) -> Any:
    """Represent undefined metrics as JSON null rather than non-standard NaN."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_complete_results(results_path: Path, registry_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    results = json.loads(results_path.read_text())
    registry = json.loads(registry_path.read_text())
    expected = [entry["id"] for entry in registry["models"]]
    actual = list(results.get("models", {}))
    missing = [model_id for model_id in expected if model_id not in actual]
    unexpected = [model_id for model_id in actual if model_id not in expected]
    if missing or unexpected:
        raise RuntimeError(
            "Refusing to summarize an incomplete/mismatched registry evaluation: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for model_id in expected:
        model = results["models"][model_id]
        primary = model.get("signal", {}).get("missing_leads", {})
        for metric in ("mse", "rmse", "mae", "pearson", "r2"):
            if not finite(primary.get(metric)):
                raise RuntimeError(f"{model_id} has invalid primary metric {metric}: {primary.get(metric)}")
        if model.get("clinical", {}).get("status") == "oracle_skipped":
            raise RuntimeError(f"{model_id} is missing oracle-based clinical evaluation")
    return results, registry


def section_payload(section: str, results: dict[str, Any]) -> dict[str, Any]:
    models = results["models"]
    if section == "signal":
        body = {model_id: model["signal"] for model_id, model in models.items()}
    elif section == "clinical":
        body = {
            model_id: {
                "macro_auroc": model["clinical"]["macro_auroc"],
                "macro_f1": model["clinical"]["macro_f1"],
                "per_class": model["clinical"]["per_class"],
            }
            for model_id, model in models.items()
        }
    elif section == "fairness":
        body = {model_id: model["fairness"] for model_id, model in models.items()}
    elif section == "robustness":
        body = {model_id: model["robustness"] for model_id, model in models.items()}
    elif section == "morphology":
        body = {model_id: model["morphology"] for model_id, model in models.items()}
    elif section == "calibration":
        body = {
            model_id: model["clinical"]["calibration"] for model_id, model in models.items()
        }
    elif section == "sensitivity_specificity":
        body = {
            model_id: {
                "macro_sensitivity": model["clinical"]["macro_sensitivity"],
                "macro_specificity": model["clinical"]["macro_specificity"],
                "per_class": {
                    name: {
                        "sensitivity": values["sensitivity"],
                        "specificity": values["specificity"],
                    }
                    for name, values in model["clinical"]["per_class"].items()
                },
            }
            for model_id, model in models.items()
        }
    elif section == "plots":
        body = {model_id: model["example_plot"] for model_id, model in models.items()}
    else:
        raise ValueError(f"Unknown section: {section}")
    return {
        "schema_version": results.get("schema_version", 2),
        "section": section,
        "protocol": results["protocol"],
        "models": body,
    }


def fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if finite(value) else "N/A"


def delta(value: Any, baseline: Any) -> float:
    """Return a numeric delta only when both operands are available."""
    if not finite(value) or not finite(baseline):
        return float("nan")
    return float(value) - float(baseline)


def make_report(results: dict[str, Any], registry: dict[str, Any]) -> str:
    models = results["models"]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in registry["models"]:
        by_family[entry["family"]].append(models[entry["id"]])

    lines = [
        "# Comprehensive ECG Reconstruction Loss Benchmark",
        "",
        "## Evaluation protocol",
        "",
        f"- Dataset: PTB-XL test split ({results['protocol']['n_samples']} records).",
        "- Primary reconstruction metrics are computed only on leads not supplied to each model.",
        "- Diagnostic labels, age, and sex are joined from PTB-XL metadata using ECG IDs.",
        "- Clinical metrics use the same frozen 150-task ECGFounder classifier for every reconstruction.",
        "- Robustness covers synthetic Gaussian noise and drift plus NSTDB baseline-wander, electrode-motion, and muscle-artifact records at 24/12/6/0 dB SNR.",
        "- Results are single-seed unless stated otherwise; no inferential significance claim is made.",
        "",
        "Every model receives the same I–II–V2 observed-lead basis. Cross-family absolute",
        "comparisons remain descriptive because architectures and optimization differ; loss conclusions",
        "primarily use within-family comparisons against that family's baseline.",
        "",
        "## Primary missing-lead reconstruction results",
        "",
        "| Model | Family | Loss | Observed leads | MSE ↓ | RMSE ↓ | Pearson ↑ | R² ↑ |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for entry in registry["models"]:
        model = models[entry["id"]]
        primary = model["signal"]["missing_leads"]
        observed = ", ".join(model["signal"]["observed_leads"]["names"])
        lines.append(
            f"| {entry['id']} | {entry['family']} | {entry['loss']} | {observed} | "
            f"{fmt(primary['mse'], 6)} | {fmt(primary['rmse'], 6)} | "
            f"{fmt(primary['pearson'])} | {fmt(primary['r2'])} |"
        )

    lines.extend(
        [
            "",
            "## Diagnostic utility and calibration",
            "",
            "| Model | Macro AUROC ↑ | Macro AP ↑ | Macro F1 ↑ | Sensitivity ↑ | Specificity ↑ | ECE ↓ | Brier ↓ |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in registry["models"]:
        clinical = models[entry["id"]]["clinical"]
        calibration = clinical["calibration"]
        lines.append(
            f"| {entry['id']} | {fmt(clinical['macro_auroc'])} | "
            f"{fmt(clinical['macro_average_precision'])} | {fmt(clinical['macro_f1'])} | {fmt(clinical['macro_sensitivity'])} | "
            f"{fmt(clinical['macro_specificity'])} | {fmt(calibration['ece'])} | "
            f"{fmt(calibration['brier'])} |"
        )

    lines.extend(
        [
            "",
            "## Within-family loss deltas",
            "",
            "Deltas are relative to the MSE/ELBO baseline in the same family. Negative ΔMSE and",
            "positive ΔPearson/ΔAUROC indicate improvement.",
            "",
            "| Model | Baseline | Δ missing-lead MSE | Δ Pearson | Δ macro AUROC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for family, family_models in by_family.items():
        baseline = next(
            (
                model
                for model in family_models
                if model["loss"] in {"mse", "elbo", "gaussian_nll"}
            ),
            None,
        )
        if baseline is None:
            continue
        for model in family_models:
            if model["id"] == baseline["id"]:
                continue
            lines.append(
                f"| {model['id']} | {baseline['id']} | "
                f"{fmt(delta(model['signal']['missing_leads']['mse'], baseline['signal']['missing_leads']['mse']), 6)} | "
                f"{fmt(delta(model['signal']['missing_leads']['pearson'], baseline['signal']['missing_leads']['pearson']))} | "
                f"{fmt(delta(model['clinical']['macro_auroc'], baseline['clinical']['macro_auroc']))} |"
            )

    lines.extend(
        [
            "",
            "## Robustness, morphology, and fairness",
            "",
            "| Model | Gaussian 6 dB ΔMSE | NSTDB-BW 6 dB ΔMSE | NSTDB-EM 6 dB ΔMSE | NSTDB-MA 6 dB ΔMSE | R-peak timing MAE (ms) | Gender AUROC gap | Age AUROC gap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in registry["models"]:
        model = models[entry["id"]]
        robustness = model["robustness"]
        conditions = robustness["conditions"]
        gaps = model["fairness"]["gaps"]
        lines.append(
            f"| {entry['id']} | {fmt(conditions['gaussian_6db']['missing_lead_mse_delta'], 6)} | "
            f"{fmt(conditions['nstdb_bw_6db']['missing_lead_mse_delta'], 6)} | "
            f"{fmt(conditions['nstdb_em_6db']['missing_lead_mse_delta'], 6)} | "
            f"{fmt(conditions['nstdb_ma_6db']['missing_lead_mse_delta'], 6)} | "
            f"{fmt(model['morphology']['r_peak_timing_mae_ms'], 2)} | "
            f"{fmt(gaps['gender_auroc_gap'])} | {fmt(gaps['age_auroc_gap'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- A lower composite training objective is not itself evidence of better reconstruction; the tables above use held-out metrics.",
            "- Observed-lead copying can inflate all-lead scores, so missing-lead scores are primary.",
            "- Cross-family comparisons are descriptive because architectures and optimization differ.",
            "- Single-seed differences require multi-seed confirmation before strong claims.",
            "- Subgroup gaps are descriptive and should be accompanied by uncertainty intervals in publication claims.",
            "",
            "Machine-readable task exports and per-model plots are stored beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/comprehensive/comprehensive_results.json")
    parser.add_argument("--registry", default="experiment_queue/comprehensive_v1/model_registry.json")
    parser.add_argument(
        "--section",
        choices=[
            "signal",
            "clinical",
            "fairness",
            "robustness",
            "morphology",
            "calibration",
            "sensitivity_specificity",
            "plots",
            "report",
        ],
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = PROJECT_ROOT / args.input
    registry_path = PROJECT_ROOT / args.registry
    output_path = PROJECT_ROOT / args.output
    results, registry = load_complete_results(input_path, registry_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.section == "report":
        output_path.write_text(make_report(results, registry))
    else:
        output_path.write_text(
            json.dumps(
                json_safe(section_payload(args.section, results)),
                indent=2,
                allow_nan=False,
            )
        )
    print(f"Saved {args.section}: {output_path}")


if __name__ == "__main__":
    main()
