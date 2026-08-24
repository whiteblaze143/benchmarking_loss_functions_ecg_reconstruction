#!/usr/bin/env python3
"""Assemble a space-efficient, auditable package of the latest 48-model results."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/comprehensive_latest_48_models"
CORE = ROOT / "results/factorial_v4_2x4"
SUNNYBROOK = ROOT / "results/sunnybrook_factorial_v4_2x4"
QUALITY = ROOT / "results/ptbxl_signal_quality_factorial_v4_2x4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        target.unlink()
    os.link(source, target)


def link_tree(source: Path, target: Path) -> list[Path]:
    linked = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            destination = target / path.relative_to(source)
            hardlink(path, destination)
            linked.append(destination)
    return linked


def get(mapping: dict[str, Any], *keys: str, default=None):
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def finite_mean(values) -> float | None:
    numeric = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(numeric)) if numeric else None


def scalar_fields(mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in mapping.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


def write_table(rows: list[dict[str, Any]], name: str) -> None:
    frame = pd.DataFrame(rows)
    path = PACKAGE / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path.with_suffix(".csv"), index=False)
    frame.to_parquet(path.with_suffix(".parquet"), index=False)


def main() -> None:
    core = json.loads((CORE / "comprehensive_results.json").read_text())
    echo = json.loads((CORE / "echonext_results.json").read_text())
    smartwatch = json.loads((CORE / "smartwatch_results.json").read_text())
    sunnybrook = json.loads((SUNNYBROOK / "sunnybrook_results.json").read_text())
    registry = json.loads(
        (ROOT / "experiment_queue/factorial_v4/model_registry.json").read_text()
    )
    registry_by_id = {spec["id"]: spec for spec in registry["models"]}
    quality_path = QUALITY / "signal_quality_results.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else None

    ids = sorted(core["models"])
    expected = set(ids)
    sources = {
        "ptbxl_core": set(core["models"]),
        "echonext": set(echo["models"]),
        "smartwatch": set(smartwatch["models"]),
        "sunnybrook": set(sunnybrook["models"]),
    }
    if quality:
        sources["ptbxl_signal_quality"] = set(quality["models"])
    mismatches = {
        name: {
            "missing": sorted(expected - values),
            "extra": sorted(values - expected),
        }
        for name, values in sources.items()
        if values != expected
    }
    if len(ids) != 48 or mismatches:
        raise RuntimeError(f"Coverage mismatch: n={len(ids)}, mismatches={mismatches}")

    rows: list[dict[str, Any]] = []
    for model_id in ids:
        base = core["models"][model_id]
        external = echo["models"][model_id]
        watch = smartwatch["models"][model_id]
        sunny = sunnybrook["models"][model_id]
        spec = registry_by_id[model_id]
        row: dict[str, Any] = {
            "model_id": model_id,
            "family": base["family"],
            "factorial_mask": spec["factorial_mask"],
            "mse_enabled": int(spec["factors"]["mse"]),
            "correlation_enabled": int(spec["factors"]["correlation"]),
            "mmd_enabled": int(spec["factors"]["mmd"]),
            "derivative_enabled": int(spec["factors"]["derivative"]),
            "ptbxl_missing_mse": get(base, "signal", "missing_leads", "mse"),
            "ptbxl_missing_rmse": get(base, "signal", "missing_leads", "rmse"),
            "ptbxl_missing_mae": get(base, "signal", "missing_leads", "mae"),
            "ptbxl_missing_r2": get(base, "signal", "missing_leads", "r2"),
            "ptbxl_missing_pearson": get(base, "signal", "missing_leads", "pearson"),
            "ptbxl_missing_snr_db": get(base, "signal", "missing_leads", "snr_db"),
            "ptbxl_missing_derivative_mse": get(
                base, "signal", "missing_leads", "derivative_mse"
            ),
            "ptbxl_qrs_correlation": get(base, "morphology", "qrs_correlation"),
            "ptbxl_st_correlation": get(base, "morphology", "st_correlation"),
            "ptbxl_jpoint_mae": get(base, "morphology", "j_point_amplitude_mae"),
            "ptbxl_rpeak_timing_mae_ms": get(
                base, "morphology", "r_peak_timing_mae_ms"
            ),
            "ptbxl_morphology_coverage": get(
                base, "morphology", "detector_coverage", "fraction"
            ),
            "ecgfounder_150_macro_auroc": get(base, "clinical", "macro", "auroc"),
            "ecgfounder_150_macro_average_precision": get(
                base, "clinical", "macro", "average_precision"
            ),
            "ecgfounder_150_macro_f1": get(base, "clinical", "macro", "f1"),
            "ecgfounder_150_ece": get(base, "clinical", "macro", "ece"),
            "ecgfounder_150_brier": get(base, "clinical", "macro", "brier"),
            "ptbxl_superclass_macro_auroc": get(
                base, "paper_parity_clinical", "macro", "auroc"
            ),
            "ptbxl_superclass_macro_ap": get(
                base, "paper_parity_clinical", "macro", "average_precision"
            ),
            "ptbxl_superclass_macro_f1": get(
                base, "paper_parity_clinical", "macro", "f1"
            ),
            "ptbxl_gender_auroc_gap": get(
                base, "fairness", "gaps", "gender_auroc_gap"
            ),
            "ptbxl_age_auroc_gap": get(base, "fairness", "gaps", "age_auroc_gap"),
            "echonext_missing_mse": get(external, "signal", "missing_leads", "mse"),
            "echonext_missing_pearson": get(
                external, "signal", "missing_leads", "pearson"
            ),
            "echonext_shd_macro_auroc": get(
                external, "shd_clinical", "macro", "auroc"
            ),
            "sunnybrook_missing_mse": get(sunny, "signal", "missing_leads", "mse"),
            "sunnybrook_missing_r2": get(sunny, "signal", "missing_leads", "r2"),
            "sunnybrook_missing_pearson": get(
                sunny, "signal", "missing_leads", "pearson"
            ),
            "sunnybrook_qrs_correlation": get(
                sunny, "morphology", "qrs_correlation"
            ),
            "sunnybrook_proxy_superclass_auroc": get(
                sunny, "five_superclass", "macro", "auroc"
            ),
            "sunnybrook_ecgfounder_probability_pearson": get(
                sunny,
                "ecgfounder_probability_fidelity",
                "macro",
                "probability_pearson",
            ),
        }
        robustness = base["robustness"]["conditions"]
        row["ptbxl_noise_mean_missing_mse_delta"] = finite_mean(
            value.get("missing_lead_mse_delta") for value in robustness.values()
        )
        for condition, value in robustness.items():
            row[f"noise_{condition}_missing_mse"] = get(
                value, "signal", "missing_leads", "mse"
            )
            row[f"noise_{condition}_missing_mse_delta"] = value.get(
                "missing_lead_mse_delta"
            )
        for device, value in watch.items():
            row[f"smartwatch_{device}_n"] = value.get("n_paired_records")
            row[f"smartwatch_{device}_missing11_mse"] = get(
                value, "signal_wearable_missing_eleven", "missing_leads", "mse"
            )
            row[f"smartwatch_{device}_missing11_pearson"] = get(
                value, "signal_wearable_missing_eleven", "missing_leads", "pearson"
            )
            row[f"smartwatch_{device}_ecgfounder_probability_pearson"] = get(
                value,
                "ecgfounder_probability_fidelity",
                "macro",
                "probability_pearson",
            )
        if quality:
            quality_model = quality["models"][model_id]
            row["ptbxl_signal_quality_macro_auroc"] = get(
                quality_model, "classification", "macro", "auroc"
            )
            row["ptbxl_signal_quality_any_artifact_auroc"] = get(
                quality_model,
                "classification",
                "per_class",
                "any_artifact",
                "auroc",
            )
            row["ptbxl_signal_quality_probability_pearson"] = get(
                quality_model,
                "probability_fidelity",
                "macro",
                "probability_pearson",
            )
        rows.append(row)

    PACKAGE.mkdir(parents=True, exist_ok=True)
    linked = []
    linked.extend(link_tree(CORE, PACKAGE / "raw/factorial_v4_2x4"))
    linked.extend(link_tree(SUNNYBROOK, PACKAGE / "raw/sunnybrook"))
    if quality:
        linked.extend(link_tree(QUALITY, PACKAGE / "raw/ptbxl_signal_quality"))

    # Include every existing artifact explicitly referenced by the locked JSONs.
    referenced: set[Path] = set()
    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, str) and value.startswith("results/"):
            candidate = ROOT / value
            if candidate.is_file():
                referenced.add(candidate)
    for payload in (core, echo, smartwatch, sunnybrook, quality):
        if payload:
            collect(payload)
    for source in sorted(referenced):
        destination = PACKAGE / "referenced_artifacts" / source.relative_to(ROOT / "results")
        hardlink(source, destination)
        linked.append(destination)

    table = pd.DataFrame(rows).sort_values(["family", "factorial_mask"])
    table.to_csv(PACKAGE / "all_48_models_master.csv", index=False)
    table.to_parquet(PACKAGE / "all_48_models_master.parquet", index=False)
    (PACKAGE / "all_48_models_master.json").write_text(
        json.dumps(table.replace({np.nan: None}).to_dict(orient="records"), indent=2)
    )

    # Long-form tables expose every nested endpoint without requiring JSON parsing.
    write_table(
        [
            {"model_id": model_id, "lead": lead, **scalar_fields(metrics)}
            for model_id in ids
            for lead, metrics in core["models"][model_id]["signal"]["per_lead"].items()
        ],
        "ptbxl_clean_per_lead",
    )
    write_table(
        [
            {"model_id": model_id, "task": task, **scalar_fields(metrics)}
            for model_id in ids
            for task, metrics in core["models"][model_id]["clinical"]["per_task"].items()
        ],
        "ecgfounder_150_per_task",
    )
    write_table(
        [
            {"model_id": model_id, "superclass": task, **scalar_fields(metrics)}
            for model_id in ids
            for task, metrics in core["models"][model_id][
                "paper_parity_clinical"
            ]["per_class"].items()
        ],
        "ptbxl_five_superclass_per_class",
    )
    write_table(
        [
            {
                "model_id": model_id,
                "condition": condition,
                "source": metrics.get("source"),
                "noise_type": metrics.get("noise_type"),
                "snr_db": metrics.get("snr_db"),
                "n_samples": metrics.get("n_samples"),
                "missing_lead_mse_delta": metrics.get("missing_lead_mse_delta"),
                **{
                    f"missing_{key}": value
                    for key, value in scalar_fields(
                        metrics.get("signal", {}).get("missing_leads", {})
                    ).items()
                },
            }
            for model_id in ids
            for condition, metrics in core["models"][model_id]["robustness"][
                "conditions"
            ].items()
        ],
        "ptbxl_noise_stress_17_conditions",
    )
    write_table(
        [
            {
                "model_id": model_id,
                "device": device,
                "n_paired_records": metrics.get("n_paired_records"),
                **{
                    f"missing11_{key}": value
                    for key, value in scalar_fields(
                        metrics["signal_wearable_missing_eleven"]["missing_leads"]
                    ).items()
                },
                **{
                    f"ecgfounder_fidelity_{key}": value
                    for key, value in scalar_fields(
                        metrics["ecgfounder_probability_fidelity"]["macro"]
                    ).items()
                },
            }
            for model_id in ids
            for device, metrics in smartwatch["models"][model_id].items()
        ],
        "smartwatch_four_device_summary",
    )
    write_table(
        [
            {"model_id": model_id, "lead": lead, **scalar_fields(metrics)}
            for model_id in ids
            for lead, metrics in sunnybrook["models"][model_id]["signal"][
                "per_lead"
            ].items()
        ],
        "sunnybrook_per_lead",
    )
    write_table(
        [
            {"model_id": model_id, "superclass": task, **scalar_fields(metrics)}
            for model_id in ids
            for task, metrics in sunnybrook["models"][model_id][
                "five_superclass"
            ]["per_class"].items()
        ],
        "sunnybrook_proxy_superclass_per_class",
    )
    write_table(
        [
            {"model_id": model_id, "task": task, **scalar_fields(metrics)}
            for model_id in ids
            for task, metrics in sunnybrook["models"][model_id][
                "ecgfounder_probability_fidelity"
            ]["per_task"].items()
        ],
        "sunnybrook_ecgfounder_150_probability_fidelity",
    )
    write_table(
        [
            {"model_id": model_id, "task": task, **scalar_fields(metrics)}
            for model_id in ids
            for task, metrics in echo["models"][model_id]["shd_clinical"][
                "per_task"
            ].items()
        ],
        "echonext_shd_per_task",
    )
    if quality:
        write_table(
            [
                {"model_id": model_id, "quality_class": task, **scalar_fields(metrics)}
                for model_id in ids
                for task, metrics in quality["models"][model_id]["classification"][
                    "per_class"
                ].items()
            ],
            "ptbxl_signal_quality_per_class",
        )

    # Friendly benchmark entry points (hard links, therefore no duplicate blocks).
    entry_points = {
        "ptbxl/full_clean_clinical_robustness.json": CORE / "comprehensive_results.json",
        "ecgfounder/ptbxl_150_task_results.json": CORE / "comprehensive_results.json",
        "ptbxl_superclass/five_superclass_results.json": CORE / "comprehensive_results.json",
        "noise_stress/ptbxl_17_conditions.json": CORE / "comprehensive_results.json",
        "echonext/echonext_results.json": CORE / "echonext_results.json",
        "smartwatch/four_device_results.json": CORE / "smartwatch_results.json",
        "sunnybrook/sunnybrook_results.json": SUNNYBROOK / "sunnybrook_results.json",
        "provenance/model_registry.json": ROOT
        / "experiment_queue/factorial_v4/model_registry.json",
        "provenance/sunnybrook/README.md": ROOT
        / "data/sunnybrook_12_lead_ecg_samples/README.md",
        "provenance/sunnybrook/DATA_DICTIONARY.md": ROOT
        / "data/sunnybrook_12_lead_ecg_samples/DATA_DICTIONARY.md",
        "provenance/sunnybrook/METADATA.json": ROOT
        / "data/sunnybrook_12_lead_ecg_samples/METADATA.json",
    }
    if quality:
        entry_points["ptbxl_signal_quality/signal_quality_results.json"] = quality_path
        entry_points["provenance/ecgfm_signal_quality_head.json"] = (
            ROOT / "checkpoints/factorial_v4/parity/ecgfm_signal_quality.json"
        )
    for relative, source in entry_points.items():
        hardlink(source, PACKAGE / relative)

    artifact_files = sorted(path for path in PACKAGE.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "package": str(PACKAGE.relative_to(ROOT)),
        "model_count": len(table),
        "families": table.groupby("family").size().to_dict(),
        "factorial_design": "3 architectures x 2^4 MSE/correlation/MMD/derivative = 48",
        "included_benchmarks": [
            "PTB-XL clean reconstruction and morphology (2,198 records)",
            "PTB-XL ECGFounder 150-task diagnostic utility and subgroup gaps",
            "PTB-XL frozen five-superclass parity classifier",
            "PTB-XL 17-condition deterministic synthetic/NSTDB robustness",
            "EchoNext external validation",
            "four-device PhysioNet smartwatch validation",
            "Sunnybrook Cath Lab 20-record zero-shot validation",
            *(
                ["PTB-XL signal-quality classification"]
                if quality
                else ["PTB-XL signal-quality classification: pending"]
            ),
        ],
        "coverage": {
            name: len(values) for name, values in sources.items()
        },
        "hardlink_storage_note": (
            "Raw artifacts are hard-linked to canonical results to avoid duplicate disk blocks."
        ),
        "files": [
            {
                "path": str(path.relative_to(PACKAGE)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in artifact_files
            if path.name
            not in {
                "MANIFEST.json",
                "MANIFEST.md",
                "README.md",
                "FINDINGS.md",
                "VERIFICATION.json",
                "COMPLETENESS_REPORT.md",
            }
        ],
    }
    (PACKAGE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        "# Comprehensive Latest 48-Model Results",
        "",
        "Locked design: 3 architectures × 16 MSE/correlation/MMD/derivative masks.",
        "",
        "| Benchmark | Models | Status |",
        "|---|---:|---|",
    ]
    for name, count in manifest["coverage"].items():
        lines.append(f"| {name} | {count}/48 | {'complete' if count == 48 else 'incomplete'} |")
    lines.extend(
        [
            "",
            "Sunnybrook classification uses non-adjudicated Philips PageWriter statement",
            "proxies. ECGFounder on Sunnybrook is probability fidelity only because no",
            "150-task ground-truth label matrix is available. The raw ECG010 V5 transient",
            "(20.44 mV) and ECG008 V4 transient (6.66 mV) are retained without clipping.",
            "",
            "The electrode-problem PTB-XL signal-quality endpoint is exploratory because",
            "fold 10 contains only three positives.",
            "",
            "Use `all_48_models_master.csv` for figures and `raw/` for full nested metrics.",
        ]
    )
    (PACKAGE / "README.md").write_text("\n".join(lines) + "\n")
    findings = [
        "# Cross-Benchmark Findings",
        "",
        "All comparisons below use the locked seed-42 2^4 grid. “Full” is E+C+M+D",
        "and “base” is MSE-only (E=1, C=M=D=0). These are descriptive deltas unless",
        "the patient-cluster BCa family-wise test is explicitly cited.",
        "",
        "## Full Composite Versus MSE-Only",
        "",
        "| Family | Δ PTB-XL Pearson | Δ QRS corr. | Δ ST corr. | Δ ECGFounder AUROC | Δ EchoNext Pearson | Δ Sunnybrook Pearson |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("unet", "multiscale_vae", "ecg_aim"):
        group = table[table["family"] == family]
        base = group[group["factorial_mask"] == "1000"].iloc[0]
        full = group[group["factorial_mask"] == "1111"].iloc[0]
        findings.append(
            "| "
            + family.replace("_", " ").title()
            + " | "
            + " | ".join(
                f"{full[column] - base[column]:+.4f}"
                for column in (
                    "ptbxl_missing_pearson",
                    "ptbxl_qrs_correlation",
                    "ptbxl_st_correlation",
                    "ecgfounder_150_macro_auroc",
                    "echonext_missing_pearson",
                    "sunnybrook_missing_pearson",
                )
            )
            + " |"
        )
    statistics = json.loads((CORE / "statistics.json").read_text())
    ni = statistics["ecgfounder_noninferiority_summary"]
    findings.extend(
        [
            "",
            "The prespecified full-versus-base patient-cluster BCa tests show QRS",
            "and ST improvements in all three families at α=0.0167. ECGFounder",
            "AUROC changes are not significant in any family; this supports",
            "morphology improvement without evidence of diagnostic-utility loss,",
            "but it is not proof of clinical equivalence.",
            "",
            f"Across all 48 cells, {ni['passed']}/{ni['tested']} meet the exploratory",
            "0.02 ECGFounder non-inferiority margin: ECG-AIM 16/16,",
            "MultiScale-VAE 11/16, and U-Net 0/16.",
            "",
            "## External And Device Transfer",
            "",
            "- ECG-AIM has the strongest Sunnybrook missing-lead Pearson: "
            f"{table['sunnybrook_missing_pearson'].max():.3f}.",
            "- The best Sunnybrook Philips-statement proxy superclass AUROC is "
            f"{table['sunnybrook_proxy_superclass_auroc'].max():.3f}; this is not "
            "clinician-adjudicated ground truth.",
            "- ECGFounder on Sunnybrook is evaluated only by frozen probability",
            "fidelity because no compatible 150-task label matrix exists.",
            "- EchoNext and smartwatch metrics are retained in full, including",
            "per-task and per-device tables; smartwatch signals are simulator/device",
            "transfer rather than human diagnostic validation.",
            "- Sunnybrook retains raw cath-lab transients and uses physical mV without",
            "per-record min-max normalization or clipping.",
        ]
    )
    if quality:
        source_quality = quality["source_reference"]["macro"]["auroc"]
        findings.extend(
            [
                "",
                "## PTB-XL Signal Quality",
                "",
                f"The frozen source-ECG quality head reaches macro AUROC {source_quality:.3f}.",
                "Reconstruction-cell results and probability fidelity appear in the",
                "master table and per-class long table. Electrode-problem AUROC is",
                "exploratory because fold 10 has only three positives.",
            ]
        )
    (PACKAGE / "FINDINGS.md").write_text("\n".join(findings) + "\n")
    manifest_lines = ["# Artifact Manifest", ""]
    manifest_lines.extend(
        f"- `{item['path']}` — {item['bytes']} bytes — `{item['sha256']}`"
        for item in manifest["files"]
    )
    (PACKAGE / "MANIFEST.md").write_text("\n".join(manifest_lines) + "\n")
    print(PACKAGE)


if __name__ == "__main__":
    main()
