#!/usr/bin/env python3
"""Build and verify the factorial-v4 poster evidence handoff.

The package contains only values derived from authoritative v4 artifacts. It
does not convert the unresolved independent audit into a supportive claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "results/factorial_v4"
CLINICAL = ROOT / "results/factorial_v4_clinical"
DEFAULT_OUTPUT = CORE / "poster_evidence"
FAMILY_LABELS = {
    "unet": "U-Net",
    "msvae": "MultiScale-VAE",
    "ecgaim": "ECG-AIM",
}
CORE_FIGURES = [
    "factorial_heatmaps",
    "factorial_heatmaps_4factor_supplementary",
    "factorial_effect_forest",
    "morphology_diagnostic_pareto",
    "nstdb_degradation",
    "echonext_external_heatmap",
    "echonext_representative_reconstructions",
]
CLINICAL_FIGURES = [
    "ptbxl_ecgfounder_task_rank",
    "ptbxl_five_superclasses",
    "echonext_shd_tasks",
    "echonext_shd_stress",
    "smartwatch_radar",
    "smartwatch_protocol_accuracy",
    "smartwatch_task_rank",
    "echonext_shd_calibration",
]
FIGURE_ROLES = {
    "factorial_heatmaps": ("main", "Architecture × loss-mask result overview"),
    "factorial_heatmaps_4factor_supplementary": (
        "supplementary",
        "Exploratory MSE-toggle context with MSVAE quarantine",
    ),
    "factorial_effect_forest": ("main", "Component marginal effects and intervals"),
    "morphology_diagnostic_pareto": (
        "main",
        "Morphology versus frozen-classifier utility trade-off",
    ),
    "nstdb_degradation": ("main", "Noise robustness and ranking changes"),
    "echonext_external_heatmap": ("main", "External waveform generalization"),
    "echonext_representative_reconstructions": (
        "main",
        "Representative external reconstructions",
    ),
    "ptbxl_ecgfounder_task_rank": (
        "optional",
        "Per-task ECGFounder retention distribution",
    ),
    "ptbxl_five_superclasses": ("optional", "Paper-parity diagnostic analysis"),
    "echonext_shd_tasks": ("optional", "Per-task external SHD performance"),
    "echonext_shd_stress": ("optional", "External SHD robustness"),
    "smartwatch_radar": ("optional", "Relative-only smartwatch profile"),
    "smartwatch_protocol_accuracy": (
        "optional",
        "Calibrated device protocol characterization",
    ),
    "smartwatch_task_rank": (
        "supplementary",
        "Smartwatch probability-fidelity proxy task distribution",
    ),
    "echonext_shd_calibration": ("optional", "External SHD calibration"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def strict_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def selected_ids() -> tuple[dict[str, str], dict[str, str]]:
    selection = load_json(CORE / "selected_masks.json")
    masks = selection["masks"]
    ids = {
        family: (
            f"{family}__e{mask[0]}c{mask[1]}m{mask[2]}d{mask[3]}__s42"
        )
        for family, mask in masks.items()
    }
    return masks, ids


def require_locked_state() -> dict[str, Any]:
    core = load_json(CORE / "completeness.json")
    clinical = load_json(CLINICAL / "completeness.json")
    final = load_json(CLINICAL / "FINAL_COMPLETION.json")
    if core.get("status") != "complete":
        raise RuntimeError("Core completeness is not complete")
    if clinical.get("status") != "PASS":
        raise RuntimeError("Clinical completeness is not PASS")
    failed = [
        name for name, check in final.get("checks", {}).items()
        if not check.get("passed")
    ]
    if failed not in ([], ["fresh_postrun_audit_pass"]):
        raise RuntimeError(f"Unexpected final-gate failures: {failed}")
    return {
        "core_completeness": core["status"],
        "clinical_completeness": clinical["status"],
        "final_gate": final["status"],
        "final_gate_passed": final["checks_passed"],
        "final_gate_total": final["checks_total"],
        "unresolved_final_checks": failed,
    }


def build_architecture_table(ids: dict[str, str]) -> pd.DataFrame:
    ptb = pd.read_parquet(CORE / "main_primary_24_cell_table.parquet").set_index("id")
    echo = load_json(CORE / "echonext_results.json")["models"]
    rows = []
    for family, model_id in ids.items():
        row = ptb.loc[model_id]
        echo_model = echo[model_id]
        rows.append({
            "family": family,
            "family_label": FAMILY_LABELS[family],
            "model_id": model_id,
            "mask_cmd": row["mask"],
            "ptbxl_mse": row["mse"],
            "ptbxl_r2": row["r2"],
            "ptbxl_pearson": row["pearson"],
            "ptbxl_qrs_correlation": row["qrs_correlation"],
            "ptbxl_st_correlation": row["st_correlation"],
            "ptbxl_ecgfounder_auroc": row["ecgfounder_auroc"],
            "echonext_r2": echo_model["signal"]["missing_leads"]["r2"],
            "echonext_pearson": echo_model["signal"]["missing_leads"]["pearson"],
            "echonext_shd_auroc": echo_model["shd_clinical"]["macro"]["auroc"],
        })
    return pd.DataFrame(rows)


def build_component_table() -> pd.DataFrame:
    effects = pd.read_parquet(CORE / "factorial_effects_bca.parquet")
    keep = effects[
        (effects["effect_type"] == "main")
        & effects["metric"].isin(
            ["r2", "qrs_correlation", "st_correlation", "ecgfounder_auroc"]
        )
    ].copy()
    columns = [
        "family", "metric", "effect", "estimate", "ci_low", "ci_high",
        "bootstrap_unit", "n", "n_patients", "n_resamples", "method",
    ]
    return keep[columns].sort_values(["metric", "family", "effect"])


def build_robustness_table(ids: dict[str, str]) -> pd.DataFrame:
    results = load_json(CORE / "comprehensive_results.json")["models"]
    rows = []
    for family, model_id in ids.items():
        model = results[model_id]
        rows.append({
            "family": family,
            "model_id": model_id,
            "condition": "clean",
            "source": "PTB-XL",
            "noise_type": "clean",
            "snr_db": None,
            "missing_lead_mse": model["signal"]["missing_leads"]["mse"],
        })
        for condition, payload in model["robustness"]["conditions"].items():
            rows.append({
                "family": family,
                "model_id": model_id,
                "condition": condition,
                "source": payload["source"],
                "noise_type": payload["noise_type"],
                "snr_db": payload["snr_db"],
                "missing_lead_mse": payload["signal"]["missing_leads"]["mse"],
            })
    return pd.DataFrame(rows)


def build_smartwatch_table(ids: dict[str, str]) -> pd.DataFrame:
    results = load_json(CORE / "smartwatch_results.json")["models"]
    rows = []
    for family, model_id in ids.items():
        for device, payload in results[model_id].items():
            signal = payload["signal_wearable_missing_eleven"]["missing_leads"]
            fidelity = payload["ecgfounder_probability_fidelity"]["macro"]
            rows.append({
                "family": family,
                "model_id": model_id,
                "device": device,
                "n_paired_records": payload["n_paired_records"],
                "r2": signal["r2"],
                "pearson": signal["pearson"],
                "mse": signal["mse"],
                "snr_db": signal["snr_db"],
                "probability_mae_proxy": fidelity["probability_mae"],
                "probability_pearson_proxy": fidelity["probability_pearson"],
                "threshold_agreement_proxy": fidelity["threshold_agreement"],
            })
    return pd.DataFrame(rows)


def build_endpoint_table() -> pd.DataFrame:
    return pd.read_parquet(CORE / "familywise_endpoint_tests.parquet")


def build_figure_inventory() -> pd.DataFrame:
    rows = []
    for root, review_path, names in [
        (CORE / "plots", CORE / "FIGURE_REVIEW.json", CORE_FIGURES),
        (
            CLINICAL / "figures",
            CLINICAL / "FIGURE_REVIEW.json",
            CLINICAL_FIGURES,
        ),
    ]:
        review = load_json(review_path)["figures"]
        for name in names:
            path = root / f"{name}.png"
            role, purpose = FIGURE_ROLES[name]
            rows.append({
                "figure": name,
                "role": role,
                "purpose": purpose,
                "path": str(path.relative_to(ROOT)),
                "png_sha256": sha256(path),
                "data_hash_gate": review[name]["data_hash_gate"],
                "visual_review": review[name]["visual_review"],
                "review_hash_matches": (
                    review[name].get("png_sha256") == sha256(path)
                ),
            })
    return pd.DataFrame(rows)


def build_claims(
    architecture: pd.DataFrame,
    effects: pd.DataFrame,
    endpoints: pd.DataFrame,
    robustness: pd.DataFrame,
    smartwatch: pd.DataFrame,
) -> list[dict[str, Any]]:
    arch = architecture.set_index("family")
    corr_r2 = effects[
        (effects.metric == "r2") & (effects.effect == "correlation")
    ].set_index("family")
    zero_db = robustness[
        robustness.condition.str.match(r"nstdb_(bw|em|ma)_0db")
    ].groupby("family").missing_lead_mse.agg(["min", "max"])
    return [
        {
            "id": "P1_architecture_balance",
            "status": "empirically_supported_pending_independent_audit",
            "statement": (
                "Among validation-selected models, ECG-AIM has the highest "
                "PTB-XL R2 and is the only family with positive EchoNext R2."
            ),
            "values": {
                family: {
                    "ptbxl_r2": float(arch.loc[family, "ptbxl_r2"]),
                    "echonext_r2": float(arch.loc[family, "echonext_r2"]),
                }
                for family in FAMILY_LABELS
            },
            "qualifiers": [
                "Selection used validation MSE with Pearson tie-breaker.",
                "EchoNext SHD uses unchanged original tabular metadata in addition to waveform.",
            ],
            "sources": [
                "architecture_selected.csv",
                "../main_primary_24_cell_table.parquet",
                "../echonext_results.json",
            ],
        },
        {
            "id": "P2_correlation_is_architecture_dependent",
            "status": "empirically_supported_pending_independent_audit",
            "statement": (
                "Correlation loss improves R2 in MultiScale-VAE and ECG-AIM "
                "but reduces R2 in U-Net."
            ),
            "values": {
                family: {
                    "estimate": float(corr_r2.loc[family, "estimate"]),
                    "ci_low": float(corr_r2.loc[family, "ci_low"]),
                    "ci_high": float(corr_r2.loc[family, "ci_high"]),
                }
                for family in FAMILY_LABELS
            },
            "qualifiers": [
                "Conditional on each seed-42 trained model.",
                "BCa resampling is paired at the PTB-XL patient-cluster level.",
            ],
            "sources": ["component_effects.csv", "../factorial_effects_bca.parquet"],
        },
        {
            "id": "P3_morphology_not_diagnostic_utility",
            "status": "empirically_supported_pending_independent_audit",
            "statement": (
                "The full objective improves QRS and ST correlations in every "
                "family, without a significant ECGFounder AUROC improvement."
            ),
            "values": {
                family: {
                    row.endpoint: {
                        "estimate": float(row.estimate),
                        "p_value": float(row.p_value),
                        "reject_null": bool(row.reject_null),
                    }
                    for row in endpoints[endpoints.family == family].itertuples()
                }
                for family in FAMILY_LABELS
            },
            "qualifiers": [
                "Alpha 0.0167 controls three prespecified endpoints within each architecture.",
                "ECGFounder macro AUROC uses 31/150 fold-10 tasks with both classes.",
            ],
            "sources": ["familywise_endpoints.csv", "../familywise_endpoint_tests.parquet"],
        },
        {
            "id": "P4_severe_noise_changes_ranking",
            "status": "empirically_supported_pending_independent_audit",
            "statement": (
                "At 0 dB NSTDB, U-Net's worse clean reconstruction can become "
                "competitive because it degrades less than the latent models."
            ),
            "values": {
                family: {
                    "zero_db_mse_min": float(zero_db.loc[family, "min"]),
                    "zero_db_mse_max": float(zero_db.loc[family, "max"]),
                }
                for family in FAMILY_LABELS
            },
            "qualifiers": [
                "Absolute MSE and degradation from clean must be shown together.",
                "Noise samples are deterministically paired across models.",
            ],
            "sources": ["selected_robustness.csv", "../comprehensive_results.json"],
        },
        {
            "id": "P5_smartwatch_domain_gap",
            "status": "empirically_supported_pending_independent_audit",
            "statement": (
                "All selected family-device smartwatch evaluations have "
                "negative missing-eleven-lead R2."
            ),
            "values": {
                "maximum_r2": float(smartwatch.r2.max()),
                "minimum_r2": float(smartwatch.r2.min()),
                "cells": int(len(smartwatch)),
            },
            "qualifiers": [
                "This is zero-shot domain-gap analysis, not clinical validation.",
                "ECGFounder values are Philips-referenced probability-fidelity proxies.",
                "The dataset contains no human disease labels.",
            ],
            "sources": ["selected_smartwatch.csv", "../smartwatch_results.json"],
        },
    ]


def write_report(
    output: Path,
    state: dict[str, Any],
    architecture: pd.DataFrame,
    claims: list[dict[str, Any]],
) -> None:
    rows = [
        "| Family | Mask | PTB-XL R² | PTB-XL Pearson | EchoNext R² | EchoNext SHD AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in architecture.itertuples():
        rows.append(
            f"| {row.family_label} | `{row.mask_cmd}` | {row.ptbxl_r2:.3f} | "
            f"{row.ptbxl_pearson:.3f} | {row.echonext_r2:.3f} | "
            f"{row.echonext_shd_auroc:.3f} |"
        )
    claim_lines = [
        f"{index}. {claim['statement']}"
        for index, claim in enumerate(claims, start=1)
    ]
    report = "\n".join([
        "# Factorial v4 Poster Evidence Handoff",
        "",
        f"Machine gates: core `{state['core_completeness']}`, clinical "
        f"`{state['clinical_completeness']}`, final "
        f"`{state['final_gate_passed']}/{state['final_gate_total']}`.",
        "",
        "The only unresolved final check is the independent postrun audit. "
        "Every statement below is therefore labeled empirical and provisional "
        "for publication wording.",
        "",
        "## Selected architecture results",
        "",
        *rows,
        "",
        "## Proposed empirical poster statements",
        "",
        *claim_lines,
        "",
        "## Required visual framing",
        "",
        "- Show absolute clean performance and degradation under noise together.",
        "- Keep morphology and diagnostic-utility endpoints visually separate.",
        "- Label smartwatch classifier outputs as probability-fidelity proxies.",
        "- State that EchoNext SHD inference also uses unchanged tabular metadata.",
        "- Do not display the quarantined MultiScale-VAE MSE-off cells as valid effects.",
        "",
    ])
    (output / "POSTER_EVIDENCE_HANDOFF.md").write_text(report)


def write_manifest(output: Path) -> None:
    content = """# Poster Evidence Package Manifest

- `POSTER_EVIDENCE_HANDOFF.md`: concise results and required visual framing.
- `poster_claims.json`: proposed empirical statements, exact values, sources,
  and mandatory qualifiers.
- `architecture_selected.csv`: selected-family clean and external results.
- `component_effects.csv`: patient-cluster BCa component effects.
- `familywise_endpoints.csv`: prespecified QRS/ST/diagnostic tests.
- `selected_robustness.csv`: clean plus all 17 paired stress conditions.
- `selected_smartwatch.csv`: four-device zero-shot transfer results.
- `figure_inventory.csv`: all 15 reviewed figures, roles, paths, and hashes.
- `gate_status.json`: current core, clinical, and final-gate state.
- `evidence_manifest.json`: SHA-256 binding for all sources and package outputs.
- `VERIFICATION.json`: fail-closed verification result.

The package is poster-layout ready but publication wording remains provisional
until the independent postrun audit and result-to-claim verdict are available.
"""
    (output / "MANIFEST.md").write_text(content)


def verify(output: Path) -> dict[str, Any]:
    manifest_path = output / "evidence_manifest.json"
    manifest = load_json(manifest_path)
    failures = []
    for path_text, expected in manifest["source_sha256"].items():
        path = ROOT / path_text
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"source:{path_text}")
    for name, expected in manifest["output_sha256"].items():
        path = output / name
        if not path.is_file() or sha256(path) != expected:
            failures.append(f"output:{name}")
    claims = load_json(output / "poster_claims.json")
    if not claims or any(
        claim.get("status") != "empirically_supported_pending_independent_audit"
        for claim in claims
    ):
        failures.append("claim_status")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "checks": {
            "source_hashes": not any(item.startswith("source:") for item in failures),
            "output_hashes": not any(item.startswith("output:") for item in failures),
            "claim_status_fail_closed": "claim_status" not in failures,
        },
        "failures": failures,
    }
    strict_json(output / "VERIFICATION.json", result)
    if failures:
        raise RuntimeError(f"Poster evidence verification failed: {failures}")
    return result


def build(output: Path) -> None:
    state = require_locked_state()
    _, ids = selected_ids()
    architecture = build_architecture_table(ids)
    effects = build_component_table()
    endpoints = build_endpoint_table()
    robustness = build_robustness_table(ids)
    smartwatch = build_smartwatch_table(ids)
    figures = build_figure_inventory()
    claims = build_claims(architecture, effects, endpoints, robustness, smartwatch)

    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "architecture_selected.csv": architecture,
        "component_effects.csv": effects,
        "familywise_endpoints.csv": endpoints,
        "selected_robustness.csv": robustness,
        "selected_smartwatch.csv": smartwatch,
        "figure_inventory.csv": figures,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    strict_json(output / "poster_claims.json", claims)
    strict_json(output / "gate_status.json", state)
    write_report(output, state, architecture, claims)
    write_manifest(output)

    sources = [
        CORE / "completeness.json",
        CLINICAL / "completeness.json",
        CLINICAL / "FINAL_COMPLETION.json",
        CORE / "selected_masks.json",
        CORE / "main_primary_24_cell_table.parquet",
        CORE / "factorial_effects_bca.parquet",
        CORE / "familywise_endpoint_tests.parquet",
        CORE / "comprehensive_results.json",
        CORE / "echonext_results.json",
        CORE / "smartwatch_results.json",
        CORE / "CLAIM_SCOPE.md",
        CORE / "exclusions/msvae_mse_toggle.json",
    ]
    output_names = [
        *outputs,
        "poster_claims.json",
        "gate_status.json",
        "POSTER_EVIDENCE_HANDOFF.md",
        "MANIFEST.md",
    ]
    manifest = {
        "schema_version": 1,
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in sources
        },
        "output_sha256": {
            name: sha256(output / name) for name in output_names
        },
        "claim_policy": (
            "Empirical interpretation only until independent postrun audit and "
            "result-to-claim verdict are available."
        ),
    }
    strict_json(output / "evidence_manifest.json", manifest)
    verify(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.verify_only:
        result = verify(output)
    else:
        build(output)
        result = load_json(output / "VERIFICATION.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
