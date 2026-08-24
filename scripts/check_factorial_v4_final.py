#!/usr/bin/env python3
"""Final fail-closed gate combining training, clinical, figures, and audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_FIGURES = {
    "factorial_heatmaps",
    "factorial_heatmaps_4factor_supplementary",
    "factorial_effect_forest",
    "morphology_diagnostic_pareto",
    "nstdb_degradation",
    "echonext_external_heatmap",
    "echonext_representative_reconstructions",
}
CLINICAL_FIGURES = {
    "ptbxl_ecgfounder_task_rank",
    "ptbxl_five_superclasses",
    "echonext_shd_tasks",
    "echonext_shd_stress",
    "smartwatch_radar",
    "smartwatch_protocol_accuracy",
    "smartwatch_task_rank",
    "echonext_shd_calibration",
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve(path).read_text())


def queue_complete(payload: dict[str, Any]) -> bool:
    jobs = payload.get("jobs", [])
    if isinstance(jobs, dict):
        jobs = list(jobs.values())
    return bool(jobs) and all(job.get("status") == "completed" for job in jobs)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_complete(
    payload: dict[str, Any],
    expected: set[str],
    figure_dir: Path,
) -> bool:
    figures = payload.get("figures", {})
    return set(figures) == expected and all(
        item.get("data_hash_gate") == "PASS"
        and item.get("visual_review") == "PASS"
        and (figure_dir / f"{name}.png").is_file()
        and item.get("png_sha256") == sha256(figure_dir / f"{name}.png")
        for name, item in figures.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-dir", type=Path, default=Path("results/factorial_v4"))
    parser.add_argument(
        "--clinical-dir",
        type=Path,
        default=Path("results/factorial_v4_clinical"),
    )
    parser.add_argument(
        "--core-queue",
        type=Path,
        default=Path("experiment_queue/factorial_v4/queue_state.json"),
    )
    parser.add_argument(
        "--clinical-queue",
        type=Path,
        default=Path("experiment_queue/factorial_v4_clinical/queue_state.json"),
    )
    parser.add_argument(
        "--postrun-audit",
        type=Path,
        default=Path("EXPERIMENT_AUDIT_POSTRUN.json"),
    )
    args = parser.parse_args()
    core = resolve(args.core_dir)
    clinical = resolve(args.clinical_dir)
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: Any) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    required = {
        "core_queue": resolve(args.core_queue),
        "clinical_queue": resolve(args.clinical_queue),
        "core_completeness": core / "completeness.json",
        "clinical_completeness": clinical / "completeness.json",
        "core_review": core / "FIGURE_REVIEW.json",
        "clinical_review": clinical / "FIGURE_REVIEW.json",
        "protocol_audit": core / "protocol_audit.json",
        "mmd_repair": core / "MMD_REPAIR_VALIDATION.json",
        "postrun_audit": resolve(args.postrun_audit),
    }
    missing = {name: str(path) for name, path in required.items() if not path.is_file()}
    add("required_artifacts_present", not missing, missing)
    if missing:
        payloads: dict[str, dict[str, Any]] = {}
    else:
        payloads = {name: json.loads(path.read_text()) for name, path in required.items()}

    if payloads:
        add(
            "core_queue_complete",
            queue_complete(payloads["core_queue"]),
            len(payloads["core_queue"].get("jobs", [])),
        )
        add(
            "clinical_queue_complete",
            queue_complete(payloads["clinical_queue"]),
            len(payloads["clinical_queue"].get("jobs", [])),
        )
        add(
            "core_completeness_pass",
            payloads["core_completeness"].get("status") == "complete",
            payloads["core_completeness"].get("status"),
        )
        add(
            "clinical_completeness_pass",
            payloads["clinical_completeness"].get("status") == "PASS",
            payloads["clinical_completeness"].get("status"),
        )
        add(
            "core_figures_reviewed_7",
            review_complete(payloads["core_review"], CORE_FIGURES, core / "plots"),
            payloads["core_review"].get("status"),
        )
        add(
            "clinical_figures_reviewed_8",
            review_complete(
                payloads["clinical_review"],
                CLINICAL_FIGURES,
                clinical / "figures",
            ),
            payloads["clinical_review"].get("status"),
        )
        add(
            "protocol_audit_pass",
            payloads["protocol_audit"].get("status") == "pass",
            payloads["protocol_audit"].get("status"),
        )
        add(
            "mmd_repair_validated",
            payloads["mmd_repair"].get("status") == "pass",
            payloads["mmd_repair"].get("status"),
        )
        add(
            "fresh_postrun_audit_pass",
            payloads["postrun_audit"].get("overall_verdict") == "PASS",
            payloads["postrun_audit"].get("overall_verdict"),
        )

    passed = all(item["passed"] for item in checks.values())
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(item["passed"] for item in checks.values()),
        "checks_total": len(checks),
        "checks": checks,
    }
    clinical.mkdir(parents=True, exist_ok=True)
    (clinical / "FINAL_COMPLETION.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    lines = [
        "# Factorial v4 Final Completion",
        "",
        f"Status: **{report['status']}**",
        "",
        *[
            f"- {'PASS' if value['passed'] else 'FAIL'} — `{name}`"
            for name, value in checks.items()
        ],
    ]
    (clinical / "FINAL_COMPLETION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
