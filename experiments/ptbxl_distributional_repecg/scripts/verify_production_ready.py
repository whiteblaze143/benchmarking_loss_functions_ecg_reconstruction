#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3

from repecg.evaluation.tasks import load_dataset_tasks


REQUIRED_GATES = {
    "R001", "R002", "R003",
    "R010", "R011", "R012", "R013", "R014", "R015", "R016",
    "R020", "R021", "R022", "R023",
    "R030", "R031", "R032", "R033",
}
FORBIDDEN_SOURCE = (
    "np.zeros(5",
    "labels_list.append(np.zeros",
    "For now we just instantiate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tracker_status(path: Path) -> dict[str, str]:
    statuses = {}
    pattern = re.compile(r"^\| (R\d{3}) \|.*?\| (DONE|TODO|IN_PROGRESS|BLOCKED) \|")
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            statuses[match.group(1)] = match.group(2)
    return statuses


def _validate_hashed_manifest(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing manifest: {path}")
        return
    payload = json.loads(path.read_text())
    root = path.parent
    for condition in payload.get("conditions", {}).values():
        archive = root / condition["archive"]
        if not archive.is_file():
            errors.append(f"missing archive: {archive}")
        elif _sha256(archive) != condition["sha256"]:
            errors.append(f"archive hash mismatch: {archive}")


def _validate_database(path: Path, task_ids: set[str], errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing production results database: {path}")
        return
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        expected = connection.execute(
            "SELECT status, COUNT(*) FROM expected_evaluations WHERE run_type='production' GROUP BY status"
        ).fetchall()
        coverage = dict(expected)
        total = sum(coverage.values())
        if total == 0:
            errors.append("production expected-evaluation plan is empty")
        if coverage.get("pending", 0) or coverage.get("failed", 0):
            errors.append(f"production database is incomplete: {coverage}")
        database_tasks = {
            row[0] for row in connection.execute(
                "SELECT DISTINCT task_id FROM expected_evaluations WHERE run_type='production'"
            )
        }
        missing_tasks = task_ids - database_tasks
        if missing_tasks:
            errors.append(f"production database omits native tasks: {sorted(missing_tasks)}")
        bad_checkpoints = connection.execute(
            """SELECT COUNT(*) FROM runs
               WHERE run_type LIKE 'production%' AND status='complete'
                 AND (checkpoint_path IS NULL OR checkpoint_sha256 IS NULL)"""
        ).fetchone()[0]
        if bad_checkpoints:
            errors.append(f"{bad_checkpoints} complete production runs lack checkpoint lineage")
    except sqlite3.Error as error:
        errors.append(f"invalid production database schema: {error}")
    finally:
        connection.close()


def verify(experiment: Path) -> dict[str, object]:
    sentinel = experiment / "outputs/PRODUCTION_READY"
    sentinel.unlink(missing_ok=True)
    errors: list[str] = []

    tracker = _tracker_status(experiment / "refine-logs/EXPERIMENT_TRACKER.md")
    for gate in sorted(REQUIRED_GATES):
        if tracker.get(gate) != "DONE":
            errors.append(f"gate {gate} is {tracker.get(gate, 'MISSING')}, not DONE")

    registry = load_dataset_tasks(experiment / "configs/dataset_tasks.json")
    task_ids = {
        str(task["task_id"])
        for specification in registry.values()
        for task in specification["tasks"]
    }
    for source_root in (experiment / "scripts", experiment / "src"):
        for source in source_root.rglob("*.py"):
            if source.resolve() == Path(__file__).resolve():
                continue
            text = source.read_text()
            for forbidden in FORBIDDEN_SOURCE:
                if forbidden in text:
                    errors.append(f"forbidden placeholder source {forbidden!r} in {source}")
            if "retired unsafe OOD evaluator" in text or "retired unsafe OOD builder" in text:
                errors.append(f"native replacement missing for retired entry point: {source}")

    _validate_hashed_manifest(
        experiment / "outputs/ptbxl_masking_control/ptbxl_masking_control_manifest.json", errors
    )
    required_results = (
        experiment / "outputs/native_task_reconciliation.json",
        experiment / "outputs/adapter_reconciliation.json",
        experiment / "outputs/paper_input_reconciliation.json",
        experiment / "outputs/ptbxl_masking_control/metrics/metrics.json",
        experiment / "outputs/ptbxl_masking_control/metrics/bootstrap.npz",
        experiment / "outputs/kingston_negative_control/metrics/metrics.json",
        experiment / "outputs/kingston_negative_control/metrics/bootstrap.npz",
    )
    for artifact in required_results:
        if not artifact.is_file():
            errors.append(f"missing required production artifact: {artifact}")
    reconciliation = experiment / "outputs/native_task_reconciliation.json"
    if reconciliation.is_file():
        payload = json.loads(reconciliation.read_text())
        if payload.get("status") != "complete" or set(payload.get("task_ids", [])) != task_ids:
            errors.append("native task reconciliation is incomplete or disagrees with the registry")
    adapter_reconciliation = experiment / "outputs/adapter_reconciliation.json"
    if adapter_reconciliation.is_file():
        payload = json.loads(adapter_reconciliation.read_text())
        if payload.get("status") != "complete" or payload.get("blocked_datasets"):
            errors.append(
                f"external adapter reconciliation is blocked: {payload.get('blocked_datasets', [])}"
            )
    paper_input_reconciliation = experiment / "outputs/paper_input_reconciliation.json"
    if paper_input_reconciliation.is_file():
        payload = json.loads(paper_input_reconciliation.read_text())
        if payload.get("status") != "complete" or payload.get("blocked_papers"):
            errors.append(
                f"paper input reconciliation is blocked: {payload.get('blocked_papers', [])}"
            )
    _validate_database(experiment / "outputs/results.sqlite3", task_ids, errors)

    result = {
        "status": "PASS" if not errors else "BLOCKED",
        "errors": errors,
        "required_gates": sorted(REQUIRED_GATES),
        "native_tasks": sorted(task_ids),
    }
    if not errors:
        temporary = sentinel.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, sentinel)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = verify(args.experiment.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
