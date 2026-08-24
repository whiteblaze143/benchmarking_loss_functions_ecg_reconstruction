#!/usr/bin/env python3
"""Fail-closed release gate and table writer for the mixed factorial study."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPATIBILITY_AUDIT = (
    ROOT / "results/checkpoint_store/compatibility_audit.json"
)
DEFAULT_PER_RECORD_MANIFEST = (
    ROOT / "results/factorial_mixed_level/per_record_manifest.json"
)
DEFAULT_TRAINING_CONTRACT = (
    ROOT / "refine-logs/factorial_training_contract.json"
)
DEFAULT_CHECKPOINT_CATALOG = (
    ROOT / "results/checkpoint_store/catalog.sqlite"
)
EXPECTED_GATE_IDS = {
    "manifest_complete",
    "completed_exact_artifacts",
    "failure_states_resolved",
    "data_contract_identical",
    "paired_outputs_complete",
    "table_writer_fail_closed",
    "placeholder_watermarked",
    "source_precision_compatible",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def expected_models(manifest_path: Path) -> set[tuple[str, str, int]]:
    manifest = json.loads(manifest_path.read_text())
    result = set()
    for phase in manifest["phases"]:
        for job in phase["jobs"]:
            mask = re.search(r"--factorial_mask\s+(\d{7})", job["cmd"]).group(1)
            seed = int(re.search(r"--seed\s+(\d+)", job["cmd"]).group(1))
            result.add((job["id"], mask, seed))
    if len(result) != 480:
        raise ValueError(f"Authoritative manifest resolves to {len(result)}, not 480")
    return result


def validate_compatibility_audit(
    path: Path,
    expected: set[tuple[str, str, int]],
    training_contract_path: Path = DEFAULT_TRAINING_CONTRACT,
) -> dict[str, tuple[str, int]]:
    payload = json.loads(path.read_text())
    contract = json.loads(training_contract_path.read_text())
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("models"), list
    ):
        raise ValueError("Compatibility audit schema is invalid")
    rows = payload["models"]
    observed = {
        (str(row.get("model_id")), str(row.get("factorial_mask")), int(row.get("seed")))
        for row in rows
    }
    if len(rows) != 480 or observed != expected:
        raise ValueError("Compatibility audit must cover the exact 480 identities")
    if not all(row.get("compatible") is True for row in rows):
        raise ValueError("Compatibility audit contains incompatible models")
    counts = payload.get("counts")
    if counts != {"compatible": 480}:
        raise ValueError("Compatibility audit counts are not exactly 480 compatible")
    if payload.get("contract") != contract:
        raise ValueError("Compatibility audit embeds a different training contract")
    expected_precision = [f"torch.{contract['state_precision']}"]
    for row in rows:
        if (
            row.get("source_bundle_sha256")
            != contract["approved_source_bundle_sha256"]
            or row.get("source_policy") != "pinned-current-exact-sources"
            or row.get("embedded_sources_valid") is not True
            or row.get("state_precision") != expected_precision
            or row.get("state_schema_sha256")
                != contract["state_schema_sha256"]
            or row.get("data_content_compatible") is not True
            or row.get("reasons") != []
        ):
            raise ValueError(
                f"{row.get('model_id')}: compatibility evidence violates contract"
            )
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(row.get("checkpoint_sha256")))
            or int(row.get("checkpoint_size_bytes", 0)) <= 0
        ):
            raise ValueError(
                f"{row.get('model_id')}: compatibility checkpoint identity invalid"
            )
    return {
        str(row["model_id"]): (
            str(row["checkpoint_sha256"]),
            int(row["checkpoint_size_bytes"]),
        )
        for row in rows
    }


def cohort_order_sha256(record_ids, patient_ids) -> str:
    return hashlib.sha256(
        "".join(
            f"{int(record_id)}\t{int(patient_id)}\n"
            for record_id, patient_id in zip(record_ids, patient_ids)
        ).encode()
    ).hexdigest()


def validate_per_record_manifest(
    path: Path,
    expected: set[tuple[str, str, int]],
    checkpoint_identities: dict[str, tuple[str, int]],
) -> dict[str, tuple[float, float]]:
    payload = json.loads(path.read_text())
    rows = payload.get("models")
    if not isinstance(rows, list):
        raise ValueError("Per-record manifest requires a models list")
    expected_by_id = {model_id: (mask, seed) for model_id, mask, seed in expected}
    ids = [str(row.get("model_id")) for row in rows]
    if len(rows) != 480 or len(ids) != len(set(ids)) or set(ids) != set(expected_by_id):
        raise ValueError("Per-record manifest must cover the exact 480 model ids")
    declared_paths = [(ROOT / str(row.get("path"))).resolve() for row in rows]
    if len(declared_paths) != len(set(declared_paths)):
        raise ValueError("Per-record manifest requires 480 distinct artifact paths")
    allowed_root = (path.parent / "per_record").resolve()
    required_columns = {
        "model_id", "checkpoint_sha256", "record_id", "patient_id",
        "missing_mse", "missing_pearson"
    }
    expected_order = (
        "5f85303e675ae817e486e693dbaba9ca1dfa4892c473f53ca1b785b9937e241d"
    )
    aggregates = {}
    for row in rows:
        model_id = str(row["model_id"])
        if (
            str(row.get("factorial_mask")) != expected_by_id[model_id][0]
            or int(row.get("seed")) != expected_by_id[model_id][1]
            or row.get("status") != "complete"
            or row.get("checkpoint_sha256")
                != checkpoint_identities[model_id][0]
            or int(row.get("checkpoint_size_bytes", 0))
                != checkpoint_identities[model_id][1]
        ):
            raise ValueError(f"{model_id}: per-record identity/status mismatch")
        artifact = (ROOT / str(row.get("path"))).resolve()
        try:
            artifact.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError(f"{model_id}: per-record path escapes result root") from error
        if artifact.suffix != ".parquet" or not artifact.is_file():
            raise ValueError(f"{model_id}: per-record Parquet is unavailable")
        if sha256_file(artifact) != row.get("file_sha256"):
            raise ValueError(f"{model_id}: per-record file SHA mismatch")
        frame = pd.read_parquet(artifact, columns=sorted(required_columns))
        if (
            frame["model_id"].nunique(dropna=False) != 1
            or str(frame["model_id"].iloc[0]) != model_id
            or frame["checkpoint_sha256"].nunique(dropna=False) != 1
            or str(frame["checkpoint_sha256"].iloc[0])
                != checkpoint_identities[model_id][0]
        ):
            raise ValueError(f"{model_id}: internal checkpoint identity mismatch")
        try:
            record_ids = pd.to_numeric(frame["record_id"], errors="raise")
            patient_ids = pd.to_numeric(frame["patient_id"], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{model_id}: nonnumeric cohort identity") from error
        if (
            record_ids.isna().any()
            or patient_ids.isna().any()
            or not np.equal(record_ids, np.floor(record_ids)).all()
            or not np.equal(patient_ids, np.floor(patient_ids)).all()
        ):
            raise ValueError(f"{model_id}: cohort identities must be integers")
        record_ids = record_ids.astype("int64")
        patient_ids = patient_ids.astype("int64")
        if (
            len(frame) != 2_198
            or record_ids.nunique() != 2_198
            or patient_ids.nunique() != 1_904
            or not np.isfinite(
                frame[["missing_mse", "missing_pearson"]].to_numpy()
            ).all()
        ):
            raise ValueError(f"{model_id}: per-record cohort/metrics mismatch")
        observed_order = cohort_order_sha256(record_ids, patient_ids)
        if (
            observed_order != expected_order
            or row.get("record_order_sha256") != expected_order
        ):
            raise ValueError(f"{model_id}: per-record order hash mismatch")
        aggregates[model_id] = (
            float(frame["missing_mse"].mean()),
            float(frame["missing_pearson"].mean()),
        )
    return aggregates


def validate_checkpoint_catalog(
    path: Path,
    expected: set[tuple[str, str, int]],
    checkpoint_identities: dict[str, tuple[str, int]],
    training_contract_path: Path,
) -> None:
    contract = json.loads(training_contract_path.read_text())
    expected_state_schema = contract["state_schema_sha256"]
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT model_id, factorial_mask, seed, size_bytes, sha256, status,
               asset_id, asset_size_bytes, asset_digest, asset_state,
               round_trip_verified_at, payload_validated_at,
               payload_tensor_count, payload_factorial_mask, payload_seed,
               payload_state_schema_sha256, metadata_json
        FROM checkpoints
        """
    ).fetchall()
    connection.close()
    expected_by_id = {model_id: (mask, seed) for model_id, mask, seed in expected}
    eligible = {row["model_id"]: row for row in rows if row["model_id"] in expected_by_id}
    if len(eligible) != 480:
        raise ValueError("Checkpoint catalog does not cover exact 480 identities")
    for model_id, row in eligible.items():
        digest, size = checkpoint_identities[model_id]
        mask, seed = expected_by_id[model_id]
        try:
            metadata = json.loads(row["metadata_json"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"{model_id}: invalid catalog metadata JSON") from error
        if (
            row["factorial_mask"] != mask
            or int(row["seed"]) != seed
            or row["status"] not in {"remote_verified", "cached"}
            or row["sha256"] != digest
            or int(row["size_bytes"]) != size
            or row["asset_id"] is None
            or int(row["asset_size_bytes"] or -1) != size
            or row["asset_digest"] != f"sha256:{digest}"
            or row["asset_state"] != "uploaded"
            or row["round_trip_verified_at"] is None
            or row["payload_validated_at"] is None
            or int(row["payload_tensor_count"] or -1) != 162
            or row["payload_factorial_mask"] != mask
            or int(row["payload_seed"] or -1) != seed
            or row["payload_state_schema_sha256"] != expected_state_schema
            or metadata.get("schema_version") != 3
            or str(metadata.get("run_name")) != model_id
            or str(metadata.get("factorial_mask")) != mask
            or int(metadata.get("seed", -1)) != seed
            or int(metadata.get("checkpoint_size_bytes", -1)) != size
            or metadata.get("checkpoint_sha256") != digest
            or int(metadata.get("state_tensor_count", -1)) != 162
            or metadata.get("state_schema_sha256") != expected_state_schema
            or metadata.get("training_contract") != contract
        ):
            raise ValueError(f"{model_id}: checkpoint catalog generation mismatch")


def require_releasable(
    report: pd.DataFrame | list[dict[str, Any]],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Reject any report containing a false or malformed gate."""
    frame = report if isinstance(report, pd.DataFrame) else pd.DataFrame(report)
    required = {"gate", "pass"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Release report requires columns {sorted(required)}")
    if frame.empty or frame["gate"].isna().any() or frame["gate"].duplicated().any():
        raise ValueError("Release report gates must be nonempty, named, and unique")
    if frame["pass"].isna().any() or not pd.api.types.is_bool_dtype(frame["pass"]):
        raise ValueError("Every release-gate pass value must be a JSON/Python boolean")
    blockers = frame.loc[~frame["pass"], "gate"].tolist()
    if blockers and not allow_partial:
        raise RuntimeError(
            "Factorial results are not releasable: " + "; ".join(blockers)
        )
    return {"releasable": not blockers, "blockers": blockers}


def build_release_table(
    summary_path: Path,
    release_report_path: Path,
    output_path: Path,
    *,
    manifest_path: Path = ROOT / "refine-logs/factorial_manifest.json",
    compatibility_audit_path: Path = DEFAULT_COMPATIBILITY_AUDIT,
    per_record_manifest_path: Path = DEFAULT_PER_RECORD_MANIFEST,
    training_contract_path: Path = DEFAULT_TRAINING_CONTRACT,
    checkpoint_catalog_path: Path = DEFAULT_CHECKPOINT_CATALOG,
    allow_partial: bool = False,
) -> Path:
    """Write a result table only after the machine-readable report passes."""
    report_payload = json.loads(release_report_path.read_text())
    if not isinstance(report_payload, dict):
        raise ValueError("Release report must be a versioned object, not a bare list")
    if report_payload.get("schema_version") != 1:
        raise ValueError("Release report schema_version must equal 1")
    if report_payload.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Release report is not bound to the authoritative manifest")
    if report_payload.get("summary_sha256") != sha256_file(summary_path):
        raise ValueError("Release report is not bound to these summary bytes")
    if report_payload.get("compatibility_audit_sha256") != sha256_file(
        compatibility_audit_path
    ):
        raise ValueError(
            "Release report is not bound to the source/precision audit"
        )
    if report_payload.get("training_contract_sha256") != sha256_file(
        training_contract_path
    ):
        raise ValueError("Release report is not bound to the training contract")
    if report_payload.get("checkpoint_catalog_sha256") != sha256_file(
        checkpoint_catalog_path
    ):
        raise ValueError("Release report is not bound to this checkpoint catalog")
    if report_payload.get("per_record_manifest_sha256") != sha256_file(
        per_record_manifest_path
    ):
        raise ValueError(
            "Release report is not bound to the per-record manifest"
        )
    report = report_payload.get("gates")
    if not isinstance(report, list):
        raise ValueError("Release report requires a gates list")
    gate_ids = [row.get("gate_id") for row in report]
    if len(gate_ids) != len(set(gate_ids)) or set(gate_ids) != EXPECTED_GATE_IDS:
        raise ValueError("Release report does not contain the exact eight gate ids")
    summary = pd.read_csv(
        summary_path,
        dtype={
            "model_id": str,
            "factorial_mask": str,
            "seed": int,
            "checkpoint_sha256": str,
        },
    )
    required = {
        "model_id", "factorial_mask", "seed", "checkpoint_sha256",
        "checkpoint_size_bytes", "missing_mse", "missing_pearson"
    }
    if set(summary.columns) != required:
        raise ValueError(
            "Summary table must use the exact registered release schema "
            f"{sorted(required)}; extra result columns require explicit "
            "finiteness and per-record recomputation rules"
        )
    observed = set(
        summary[["model_id", "factorial_mask", "seed"]]
        .itertuples(index=False, name=None)
    )
    expected = expected_models(manifest_path)
    if (
        len(summary) != 480
        or summary["model_id"].duplicated().any()
        or observed != expected
    ):
        raise ValueError(
            "Summary must match all 480 authoritative model/mask/seed identities"
        )
    if not np.isfinite(
        summary[["missing_mse", "missing_pearson"]].to_numpy()
    ).all():
        raise ValueError("Summary result metrics must all be finite")
    checkpoint_identities = validate_compatibility_audit(
        compatibility_audit_path, expected, training_contract_path
    )
    per_record_aggregates = validate_per_record_manifest(
        per_record_manifest_path, expected, checkpoint_identities
    )
    validate_checkpoint_catalog(
        checkpoint_catalog_path,
        expected,
        checkpoint_identities,
        training_contract_path,
    )
    for row in summary.itertuples(index=False):
        expected_digest, expected_size = checkpoint_identities[row.model_id]
        expected_mse, expected_pearson = per_record_aggregates[row.model_id]
        if (
            row.checkpoint_sha256 != expected_digest
            or int(row.checkpoint_size_bytes) != expected_size
            or not np.isclose(row.missing_mse, expected_mse, rtol=0, atol=1e-12)
            or not np.isclose(
                row.missing_pearson, expected_pearson, rtol=0, atol=1e-12
            )
        ):
            raise ValueError(f"{row.model_id}: summary/checkpoint/record mismatch")
    require_releasable(report, allow_partial=allow_partial)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--release-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "refine-logs/factorial_manifest.json",
    )
    parser.add_argument(
        "--compatibility-audit",
        type=Path,
        default=DEFAULT_COMPATIBILITY_AUDIT,
    )
    parser.add_argument(
        "--per-record-manifest",
        type=Path,
        default=DEFAULT_PER_RECORD_MANIFEST,
    )
    parser.add_argument(
        "--training-contract",
        type=Path,
        default=DEFAULT_TRAINING_CONTRACT,
    )
    parser.add_argument(
        "--checkpoint-catalog",
        type=Path,
        default=DEFAULT_CHECKPOINT_CATALOG,
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exploratory only; never use for a released results table.",
    )
    args = parser.parse_args()
    output = build_release_table(
        args.summary,
        args.release_report,
        args.output,
        manifest_path=args.manifest,
        compatibility_audit_path=args.compatibility_audit,
        per_record_manifest_path=args.per_record_manifest,
        training_contract_path=args.training_contract,
        checkpoint_catalog_path=args.checkpoint_catalog,
        allow_partial=args.allow_partial,
    )
    print(output)


if __name__ == "__main__":
    main()
