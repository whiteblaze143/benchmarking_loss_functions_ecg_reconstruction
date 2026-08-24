#!/usr/bin/env python3
"""Fail-closed source and precision compatibility audit for factorial models."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


LEGACY_SOURCE_POLICY = {
    # The old lead-consistency implementation had a broken active branch.
    # It is compatible only where digit 5 is zero and that branch is dormant.
    "75ae689d2284719ba1e3b9cbb1ca3b8fb50cad29fcbfb2699ea20fde76bbf2ec": {
        "name": "legacy-lead-branch-dormant",
        "allowed_lead_digits": ["0"],
        "source_file_sha256": {
            "scripts/common_loss.py": "5462b637a07d01ab7e0a8fa2b019ae2d4f725c2a1a6a950dbb410740a388b275",
            "scripts/train_factorial.py": "9308bbe873e6e5cdb9451e11603955b1baf393e00b57f0766f4bfb9476319def",
            "scripts/train_mcma_3lead.py": "fb51a2b2d1793b848bbe44bbffeab3b4f5392f1d264a161d53c991fe5cc2915f",
        },
    },
    # Corrected lead-consistency implementation; clean committed runs only.
    "3bd89c0f1b12f594e8c7df386fedeabad3b50d58a1b226f48c8f156794816f3f": {
        "name": "legacy-lead-branch-corrected",
        "allowed_lead_digits": ["0", "1"],
        "source_file_sha256": {
            "scripts/common_loss.py": "b415002219d749b5ca3f015ea2e501f3acbb4e3ea5d13ec6cb015861fa86eafc",
            "scripts/train_factorial.py": "3ed026dce6842326a10a6c56888afaf632741a5ef58ade07f1ae2b73af80cd25",
            "scripts/train_mcma_3lead.py": "fb51a2b2d1793b848bbe44bbffeab3b4f5392f1d264a161d53c991fe5cc2915f",
        },
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def embedded_sources_valid(provenance: dict) -> bool:
    encoded = provenance.get("source_file_contents_base64")
    expected = provenance.get("source_file_sha256")
    if not isinstance(encoded, dict) or not isinstance(expected, dict):
        return False
    if set(encoded) != set(expected):
        return False
    try:
        observed = {
            name: hashlib.sha256(base64.b64decode(value, validate=True)).hexdigest()
            for name, value in encoded.items()
        }
    except Exception:
        return False
    if observed != expected:
        return False
    bundle = hashlib.sha256()
    for name, digest in sorted(observed.items()):
        bundle.update(f"{name}:{digest}\n".encode())
    if bundle.hexdigest() != provenance.get("source_bundle_sha256"):
        return False
    try:
        diff = base64.b64decode(
            provenance["source_diff_base64"], validate=True
        )
    except Exception:
        return False
    if hashlib.sha256(diff).hexdigest() != provenance.get("source_diff_sha256"):
        return False
    return bool(diff) == bool(provenance.get("git_dirty_for_sources"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", type=Path, default=Path("results/checkpoint_store/catalog.sqlite")
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("refine-logs/factorial_training_contract.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/checkpoint_store/compatibility_audit.json"),
    )
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT model_id, factorial_mask, seed, status, size_bytes, sha256,
               metadata_json,
               payload_state_schema_sha256
        FROM checkpoints
        WHERE status IN ('remote_verified', 'cached')
          AND model_id LIKE 'f_%'
        ORDER BY model_id
        """
    ).fetchall()
    connection.close()

    audited = []
    for row in rows:
        metadata = json.loads(row["metadata_json"] or "{}")
        provenance = metadata.get("architecture_provenance") or {}
        bundle = provenance.get("source_bundle_sha256")
        dirty = provenance.get("git_dirty_for_sources")
        embedded_valid = embedded_sources_valid(provenance)
        is_current = bundle == contract["approved_source_bundle_sha256"]
        legacy = LEGACY_SOURCE_POLICY.get(bundle)
        lead_digit = str(row["factorial_mask"])[5]

        if is_current:
            source_policy = "pinned-current-exact-sources"
            source_compatible = embedded_valid
        elif legacy is not None:
            source_policy = legacy["name"]
            source_compatible = (
                dirty is False
                and lead_digit in legacy["allowed_lead_digits"]
                and provenance.get("source_file_sha256")
                    == legacy["source_file_sha256"]
            )
        else:
            source_policy = "unapproved"
            source_compatible = False

        precision = metadata.get("state_precision")
        if isinstance(precision, str):
            precision = [precision if precision.startswith("torch.") else f"torch.{precision}"]
        expected_precision = [f"torch.{contract['state_precision']}"]
        precision_compatible = precision == expected_precision
        schema_compatible = (
            row["payload_state_schema_sha256"]
            == contract["state_schema_sha256"]
        )
        recorded_contract = metadata.get("provenance", {}).get(
            "training_contract"
        )
        data_content_compatible = (
            metadata.get("provenance", {}).get("split_content_roots")
            == contract["split_content_roots"]
            and recorded_contract == contract
        )
        compatible = (
            source_compatible
            and precision_compatible
            and schema_compatible
            and data_content_compatible
        )
        reasons = []
        if not source_compatible:
            reasons.append("source_policy")
        if not precision_compatible:
            reasons.append("state_precision")
        if not schema_compatible:
            reasons.append("state_schema")
        if not data_content_compatible:
            reasons.append("data_content_contract")
        audited.append(
            {
                "model_id": row["model_id"],
                "factorial_mask": row["factorial_mask"],
                "seed": row["seed"],
                "checkpoint_size_bytes": row["size_bytes"],
                "checkpoint_sha256": row["sha256"],
                "source_bundle_sha256": bundle,
                "source_policy": source_policy,
                "git_dirty_for_sources": dirty,
                "embedded_sources_valid": embedded_valid,
                "state_precision": precision,
                "state_schema_sha256": row["payload_state_schema_sha256"],
                "data_content_compatible": data_content_compatible,
                "compatible": compatible,
                "reasons": reasons,
            }
        )

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "contract": contract,
        "legacy_source_policy": LEGACY_SOURCE_POLICY,
        "counts": {
            "compatible": sum(row["compatible"] for row in audited),
            "incompatible": sum(not row["compatible"] for row in audited),
        },
        "reason_counts": dict(Counter(reason for r in audited for reason in r["reasons"])),
        "models": audited,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report["counts"], sort_keys=True))
    if report["counts"].get("incompatible", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
