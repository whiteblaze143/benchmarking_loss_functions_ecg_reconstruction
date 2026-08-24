#!/usr/bin/env python3
"""Replace an exact model subset in a result JSON with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-prefix", default="msvae__e0")
    parser.add_argument("--expected-count", type=int, default=8)
    args = parser.parse_args()

    base = json.loads(args.base.read_text())
    replacement = json.loads(args.replacement.read_text())
    attestation = json.loads(args.attestation.read_text())
    if attestation.get("status") != "PASS":
        raise ValueError("Repair attestation is not PASS")
    replacement_ids = set(replacement.get("models", {}))
    if len(replacement_ids) != args.expected_count or not all(
        model_id.startswith(args.expected_prefix) for model_id in replacement_ids
    ):
        raise ValueError(
            f"Expected exactly {args.expected_count} {args.expected_prefix}* models; "
            f"got {sorted(replacement_ids)}"
        )
    missing = replacement_ids - set(base.get("models", {}))
    if missing:
        raise ValueError(f"Replacement models absent from base: {sorted(missing)}")

    base["models"].update(replacement["models"])
    base.setdefault("repairs", {})["msvae_mse_toggle_2x4"] = {
        "status": "replaced_and_attested",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_ids": sorted(replacement_ids),
        "base_sha256": sha256(args.base),
        "replacement_sha256": sha256(args.replacement),
        "attestation_sha256": sha256(args.attestation),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(base, indent=2, allow_nan=False) + "\n")
    print(f"Merged {len(replacement_ids)} repaired models into {args.output}")


if __name__ == "__main__":
    main()
