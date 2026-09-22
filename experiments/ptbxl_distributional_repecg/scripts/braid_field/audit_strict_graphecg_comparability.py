#!/usr/bin/env python3
"""Admit the fixed PTB-XL Braid battery to the GraphECG comparison contract.

This audit deliberately checks only the common Fold-8 population and the
eight independent physical leads.  The V3-V2 ICM construction is excluded:
GraphECG receives V2 and V3 as graph nodes whereas SetOperator receives the
derived difference, so it is not a cross-family comparison cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_required(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = ("ecg_ids", "patient_ids", "labels")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        return {key: np.asarray(payload[key]) for key in required}


def require_equal(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], name: str) -> None:
    for key in ("ecg_ids", "patient_ids", "labels"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{name} is not exactly aligned to the canonical Fold-8 {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--braid-validation", type=Path, required=True)
    parser.add_argument("--canonical-fold8", type=Path, required=True)
    parser.add_argument("--setoperator-fold8", type=Path, required=True)
    parser.add_argument("--graphecg-checkpoint", type=Path, required=True)
    parser.add_argument("--braid-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite comparability audit: {args.output}")
    all_paths = [args.braid_validation, args.canonical_fold8, args.setoperator_fold8, args.graphecg_checkpoint, *args.braid_checkpoint]
    absent = [str(path) for path in all_paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"missing comparison inputs: {absent}")
    canonical = load_required(args.canonical_fold8)
    braid = load_required(args.braid_validation)
    setoperator = load_required(args.setoperator_fold8)
    require_equal(canonical, braid, "Braid validation")
    require_equal(canonical, setoperator, "SetOperator validation")
    if len(canonical["ecg_ids"]) != 2173:
        raise ValueError(f"expected the frozen 2,173-record Fold-8 cohort, got {len(canonical['ecg_ids'])}")
    output = {
        "schema_version": "strict_braid_graphecg_fold8_comparability_v1",
        "status": "admitted",
        "cohort": {"records": int(len(canonical["ecg_ids"])), "labels": int(canonical["labels"].shape[1])},
        "admitted_configurations": [
            "all non-empty subsets of Q8 independent physical leads I, II, V1-V6",
        ],
        "excluded_configurations": {
            "S_icm": "invalid cross-family cell: GraphECG receives V2 and V3 nodes while SetOperator receives V3-V2",
        },
        "input_sha256": {str(path): sha256(path) for path in all_paths},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], **output["cohort"]}))


if __name__ == "__main__":
    main()
