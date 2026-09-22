#!/usr/bin/env python3
"""Admit Sunnybrook only when a real within-patient adaptation holdout exists."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patient_key(path: Path) -> str:
    root = ET.parse(path).getroot()
    values = [
        (element.text or "").strip()
        for element in root.iter()
        if element.tag.split("}")[-1].lower() == "mrn" and (element.text or "").strip()
    ]
    if len(values) != 1:
        raise ValueError(f"{path.name} requires exactly one non-empty retained MRN")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite eligibility audit: {args.output}")
    files = sorted(args.xml_dir.glob("ECG*.xml"))
    if len(files) != 20:
        raise ValueError(f"expected exactly 20 Sunnybrook XML records, found {len(files)}")
    groups = Counter(_patient_key(path) for path in files)
    repeated = sorted(count for count in groups.values() if count >= 2)
    payload = {
        "schema_version": "sunnybrook_patient_adaptation_eligibility_v1",
        "records": len(files),
        "retained_patient_keys": len(groups),
        "patients_with_multiple_records": len(repeated),
        "maximum_records_per_patient": max(groups.values()),
        "status": "eligible" if repeated else "not_eligible_no_within_patient_holdout",
        "reason": (
            "A within-patient adaptation experiment requires at least two time-separated ECGs for one patient."
            if not repeated else "At least one retained patient key has multiple ECGs."
        ),
        "source_sha256": {path.name: _sha256(path) for path in files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "records", "retained_patient_keys", "patients_with_multiple_records")}))


if __name__ == "__main__":
    main()
