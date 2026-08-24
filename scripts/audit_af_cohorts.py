#!/usr/bin/env python3
"""Audit AF labels and frozen train/validation/test cohorts without scoring tests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.af_protocol import DEFAULT_LABEL_MAP, load_label_map, parse_scp_codes, ptbxl_af_label, rdb_af_label, sha256_file


def ptbxl_audit(path: Path) -> dict:
    frame = pd.read_csv(path, usecols=["ecg_id", "patient_id", "strat_fold", "scp_codes"])
    split_for = lambda fold: "train" if fold <= 8 else "val" if fold == 9 else "test" if fold == 10 else "invalid"
    counts, conflicts, patients = defaultdict(Counter), [], defaultdict(set)
    for row in frame.itertuples(index=False):
        split = split_for(int(row.strat_fold)); label, group = ptbxl_af_label(parse_scp_codes(row.scp_codes))
        counts[split][group] += 1
        if label is None: conflicts.append(int(row.ecg_id))
        patients[split].add(str(row.patient_id))
    overlaps = {f"{a}_{b}": len(patients[a] & patients[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    return {"metadata_sha256": sha256_file(path), "records": len(frame), "counts": {k: dict(v) for k, v in counts.items()},
            "conflicting_AFIB_AFLT_ecg_ids": conflicts, "patient_counts": {k: len(v) for k, v in patients.items()}, "patient_overlaps": overlaps}


def rdb_audit(path: Path) -> dict:
    manifest = json.loads(path.read_text()); counts, patients = defaultdict(Counter), defaultdict(set)
    released_aliases = defaultdict(Counter)
    for row in manifest["records"]:
        split = row["split"]; label, group = rdb_af_label(row["canonical_rhythm"])
        counts[split][group] += 1; patients[split].add(str(row["patient_id"]))
        released_aliases[group][str(row["released_rhythm"])] += 1
    overlaps = {f"{a}_{b}": len(patients[a] & patients[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    if any(overlaps.values()): raise RuntimeError(f"RDB patient leakage: {overlaps}")
    return {"manifest_sha256": sha256_file(path), "records": len(manifest["records"]), "counts": {k: dict(v) for k, v in counts.items()},
            "released_aliases_by_canonical_group": {k: dict(v) for k, v in released_aliases.items()},
            "patient_counts": {k: len(v) for k, v in patients.items()}, "patient_overlaps": overlaps,
            "test_role": manifest["split"]["test_role"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-map", type=Path, default=DEFAULT_LABEL_MAP)
    parser.add_argument("--ptbxl", type=Path, default=ROOT / "data/ptb_xl/ptbxl_database.csv")
    parser.add_argument("--rdb-manifest", type=Path, default=ROOT / "data/rdb_wavelet_delineation_cache/manifest.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); mapping = load_label_map(args.label_map)
    result = {"version": 1, "purpose": "cohort_audit_only_no_test_scoring", "label_map": str(args.label_map),
              "label_map_sha256": sha256_file(args.label_map), "mapping": mapping,
              "ptbxl": ptbxl_audit(args.ptbxl), "rdb": rdb_audit(args.rdb_manifest)}
    if any(result["ptbxl"]["patient_overlaps"].values()): raise RuntimeError("PTB-XL patient overlap across official folds")
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output: args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__": main()
