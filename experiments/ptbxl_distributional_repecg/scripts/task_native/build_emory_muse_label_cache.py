#!/usr/bin/env python3
"""Materialize the train-derived sparse label contract for admitted Emory-MUSE rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_codes(text: str) -> tuple[str, ...]:
    values = tuple(code for code in str(text).split(",") if code)
    if not values:
        raise ValueError(f"invalid admitted diagnosis-code list: {text!r}")
    # The released 12SL table can repeat a code in one record.  This is one
    # multilabel target, not a frequency signal, so retain first occurrence.
    return tuple(dict.fromkeys(values))


def encode(frame: pd.DataFrame, vocabulary: dict[str, int]) -> tuple[sparse.csr_matrix, Counter[str]]:
    rows: list[int] = []
    columns: list[int] = []
    unseen: Counter[str] = Counter()
    for row_index, codes in enumerate(frame["codes"]):
        for code in parse_codes(codes):
            column = vocabulary.get(code)
            if column is None:
                unseen[code] += 1
            else:
                rows.append(row_index)
                columns.append(column)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(frame), len(vocabulary)),
        dtype=np.uint8,
    )
    return matrix, unseen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    admitted_path = args.admission_dir / "admitted_records.csv"
    admission_manifest_path = args.admission_dir / "manifest.json"
    if not admitted_path.is_file() or not admission_manifest_path.is_file():
        raise FileNotFoundError("admission directory lacks admitted_records.csv or manifest.json")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")

    admitted = pd.read_csv(admitted_path, dtype=str)
    required = {"wfdb_path", "patient_id", "split", "codes"}
    if set(admitted.columns) != required or admitted["wfdb_path"].duplicated().any():
        raise ValueError("admitted Emory table has invalid columns or duplicate records")
    if set(admitted["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("admitted Emory table does not contain the frozen three-way split")

    train = admitted.loc[admitted["split"].eq("train")].reset_index(drop=True)
    vocabulary = sorted({code for codes in train["codes"] for code in parse_codes(codes)})
    if not vocabulary:
        raise ValueError("training split has no diagnosis codes")
    vocabulary_index = {code: index for index, code in enumerate(vocabulary)}

    args.output_dir.mkdir(parents=True)
    split_summary: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        frame = admitted.loc[admitted["split"].eq(split)].reset_index(drop=True)
        matrix, unseen = encode(frame, vocabulary_index)
        sparse.save_npz(args.output_dir / f"{split}_labels_csr.npz", matrix, compressed=True)
        split_summary[split] = {
            "records": int(len(frame)),
            "patients": int(frame["patient_id"].nunique()),
            "positive_entries": int(matrix.nnz),
            "unseen_code_occurrences": {code: int(count) for code, count in sorted(unseen.items())},
        }

    (args.output_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2) + "\n")
    manifest = {
        "schema_version": "emory_muse_sparse_labels_v1",
        "admission_manifest_sha256": sha256_file(admission_manifest_path),
        "admitted_records_sha256": sha256_file(admitted_path),
        "vocabulary_source": "training split only",
        "n_classes": len(vocabulary),
        "splits": split_summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "complete", "n_classes": len(vocabulary), "splits": split_summary}))


if __name__ == "__main__":
    main()
