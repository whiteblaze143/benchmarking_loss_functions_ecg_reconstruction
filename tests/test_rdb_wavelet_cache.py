from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_rdb_wavelet_delineation_cache import (
    FIDUCIAL_NAMES, dense_labels, stratified_assignments,
)
from scripts.rdb_oracle import read_mapping


def selected_rows():
    mapping = ROOT / "data/rdb/rdb_chapman_mapping.xlsx"
    return [row for row in read_mapping(mapping) if row["note"] != "duplicate_rdb_record"]


def test_rdb_split_is_deterministic_stratified_and_subject_disjoint():
    rows = selected_rows()
    first = stratified_assignments(rows, 20260822, 0.15, 0.15)
    second = stratified_assignments(rows, 20260822, 0.15, 0.15)
    assert first == second
    assert Counter(first.values()) == {"train": 1678, "val": 360, "test": 360}
    assert len(rows) == len({row["chapman_record_id"] for row in rows}) == len(first)


def test_cross_class_overlap_is_masked_invalid_not_overwritten():
    labels, valid, heatmaps, fiducial_valid, hashes, ambiguous = dense_labels(
        ROOT / "data/rdb", "SB0101", Counter()
    )
    assert labels.shape == valid.shape == (12, 5000)
    assert len(hashes) == 13
    assert ambiguous == 43
    assert int((~valid).sum()) == 43
    assert torch.equal(torch.from_numpy(valid), torch.from_numpy(labels != -1))
    assert heatmaps.shape == (12, 6, 5000)
    assert fiducial_valid.shape == (12, 6)


def test_rdb_boundary_head_contains_only_annotated_start_end_channels():
    labels, valid, heatmaps, landmark_valid, _, _ = dense_labels(
        ROOT / "data/rdb", "AF0001", Counter()
    )
    assert FIDUCIAL_NAMES == (
        "P_onset", "P_offset", "QRS_onset", "QRS_offset", "T_onset", "T_offset",
    )
    # AF0001 lead I's first released QRS and T intervals are [84,128]
    # and [168,253]; inclusive END is the offset target.
    assert heatmaps[0, 2, 84] == 1 and heatmaps[0, 3, 128] == 1
    assert heatmaps[0, 4, 168] == 1 and heatmaps[0, 5, 253] == 1
    assert landmark_valid[0, [2, 3, 4, 5]].all()
    assert valid[0, 84] and labels[0, 84] == 2


def test_materialized_cache_reserves_untouched_test_partition():
    manifest = json.loads((ROOT / "data/rdb_wavelet_delineation_cache/manifest.json").read_text())
    assert manifest["split"]["counts"] == {"train": 1678, "val": 360, "test": 360}
    assert manifest["split"]["test_role"].startswith("untouched")
    patients = {split: set() for split in ("train", "val", "test")}
    for record in manifest["records"]:
        patients[record["split"]].add(record["patient_id"])
    assert not patients["train"] & patients["val"]
    assert not patients["train"] & patients["test"]
    assert not patients["val"] & patients["test"]
