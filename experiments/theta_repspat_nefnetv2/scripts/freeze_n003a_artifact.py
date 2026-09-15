#!/usr/bin/env python3
"""Freeze the completed N003a seed-123 artifact before external evaluation."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
RUN = ROOT / "results/n003a_stage1_fixed_norm_seed123"
FREEZE = RUN / "frozen_artifact"
PTB_ROOT = Path("/home/mithunmanivannan/data/ptb_xl")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    manifest = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        manifest[path.relative_to(root).as_posix()] = sha256(path)
    return manifest


def split_frame(metadata: pd.DataFrame, split: str, folds: set[int]) -> pd.DataFrame:
    tensor_ids = sorted(int(path.stem) for path in (PTB_ROOT / "tensors" / split).glob("*.pt"))
    frame = metadata.loc[tensor_ids, ["patient_id", "strat_fold"]].reset_index()
    actual_folds = set(frame["strat_fold"].astype(int).unique())
    if actual_folds != folds:
        raise RuntimeError(f"{split} tensor folds {actual_folds} != expected {folds}")
    if frame["patient_id"].isna().any():
        raise RuntimeError(f"{split} has missing patient IDs")
    return frame


def main():
    required = [RUN / "model_best.pt", RUN / "model_final.pt", RUN / "last_training_state.pt",
                RUN / "train_metrics.jsonl", RUN / "validation_manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required N003a artifacts: {missing}")
    state = torch.load(RUN / "last_training_state.pt", map_location="cpu", weights_only=True)
    if int(state["epoch"]) != 99:
        raise RuntimeError(f"Expected completed epoch 100, got {int(state['epoch']) + 1}")

    metadata = pd.read_csv(PTB_ROOT / "ptbxl_database.csv", index_col="ecg_id")
    train = split_frame(metadata, "train", set(range(1, 9)))
    val = split_frame(metadata, "val", {9})
    train_patients, val_patients = set(train.patient_id), set(val.patient_id)
    if train_patients & val_patients:
        raise RuntimeError("Patient leakage between PTB-XL train and validation tensors")

    author = tree_manifest(ROOT / "author_code/nefnet_v2")
    upstream = tree_manifest(REPO / "external/NEFNET-v2-main")
    if author != upstream:
        raise RuntimeError("Vendored author tree differs from upstream reference")

    FREEZE.mkdir(parents=True, exist_ok=True)
    train.to_csv(FREEZE / "ptbxl_train_records_patients.csv", index=False)
    val.to_csv(FREEZE / "ptbxl_val_records_patients.csv", index=False)
    (FREEZE / "author_tree_manifest.json").write_text(json.dumps(author, indent=2) + "\n")

    source_paths = [
        ROOT / "scripts/train_n003_stage1_clinical.py",
        ROOT / "src/theta_repspat/clinical_dataset.py",
        ROOT / "src/theta_repspat/vendor.py",
    ]
    config = {
        "experiment": "N003a_clean_stage1_seed123",
        "selection_status": "FROZEN_BEFORE_EXTERNAL_EVALUATION",
        "selected_checkpoint": "model_best.pt",
        "selected_checkpoint_sha256": sha256(RUN / "model_best.pt"),
        "model_final_sha256": sha256(RUN / "model_final.pt"),
        "training_state_sha256": sha256(RUN / "last_training_state.pt"),
        "training_metrics_sha256": sha256(RUN / "train_metrics.jsonl"),
        "validation_manifest_sha256": sha256(RUN / "validation_manifest.json"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "executed_source_sha256": {str(path.relative_to(REPO)): sha256(path) for path in source_paths},
        "author_tree_matches_upstream": True,
        "author_tree_manifest_sha256": sha256(FREEZE / "author_tree_manifest.json"),
        "data": {
            "train_folds": list(range(1, 9)), "validation_folds": [9], "test_fold": 10,
            "train_records": len(train), "validation_records": len(val),
            "train_patients": len(train_patients), "validation_patients": len(val_patients),
            "train_validation_patient_overlap": 0,
        },
        "model": {"architecture": "official nefnet_plus.layer", "super_mode": "pretrain"},
        "optimization": {"seed": 123, "epochs": 100, "batch_size": 64, "optimizer": "AdamW",
                         "learning_rate": 1e-3, "weight_decay": 0.01,
                         "scheduler": "MultiStepLR", "milestones": [30, 60, 80], "gamma": 0.5},
        "amplitude_transform": {"lower_mv": -4.0, "upper_mv": 4.0, "clip_epsilon": 1e-4},
        "canonical_leads": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6", "III", "aVR", "aVL", "aVF"],
        "disk_to_canonical": [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5],
        "canonical_angles_rad": [
            [1.5707963705062866, 1.5707963705062866], [2.6179938316345215, 1.5707963705062866],
            [1.5707963705062866, -0.1745329201221466], [1.5707963705062866, 0.1745329201221466],
            [1.6580628156661987, 0.2617993950843811], [1.7278759479522705, 0.5235987901687622],
            [1.675516128540039, 1.0471975803375244], [1.675516128540039, 1.5707963705062866],
            [2.6179938316345215, -1.5707963705062866], [1.0471975803375244, -1.5707963705062866],
            [1.0471975803375244, 1.5707963705062866], [3.1415927410125732, 1.5707963705062866],
        ],
        "checkpoint": {"epoch": 100, "best_validation_l1": float(state["best_val_l1"]),
                       "state_keys": sorted(state)},
    }
    config_path = FREEZE / "freeze_manifest.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))
    print(f"freeze_manifest_sha256={sha256(config_path)}")


if __name__ == "__main__":
    main()
