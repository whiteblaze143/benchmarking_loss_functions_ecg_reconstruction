#!/usr/bin/env python3
"""One-batch CPU witness for the author-component LUDB Mean Teacher path."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
VENDOR_SRC = ROOT / "external/semiseg/semi-seg-ecg/src"
sys.path.insert(0, str(VENDOR_SRC))

from algorithms.base import init_model_from_cfg  # noqa: E402
from algorithms.mean_teacher import train_one_epoch  # noqa: E402
from utils.misc import NativeScalerWithGradNormCount  # noqa: E402
from utils.optimizer import get_optimizer_from_config  # noqa: E402
from utils.semi_dataset import build_seg_dataset  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/semiseg_ludb_mt_full_cpu.yaml").read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.set_num_threads(7)

    unlabeled = build_seg_dataset(config["dataset"], split="train_unlabeled")
    labeled = build_seg_dataset(config["dataset"], split="train_labeled", num_unlabeled=len(unlabeled))
    labeled_loader = DataLoader(Subset(labeled, [0, 1]), batch_size=2, num_workers=0)
    unlabeled_loader = DataLoader(Subset(unlabeled, [0, 1]), batch_size=2, num_workers=0)

    student = init_model_from_cfg(config).cpu()
    teacher = copy.deepcopy(student).cpu().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    before_student = next(student.parameters()).detach().clone()
    before_teacher = next(teacher.parameters()).detach().clone()
    optimizer = get_optimizer_from_config(config["train"], student.parameters())
    scaler = NativeScalerWithGradNormCount()
    stats = train_one_epoch(
        student,
        teacher,
        labeled_loader,
        unlabeled_loader,
        optimizer,
        torch.device("cpu"),
        1,
        scaler,
        use_amp=False,
        config=config["train"],
    )
    student_delta = float((next(student.parameters()) - before_student).abs().max())
    teacher_delta = float((next(teacher.parameters()) - before_teacher).abs().max())
    if not all(np.isfinite(value) for value in stats.values()) or student_delta <= 0 or teacher_delta <= 0:
        raise RuntimeError("training or EMA update witness failed")
    print(json.dumps({
        "event": "WITNESS_SEMISEG_MT_CPU",
        "labeled_streams": len(labeled),
        "unlabeled_streams": len(unlabeled),
        "batch_size": 2,
        "threads": torch.get_num_threads(),
        "cuda_visible": torch.cuda.is_available(),
        "student_delta": student_delta,
        "teacher_delta": teacher_delta,
        "stats": stats,
    }), flush=True)


if __name__ == "__main__":
    main()
