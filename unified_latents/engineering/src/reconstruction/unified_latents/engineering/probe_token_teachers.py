"""Probe token-teacher shapes for the sequential refiner study."""

from __future__ import annotations

import argparse
import ast
import csv
import os
import sys
sys.path.insert(0, '/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction/unified_latents/engineering')

import torch
import torch.nn.functional as F

sys.path.append(os.getcwd())

from src.reconstruction.unified_latents.engineering.common import TensorFolderDataset
from src.reconstruction.unified_latents.engineering.token_refiner import (
    DEFAULT_ECGFM_CKPT,
    DEFAULT_HUBERT_MODEL,
    probe_token_teacher_shapes,
)

MI_CODES = {"AMI", "IMI", "ILMI", "ASMI", "LMI", "IPLMI", "IPMI", "ALMI", "PMI"}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_dir", type=str, default="/home/mithunmanivannan/data/ptb_xl/tensors")
    parser.add_argument(
        "--ptbxl_database",
        type=str,
        default="/home/mithunmanivannan/data/ptb_xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv",
    )
    parser.add_argument("--ecgfm_checkpoint", type=str, default=DEFAULT_ECGFM_CKPT)
    parser.add_argument("--hubert_checkpoint", type=str, default=DEFAULT_HUBERT_MODEL)
    parser.add_argument("--teacher_common_token_length", type=int, default=625)
    parser.add_argument("--frozen_vae_checkpoint", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default="/home/mithunmanivannan/reports/token_refiner/teacher_shape_probe.json",
    )
    return parser.parse_args()


def _load_tensor(path: str) -> torch.Tensor:
    x = torch.load(path, weights_only=True)
    return torch.clamp(x, min=-5.0, max=5.0).float()


def _make_sample_batch(dataset: TensorFolderDataset, sample_index: int, batch_size: int) -> torch.Tensor:
    if len(dataset) == 0:
        raise ValueError("No tensors found for probe split.")
    items = []
    for offset in range(max(1, int(batch_size))):
        idx = (int(sample_index) + offset) % len(dataset)
        x, _y, _meta = dataset[idx]
        items.append(x.float())
    return torch.stack(items, dim=0)


def _select_pathology_pair(base_dir: str, split: str, ptbxl_database: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(ptbxl_database):
        return None
    norm_path = None
    mi_path = None
    with open(ptbxl_database, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ecg_id = str(row.get("ecg_id", "")).strip()
            tensor_path = os.path.join(split_dir, f"{ecg_id}.pt")
            if not ecg_id or not os.path.exists(tensor_path):
                continue
            try:
                codes = set(ast.literal_eval(row.get("scp_codes", "{}")).keys())
            except (SyntaxError, ValueError):
                continue
            if norm_path is None and "NORM" in codes and not (codes & MI_CODES):
                norm_path = tensor_path
            if mi_path is None and codes & MI_CODES:
                mi_path = tensor_path
            if norm_path is not None and mi_path is not None:
                break
    if norm_path is None or mi_path is None:
        return None
    return _load_tensor(norm_path).unsqueeze(0), _load_tensor(mi_path).unsqueeze(0)


def main() -> None:
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorFolderDataset(os.path.join(args.base_dir, args.sample_split))
    sample = _make_sample_batch(dataset, args.sample_index, args.batch_size).to(device=device, dtype=torch.float32)
    pathology_pair = _select_pathology_pair(args.base_dir, args.sample_split, args.ptbxl_database)
    if pathology_pair is not None:
        pathology_pair = tuple(item.to(device=device, dtype=torch.float32) for item in pathology_pair)
    report = probe_token_teacher_shapes(
        sample,
        output_path=args.output,
        ecgfm_checkpoint=args.ecgfm_checkpoint,
        hubert_checkpoint=args.hubert_checkpoint,
        teacher_common_token_length=args.teacher_common_token_length,
        frozen_vae_checkpoint=args.frozen_vae_checkpoint,
        pathology_pair=pathology_pair,
    )
    print(f"Saved token-teacher probe to {args.output}")
    for name, item in report.items():
        if isinstance(item, dict) and "shape" in item:
            print(
                f"{name}: shape={item['shape']} aligned={item.get('aligned_shape')} "
                f"finite={item.get('finite')} std={item.get('std', 0.0):.6f}"
            )
        else:
            print(f"{name}: {item}")


if __name__ == "__main__":
    main()
