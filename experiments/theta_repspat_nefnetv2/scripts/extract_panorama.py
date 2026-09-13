#!/usr/bin/env python3
"""Extract frozen Nef-Net v2 canonical panoramas from eight-lead ECG arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from theta_repspat.panorama import INPUT_INDICES, build_canonical_panorama, global_minmax_normalize
from theta_repspat.vendor import load_author_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="NPY array [records,8,time]")
    parser.add_argument("--checkpoint", type=Path, required=True, help="raw author state_dict")
    parser.add_argument("--output", type=Path, required=True, help="output NPZ")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    records = np.load(args.input, allow_pickle=False)
    if records.ndim != 3 or records.shape[1] != 8:
        raise ValueError("input must have shape [records,8,time]")
    if records.shape[-1] % 4:
        raise ValueError("time length must be divisible by four")

    root = Path(__file__).resolve().parents[1]
    device = torch.device(args.device)
    model = load_author_model(root / "author_code" / "nefnet_v2", args.checkpoint, device)
    normalized = np.stack([global_minmax_normalize(record) for record in records])
    outputs = []
    for start in range(0, len(normalized), args.batch_size):
        batch = torch.from_numpy(normalized[start : start + args.batch_size, list(INPUT_INDICES)]).to(device)
        outputs.append(build_canonical_panorama(model, batch).cpu().numpy())
    panorama = np.concatenate(outputs, axis=0)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "representation": "theta_repspat_v1",
        "lead_order": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"],
        "input_leads": ["I", "II", "V3"],
        "normalization": "author_global_record_minmax",
        "checkpoint": str(args.checkpoint.resolve()),
    }
    np.savez_compressed(args.output, panorama=panorama, metadata=json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
