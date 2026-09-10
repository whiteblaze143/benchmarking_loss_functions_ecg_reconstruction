#!/usr/bin/env python3
"""Extract frozen structure/dynamics/rhythm embeddings from a trained branch."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_ecg_aim_lvcg_embedding import BeatBoundaryDataset, collate
from unified_latents.engineering.models.ecg_aim_lvcg_variant import LVCGStyleLeadIEmbedding


def extract(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    arrays = {name: [] for name in ("ecg_emb", "emb_struct", "emb_dynamic", "emb_rhythm")}
    with torch.inference_mode():
        for lead_i, bounds, mask in loader:
            result = model(lead_i.to(device), bounds.to(device), mask.to(device))
            for name in arrays:
                arrays[name].append(result[name].float().cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in arrays.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--bounds-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-per-split", type=int)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "model_kwargs" not in checkpoint:
        raise ValueError("checkpoint must contain model_state_dict and model_kwargs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.enabled = False
    model = LVCGStyleLeadIEmbedding(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        dataset = BeatBoundaryDataset(
            args.tensor_root / split, args.bounds_root / f"{split}.jsonl", args.limit_per_split
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate,
        )
        arrays = extract(model, loader, device)
        record_ids = np.asarray([row["record_id"] for row in dataset.rows])
        np.savez_compressed(args.output_root / f"{split}.npz", record_id=record_ids, **arrays)
        print(f"{split}: {len(dataset)} records, embedding dim {arrays['ecg_emb'].shape[1]}", flush=True)


if __name__ == "__main__":
    main()

