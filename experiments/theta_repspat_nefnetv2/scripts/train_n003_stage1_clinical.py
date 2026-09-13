#!/usr/bin/env python3
"""Stage I Clinical Any-Pairs Pretraining for N003a (PTB-XL).

Implements the official NefNet-v2 variable-cardinality pretraining workflow on PTB-XL,
enforcing:
1. Architecture & Invariants:
   - super_mode='pretrain' in GeoVT (nefnet_plus.layer).
   - Dynamic input cardinality L_input = 2 + k with k in {1, 2, 3} (L_input in {3, 4, 5}).
   - Runtime dynamic slot dimension L = x.shape[1] (no fixed module lists).
   - Anchor views: Lead I (0) and Lead II (1) always present in slots 0 and 1.
2. Normalization Contract:
   - Prospectively fixed observed-lead normalization:
       m_obs = min_{j in O, t} x_j(t)
       M_obs = max_{j in O, t} x_j(t)
       x_tilde = (x - m_obs) / (M_obs - m_obs)
     with zero query-view statistics leakage.
3. Separation:
   - D_N003a_train cap D_repSpat_eval = emptyset (LUDB, ISP, RDB, Emory excluded).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

exp_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(exp_root / "src"))

from theta_repspat.clinical_dataset import PTBXL12LeadDataset, STANDARD_12LEAD_ANGLES_RAD
from theta_repspat.vendor import instantiate_author_model


def parse_args():
    parser = argparse.ArgumentParser(description="Stage I Clinical Pretraining for N003a")
    parser.add_argument(
        "--tensors-dir",
        type=Path,
        default=Path("/home/mithunmanivannan/data/ptb_xl/tensors"),
        help="Path to pre-extracted PTB-XL tensors directory",
    )
    parser.add_argument("--output", type=Path, default=exp_root / "results/n003a_stage1_clinical")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def atomic_save(payload, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN to bypass PyTorch 2.6 + CUDA 13.0 ptrDesc driver incompatibility on A100
    torch.backends.cudnn.enabled = False

    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = PTBXL12LeadDataset(
        tensors_dir=args.tensors_dir,
        split="train",
        seed=args.seed,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        generator=generator,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )

    # Instantiate GeoVT strictly in pretrain mode:
    # super_mode='pretrain' enforces shared mlp3 / W_encoder3 and dynamic x.shape[1]
    model = instantiate_author_model(
        exp_root / "author_code" / "nefnet_v2",
        device,
        super_mode="pretrain",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 80], gamma=0.5
    )

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        generator.set_state(state["loader_rng_state"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Resumed from epoch {start_epoch}")

    # Standard lead angles pre-converted to torch tensor
    angles_tensor = torch.from_numpy(STANDARD_12LEAD_ANGLES_RAD).to(device)  # [12, 2]

    print(f"================== N003a STAGE I CLINICAL PRETRAINING ==================")
    print(f"Dataset: PTB-XL (N={len(dataset)} records, strictly separated from evaluation cohorts)")
    print(f"Architecture: official GeoVT (nefnet_plus.layer) in super_mode='pretrain'")
    print(f"Variable Cardinality: L_input = 2 + k with k in {{1, 2, 3}} (sampled dynamically)")
    print(f"Anchors: Lead I (0) and Lead II (1) always in slots 0 and 1")
    print(f"Normalization: strictly observed-lead affine mapping (no query statistics leakage)")
    print(f"Batch Size: {args.batch_size}, Workers: {args.workers}, LR: {args.lr}")
    print(f"Output Directory: {args.output}")
    print(f"========================================================================\n")

    metrics_path = args.output / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_items = 0

        # Batch-level RNG for reproducible dynamic cardinality & slot sampling
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))

        for batch_idx, data_12lead in enumerate(loader):
            # data_12lead: [B, 12, 4608]
            B = data_12lead.size(0)

            # 1. Dynamically sample cardinality k in {1, 2, 3} -> L_input in {3, 4, 5}
            k = int(rng.choice([1, 2, 3]))
            available_additional = list(range(2, 12))
            selected_additional = rng.choice(available_additional, size=k, replace=False).tolist()
            input_indices = [0, 1] + sorted(selected_additional)

            unused_leads = [i for i in range(12) if i not in input_indices]
            target_idx = int(rng.choice(unused_leads))

            # 2. Extract input views and target view
            input_waveforms = data_12lead[:, input_indices, :].to(device, non_blocking=True)  # [B, L_in, 4608]
            target_waveform = data_12lead[:, target_idx : target_idx + 1, :].to(device, non_blocking=True)  # [B, 1, 4608]

            # 3. Prospective observed-lead normalization:
            # Min and max computed strictly across observed input slots!
            m_obs = input_waveforms.amin(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
            M_obs = input_waveforms.amax(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
            diff = torch.clamp(M_obs - m_obs, min=1e-6)

            input_norm = (input_waveforms - m_obs) / diff
            target_norm = (target_waveform - m_obs) / diff

            # Angles
            input_angles = angles_tensor[input_indices].unsqueeze(0).repeat(B, 1, 1)  # [B, L_in, 2]
            target_angle = angles_tensor[target_idx].unsqueeze(0).repeat(B, 1)        # [B, 2]

            optimizer.zero_grad(set_to_none=True)
            prediction = model(input_norm, input_angles, target_angle)
            loss = torch.nn.functional.l1_loss(prediction, target_norm)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.detach()) * B
            total_items += B

        scheduler.step()
        avg_l1 = total_loss / total_items
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{args.epochs:3d} | Train L1: {avg_l1:.5f} | LR: {current_lr:.6f}", flush=True)

        record = {"epoch": epoch, "train_l1": avg_l1, "lr": current_lr}
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
            "loader_rng_state": generator.get_state(),
        }
        atomic_save(checkpoint, args.output / "last_training_state.pt")
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            atomic_save(model.state_dict(), args.output / f"model_epoch_{epoch+1}.pt")
            atomic_save(model.state_dict(), args.output / "model_final.pt")

    print(f"\nN003a Stage I Pretraining complete! Final model saved to {args.output / 'model_final.pt'}")


if __name__ == "__main__":
    main()
