#!/usr/bin/env python3
"""Train the copied author Nef-Net v2 architecture from scratch on PanoBench."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from theta_repspat.panobench import ReleasedPanoBench
from theta_repspat.vendor import instantiate_author_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/data/mithunmanivannan/panobench/train"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def atomic_save(payload, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("invalid training arguments")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # cuDNN 9.2.x (bundled in PyTorch 2.6+cu124) is incompatible with the
    # system CUDA 13.0 driver (580.173.02): every conv throws ptrDesc->finalize().
    # Disable cuDNN entirely to fall back to the native CUDA conv path, which
    # is still GPU-accelerated and fully correct on A100.
    torch.backends.cudnn.enabled = False

    root = Path(__file__).resolve().parents[1]
    device = torch.device("cuda")
    dataset = ReleasedPanoBench(args.data_root, seed=args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        generator=generator,
        persistent_workers=args.workers > 0,
        prefetch_factor=4 if args.workers > 0 else None,
    )
    model = instantiate_author_model(root / "author_code" / "nefnet_v2", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.5
    )
    start_epoch = 0

    state: dict = {}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        required = {
            "epoch", "model", "optimizer", "scheduler", "rng_state",
            "cuda_rng_state", "loader_rng_state",
        }
        optional = {"scaler"}
        unknown = set(state) - required - optional
        if unknown:
            raise ValueError(f"unexpected resume checkpoint keys: {sorted(unknown)}")
        if not required.issubset(state):
            raise ValueError(f"missing resume checkpoint keys: {sorted(required - set(state))}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        generator.set_state(state["loader_rng_state"])
        start_epoch = int(state["epoch"]) + 1

    # ── throughput optimisations (no effect on training contract) ─────────
    # persistent_workers + prefetch_factor keep the A100 fed from disk.
    # torch.compile and AMP both trigger cuDNN descriptor bugs on GeoVT conv1d
    # layers (ptrDesc->finalize() under PyTorch 2.6 + CUDA 12.4) and are
    # intentionally omitted.

    metrics_path = args.output / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            inputs = batch["input"].to(device, non_blocking=True)
            input_angles = batch["input_angles"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            target_angles = batch["target_angle"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(inputs, input_angles, target_angles)
            loss = torch.nn.functional.l1_loss(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(inputs)
            total_items += len(inputs)
        scheduler.step()
        record = {"epoch": epoch, "train_l1": total_loss / total_items, "lr": scheduler.get_last_lr()[0]}
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
        if epoch == args.epochs - 1:
            atomic_save(model.state_dict(), args.output / "model_final.pt")
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
