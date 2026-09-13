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
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=0.1)
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
    )
    model = instantiate_author_model(root / "author_code" / "nefnet_v2", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)
    start_epoch = 0

    args.output.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        required = {
            "epoch", "model", "optimizer", "scheduler", "rng_state",
            "cuda_rng_state", "loader_rng_state",
        }
        if set(state) != required:
            raise ValueError(f"resume checkpoint keys must be exactly {sorted(required)}")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        generator.set_state(state["loader_rng_state"])
        start_epoch = int(state["epoch"]) + 1

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
