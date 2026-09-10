#!/usr/bin/env python3
"""Self-supervised pretraining for the beat-structured ECG-AIM embedding."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from unified_latents.engineering.models.ecg_aim_lvcg_variant import LVCGStyleLeadIEmbedding


class BeatBoundaryDataset(Dataset):
    def __init__(
        self, tensor_dir: Path, index_path: Path, limit: int | None = None,
        boundary_control: str = "detected",
    ) -> None:
        rows = [json.loads(line) for line in index_path.read_text().splitlines()]
        if limit is not None:
            rows = rows[:limit]
        if not rows:
            raise ValueError(f"empty beat-boundary index: {index_path}")
        self.tensor_dir = tensor_dir
        self.rows = rows
        if boundary_control not in {"detected", "uniform"}:
            raise ValueError("boundary_control must be 'detected' or 'uniform'")
        self.boundary_control = boundary_control

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        waveform = torch.load(
            self.tensor_dir / f"{row['record_id']}.pt", map_location="cpu", weights_only=True
        )
        if waveform.shape != (12, 5000):
            raise ValueError(f"record {row['record_id']} has shape {tuple(waveform.shape)}")
        bounds = row["beat_bounds"]
        if self.boundary_control == "uniform":
            edges = torch.linspace(0, waveform.shape[-1], len(bounds) + 1).round().long()
            bounds = torch.stack((edges[:-1], edges[1:]), dim=1)
        else:
            bounds = torch.tensor(bounds, dtype=torch.long)
        return waveform[0:1].float(), bounds


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    leads, bounds = zip(*batch)
    maximum = max(item.shape[0] for item in bounds)
    padded = torch.zeros(len(batch), maximum, 2, dtype=torch.long)
    mask = torch.zeros(len(batch), maximum, dtype=torch.bool)
    for index, item in enumerate(bounds):
        padded[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return torch.stack(leads), padded, mask


def atomic_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def epoch(model, loader, device, optimizer=None) -> dict[str, float]:
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "beat": 0.0, "temporal": 0.0, "records": 0}
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for lead_i, bounds, mask in loader:
            lead_i, bounds, mask = lead_i.to(device), bounds.to(device), mask.to(device)
            result = model(lead_i, bounds, mask)
            loss = result["embedding_ssl_loss"]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            count = lead_i.shape[0]
            totals["loss"] += float(loss) * count
            totals["beat"] += float(result["beat_reconstruction_loss"]) * count
            totals["temporal"] += float(result["temporal_prediction_loss"]) * count
            totals["records"] += count
    records = int(totals.pop("records"))
    return {name: value / records for name, value in totals.items()} | {"records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--bounds-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--small-model", action="store_true", help="Reduced architecture for smoke tests only")
    parser.add_argument("--field-mode", choices=("learned", "triplicate"), default="learned")
    parser.add_argument("--boundary-control", choices=("detected", "uniform"), default="detected")
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Matches the main ECG-AIM trainer on this cuDNN 9/A100 stack.
        torch.backends.cudnn.enabled = False
    train_data = BeatBoundaryDataset(
        args.tensor_root / "train", args.bounds_root / "train.jsonl",
        args.limit_per_split, args.boundary_control,
    )
    val_data = BeatBoundaryDataset(
        args.tensor_root / "val", args.bounds_root / "val.jsonl",
        args.limit_per_split, args.boundary_control,
    )
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )
    model_kwargs = {"field_mode": args.field_mode}
    if args.small_model:
        model_kwargs = {
            "beat_len": 32, "state_dim": 16, "rhythm_dim": 8, "stem_channels": 8,
            "stage_channels": (8, 8), "stage_blocks": (1, 1), "dropout": 0.0,
            "field_mode": args.field_mode,
        }
    model = LVCGStyleLeadIEmbedding(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for number in range(1, args.epochs + 1):
        train_metrics = epoch(model, train_loader, device, optimizer)
        val_metrics = epoch(model, val_loader, device)
        row = {"epoch": number, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        atomic_save({"model_state_dict": model.state_dict(), "model_kwargs": model_kwargs, "epoch": number}, args.output_dir / "last.pt")
        if val_metrics["loss"] < best:
            best = val_metrics["loss"]
            atomic_save(
                {"model_state_dict": model.state_dict(), "model_kwargs": model_kwargs, "epoch": number,
                 "val_embedding_ssl_loss": best},
                args.output_dir / "best.pt",
            )
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")


if __name__ == "__main__":
    main()
