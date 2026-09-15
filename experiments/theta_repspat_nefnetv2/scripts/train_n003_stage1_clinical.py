#!/usr/bin/env python3
"""Clean Stage I clinical Any-Pairs pretraining for N003a on PTB-XL."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

exp_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(exp_root / "src"))
from theta_repspat.clinical_dataset import (  # noqa: E402
    CANONICAL_12LEAD_ANGLES_RAD, NORMALIZATION_EPS, PRECORDIAL_LEAD_INDICES,
    PTBXL12LeadDataset, TRAIN_LOWER_MV, TRAIN_UPPER_MV, normalize_fixed_train_bounds,
)
from theta_repspat.vendor import instantiate_author_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Clean Stage I clinical pretraining for N003a")
    parser.add_argument("--tensors-dir", type=Path, default=Path("/home/mithunmanivannan/data/ptb_xl/tensors"))
    parser.add_argument("--output", type=Path, default=exp_root / "results/n003a_stage1_fixed_norm_seed123")
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


def sample_problem(rng: np.random.Generator, k: int, query_idx: int | None = None):
    query_idx = int(rng.choice(PRECORDIAL_LEAD_INDICES)) if query_idx is None else query_idx
    candidates = [idx for idx in PRECORDIAL_LEAD_INDICES if idx != query_idx]
    observed = [0, 1] + sorted(rng.choice(candidates, size=k, replace=False).tolist())
    return observed, query_idx


def expected_validation_manifest(dataset, seed: int) -> list[dict]:
    cells = [(query_idx, k) for query_idx in PRECORDIAL_LEAD_INDICES for k in (1, 2, 3)]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x4E303033]))
    rng.shuffle(cells)
    manifest = []
    for record_index, filepath in enumerate(dataset.files):
        query_idx, k = cells[record_index % len(cells)]
        observed, _ = sample_problem(rng, k, query_idx)
        manifest.append({"record_index": record_index, "record_id": filepath.stem,
                         "query_index": query_idx, "k": k, "observed_indices": observed})
    return manifest


def freeze_validation_manifest(dataset, path: Path, seed: int) -> list[dict]:
    expected = expected_validation_manifest(dataset, seed)
    if path.exists():
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"Existing validation manifest violates frozen contract: {path}")
        return actual
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return expected


def ensure_query_dimension(query_angles: torch.Tensor) -> torch.Tensor:
    if query_angles.dim() == 2:
        query_angles = query_angles.unsqueeze(1)
    return query_angles


def evaluate(model, loader, manifest, angles, device):
    sums, counts = defaultdict(float), defaultdict(int)
    model.eval()
    offset = 0
    with torch.no_grad():
        for waveforms in loader:
            batch_manifest = manifest[offset:offset + len(waveforms)]
            offset += len(waveforms)
            waveforms = waveforms.to(device, non_blocking=True)
            for k in (1, 2, 3):
                positions = [i for i, item in enumerate(batch_manifest) if item["k"] == k]
                if not positions:
                    continue
                items = [batch_manifest[i] for i in positions]
                rows = torch.tensor(positions, device=device)
                observed = torch.tensor([item["observed_indices"] for item in items], device=device)
                queries = torch.tensor([item["query_index"] for item in items], device=device)
                selected = waveforms[rows]
                inputs = torch.gather(selected, 1, observed[:, :, None].expand(-1, -1, selected.shape[-1]))
                targets = torch.gather(selected, 1, queries[:, None, None].expand(-1, 1, selected.shape[-1]))
                prediction = model(normalize_fixed_train_bounds(inputs), angles[observed],
                                   ensure_query_dimension(angles[queries]))
                losses = torch.nn.functional.l1_loss(
                    prediction, normalize_fixed_train_bounds(targets), reduction="none").mean((1, 2))
                for item, loss in zip(items, losses.tolist()):
                    cell = (item["query_index"], item["k"])
                    sums[cell] += loss
                    counts[cell] += 1
    cells = {(query_idx, k) for query_idx in PRECORDIAL_LEAD_INDICES for k in (1, 2, 3)}
    if set(counts) != cells:
        raise RuntimeError("Validation manifest does not cover all 18 query/cardinality cells")
    means = {cell: sums[cell] / counts[cell] for cell in cells}
    balanced = sum(means.values()) / 18
    per_lead = {f"V{q - 1}": sum(means[q, k] for k in (1, 2, 3)) / 3
                for q in PRECORDIAL_LEAD_INDICES}
    per_cardinality = {str(k): sum(means[q, k] for q in PRECORDIAL_LEAD_INDICES) / 6
                       for k in (1, 2, 3)}
    return balanced, per_lead, per_cardinality


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    if args.resume is None and any((args.output / name).exists() for name in
                                   ("train_metrics.jsonl", "last_training_state.pt", "model_final.pt")):
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda")
    args.output.mkdir(parents=True, exist_ok=True)
    train_dataset = PTBXL12LeadDataset(args.tensors_dir, "train", args.seed)
    val_dataset = PTBXL12LeadDataset(args.tensors_dir, "val", args.seed)
    manifest = freeze_validation_manifest(val_dataset, args.output / "validation_manifest.json", args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    common = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": True,
              "prefetch_factor": 4 if args.workers > 0 else None}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, generator=generator, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    model = instantiate_author_model(exp_root / "author_code/nefnet_v2", device, super_mode="pretrain")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.5)
    start_epoch, best_val_l1 = 0, float("inf")
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); torch.set_rng_state(state["rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state"]); generator.set_state(state["loader_rng_state"])
        best_val_l1 = float(state["best_val_l1"]); start_epoch = int(state["epoch"]) + 1
    angles = torch.from_numpy(CANONICAL_12LEAD_ANGLES_RAD).to(device)
    print("================ N003a CLEAN STAGE I PRETRAINING ================")
    print(f"Split: train={len(train_dataset)}, val={len(val_dataset)}, test=untouched")
    print("Independent sampling: anchors I/II; observed/query leads V1-V6 only")
    print(f"Normalization: clip((x-{TRAIN_LOWER_MV})/{TRAIN_UPPER_MV-TRAIN_LOWER_MV}, {NORMALIZATION_EPS}, {1-NORMALIZATION_EPS})")
    print("Validation: frozen panel; equal mean over 6 query x 3 cardinality cells")
    print(f"Output: {args.output}", flush=True)
    metrics_path = args.output / "train_metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        train_dataset.set_epoch(epoch); model.train(); total_loss = 0.0; total_items = 0
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, epoch]))
        for data in train_loader:
            batch_size = data.shape[0]; k = int(rng.choice([1, 2, 3])); observed, query = sample_problem(rng, k)
            inputs = data[:, observed].to(device, non_blocking=True)
            targets = data[:, query:query + 1].to(device, non_blocking=True)
            input_angles = angles[observed].unsqueeze(0).repeat(batch_size, 1, 1)
            query_angles = ensure_query_dimension(angles[query].unsqueeze(0).repeat(batch_size, 1))
            optimizer.zero_grad(set_to_none=True)
            prediction = model(normalize_fixed_train_bounds(inputs), input_angles, query_angles)
            loss = torch.nn.functional.l1_loss(prediction, normalize_fixed_train_bounds(targets))
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * batch_size; total_items += batch_size
        scheduler.step()
        record = {"epoch": epoch, "train_l1": total_loss / total_items, "lr": scheduler.get_last_lr()[0]}
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            val_l1, per_lead, per_cardinality = evaluate(model, val_loader, manifest, angles, device)
            record.update(val_balanced_l1=val_l1, val_l1_per_lead=per_lead,
                          val_l1_per_cardinality=per_cardinality)
            if val_l1 < best_val_l1:
                best_val_l1 = val_l1; atomic_save(model.state_dict(), args.output / "model_best.pt")
            atomic_save(model.state_dict(), args.output / f"model_epoch_{epoch + 1}.pt")
            atomic_save(model.state_dict(), args.output / "model_final.pt")
            print(f"Epoch {epoch+1:3d}/{args.epochs:3d} | train={record['train_l1']:.5f} | val={val_l1:.5f} | lr={record['lr']:.6f}", flush=True)
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs:3d} | train={record['train_l1']:.5f} | lr={record['lr']:.6f}", flush=True)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        atomic_save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "rng_state": torch.get_rng_state(),
                     "cuda_rng_state": torch.cuda.get_rng_state_all(), "loader_rng_state": generator.get_state(),
                     "best_val_l1": best_val_l1}, args.output / "last_training_state.pt")
    print(f"N003a Stage I pretraining complete: {args.output / 'model_final.pt'}")


if __name__ == "__main__":
    main()
