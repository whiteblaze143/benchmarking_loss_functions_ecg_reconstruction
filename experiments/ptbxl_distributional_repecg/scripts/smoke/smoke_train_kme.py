from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from repecg.common.kernels import NystromMap, WhiteningTransform
from repecg.common.models import PhaseCNN


def load_cache(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    manifest = json.loads((path / "manifest.json").read_text())
    beats = []
    labels = []
    for row in manifest["rows"]:
        if not row["eligible"]:
            continue
        item = np.load(path / row["artifact"])
        beats.append(np.asarray(item["beats"], dtype=np.float64))
        labels.append(np.asarray(item["labels"], dtype=np.float32))
    if len(beats) < 16:
        raise ValueError("smoke cache has too few eligible records")
    return beats, np.stack(labels)


def representations(beats: list[np.ndarray], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    atoms = np.concatenate([value.reshape(-1, 8) for value in beats], axis=0)
    reservoir = atoms[rng.choice(len(atoms), size=min(50_000, len(atoms)), replace=False)]
    whitening = WhiteningTransform.fit(reservoir)
    mapping = NystromMap.fit(whitening.transform(reservoir), landmarks=128, seed=seed)
    kernel_records = []
    moment_records = []
    for value in beats:
        cells = value.reshape(value.shape[0], 16, 16, 8)
        phase = cells.transpose(1, 0, 2, 3).reshape(16, -1, 8)
        kernel_records.append(np.stack([mapping.mean(whitening.transform(cell)) for cell in phase]))
        moment_records.append(np.concatenate((phase.mean(axis=1), phase.std(axis=1)), axis=1))
    return np.asarray(kernel_records, dtype=np.float32), np.asarray(moment_records, dtype=np.float32)


def train_variant(x: np.ndarray, y: np.ndarray, name: str, epochs: int) -> dict[str, float | str]:
    device = torch.device("cuda")
    model = PhaseCNN(x.shape[-1]).to(device)
    features = torch.from_numpy(x).to(device)
    targets = torch.from_numpy(y).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    final = float("nan")
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = float(loss.detach())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "variant": name,
        "records": len(x),
        "epochs": epochs,
        "loss": final,
        "elapsed_seconds": elapsed,
        "record_epochs_per_second": len(x) * epochs / elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    # cuDNN 9 Conv1d fails with ptrDesc->finalize() on this A100 host.
    # CUDA ATen remains enabled and is independently witnessed by the env gate.
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    beats, labels = load_cache(args.cache)
    kernel, moments = representations(beats, args.seed)
    results = [
        train_variant(kernel, labels, "imq_kme", args.epochs),
        train_variant(moments, labels, "mean_std", args.epochs),
    ]
    payload = {
        "kind": "non_scientific_m1_gpu_smoke",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
