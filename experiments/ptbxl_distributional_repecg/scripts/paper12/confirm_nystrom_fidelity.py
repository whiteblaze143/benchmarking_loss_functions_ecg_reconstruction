"""Confirm a frozen Phase-KME map on an untouched fold-7 pair sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cells(cache: Path, count: int, seed: int) -> list[np.ndarray]:
    frame = pd.read_csv(cache / "qc.csv")
    frame = frame[frame.eligible & (frame.fold == 7)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(frame), size=min(count, len(frame)), replace=False)
    cells: list[np.ndarray] = []
    for index in chosen:
        with np.load(cache / str(frame.iloc[index].artifact)) as item:
            beats = np.asarray(item["beats"], dtype=np.float32)
        shaped = beats.reshape(len(beats), 16, 16, 8)
        cells.extend(shaped[:, phase].reshape(-1, 8) for phase in range(16))
    rng.shuffle(cells)
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--kernel-fit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=1000)
    parser.add_argument("--records", type=int, default=256)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()

    with np.load(args.kernel_fit) as fit:
        mean = torch.as_tensor(fit["whitening_mean"], device="cuda", dtype=torch.float32)
        components = torch.as_tensor(fit["whitening_components"], device="cuda", dtype=torch.float32)
        scales = torch.as_tensor(fit["whitening_scales"], device="cuda", dtype=torch.float32)
        landmarks = torch.as_tensor(fit["landmarks"], device="cuda", dtype=torch.float32)
        inverse_root = torch.as_tensor(fit["inverse_root"], device="cuda", dtype=torch.float32)
        c2 = float(fit["c2"])
    cells = load_cells(args.cache, args.records, args.seed)
    rng = np.random.default_rng(args.seed)
    exact = np.empty(args.pairs, dtype=np.float64)
    approximate = np.empty(args.pairs, dtype=np.float64)
    with torch.inference_mode():
        for pair in range(args.pairs):
            left, right = rng.choice(len(cells), size=2, replace=False)
            x = (torch.as_tensor(cells[left], device="cuda") - mean) @ components * scales
            y = (torch.as_tensor(cells[right], device="cuda") - mean) @ components * scales
            exact[pair] = float(
                torch.rsqrt(torch.cdist(x, x).square() + c2).mean()
                + torch.rsqrt(torch.cdist(y, y).square() + c2).mean()
                - 2 * torch.rsqrt(torch.cdist(x, y).square() + c2).mean()
            )
            fx = torch.rsqrt(torch.cdist(x, landmarks).square() + c2) @ inverse_root
            fy = torch.rsqrt(torch.cdist(y, landmarks).square() + c2) @ inverse_root
            approximate[pair] = float((fx.mean(0) - fy.mean(0)).square().sum())

    positive = exact[exact > 0]
    epsilon = max(1e-8, 0.001 * float(np.median(positive)))
    relative = np.abs(approximate - exact) / np.maximum(exact, epsilon)
    rho = float(spearmanr(exact, approximate).statistic)
    quartiles = np.quantile(exact, [0, 0.25, 0.5, 0.75, 1])
    by_quartile = {}
    for index in range(4):
        upper = exact <= quartiles[index + 1] if index == 3 else exact < quartiles[index + 1]
        mask = (exact >= quartiles[index]) & upper
        by_quartile[str(index + 1)] = {
            "count": int(mask.sum()),
            "exact_min": float(quartiles[index]),
            "exact_max": float(quartiles[index + 1]),
            "median_relative_error": float(np.median(relative[mask])),
        }
    payload = {
        "kind": "paper12_nystrom_fold7_confirmatory_fidelity",
        "fold": 7,
        "pairs": args.pairs,
        "records_sampled": min(args.records, len(cells) // 16),
        "seed": args.seed,
        "kernel_fit_sha256": sha256(args.kernel_fit),
        "relative_error_definition": "abs(approx-exact)/max(exact,epsilon)",
        "epsilon": epsilon,
        "spearman": rho,
        "median_relative_error": float(np.median(relative)),
        "relative_error_q90": float(np.quantile(relative, 0.90)),
        "relative_error_q95": float(np.quantile(relative, 0.95)),
        "relative_error_q99": float(np.quantile(relative, 0.99)),
        "distance_quartiles": by_quartile,
        "passed": bool(rho >= 0.95 and np.median(relative) < 0.10),
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
