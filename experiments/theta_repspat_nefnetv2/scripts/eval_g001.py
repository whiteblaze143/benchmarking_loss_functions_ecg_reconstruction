#!/usr/bin/env python3
"""G001: Evaluate N002 checkpoint on the held-out PanoBench test split.

Metrics: per-record L1, PSNR, SSIM.
Angular stratification: reconstruction quality vs angular distance between
input centroid and query target, to check whether GeoVT degrades for
far-from-input queries.

Scientific contract:
  - Checkpoint: panobench_fixed_triplet_randview3_geovt_seed123/model_final.pt
  - Architecture: nefnet_plus.layer, super_mode='optimization', lead_num=3
  - Third view: random from indices 2-43 per record (same as training)
  - Normalization: global min-max across all 44 channels (same as training)
  - This is a fixed-triplet random-third-view proof-of-concept, NOT Any-Pairs.
  - Test split is strictly held out: never seen during training.

Output: results/g001_eval/g001_results.json  (per-record)
        results/g001_eval/g001_angular_strat.json  (binned by query angle dist)
        results/g001_eval/g001_summary.json  (aggregate)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.io
import torch

from theta_repspat.panobench import PANOBENCH_ANGLES_RAD, _upsample2x
from theta_repspat.vendor import load_author_model

# ── constants ────────────────────────────────────────────────────────────────
EVAL_SEED = 42          # distinct from training seed 123
N_QUERY_SEEDS = 5       # number of random query/third-view samples per record
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Peak signal-to-noise ratio assuming signal range [0,1]."""
    mse = float(np.mean((pred - target) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * math.log10(1.0 / mse))


def ssim_1d(pred: np.ndarray, target: np.ndarray,
            window: int = 11, k1: float = 0.01, k2: float = 0.03) -> float:
    """Simplified 1-D SSIM for ECG signals. Input shape: (T,), values in [0,1]."""
    assert pred.shape == target.shape and pred.ndim == 1
    T = len(pred)
    if T < window:
        return float("nan")
    C1, C2 = k1 ** 2, k2 ** 2
    ssim_vals = []
    for i in range(0, T - window + 1, window // 2):
        p = pred[i: i + window].astype(np.float64)
        t = target[i: i + window].astype(np.float64)
        mu_p, mu_t = p.mean(), t.mean()
        sig_p = p.var()
        sig_t = t.var()
        sig_pt = ((p - mu_p) * (t - mu_t)).mean()
        num = (2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)
        den = (mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)
        ssim_vals.append(num / den if den != 0 else float("nan"))
    valid = [v for v in ssim_vals if not math.isnan(v)]
    return float(np.mean(valid)) if valid else float("nan")


def angular_distance_deg(a1: np.ndarray, a2: np.ndarray) -> float:
    """Euclidean distance in (theta,phi) angle space, converted to degrees."""
    return float(np.degrees(np.linalg.norm(a1 - a2)))


def evaluate_record(mat_path: Path, model, rng: np.random.Generator,
                    n_samples: int) -> list[dict]:
    """Evaluate one .mat record with n_samples random (third-view, query) pairs."""
    raw = scipy.io.loadmat(mat_path)["Panobench"][:44].astype(np.float32)
    if raw.shape != (44, 2500) or not np.isfinite(raw).all():
        raise ValueError(f"invalid record: {mat_path}")

    values = _upsample2x(raw)   # (44, 5000)
    crop_start = int(rng.integers(0, 5000 - 4608 + 1))
    values = values[:, crop_start: crop_start + 4608]
    lo, hi = float(values.min()), float(values.max())
    if not hi > lo:
        raise ValueError(f"constant record: {mat_path}")
    values = (values - lo) / (hi - lo)  # global norm across all 44 (same as training)

    results = []
    for _ in range(n_samples):
        third_input = int(rng.integers(2, 44))
        candidates = np.delete(np.arange(2, 44), third_input - 2)
        query_idx = int(rng.choice(candidates))
        input_indices = np.array([0, 1, third_input])

        inp = torch.from_numpy(values[input_indices]).unsqueeze(0).to(DEVICE)
        inp_angles = torch.from_numpy(
            PANOBENCH_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)
        tgt_angle = torch.from_numpy(
            PANOBENCH_ANGLES_RAD[query_idx]).unsqueeze(0).to(DEVICE)
        target_np = values[query_idx]

        with torch.no_grad():
            pred = model(inp, inp_angles, tgt_angle).squeeze().cpu().numpy()

        l1 = float(np.mean(np.abs(pred - target_np)))
        ps = psnr(pred, target_np)
        ss = ssim_1d(pred, target_np)
        input_angle_centroid = PANOBENCH_ANGLES_RAD[input_indices].mean(axis=0)
        ang_dist = angular_distance_deg(
            input_angle_centroid, PANOBENCH_ANGLES_RAD[query_idx])

        results.append({
            "record": mat_path.stem,
            "third_view_idx": third_input,
            "query_idx": query_idx,
            "l1": l1,
            "psnr_db": ps,
            "ssim": ss,
            "query_ang_dist_deg": ang_dist,
        })
    return results


def angular_bins(records: list[dict], n_bins: int = 6) -> list[dict]:
    dists = [r["query_ang_dist_deg"] for r in records]
    edges = np.linspace(0, max(dists) + 1e-6, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = [r for r in records if lo <= r["query_ang_dist_deg"] < hi]
        if not bucket:
            continue
        bins.append({
            "ang_dist_range_deg": [round(lo, 1), round(hi, 1)],
            "n": len(bucket),
            "l1_mean": float(np.mean([r["l1"] for r in bucket])),
            "l1_std": float(np.std([r["l1"] for r in bucket])),
            "psnr_mean_db": float(np.mean([r["psnr_db"] for r in bucket])),
            "ssim_mean": float(np.mean([r["ssim"] for r in bucket])),
        })
    return bins


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/panobench_fixed_triplet_randview3_geovt_seed123/model_final.pt"))
    p.add_argument("--test-root", type=Path,
                   default=Path("/data/mithunmanivannan/panobench/test"))
    p.add_argument("--output", type=Path,
                   default=Path("results/g001_eval"))
    p.add_argument("--n-samples", type=int, default=N_QUERY_SEEDS)
    p.add_argument("--max-records", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    root = Path(__file__).resolve().parents[1]
    model = load_author_model(
        root / "author_code" / "nefnet_v2",
        args.checkpoint,
        DEVICE,
    )
    torch.backends.cudnn.enabled = False   # cuDNN compat workaround

    mat_files = sorted(args.test_root.glob("*.mat"), key=lambda p: int(p.stem))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files under {args.test_root}")
    if args.max_records:
        mat_files = mat_files[: args.max_records]

    rng = np.random.default_rng(EVAL_SEED)
    all_results: list[dict] = []
    for i, mat_path in enumerate(mat_files):
        recs = evaluate_record(mat_path, model, rng, args.n_samples)
        all_results.extend(recs)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(mat_files)} records, "
                  f"running L1={np.mean([r['l1'] for r in all_results]):.5f}")

    (args.output / "g001_results.json").write_text(
        json.dumps(all_results, indent=2))

    strat = angular_bins(all_results)
    (args.output / "g001_angular_strat.json").write_text(
        json.dumps(strat, indent=2))

    l1s = [r["l1"] for r in all_results]
    psnrs = [r["psnr_db"] for r in all_results]
    ssims = [r["ssim"] for r in all_results if not math.isnan(r["ssim"])]
    summary = {
        "checkpoint": str(args.checkpoint),
        "test_root": str(args.test_root),
        "n_records": len(mat_files),
        "n_samples_per_record": args.n_samples,
        "n_total_evaluations": len(all_results),
        "eval_seed": EVAL_SEED,
        "scientific_contract": {
            "model": "nefnet_plus.layer",
            "super_mode": "optimization",
            "lead_num": 3,
            "third_view": "random from PanoBench indices 2-43 (same as training)",
            "normalization": "global min-max across all 44 channels (author-faithful, mild leakage from unobserved views)",
            "any_pairs": False,
            "note": "Fixed-triplet random-third-view proof-of-concept. NOT Any-Pairs. NOT the final frozen Theta-repSpat representation.",
        },
        "l1": {"mean": float(np.mean(l1s)), "std": float(np.std(l1s)),
               "p5": float(np.percentile(l1s, 5)), "p95": float(np.percentile(l1s, 95))},
        "psnr_db": {"mean": float(np.mean(psnrs)), "std": float(np.std(psnrs))},
        "ssim": {"mean": float(np.mean(ssims)), "std": float(np.std(ssims))},
    }
    (args.output / "g001_summary.json").write_text(
        json.dumps(summary, indent=2))

    print("\n=== G001 Summary ===")
    print(f"  Records: {summary['n_records']} x {args.n_samples} = {len(all_results)} evals")
    print(f"  L1   mean={summary['l1']['mean']:.5f}  std={summary['l1']['std']:.5f}")
    print(f"  PSNR mean={summary['psnr_db']['mean']:.2f} dB")
    print(f"  SSIM mean={summary['ssim']['mean']:.4f}")
    print(f"\n  Angular stratification:")
    for b in strat:
        print(f"    {b['ang_dist_range_deg'][0]:5.1f}–{b['ang_dist_range_deg'][1]:5.1f}°: "
              f"L1={b['l1_mean']:.5f}  PSNR={b['psnr_mean_db']:.2f}dB  n={b['n']}")
    print(f"\n  Output: {args.output}/")


if __name__ == "__main__":
    main()
