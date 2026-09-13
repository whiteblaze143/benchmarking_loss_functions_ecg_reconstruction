#!/usr/bin/env python3
"""G001-B: Evaluate N002 checkpoint with observation-only normalization.

Scientific contract:
  - Checkpoint: panobench_3view_variablethird_geovt_seed123/model_final.pt (frozen)
  - Same held-out test split: /data/mithunmanivannan/panobench/test (1030 records)
  - Identical sampling protocol (EVAL_SEED=42, 5 samples per record) to enable
    exact paired comparison with G001-A.
  - Normalization: strictly observation-only:
      m_obs = min_{j in {0, 1, third_input}, t} V_j(t)
      M_obs = max_{j in {0, 1, third_input}, t} V_j(t)
    No information from the other 41 unobserved views leaks into input scaling.
  - Metrics computed both in normalized scale and physical unscaled units,
    and paired differences (Delta = B - A) computed against G001-A results.

Output: results/g001_b_eval/g001_b_results.json
        results/g001_b_eval/g001_b_angular_strat.json
        results/g001_b_eval/g001_b_summary.json
        results/g001_b_eval/g001_b_paired_comparison.json
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

EVAL_SEED = 42
N_QUERY_SEEDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(pred: np.ndarray, target: np.ndarray, val_range: float = 1.0) -> float:
    mse = float(np.mean((pred - target) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10 * math.log10((val_range ** 2) / mse))


def ssim_1d(pred: np.ndarray, target: np.ndarray,
            window: int = 11, k1: float = 0.01, k2: float = 0.03,
            val_range: float = 1.0) -> float:
    assert pred.shape == target.shape and pred.ndim == 1
    T = len(pred)
    if T < window:
        return float("nan")
    C1 = (k1 * val_range) ** 2
    C2 = (k2 * val_range) ** 2
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
    return float(np.degrees(np.linalg.norm(a1 - a2)))


def evaluate_record_obs_only(mat_path: Path, model, rng: np.random.Generator,
                             n_samples: int) -> list[dict]:
    raw = scipy.io.loadmat(mat_path)["Panobench"][:44].astype(np.float32)
    if raw.shape != (44, 2500) or not np.isfinite(raw).all():
        raise ValueError(f"invalid record: {mat_path}")

    values_raw = _upsample2x(raw)  # (44, 5000)
    crop_start = int(rng.integers(0, 5000 - 4608 + 1))
    values_raw = values_raw[:, crop_start: crop_start + 4608]

    # Global min-max (needed only to replicate exact G001-A input in paired comparison)
    lo_all, hi_all = float(values_raw.min()), float(values_raw.max())
    span_all = hi_all - lo_all if hi_all > lo_all else 1.0

    results = []
    for _ in range(n_samples):
        third_input = int(rng.integers(2, 44))
        candidates = np.delete(np.arange(2, 44), third_input - 2)
        query_idx = int(rng.choice(candidates))
        input_indices = np.array([0, 1, third_input])

        # OBSERVED-ONLY NORMALIZATION:
        raw_obs = values_raw[input_indices]  # (3, 4608)
        lo_obs, hi_obs = float(raw_obs.min()), float(raw_obs.max())
        if not hi_obs > lo_obs:
            raise ValueError(f"constant observation in {mat_path}")
        span_obs = hi_obs - lo_obs

        # Model input normalized strictly using observed extrema
        inp_norm = (raw_obs - lo_obs) / span_obs

        inp_t = torch.from_numpy(inp_norm).unsqueeze(0).to(DEVICE)
        inp_angles = torch.from_numpy(
            PANOBENCH_ANGLES_RAD[input_indices]).unsqueeze(0).to(DEVICE)
        tgt_angle = torch.from_numpy(
            PANOBENCH_ANGLES_RAD[query_idx]).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()

        # Unscale prediction to physical space using observed scaling
        pred_unscaled = pred_norm * span_obs + lo_obs
        target_raw = values_raw[query_idx]

        # Target scaled into observation-relative [0, 1] frame
        target_obs_norm = (target_raw - lo_obs) / span_obs

        # Target and pred in author global frame (for direct metric comparison)
        pred_author_norm = (pred_unscaled - lo_all) / span_all
        target_author_norm = (target_raw - lo_all) / span_all

        # Metrics on author-normalized scale (directly comparable to G001-A numbers)
        l1_author_scale = float(np.mean(np.abs(pred_author_norm - target_author_norm)))
        psnr_author_scale = psnr(pred_author_norm, target_author_norm, val_range=1.0)
        ssim_author_scale = ssim_1d(pred_author_norm, target_author_norm, val_range=1.0)

        # Metrics on observation-normalized scale
        l1_obs_scale = float(np.mean(np.abs(pred_norm - target_obs_norm)))

        # Physical unscaled L1 (mV)
        l1_physical = float(np.mean(np.abs(pred_unscaled - target_raw)))

        input_angle_centroid = PANOBENCH_ANGLES_RAD[input_indices].mean(axis=0)
        ang_dist = angular_distance_deg(
            input_angle_centroid, PANOBENCH_ANGLES_RAD[query_idx])

        results.append({
            "record": mat_path.stem,
            "third_view_idx": third_input,
            "query_idx": query_idx,
            "l1": l1_author_scale,       # scale-matched to G001-A
            "psnr_db": psnr_author_scale, # scale-matched to G001-A
            "ssim": ssim_author_scale,    # scale-matched to G001-A
            "l1_obs_scale": l1_obs_scale,
            "l1_physical": l1_physical,
            "lo_obs": lo_obs,
            "hi_obs": hi_obs,
            "lo_all": lo_all,
            "hi_all": hi_all,
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
            "l1_physical_mean": float(np.mean([r["l1_physical"] for r in bucket])),
        })
    return bins


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/panobench_3view_variablethird_geovt_seed123/model_final.pt"))
    p.add_argument("--test-root", type=Path,
                   default=Path("/data/mithunmanivannan/panobench/test"))
    p.add_argument("--g001-a-results", type=Path,
                   default=Path("results/g001_eval/g001_results.json"))
    p.add_argument("--output", type=Path,
                   default=Path("results/g001_b_eval"))
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
    torch.backends.cudnn.enabled = False

    mat_files = sorted(args.test_root.glob("*.mat"), key=lambda p: int(p.stem))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files under {args.test_root}")
    if args.max_records:
        mat_files = mat_files[: args.max_records]

    rng = np.random.default_rng(EVAL_SEED)
    all_results: list[dict] = []
    for i, mat_path in enumerate(mat_files):
        recs = evaluate_record_obs_only(mat_path, model, rng, args.n_samples)
        all_results.extend(recs)
        if (i + 1) % 50 == 0 or i == len(mat_files) - 1:
            print(f"  {i+1}/{len(mat_files)} records, "
                  f"running L1={np.mean([r['l1'] for r in all_results]):.5f}, "
                  f"PSNR={np.mean([r['psnr_db'] for r in all_results]):.2f} dB")

    (args.output / "g001_b_results.json").write_text(
        json.dumps(all_results, indent=2))

    strat = angular_bins(all_results)
    (args.output / "g001_b_angular_strat.json").write_text(
        json.dumps(strat, indent=2))

    l1s = [r["l1"] for r in all_results]
    psnrs = [r["psnr_db"] for r in all_results]
    ssims = [r["ssim"] for r in all_results if not math.isnan(r["ssim"])]
    l1_phys = [r["l1_physical"] for r in all_results]

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
            "normalization": "observed_only (m_obs, M_obs from 3 input channels only; no unobserved view leakage)",
            "evaluation_scale": "author-equivalent scale [0, 1] for direct G001-A delta",
            "note": "G001-B evaluates realistic test-time deployment where unobserved leads cannot leak min/max.",
        },
        "l1": {"mean": float(np.mean(l1s)), "std": float(np.std(l1s)),
               "p5": float(np.percentile(l1s, 5)), "p95": float(np.percentile(l1s, 95))},
        "psnr_db": {"mean": float(np.mean(psnrs)), "std": float(np.std(psnrs))},
        "ssim": {"mean": float(np.mean(ssims)), "std": float(np.std(ssims))},
        "l1_physical": {"mean": float(np.mean(l1_phys)), "std": float(np.std(l1_phys))},
    }
    (args.output / "g001_b_summary.json").write_text(
        json.dumps(summary, indent=2))

    # Paired comparison with G001-A if available
    if args.g001_a_results.exists():
        g001_a_data = json.loads(args.g001_a_results.read_text())
        if len(g001_a_data) == len(all_results):
            delta_l1 = [b["l1"] - a["l1"] for a, b in zip(g001_a_data, all_results)]
            delta_psnr = [b["psnr_db"] - a["psnr_db"] for a, b in zip(g001_a_data, all_results)]
            delta_ssim = [b["ssim"] - a["ssim"] for a, b in zip(g001_a_data, all_results)]
            paired = {
                "n_pairs": len(delta_l1),
                "delta_l1 (B - A)": {
                    "mean": float(np.mean(delta_l1)),
                    "std": float(np.std(delta_l1)),
                    "p5": float(np.percentile(delta_l1, 5)),
                    "p95": float(np.percentile(delta_l1, 95)),
                },
                "delta_psnr_db (B - A)": {
                    "mean": float(np.mean(delta_psnr)),
                    "std": float(np.std(delta_psnr)),
                },
                "delta_ssim (B - A)": {
                    "mean": float(np.mean(delta_ssim)),
                    "std": float(np.std(delta_ssim)),
                },
                "verdict": (
                    "OBSERVED_NORMALIZATION_STABLE"
                    if float(np.mean(delta_l1)) < 0.005 and float(np.mean(delta_psnr)) > -3.0
                    else "SENSITIVE_TO_ALL_VIEW_LEAKAGE"
                ),
            }
            (args.output / "g001_b_paired_comparison.json").write_text(
                json.dumps(paired, indent=2))
            print("\n=== Paired Comparison (G001-B vs G001-A) ===")
            print(f"  Delta L1   (B - A): {paired['delta_l1 (B - A)']['mean']:+.6f}")
            print(f"  Delta PSNR (B - A): {paired['delta_psnr_db (B - A)']['mean']:+.2f} dB")
            print(f"  Delta SSIM (B - A): {paired['delta_ssim (B - A)']['mean']:+.4f}")
            print(f"  Verdict: {paired['verdict']}")

    print("\n=== G001-B Summary ===")
    print(f"  Records: {summary['n_records']} x {args.n_samples} = {len(all_results)} evals")
    print(f"  L1   mean={summary['l1']['mean']:.5f}  std={summary['l1']['std']:.5f}")
    print(f"  PSNR mean={summary['psnr_db']['mean']:.2f} dB")
    print(f"  SSIM mean={summary['ssim']['mean']:.4f}")
    print(f"\n  Output: {args.output}/")


if __name__ == "__main__":
    main()
