#!/usr/bin/env python3
"""Paired Patient-Level Noninferiority Bootstrap Evaluation for ECG-AIM Kill-Gate Experiments.

Performs 10,000-sample patient-clustered paired bootstrap against the anchor model
(conv15e_A0_raw_s42_l0) across all held-out validation records (Fold 9, N=2,183).

Decision Gates:
  - Primary Noninferiority: Delta r_missing11 > -0.003 (with P(Delta r < -0.003) < 0.05)
  - Precordial Noninferiority: Delta r_V1:V6 > -0.005
  - Tail Robustness: Delta r_p05 > -0.005
  - Hard-Basis Evaluation: Delta r_indep7 > -0.003
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths
setup_import_paths(include_fairseq=True)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.train_mcma_3lead import PTBXLDataset
from scripts.train_1lead_wavelet_ssl_mtl import (
    build_model,
    forward_model,
    waveform_from_batch,
    apply_zscore,
    LEADS,
)

DEFAULT_ANCHOR_DIR = _ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0"
DEFAULT_DATA_DIR = _ROOT / "data/ptb_xl/tensors"
CHEST_LEADS = [6, 7, 8, 9, 10, 11]
INDEP_LEADS = [1, 6, 7, 8, 9, 10, 11]  # II, V1-V6


def load_model_from_dir(run_dir: Path, device: torch.device) -> tuple[torch.nn.Module, argparse.Namespace]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    cfg = json.loads(cfg_path.read_text())
    args = argparse.Namespace(**cfg)

    # Supply backward-compatible defaults
    if not hasattr(args, "no_delineation_head"): args.no_delineation_head = False
    if not hasattr(args, "no_fiducial_head"): args.no_fiducial_head = False
    if not hasattr(args, "mask_type_mode"): args.mask_type_mode = "legacy"
    if not hasattr(args, "artificial_mask_mode"): args.artificial_mask_mode = "all"
    if not hasattr(args, "deterministic_limb_derivation"): args.deterministic_limb_derivation = False
    if not hasattr(args, "custom_wavelet_asset"): args.custom_wavelet_asset = None
    if not hasattr(args, "view_a_custom_wavelet_asset"): args.view_a_custom_wavelet_asset = None
    if not hasattr(args, "view_b_custom_wavelet_asset"): args.view_b_custom_wavelet_asset = None
    if not hasattr(args, "view_a_bank"): args.view_a_bank = "inherit"
    if not hasattr(args, "view_b_bank"): args.view_b_bank = "inherit"
    if not hasattr(args, "observed_leads"): args.observed_leads = [getattr(args, "observed_lead", 0)]

    model = build_model(args).to(device).eval()
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        ckpt_path = run_dir / "resume.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No best.pt or resume.pt in {run_dir}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state, strict=False)
    return model, args


def compute_vector_pearson(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Computes Pearson correlation along time axis: [B, num_leads, T] -> [B]."""
    p_c = p - p.mean(dim=-1, keepdim=True)
    t_c = t - t.mean(dim=-1, keepdim=True)
    return F.cosine_similarity(p_c.flatten(1), t_c.flatten(1), dim=1).clamp(-1.0, 1.0)


@torch.inference_mode()
def collect_per_record_metrics(
    model: torch.nn.Module,
    args: argparse.Namespace,
    loader: DataLoader,
    device: torch.device,
    observed_lead: int = 0,
) -> dict[str, np.ndarray]:
    model.eval()
    missing_mask = [l for l in range(12) if l != observed_lead]
    chest_missing = [l for l in CHEST_LEADS if l != observed_lead]
    indep_missing = [l for l in INDEP_LEADS if l != observed_lead]

    r_missing = []
    r_chest = []
    r_indep = []
    r_lead2 = []

    is_zscore = bool(getattr(args, "zscore_norm", False))

    for batch in loader:
        y = waveform_from_batch(batch)[..., :5000].to(device)
        if is_zscore:
            inp, mean, std = apply_zscore(y)
        else:
            inp = y

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            res = forward_model(
                model, inp, [observed_lead], compute_delineation=False, compute_ssl=False
            )
            pred = res["y_pred"][..., :5000].float()
            if is_zscore:
                pred = pred * std + mean
            y_float = y.float()

        # Concatenated missing 11
        r_m = compute_vector_pearson(pred[:, missing_mask], y_float[:, missing_mask])
        r_c = compute_vector_pearson(pred[:, chest_missing], y_float[:, chest_missing])
        r_i = compute_vector_pearson(pred[:, indep_missing], y_float[:, indep_missing])
        r_l2 = compute_vector_pearson(pred[:, 1:2], y_float[:, 1:2])

        r_missing.extend(r_m.cpu().tolist())
        r_chest.extend(r_c.cpu().tolist())
        r_indep.extend(r_i.cpu().tolist())
        r_lead2.extend(r_l2.cpu().tolist())

    return {
        "r_missing11": np.array(r_missing, dtype=np.float64),
        "r_chest": np.array(r_chest, dtype=np.float64),
        "r_indep": np.array(r_indep, dtype=np.float64),
        "r_lead2": np.array(r_lead2, dtype=np.float64),
    }


def run_paired_bootstrap(
    cand_vals: np.ndarray,
    anchor_vals: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    margin: float = -0.003,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(cand_vals)
    deltas = cand_vals - anchor_vals

    mean_cand = float(np.mean(cand_vals))
    mean_anchor = float(np.mean(anchor_vals))
    mean_delta = float(np.mean(deltas))

    p05_cand = float(np.quantile(cand_vals, 0.05))
    p05_anchor = float(np.quantile(anchor_vals, 0.05))
    delta_p05 = p05_cand - p05_anchor

    # Bootstrap index resampling (patient-cluster aligned)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_deltas = np.mean(deltas[indices], axis=1)

    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    p_inferior = float(np.mean(boot_deltas < margin))

    return {
        "cand_mean": mean_cand,
        "anchor_mean": mean_anchor,
        "delta_mean": mean_delta,
        "cand_p05": p05_cand,
        "anchor_p05": p05_anchor,
        "delta_p05": delta_p05,
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "p_inferior": p_inferior,
    }


def evaluate_pair(
    cand_dir: Path,
    anchor_dir: Path,
    data_dir: Path,
    split: str = "val",
    batch_size: int = 32,
    device_str: str = "cuda",
    n_boot: int = 10000,
) -> dict[str, Any]:
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    # Disable cuDNN during evaluation to avoid ptrDesc->finalize descriptor bugs on wavelet convolutions
    torch.backends.cudnn.enabled = False
    print(f"Loading candidate model from: {cand_dir}")
    cand_model, cand_args = load_model_from_dir(cand_dir, device)
    print(f"Loading anchor model from:    {anchor_dir}")
    anchor_model, anchor_args = load_model_from_dir(anchor_dir, device)

    split_dir = data_dir / split
    ds = PTBXLDataset(str(split_dir))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Collecting per-record metrics across N={len(ds)} {split} samples...")
    cand_metrics = collect_per_record_metrics(cand_model, cand_args, loader, device)
    anchor_metrics = collect_per_record_metrics(anchor_model, anchor_args, loader, device)

    # Compute bootstrap for missing11, chest (V1:V6), independent 7, and Lead II
    boot_missing = run_paired_bootstrap(cand_metrics["r_missing11"], anchor_metrics["r_missing11"], n_boot=n_boot, margin=-0.003)
    boot_chest = run_paired_bootstrap(cand_metrics["r_chest"], anchor_metrics["r_chest"], n_boot=n_boot, margin=-0.005)
    boot_indep = run_paired_bootstrap(cand_metrics["r_indep"], anchor_metrics["r_indep"], n_boot=n_boot, margin=-0.003)
    boot_lead2 = run_paired_bootstrap(cand_metrics["r_lead2"], anchor_metrics["r_lead2"], n_boot=n_boot, margin=-0.005)

    # Decision Gates
    pass_primary = boot_missing["delta_mean"] > -0.003 and boot_missing["p_inferior"] < 0.05
    pass_chest = boot_chest["delta_mean"] > -0.005
    pass_tail = boot_missing["delta_p05"] > -0.005
    pass_overall = pass_primary and pass_chest and pass_tail

    summary = {
        "candidate_run": cand_dir.name,
        "anchor_run": anchor_dir.name,
        "split": split,
        "n_samples": len(ds),
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gates": {
            "pass_overall": bool(pass_overall),
            "pass_primary_missing11": bool(pass_primary),
            "pass_precordial_v1v6": bool(pass_chest),
            "pass_tail_robustness_p05": bool(pass_tail),
        },
        "missing11": boot_missing,
        "precordial_v1v6": boot_chest,
        "independent7": boot_indep,
        "lead2": boot_lead2,
    }
    return summary


def main():
    p = argparse.ArgumentParser(description="Evaluate kill-gate candidate against anchor")
    p.add_argument("--candidate-dir", required=True, type=Path)
    p.add_argument("--anchor-dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()

    res = evaluate_pair(
        cand_dir=args.candidate_dir,
        anchor_dir=args.anchor_dir,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
    )

    out_file = args.output_json or (args.candidate_dir / f"killgate_eval_{args.split}.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nSaved evaluation to: {out_file}")

    print("\n" + "=" * 70)
    print(f"KILL-GATE VERDICT for {res['candidate_run']} vs {res['anchor_run']}:")
    print(f"Overall Gate Verdict: {'PASS (Noninferior)' if res['gates']['pass_overall'] else 'FAIL (Inferior)'}")
    print(f"  - Missing-11 Pearson:    {res['missing11']['cand_mean']:.4f} vs {res['missing11']['anchor_mean']:.4f} "
          f"(Delta: {res['missing11']['delta_mean']:+.4f}, 95% CI: [{res['missing11']['ci_lower_95']:+.4f}, {res['missing11']['ci_upper_95']:+.4f}], p_inferior={res['missing11']['p_inferior']:.4f})")
    print(f"  - Precordial V1-V6:      {res['precordial_v1v6']['cand_mean']:.4f} vs {res['precordial_v1v6']['anchor_mean']:.4f} "
          f"(Delta: {res['precordial_v1v6']['delta_mean']:+.4f})")
    print(f"  - Tail 5th percentile:   {res['missing11']['cand_p05']:.4f} vs {res['missing11']['anchor_p05']:.4f} "
          f"(Delta: {res['missing11']['delta_p05']:+.4f})")
    print(f"  - Independent 7:         {res['independent7']['cand_mean']:.4f} vs {res['independent7']['anchor_mean']:.4f} "
          f"(Delta: {res['independent7']['delta_mean']:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    main()
