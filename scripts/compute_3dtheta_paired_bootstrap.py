#!/usr/bin/env python3
"""Patient-level paired uncertainty analysis and bootstrap estimation for Stage A/A.1 3D Theta models.

Computes:
1. Patient-level metric distributions across N=2,183 held-out validation patients.
2. Paired differences Delta_i = r_i(Cell A) - r_i(Cell B) for key causal contrasts:
   - Geometry effect: D3 - D5
   - Multiplicative fusion on theta: D3 - D6
   - Multiplicative fusion on learned codes: D4 - D7
   - Trigonometric structure vs random: D3 - D8
   - Trigonometric code vs learned code: D3 - D4
3. 10,000-sample bootstrap 95% Confidence Intervals: CI_95%[E(Delta_i)].
4. Exports paired bootstrap summary to 3DTHETA_BOOTSTRAP_REPORT.md and JSON.
"""

from __future__ import annotations
import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from unified_latents.engineering.models.three_d_theta_reconstruction import (
    LEAD_NAMES,
    ThreeDThetaECGAIM,
)
from scripts.train_3dtheta_ablation import CELL_CONFIGS, SimplePTBXLDataset

OUTPUT_MD = ROOT / "3DTHETA_BOOTSTRAP_REPORT.md"
OUTPUT_JSON = ROOT / "results" / "3dtheta_paired_bootstrap_summary.json"


def compute_patient_level_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Computes per-patient scalar metrics across all records in validation loader.

    Returns dict mapping:
      'missing_r': [N]
      'chest_r': [N]
      'precordial_trans_r': [N]
      'l1_err': [N]
    """
    model.eval()
    missing_leads = list(range(1, 12))
    chest_leads = [6, 7, 8, 9, 10, 11]

    patient_missing_r = []
    patient_chest_r = []
    patient_trans_r = []
    patient_l1 = []

    with torch.inference_mode():
        for batch in loader:
            targets = batch.to(device)  # [B, 12, 5000]
            x_src = targets[:, 0:1, :]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(x_src, obs_lead_idx=0)["y_pred"].float()

            # Center for Pearson: [B, 12, 5000]
            p_c = preds - preds.mean(dim=-1, keepdim=True)
            t_c = targets - targets.mean(dim=-1, keepdim=True)
            cov = (p_c * t_c).sum(dim=-1)
            std_p = torch.sqrt((p_c**2).sum(dim=-1).clamp_min(1e-8))
            std_t = torch.sqrt((t_c**2).sum(dim=-1).clamp_min(1e-8))
            pearsons = (cov / (std_p * std_t)).clamp(-1.0, 1.0)  # [B, 12]

            # Missing leads Pearson (mean across leads 1..11 for each patient)
            m_r = pearsons[:, missing_leads].mean(dim=-1).cpu().numpy()
            c_r = pearsons[:, chest_leads].mean(dim=-1).cpu().numpy()

            # Precordial transition: delta V_k = V_{k+1} - V_k
            pred_v = preds[:, chest_leads, :]  # [B, 6, 5000]
            targ_v = targets[:, chest_leads, :]
            pred_delta = pred_v[:, 1:, :] - pred_v[:, :-1, :]  # [B, 5, 5000]
            targ_delta = targ_v[:, 1:, :] - targ_v[:, :-1, :]
            pd_c = pred_delta - pred_delta.mean(dim=-1, keepdim=True)
            td_c = targ_delta - targ_delta.mean(dim=-1, keepdim=True)
            cov_d = (pd_c * td_c).sum(dim=-1)
            std_pd = torch.sqrt((pd_c**2).sum(dim=-1).clamp_min(1e-8))
            std_td = torch.sqrt((td_c**2).sum(dim=-1).clamp_min(1e-8))
            trans_r = (cov_d / (std_pd * std_td)).clamp(-1.0, 1.0).mean(dim=-1).cpu().numpy()

            # Missing L1
            diff = torch.abs(preds[:, missing_leads, :] - targets[:, missing_leads, :])
            l1_val = diff.mean(dim=(-2, -1)).cpu().numpy()

            patient_missing_r.extend(m_r)
            patient_chest_r.extend(c_r)
            patient_trans_r.extend(trans_r)
            patient_l1.extend(l1_val)

    return {
        "missing_r": np.array(patient_missing_r, dtype=np.float64),
        "chest_r": np.array(patient_chest_r, dtype=np.float64),
        "precordial_trans_r": np.array(patient_trans_r, dtype=np.float64),
        "l1_err": np.array(patient_l1, dtype=np.float64),
    }


def bootstrap_ci(
    delta: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Computes empirical bootstrap confidence interval and two-tailed p-value."""
    rng = np.random.default_rng(seed)
    n = len(delta)
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(delta[indices], axis=1)

    point_estimate = float(np.mean(delta))
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    std_err = float(np.std(boot_means))

    # Two-tailed empirical bootstrap p-value
    p_pos = np.mean(boot_means <= 0.0)
    p_neg = np.mean(boot_means >= 0.0)
    p_val = float(2.0 * min(p_pos, p_neg))

    return {
        "mean": point_estimate,
        "ci_lower": lower,
        "ci_upper": upper,
        "std_err": std_err,
        "p_value": min(p_val, 1.0),
    }


def load_model(cell_name: str, seed: int, device: torch.device) -> ThreeDThetaECGAIM | None:
    ckpt_path = ROOT / f"refine-logs/convergence_10e/runs/{cell_name}_s{seed}_l0/best.pt"
    if not ckpt_path.exists():
        return None

    cfg = CELL_CONFIGS[cell_name]
    model = ThreeDThetaECGAIM(
        code_mode=cfg["code_mode"],
        fusion=cfg["fusion"],
        width=768,
        encoder_depth=8,
        decoder_depth=4,
        heads=12,
    ).to(device)

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = {
        k: v.float() if v.is_floating_point() else v
        for k, v in payload["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Paired bootstrap uncertainty analysis")
    parser.add_argument("--seed", type=int, default=42, help="Seed to evaluate")
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap resamples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading PTB-XL validation cohort on {device}...")

    val_ds = SimplePTBXLDataset(ROOT / "data" / "ptb_xl" / "tensors" / "val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)
    n_patients = len(val_ds)
    print(f"Cohort size: N = {n_patients} patients")

    # Load candidate cells
    cells_to_eval = [
        "D3_theta_mul_l1",
        "D5_permuted_theta_mul_l1",
        "D6_theta_add_l1",
        "D4_learned12_mul_l1",
        "D7_learned12_add_l1",
        "D8_random12_mul_l1",
    ]

    patient_data = {}
    for cell in cells_to_eval:
        print(f"Extracting patient metrics for {cell} (seed {args.seed})...")
        m = load_model(cell, args.seed, device)
        if m is None:
            print(f"  [WARN] Checkpoint not found for {cell}_s{args.seed}_l0, skipping.")
            continue
        patient_data[cell] = compute_patient_level_metrics(m, val_loader, device)

    if "D3_theta_mul_l1" not in patient_data or "D5_permuted_theta_mul_l1" not in patient_data:
        print("Error: Minimum cells D3 and D5 are required for geometry bootstrap analysis.")
        return

    # Defined causal contrasts
    contrasts = [
        ("D3_theta_mul_l1", "D5_permuted_theta_mul_l1", "Physical Geometry Effect (D3 - D5)"),
        ("D3_theta_mul_l1", "D6_theta_add_l1", "Multiplicative Fusion on Physical Theta (D3 - D6)"),
        ("D4_learned12_mul_l1", "D7_learned12_add_l1", "Multiplicative Fusion on Learned Code (D4 - D7)"),
        ("D3_theta_mul_l1", "D8_random12_mul_l1", "Trigonometric Structure vs Fixed Random (D3 - D8)"),
        ("D3_theta_mul_l1", "D4_learned12_mul_l1", "Physical Theta vs Learned Matrix (D3 - D4)"),
    ]

    results = {}
    report_rows = []

    for cell_a, cell_b, desc in contrasts:
        if cell_a not in patient_data or cell_b not in patient_data:
            print(f"Skipping contrast {desc}: one or both cells missing.")
            continue

        results[desc] = {}
        for metric in ["missing_r", "chest_r", "precordial_trans_r", "l1_err"]:
            val_a = patient_data[cell_a][metric]
            val_b = patient_data[cell_b][metric]
            delta = val_a - val_b
            # For L1 error, negative delta is improvement
            stats = bootstrap_ci(delta, n_boot=args.n_boot, seed=args.seed)
            results[desc][metric] = stats

        # Build row for report table
        r_miss = results[desc]["missing_r"]
        r_chest = results[desc]["chest_r"]
        r_trans = results[desc]["precordial_trans_r"]
        report_rows.append({
            "contrast": desc,
            "mean_delta_r": r_miss["mean"],
            "ci_missing": f"[{r_miss['ci_lower']:+.4f}, {r_miss['ci_upper']:+.4f}]",
            "p_missing": r_miss["p_value"],
            "mean_chest_delta": r_chest["mean"],
            "ci_chest": f"[{r_chest['ci_lower']:+.4f}, {r_chest['ci_upper']:+.4f}]",
            "mean_trans_delta": r_trans["mean"],
            "ci_trans": f"[{r_trans['ci_lower']:+.4f}, {r_trans['ci_upper']:+.4f}]",
        })

    # Save JSON summary
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved bootstrap metrics to: {OUTPUT_JSON}")

    # Build Markdown Report
    now_str = dt.datetime.now(dt.timezone.utc).isoformat()
    lines = [
        "# Paired Patient-Level Bootstrap Uncertainty Report (3D Theta Spatial Benchmark)",
        "",
        f"Generated at: {now_str} UTC  ",
        f"Evaluation Cohort: PTB-XL held-out validation cohort ($N = {n_patients}$ patients)  ",
        f"Bootstrap Resamples: $B = {args.n_boot:,}$ iterations  ",
        "",
        "## 1. Causal Mechanism Contrasts & 95% Confidence Intervals",
        "",
        "| Causal Hypothesis Contrast | Missing Lead $\\mathbb{E}[\\Delta_i]$ | 95% Bootstrap CI | Emp. $p$-val | Precordial $V_1$–$V_6$ $\\Delta$ | 95% CI (Chest) | Precordial Transition $\\Delta V_k$ | 95% CI (Transition) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for row in report_rows:
        lines.append(
            f"| **{row['contrast']}** | **{row['mean_delta_r']:+.4f}** | `{row['ci_missing']}` | {row['p_missing']:.4f} | "
            f"**{row['mean_chest_delta']:+.4f}** | `{row['ci_chest']}` | **{row['mean_trans_delta']:+.4f}** | `{row['ci_trans']}` |"
        )

    lines.extend([
        "",
        "## 2. Epistemological Interpretation Guidelines",
        "- **Physical Geometry ($D_3 - D_5$)**: If the 95% confidence interval for $\\Delta_i$ overlaps zero or is tightly concentrated below $+0.003$, the hypothesis that explicit 3D coordinate assignment provides meaningful anatomical advantage is rejected in favor of the structured target code hypothesis.",
        "- **Multiplicative Modulation ($D_3 - D_6$ and $D_4 - D_7$)**: Isolates whether multiplicative scaling ($H \\odot g_l$) provides an authentic inductive bias over additive broadcast ($H + g_l$) when holding spatial codes strictly constant.",
        "- **Trigonometric Representation ($D_3 - D_8$)**: Distinguishes whether trigonometric coordinates $\\Theta(\\theta, \\phi)$ provide an inductive smoothing bias over arbitrary fixed continuous codes.",
        "",
    ])

    OUTPUT_MD.write_text("\n".join(lines))
    print(f"Saved Markdown report to: {OUTPUT_MD}")
    print("\n=== Paired Bootstrap Analysis Complete ===")


if __name__ == "__main__":
    main()
