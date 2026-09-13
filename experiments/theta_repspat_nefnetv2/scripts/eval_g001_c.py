#!/usr/bin/env python3
"""G001-C: Clinical zero-shot transfer evaluation of N002 on real 12-lead ECGs.

Scientific contract:
  - Checkpoint: panobench_3view_variablethird_geovt_seed123/model_final.pt (frozen)
  - Task: Zero-shot reconstruction of precordial leads {V1, V2, V4, V5, V6}
    from clinical inputs {I, II, V3} using author PTB-XL canonical angles:
      a_I  = (90°, 90°)
      a_II = (150°, 90°)
      a_V3 = (95°, 15°)
  - Dataset: Held-out real clinical 12-lead ECGs from HEEDB Emory WFDB dataset
    (never seen during training, distinct clinical population from PanoBench).
  - Ground truth: Actual measured patient leads V1, V2, V4, V5, V6.
  - Evaluation:
      - Pointwise metrics: L1 (normalized and physical mV), PSNR (dB), SSIM (1D).
      - Morphological metric: Pearson correlation r per lead.
      - Normalization modes evaluated:
          1) observed_only (I, II, V3 extrema only — clinically realistic)
          2) global_12lead (author-faithful global extrema)
  - Zero-shot: NO fine-tuning, NO patient calibration, NO repSpat tuning.

Output: results/g001_c_eval/g001_c_results.json
        results/g001_c_eval/g001_c_per_lead.json
        results/g001_c_eval/g001_c_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.stats
import torch

from theta_repspat.vendor import load_author_model

# ── Canonical geometry from author codes/dataset/PTBXL.py ───────────────────
# Canonical lead ordering: [I, II, V1, V2, V3, V4, V5, V6]
CANONICAL_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
CANONICAL_ANGLES_RAD = np.array([
    [np.pi / 2, np.pi / 2],           # I  (idx 0)
    [np.pi * 5 / 6, np.pi / 2],       # II (idx 1)
    [np.pi / 2, -np.pi / 18],         # V1 (idx 2)
    [np.pi / 2, np.pi / 18],          # V2 (idx 3)
    [np.pi * (19 / 36), np.pi / 12],  # V3 (idx 4)
    [np.pi * (11 / 20), np.pi / 6],   # V4 (idx 5)
    [np.pi * (16 / 30), np.pi / 3],   # V5 (idx 6)
    [np.pi * (16 / 30), np.pi / 2],   # V6 (idx 7)
], dtype=np.float32)

INPUT_LEAD_NAMES = ["I", "II", "V3"]
INPUT_CANONICAL_INDICES = [0, 1, 4]

QUERY_LEAD_NAMES = ["V1", "V2", "V4", "V5", "V6"]
QUERY_CANONICAL_INDICES = [2, 3, 5, 6, 7]

EVAL_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPECTED_LEN = 4608


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


def pearson_r(pred: np.ndarray, target: np.ndarray) -> float:
    if np.std(pred) < 1e-8 or np.std(target) < 1e-8:
        return 0.0
    r, _ = scipy.stats.pearsonr(pred, target)
    return float(r) if np.isfinite(r) else 0.0


def load_wfdb_record(hea_path: Path) -> tuple[np.ndarray, dict]:
    """Load GE MUSE WFDB record into [8, 4608] canonical lead array and metadata."""
    dat_path = hea_path.with_suffix(".dat")
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing .dat for {hea_path}")

    lines = hea_path.read_text().strip().split("\n")
    info = lines[0].split()
    n_leads = int(info[1])
    fs = int(info[2])
    n_samples = int(info[3])
    raw_lead_names = [l.split()[-1] for l in lines[1: 1 + n_leads]]

    # Parse ADC gain (default 1000 ADC / mV)
    gain = 1000.0
    try:
        gain_str = lines[1].split()[2].split("(")[0].split("/")[0]
        gain = float(gain_str)
    except Exception:
        pass

    # Read binary 16-bit interleaved
    raw_data = np.fromfile(dat_path, dtype=np.int16).reshape((n_samples, n_leads)).T
    # Convert to float32 physical mV
    data_mv = raw_data.astype(np.float32) / gain

    # Map raw leads to canonical order: [I, II, V1, V2, V3, V4, V5, V6]
    name_to_idx = {name: i for i, name in enumerate(raw_lead_names)}
    canonical_data = np.zeros((len(CANONICAL_LEADS), n_samples), dtype=np.float32)
    for c_idx, lead_name in enumerate(CANONICAL_LEADS):
        if lead_name in name_to_idx:
            canonical_data[c_idx] = data_mv[name_to_idx[lead_name]]
        else:
            raise ValueError(f"Lead {lead_name} not found in {hea_path}")

    meta = {
        "record_id": hea_path.stem,
        "fs": fs,
        "n_samples": n_samples,
        "gain": gain,
    }
    return canonical_data, meta


def evaluate_record_clinical(record_data_mv: np.ndarray, model,
                             norm_mode: str = "observed_only") -> dict:
    """Evaluate zero-shot transfer on one 8-lead clinical ECG."""
    # Center-crop 4608 samples from 5000 (standard 10s @ 500Hz)
    n_samples = record_data_mv.shape[1]
    if n_samples < EXPECTED_LEN:
        pad_len = EXPECTED_LEN - n_samples
        data_crop = np.pad(record_data_mv, ((0, 0), (0, pad_len)), mode="edge")
    elif n_samples == EXPECTED_LEN:
        data_crop = record_data_mv
    else:
        # Centered crop
        start = (n_samples - EXPECTED_LEN) // 2
        data_crop = record_data_mv[:, start: start + EXPECTED_LEN]

    input_data = data_crop[INPUT_CANONICAL_INDICES]  # (3, 4608): I, II, V3

    if norm_mode == "observed_only":
        lo = float(input_data.min())
        hi = float(input_data.max())
    else:  # global_8lead
        lo = float(data_crop.min())
        hi = float(data_crop.max())

    span = hi - lo if hi > lo else 1.0
    input_norm = (input_data - lo) / span

    inp_t = torch.from_numpy(input_norm).unsqueeze(0).to(DEVICE)
    inp_angles = torch.from_numpy(
        CANONICAL_ANGLES_RAD[INPUT_CANONICAL_INDICES]).unsqueeze(0).to(DEVICE)

    lead_metrics = {}
    for q_name, q_idx in zip(QUERY_LEAD_NAMES, QUERY_CANONICAL_INDICES):
        tgt_angle = torch.from_numpy(
            CANONICAL_ANGLES_RAD[q_idx]).unsqueeze(0).to(DEVICE)
        target_raw_mv = data_crop[q_idx]

        with torch.no_grad():
            pred_norm = model(inp_t, inp_angles, tgt_angle).squeeze().cpu().numpy()

        pred_mv = pred_norm * span + lo
        target_norm = (target_raw_mv - lo) / span

        l1_norm = float(np.mean(np.abs(pred_norm - target_norm)))
        l1_mv = float(np.mean(np.abs(pred_mv - target_raw_mv)))
        p_db = psnr(pred_norm, target_norm, val_range=1.0)
        s_val = ssim_1d(pred_norm, target_norm, val_range=1.0)
        r_val = pearson_r(pred_mv, target_raw_mv)

        lead_metrics[q_name] = {
            "l1_norm": l1_norm,
            "l1_mv": l1_mv,
            "psnr_db": p_db,
            "ssim": s_val,
            "pearson_r": r_val,
        }

    return lead_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("results/panobench_3view_variablethird_geovt_seed123/model_final.pt"))
    p.add_argument("--data-root", type=Path,
                   default=Path("/data/mithunmanivannan/heedb_emory/WFDB/2010"))
    p.add_argument("--output", type=Path,
                   default=Path("results/g001_c_eval"))
    p.add_argument("--max-records", type=int, default=1000)
    p.add_argument("--norm-mode", choices=["observed_only", "global_8lead"],
                   default="observed_only")
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

    # Collect .hea files deterministically using fast scandir
    print(f"Discovering records in {args.data_root}...", flush=True)
    hea_files = []
    import os
    with os.scandir(args.data_root) as it:
        for entry in it:
            if entry.name.endswith(".hea"):
                hea_files.append(Path(entry.path))
                if args.max_records and len(hea_files) >= args.max_records:
                    break
    hea_files.sort(key=lambda p: p.name)
    if not hea_files:
        raise FileNotFoundError(f"No .hea records found in {args.data_root}")
    print(f"Evaluating {len(hea_files)} clinical records under norm_mode='{args.norm_mode}'...", flush=True)

    all_results = []
    per_lead_accum = {lead: {"l1_norm": [], "l1_mv": [], "psnr_db": [], "ssim": [], "pearson_r": []}
                      for lead in QUERY_LEAD_NAMES}

    for i, hea_path in enumerate(hea_files):
        try:
            canonical_data, meta = load_wfdb_record(hea_path)
            record_eval = evaluate_record_clinical(canonical_data, model, args.norm_mode)
        except Exception as e:
            continue

        record_entry = {"record_id": hea_path.stem, "leads": record_eval}
        all_results.append(record_entry)

        for lead, m in record_eval.items():
            for k in per_lead_accum[lead]:
                if not math.isnan(m[k]):
                    per_lead_accum[lead][k].append(m[k])

        if (i + 1) % 100 == 0 or i == len(hea_files) - 1:
            mean_l1_norm = np.mean([
                np.mean([r["leads"][l]["l1_norm"] for l in QUERY_LEAD_NAMES])
                for r in all_results
            ])
            mean_r = np.mean([
                np.mean([r["leads"][l]["pearson_r"] for l in QUERY_LEAD_NAMES])
                for r in all_results
            ])
            print(f"  {i+1}/{len(hea_files)} records, running L1_norm={mean_l1_norm:.5f}, "
                  f"running Pearson r={mean_r:.4f}", flush=True)

    # Compute summary per lead
    per_lead_summary = {}
    for lead in QUERY_LEAD_NAMES:
        per_lead_summary[lead] = {
            k: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            for k, vals in per_lead_accum[lead].items()
        }

    # Overall aggregate
    all_l1_norm = [v for l in QUERY_LEAD_NAMES for v in per_lead_accum[l]["l1_norm"]]
    all_l1_mv = [v for l in QUERY_LEAD_NAMES for v in per_lead_accum[l]["l1_mv"]]
    all_psnr = [v for l in QUERY_LEAD_NAMES for v in per_lead_accum[l]["psnr_db"]]
    all_ssim = [v for l in QUERY_LEAD_NAMES for v in per_lead_accum[l]["ssim"]]
    all_r = [v for l in QUERY_LEAD_NAMES for v in per_lead_accum[l]["pearson_r"]]

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(args.data_root),
        "n_records": len(all_results),
        "norm_mode": args.norm_mode,
        "input_leads": INPUT_LEAD_NAMES,
        "queried_leads": QUERY_LEAD_NAMES,
        "overall": {
            "l1_norm": {"mean": float(np.mean(all_l1_norm)), "std": float(np.std(all_l1_norm))},
            "l1_mv": {"mean": float(np.mean(all_l1_mv)), "std": float(np.std(all_l1_mv))},
            "psnr_db": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr))},
            "ssim": {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim))},
            "pearson_r": {"mean": float(np.mean(all_r)), "std": float(np.std(all_r))},
        },
        "per_lead": per_lead_summary,
        "verdict": {
            "transfer_status": (
                "ZERO_SHOT_TRANSFER_SUCCESS"
                if float(np.mean(all_r)) > 0.70 and float(np.mean(all_l1_norm)) < 0.05
                else "TRANSFER_FAIL_REQUIRES_DOMAIN_CANONICALIZATION"
            ),
            "clinical_morphology_preserved": bool(float(np.mean(all_r)) > 0.70),
        },
    }

    (args.output / "g001_c_results.json").write_text(
        json.dumps(all_results, indent=2))
    (args.output / "g001_c_per_lead.json").write_text(
        json.dumps(per_lead_summary, indent=2))
    (args.output / "g001_c_summary.json").write_text(
        json.dumps(summary, indent=2))

    print("\n=== G001-C Zero-Shot Clinical Transfer Summary ===")
    print(f"  Records evaluated: {len(all_results)}")
    print(f"  Input:  {INPUT_LEAD_NAMES}")
    print(f"  Query:  {QUERY_LEAD_NAMES}")
    print(f"  Overall L1 (norm): {summary['overall']['l1_norm']['mean']:.5f}")
    print(f"  Overall L1 (mV):   {summary['overall']['l1_mv']['mean']:.4f} mV")
    print(f"  Overall PSNR:      {summary['overall']['psnr_db']['mean']:.2f} dB")
    print(f"  Overall SSIM:      {summary['overall']['ssim']['mean']:.4f}")
    print(f"  Overall Pearson r: {summary['overall']['pearson_r']['mean']:.4f}")
    print("\n  Per-Lead Performance:")
    for lead, m in per_lead_summary.items():
        print(f"    {lead:3s}: r={m['pearson_r']['mean']:.4f}  "
              f"L1={m['l1_norm']['mean']:.5f}  "
              f"L1_mV={m['l1_mv']['mean']:.4f}mV  "
              f"PSNR={m['psnr_db']['mean']:.2f}dB")
    print(f"\n  Verdict: {summary['verdict']['transfer_status']}")
    print(f"  Output:  {args.output}/")


if __name__ == "__main__":
    main()
