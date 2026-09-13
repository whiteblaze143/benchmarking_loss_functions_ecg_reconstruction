#!/usr/bin/env python3
"""Audit PTB-XL observed normalization vs Nef decoder sigmoid output support.

Evaluates:
    p_-   = P(y_tilde < 0)
    p_+   = P(y_tilde > 1)
    p_OOR = P(y_tilde not in [0, 1])
    L_floor = (1/N) * sum max(-y_i, 0, y_i - 1)

Stratified across query leads V1..V6 on all PTB-XL train (N=17,418) and val (N=2,183) records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


# Disk order: 0:I, 1:II, 2:III, 3:aVR, 4:aVL, 5:aVF, 6:V1, 7:V2, 8:V3, 9:V4, 10:V5, 11:V6
# Nef canonical order: 0:I, 1:II, 2:V1, 3:V2, 4:V3, 5:V4, 6:V5, 7:V6, 8:III, 9:aVR, 10:aVL, 11:aVF
DISK_TO_CANONICAL = [0, 1, 6, 7, 8, 9, 10, 11, 2, 3, 4, 5]
LEAD_NAMES = ["V1", "V2", "V3", "V4", "V5", "V6"]


def run_audit(split: str, sample_limit: int | None = None) -> dict:
    tensors_dir = Path(f"/home/mithunmanivannan/data/ptb_xl/tensors/{split}")
    files = sorted(tensors_dir.glob("*.pt"))
    if sample_limit is not None:
        files = files[:sample_limit]
    
    print(f"=== Auditing {split.upper()} ({len(files)} records) ===")
    stats = {lead: {"total": 0, "neg": 0, "gt1": 0, "oor": 0, "l_floor_sum": 0.0} for lead in LEAD_NAMES}
    overall = {"total": 0, "neg": 0, "gt1": 0, "oor": 0, "l_floor_sum": 0.0}
    
    rng = np.random.default_rng(123)
    
    for idx, f in enumerate(files):
        data = torch.load(f, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            data = data["ecg"]
        cdata = data[DISK_TO_CANONICAL]  # shape [12, 5000]
        
        # Test each precordial lead as target query
        for q_idx, q_name in enumerate(LEAD_NAMES, start=2):
            other_precordials = [i for i in range(2, 8) if i != q_idx]
            k = int(rng.choice([1, 2, 3]))
            chosen = rng.choice(other_precordials, size=k, replace=False).tolist()
            obs_indices = [0, 1] + sorted(chosen)
            
            obs_waveforms = cdata[obs_indices]  # [2+k, 5000]
            target_waveform = cdata[q_idx]      # [5000]
            
            m_obs = float(obs_waveforms.min())
            M_obs = float(obs_waveforms.max())
            diff = max(M_obs - m_obs, 1e-6)
            
            y = (target_waveform - m_obs) / diff
            
            n_samples = y.numel()
            n_neg = int((y < 0.0).sum())
            n_gt1 = int((y > 1.0).sum())
            n_oor = int(((y < 0.0) | (y > 1.0)).sum())
            
            err_floor = torch.clamp(-y, min=0.0) + torch.clamp(y - 1.0, min=0.0)
            l_floor = float(err_floor.sum())
            
            stats[q_name]["total"] += n_samples
            stats[q_name]["neg"] += n_neg
            stats[q_name]["gt1"] += n_gt1
            stats[q_name]["oor"] += n_oor
            stats[q_name]["l_floor_sum"] += l_floor
            
            overall["total"] += n_samples
            overall["neg"] += n_neg
            overall["gt1"] += n_gt1
            overall["oor"] += n_oor
            overall["l_floor_sum"] += l_floor
            
        if (idx + 1) % 5000 == 0 or (idx + 1) == len(files):
            print(f"  Processed {idx + 1}/{len(files)} records...")
            
    p_neg = overall["neg"] / overall["total"]
    p_gt1 = overall["gt1"] / overall["total"]
    p_oor = overall["oor"] / overall["total"]
    l_floor_mean = overall["l_floor_sum"] / overall["total"]
    
    result = {
        "split": split,
        "n_records": len(files),
        "overall": {
            "p_neg": p_neg,
            "p_gt1": p_gt1,
            "p_oor": p_oor,
            "l_floor": l_floor_mean,
        },
        "stratified": {},
    }
    
    print(f"\nResults for {split.upper()} (N={len(files)}):")
    print(f"  OVERALL: p_- = {p_neg:.4%}, p_+ = {p_gt1:.4%}, p_OOR = {p_oor:.4%}, L_floor = {l_floor_mean:.6f}")
    for lead in LEAD_NAMES:
        s = stats[lead]
        tot = s["total"]
        lead_res = {
            "p_neg": s["neg"] / tot,
            "p_gt1": s["gt1"] / tot,
            "p_oor": s["oor"] / tot,
            "l_floor": s["l_floor_sum"] / tot,
        }
        result["stratified"][lead] = lead_res
        print(f"  {lead:4s}: p_- = {lead_res['p_neg']:.2%}, p_+ = {lead_res['p_gt1']:.2%}, p_OOR = {lead_res['p_oor']:.2%}, L_floor = {lead_res['l_floor']:.6f}")
    print()
    return result


def main():
    results = {}
    for split in ["train", "val"]:
        results[split] = run_audit(split)
        
    out_path = Path("experiments/theta_repspat_nefnetv2/results/audit_normalization_support.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved audit results to: {out_path}")


if __name__ == "__main__":
    main()
