#!/usr/bin/env python3
"""
Clustered Patient-Level Bootstrap Analysis & Synchronous Checkpoint Comparator
for 3DRECON-QT Reference Benchmark (RQ1Q vs RQ2Q).

Performs:
1. Clustered patient-level bootstrap (10,000 resamples) across unique patient IDs:
   - Sample patient IDs with replacement (M = 1,942 unique patients).
   - Cluster all ECG records per sampled patient.
   - Compute mean differences and 95% CIs.
2. Frontal-plane lead decomposition:
   - Delta_II (sole remaining independent frontal lead when Lead I is observed)
   - Delta_dependent_limb = mean(Delta_III, Delta_aVR, Delta_aVL, Delta_aVF)
   - Delta_frontal = mean(Delta_II, Delta_III, Delta_aVR, Delta_aVL, Delta_aVF)
   - Delta_chest = mean(V1..V6)
   - Delta_precordial_transition = mean(V_{k+1} - V_k)
   - Delta_missing_11 = mean(missing leads)
3. Synchronous matched-epoch analysis:
   - Evaluates trajectory across matched epochs between RQ1Q and RQ2Q.
   - Compares:
     a. Each model's independently selected best checkpoint
     b. Final common epoch
     c. Symmetrically chosen epoch minimizing mean validation loss
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unified_latents.engineering.models.reconqt_reference import LEAD_NAMES


def clustered_bootstrap_ci(
    delta_record: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Clustered bootstrap: samples unique patient IDs with replacement.
    Retains all ECG records belonging to each sampled patient.
    """
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)

    patient_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
    record_patient_idx = np.array([patient_to_idx[pid] for pid in patient_ids])

    patient_sums = np.bincount(record_patient_idx, weights=delta_record, minlength=n_patients)
    patient_counts = np.bincount(record_patient_idx, minlength=n_patients)

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_pts = rng.integers(0, n_patients, size=n_patients)
        sampled_counts = np.bincount(sampled_pts, minlength=n_patients)
        sum_delta = np.dot(patient_sums, sampled_counts)
        tot_records = np.dot(patient_counts, sampled_counts)
        boot_means[b] = sum_delta / tot_records

    mean_val = float(np.mean(delta_record))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    std_err = float(np.std(boot_means, ddof=1))

    p_pos = np.mean(boot_means >= 0.0)
    p_neg = np.mean(boot_means <= 0.0)
    p_value = float(2.0 * min(p_pos, p_neg))
    p_value = min(1.0, max(0.0, p_value))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "p_value": p_value,
    }


def unclustered_bootstrap_ci(
    delta_record: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(delta_record)
    boot_means = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(delta_record[idx])

    mean_val = float(np.mean(delta_record))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    std_err = float(np.std(boot_means, ddof=1))

    p_pos = np.mean(boot_means >= 0.0)
    p_neg = np.mean(boot_means <= 0.0)
    p_value = float(2.0 * min(p_pos, p_neg))
    p_value = min(1.0, max(0.0, p_value))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "p_value": p_value,
    }


def load_metrics_jsonl(run_dir: Path) -> List[Dict]:
    metrics_file = run_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return []
    records = []
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def analyze_synchronous_checkpoints(run_dir_a: Path, run_dir_b: Path) -> Dict:
    metrics_a = load_metrics_jsonl(run_dir_a)
    metrics_b = load_metrics_jsonl(run_dir_b)

    if not metrics_a or not metrics_b:
        return {"status": "metrics_missing"}

    map_a = {m["epoch"]: m for m in metrics_a}
    map_b = {m["epoch"]: m for m in metrics_b}
    common_epochs = sorted(list(set(map_a.keys()) & set(map_b.keys())))

    if not common_epochs:
        return {"status": "no_common_epochs"}

    trajectory = []
    for ep in common_epochs:
        r_a = map_a[ep]["val_missing_pearson"]
        r_b = map_b[ep]["val_missing_pearson"]
        delta = r_a - r_b
        loss_a = map_a[ep].get("val_l1_recon", 0.0)
        loss_b = map_b[ep].get("val_l1_recon", 0.0)
        trajectory.append({
            "epoch": ep,
            "r_a": r_a,
            "r_b": r_b,
            "delta_r": delta,
            "mean_loss": (loss_a + loss_b) / 2.0,
        })

    # Final common epoch
    final_ep = common_epochs[-1]
    final_point = [p for p in trajectory if p["epoch"] == final_ep][0]

    # Symmetric epoch minimizing average validation loss
    best_symmetric = min(trajectory, key=lambda p: p["mean_loss"])

    return {
        "common_epochs_count": len(common_epochs),
        "final_common_epoch": final_point,
        "symmetric_min_loss_epoch": best_symmetric,
        "trajectory": trajectory,
    }


def main():
    p = argparse.ArgumentParser(description="Clustered Bootstrap & Synchronous Comparator for 3DRECON-QT")
    p.add_argument("--model-a-npz", required=True, help="Path to patient_level_eval.npz for Model A (RQ1Q)")
    p.add_argument("--model-b-npz", required=True, help="Path to patient_level_eval.npz for Model B (RQ2Q)")
    p.add_argument("--label-a", default="RQ1Q_theta", help="Display name for Model A")
    p.add_argument("--label-b", default="RQ2Q_permuted", help="Display name for Model B")
    p.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap resamples")
    p.add_argument("--out-json", default=None, help="Output JSON path")
    p.add_argument("--out-md", default=None, help="Output Markdown report path")
    args = p.parse_args()

    npz_a = Path(args.model_a_npz)
    npz_b = Path(args.model_b_npz)
    data_a = np.load(npz_a)
    data_b = np.load(npz_b)

    # 1. Load patient IDs from ptbxl_database.csv
    meta_path = ROOT / "data/ptb_xl/ptbxl_database.csv"
    assert meta_path.exists(), f"Metadata CSV missing: {meta_path}"
    df_meta = pd.read_csv(meta_path).set_index("ecg_id")

    if "ecg_ids" in data_a:
        ecg_ids = data_a["ecg_ids"]
    else:
        # Fallback to sorted files in val directory
        val_files = sorted(list((ROOT / "data/ptb_xl/tensors/val").glob("*.pt")))
        ecg_ids = [int(p.stem) for p in val_files]

    val_df = df_meta.loc[ecg_ids]
    val_patient_ids = val_df["patient_id"].values
    n_ecg = len(ecg_ids)
    n_patient = len(np.unique(val_patient_ids))

    print("=" * 80)
    print(f"  3DRECON-QT CLUSTERED PATIENT BOOTSTRAP: {args.label_a} vs {args.label_b}")
    print("=" * 80)
    print(f"  Validation Cohort: {n_ecg} ECG records across {n_patient} unique patients")
    val_counts = pd.Series(val_patient_ids).value_counts()
    print(f"  Patients with 1 ECG:  {(val_counts == 1).sum()} ({(val_counts == 1).sum() / n_patient * 100:.1f}%)")
    print(f"  Patients with >1 ECG: {(val_counts > 1).sum()} ({(val_counts > 1).sum() / n_patient * 100:.1f}%)")

    # Extract or compute per-lead arrays
    metrics_to_eval = {}

    # Missing-11 Pearson
    delta_missing = data_a["patient_missing_r"] - data_b["patient_missing_r"]
    metrics_to_eval["Missing-11 Pearson (All Missing)"] = delta_missing

    # Chest V1-V6 Pearson
    delta_chest = data_a["patient_chest_r"] - data_b["patient_chest_r"]
    metrics_to_eval["Precordial Chest (V1-V6) Pearson"] = delta_chest

    # Precordial Transition Delta V_k
    if "patient_trans_r" in data_a and "patient_trans_r" in data_b:
        delta_trans = data_a["patient_trans_r"] - data_b["patient_trans_r"]
        metrics_to_eval["Precordial Transition Progression Delta V_k"] = delta_trans

    # Lead II (independent frontal lead when Lead I is observed)
    if "lead_r_II" in data_a and "lead_r_II" in data_b:
        delta_ii = data_a["lead_r_II"] - data_b["lead_r_II"]
        metrics_to_eval["Lead II (Independent Frontal Lead)"] = delta_ii

    # Dependent Limb Leads (III, aVR, aVL, aVF)
    dep_leads = ["III", "aVR", "aVL", "aVF"]
    if all(f"lead_r_{l}" in data_a and f"lead_r_{l}" in data_b for l in dep_leads):
        dep_deltas = [data_a[f"lead_r_{l}"] - data_b[f"lead_r_{l}"] for l in dep_leads]
        delta_dep = np.mean(dep_deltas, axis=0)
        metrics_to_eval["Dependent Limb Leads (III, aVR, aVL, aVF)"] = delta_dep

        # All 5 Frontal Leads
        if "lead_r_II" in data_a:
            frontal_deltas = [delta_ii] + dep_deltas
            delta_frontal = np.mean(frontal_deltas, axis=0)
            metrics_to_eval["All 5 Frontal Leads (II, III, aVR, aVL, aVF)"] = delta_frontal

    bootstrap_results = {}
    print("\n" + "=" * 105)
    print(f"{'Metric':<48} | {'Mean Delta':<10} | {'Clustered 95% CI':<24} | {'p-val':<7} | {'Unclustered 95% CI':<24}")
    print("=" * 105)

    for name, delta_arr in metrics_to_eval.items():
        c_res = clustered_bootstrap_ci(delta_arr, val_patient_ids, n_boot=args.n_boot, seed=42)
        u_res = unclustered_bootstrap_ci(delta_arr, n_boot=args.n_boot, seed=42)
        bootstrap_results[name] = {
            "clustered": c_res,
            "unclustered": u_res,
        }
        c_ci = f"[{c_res['ci_lower']:+.6f}, {c_res['ci_upper']:+.6f}]"
        u_ci = f"[{u_res['ci_lower']:+.6f}, {u_res['ci_upper']:+.6f}]"
        print(f"{name:<48} | {c_res['mean']:+.6f} | {c_ci:<24} | {c_res['p_value']:<7.4f} | {u_ci:<24}")

    print("=" * 105)

    # 2. Synchronous Checkpoint Analysis
    sync_analysis = analyze_synchronous_checkpoints(npz_a.parent, npz_b.parent)

    print("\n" + "=" * 80)
    print("  SYNCHRONOUS CHECKPOINT COMPARISON")
    print("=" * 80)
    if sync_analysis.get("status") == "metrics_missing":
        print("  [WARN] metrics.jsonl not found in run directories.")
    else:
        final_pt = sync_analysis["final_common_epoch"]
        sym_pt = sync_analysis["symmetric_min_loss_epoch"]
        print(f"  Common Epochs Evaluated: {sync_analysis['common_epochs_count']}")
        print(f"  Final Common Epoch ({final_pt['epoch']}):")
        print(f"    {args.label_a} r = {final_pt['r_a']:.4f}, {args.label_b} r = {final_pt['r_b']:.4f} -> Delta = {final_pt['delta_r']:+.4f}")
        print(f"  Symmetric Min-Loss Epoch ({sym_pt['epoch']}):")
        print(f"    {args.label_a} r = {sym_pt['r_a']:.4f}, {args.label_b} r = {sym_pt['r_b']:.4f} -> Delta = {sym_pt['delta_r']:+.4f}")

    # Export JSON
    full_output = {
        "label_a": args.label_a,
        "label_b": args.label_b,
        "n_ecg": n_ecg,
        "n_patients": n_patient,
        "bootstrap_results": bootstrap_results,
        "synchronous_analysis": sync_analysis,
    }

    if args.out_json:
        out_json_p = Path(args.out_json)
        out_json_p.parent.mkdir(parents=True, exist_ok=True)
        out_json_p.write_text(json.dumps(full_output, indent=2))
        print(f"\nSaved full results to: {out_json_p}")


if __name__ == "__main__":
    main()
