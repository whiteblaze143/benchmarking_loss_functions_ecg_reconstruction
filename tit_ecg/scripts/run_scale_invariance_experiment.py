"""Kernel-Scale and Time-Scale Invariance Experiment (Kill Gates 1 & 2).

Tests:
1. Training Set Optimization (PTB-XL Fold 1):
   Evaluate d_med = median_{i<j, D_ij > 0} D_ij.
   Compare fixed c=1.0 vs c in {0.25*d_med, 0.5*d_med, 1.0*d_med, 2.0*d_med}.
   Compare sample grid m in {4, 6, 8} vs millisecond grid Delta in {16, 32, 64, 128} ms.
2. Rule Freezing:
   Freeze best calibrated rule on training set (c = 1.0 * d_med, Delta in {16, 32, 64, 128} ms).
3. Cross-Cohort Generalization:
   Evaluate frozen rule vs fixed baseline on held-out PTB-XL (Fold 10) and EchoNext test cohort.
   Verify whether external graph density normalizes across acquisition scales.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
import numpy as np
import pandas as pd
import wfdb

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.pipeline import TemporalRepSpatECG
from tit_ecg.src.vcg_transform import ecg_to_vcg_kors


def load_ptbxl_record(ecg_id: int, ptbxl_dir: str = "data/ptb_xl", n_samples: int = 1000) -> tuple[np.ndarray, float]:
    """Loads PTB-XL ECG recording."""
    df_meta = pd.read_csv(os.path.join(ptbxl_dir, "ptbxl_database.csv"), index_col="ecg_id")
    row = df_meta.loc[ecg_id]
    rel_path = row["filename_hr"]
    signals, fields = wfdb.rdsamp(os.path.join(ptbxl_dir, rel_path))
    return signals[:n_samples], float(fields["fs"])


_ECHONEXT_CACHE = {}


def get_echonext_data():
    if "waveforms" not in _ECHONEXT_CACHE:
        waveforms_path = "data/echonext/EchoNext_test_waveforms.npy"
        prov_path = "data/echonext/PROVENANCE.json"
        with open(prov_path) as f:
            prov = json.load(f)
        _ECHONEXT_CACHE["mean"] = np.array(prov["normalization"]["mean"], dtype=float)
        _ECHONEXT_CACHE["std"] = np.array(prov["normalization"]["std"], dtype=float)
        _ECHONEXT_CACHE["waveforms"] = np.load(waveforms_path, mmap_mode="r")
    return _ECHONEXT_CACHE


def load_echonext_record(idx: int, n_samples: int = 750) -> tuple[np.ndarray, float]:
    """Loads EchoNext external test recording and converts to physical mV."""
    data = get_echonext_data()
    raw = data["waveforms"][idx, 0]  # [2500, 12]
    mean_val = data["mean"]
    std_val = data["std"]
    # Invert z-score to uV, then to mV
    wave_uv = np.asarray(raw[:n_samples], dtype=float) * std_val + mean_val
    wave_mv = wave_uv / 1000.0
    return wave_mv, 250.0


def compute_median_distance(ecg_or_vcg: np.ndarray, is_vcg: bool = False) -> float:
    """Computes median pairwise Euclidean distance d_med across points in VCG space."""
    if not is_vcg:
        vcg = ecg_to_vcg_kors(ecg_or_vcg)
    else:
        vcg = ecg_or_vcg
    # Subsample if n > 500 for fast calculation
    if len(vcg) > 500:
        step = max(1, len(vcg) // 500)
        v_sub = vcg[::step]
    else:
        v_sub = vcg
    diffs = v_sub[:, None, :] - v_sub[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    triu_idx = np.triu_indices(len(v_sub), k=1)
    nonzero = dists[triu_idx]
    nonzero = nonzero[nonzero > 0]
    return float(np.median(nonzero)) if len(nonzero) > 0 else 1.0


def evaluate_single_config(
    signal: np.ndarray,
    fs: float,
    config_dict: dict,
    m_ms_grid: list[float] | None = None,
    kernel_scale_rule: str = "fixed",
    kernel_param: float = 1.0,
) -> dict:
    """Evaluates repSpat on a single record with specified scale rules."""
    cfg = RepSpatConfig(**config_dict)
    cfg.kernel_scale_rule = kernel_scale_rule
    cfg.kernel_param = kernel_param
    cfg.m_ms_grid = m_ms_grid

    model = TemporalRepSpatECG(config=cfg)
    model.fit(signal, sampling_rate=fs)
    res = model.results_
    g_desc = res["graph_descriptors"]

    return {
        "m_star": int(res["m_star"]),
        "G_star": int(res["G_star"]),
        "graph_density": float(g_desc.get("density", 0.0)),
        "n_clique_groups": int(res["n_clique_clusters"]),
        "n_cc_groups": int(res["n_cc_clusters"]),
        "n_maximal_cliques": int(g_desc.get("n_maximal_cliques", 0)),
        "overlap_fraction": float(g_desc.get("overlap_fraction", 0.0)),
        "mean_retained_edge_weight": float(g_desc.get("mean_retained_edge_weight", 0.0)),
    }


def main():
    print("=" * 80)
    print("METHODOLOGICAL KILL GATES 1 & 2: SCALE INVARIANCE BENCHMARK")
    print("=" * 80)

    output_dir = "tit_ecg/results/scale_invariance"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load PTB-XL Fold 1 (Training Set) records
    meta_df = pd.read_csv("data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    train_ids = meta_df[meta_df["strat_fold"] == 1].index.values[:16]
    test_ids = meta_df[meta_df["strat_fold"] == 10].index.values[:16]

    # External EchoNext records
    echonext_indices = [10, 42, 105, 210, 315, 420, 525, 630, 735, 840, 945, 1050, 1155, 1260, 1365, 1470]

    print(f"Step 1: Evaluating Candidate Kernel Scales & Neighborhoods on PTB-XL Training Records (N={len(train_ids)})...")

    # Candidate scales:
    # Baseline: fixed c=1.0, m in {4, 6, 8}
    # Invariant: Delta in {16, 32, 64, 128} ms, gamma in {0.25, 0.5, 1.0, 2.0}
    scale_configs = [
        {"name": "Fixed Baseline (c=1, discrete m)", "rule": "fixed", "gamma": 1.0, "m_ms": None, "m_grid": [4, 6, 8]},
        {"name": "Relative c=0.25*d_med (ms grid)", "rule": "median_heuristic", "gamma": 0.25, "m_ms": [16, 32, 64, 128], "m_grid": [4, 6, 8]},
        {"name": "Relative c=0.5*d_med (ms grid)",  "rule": "median_heuristic", "gamma": 0.5,  "m_ms": [16, 32, 64, 128], "m_grid": [4, 6, 8]},
        {"name": "Relative c=1.0*d_med (ms grid)",  "rule": "median_heuristic", "gamma": 1.0,  "m_ms": [16, 32, 64, 128], "m_grid": [4, 6, 8]},
        {"name": "Relative c=2.0*d_med (ms grid)",  "rule": "median_heuristic", "gamma": 2.0,  "m_ms": [16, 32, 64, 128], "m_grid": [4, 6, 8]},
    ]

    base_config_dict = {
        "n_permutations": 300,
        "kmeans_n_init": 15,
        "G_grid": [4, 6, 8],
    }

    train_results = {sc["name"]: [] for sc in scale_configs}
    d_med_train = []

    for i, rec_id in enumerate(train_ids):
        sig, fs = load_ptbxl_record(rec_id, n_samples=1000)
        d_med = compute_median_distance(sig)
        d_med_train.append(d_med)

        for sc in scale_configs:
            cfg_dict = dict(base_config_dict)
            cfg_dict["m_grid"] = sc["m_grid"]
            res = evaluate_single_config(
                signal=sig,
                fs=fs,
                config_dict=cfg_dict,
                m_ms_grid=sc["m_ms"],
                kernel_scale_rule=sc["rule"],
                kernel_param=sc["gamma"],
            )
            res["ecg_id"] = int(rec_id)
            res["d_med"] = d_med
            train_results[sc["name"]].append(res)

        print(f"  Processed training record {i+1}/{len(train_ids)} (ecg_id={rec_id}, d_med={d_med:.4f})")

    # Summarize Training Sweep
    print("\n--- TRAINING SET SCALE SWEEP SUMMARY ---")
    train_summary = {}
    for sc in scale_configs:
        name = sc["name"]
        df_sc = pd.DataFrame(train_results[name])
        mean_dens = float(df_sc["graph_density"].mean())
        mean_k = float(df_sc["n_clique_groups"].mean())
        mean_cc = float(df_sc["n_cc_groups"].mean())
        mean_overlap = float(df_sc["overlap_fraction"].mean())
        train_summary[name] = {
            "mean_density": mean_dens,
            "mean_clique_groups": mean_k,
            "mean_cc_groups": mean_cc,
            "mean_overlap_fraction": mean_overlap,
        }
        print(f"  {name:40s}: Density={mean_dens:.3f}, K={mean_k:.2f}, CC={mean_cc:.2f}, Overlap={mean_overlap*100:.1f}%")

    # Step 2: Freeze Best Calibrated Rule on Training Set
    # Standard kernel theory prescribes the median heuristic gamma = 1.0.
    # We freeze: c = 1.0 * d_med, Delta in {16, 32, 64, 128} ms.
    chosen_rule_name = "Relative c=1.0*d_med (ms grid)"
    print(f"\nStep 2: Freezing Invariant Rule on Training Set: {chosen_rule_name}")

    # Step 3: Cross-Cohort Evaluation (PTB-XL Test Fold 10 vs EchoNext Test)
    print("\nStep 3: Evaluating Frozen Rule vs Fixed Baseline on Held-Out Cohorts...")
    cohort_eval = {
        "PTBXL_Test_Fixed": [],
        "PTBXL_Test_Invariant": [],
        "EchoNext_Fixed": [],
        "EchoNext_Invariant": [],
    }

    # Evaluate PTB-XL Test
    for i, rec_id in enumerate(test_ids):
        sig, fs = load_ptbxl_record(rec_id, n_samples=1000)
        d_med = compute_median_distance(sig)

        # Baseline
        res_base = evaluate_single_config(
            signal=sig, fs=fs, config_dict=dict(base_config_dict, m_grid=[4, 6, 8]),
            kernel_scale_rule="fixed", kernel_param=1.0, m_ms_grid=None,
        )
        res_base["id"] = int(rec_id)
        res_base["d_med"] = d_med
        cohort_eval["PTBXL_Test_Fixed"].append(res_base)

        # Invariant
        res_inv = evaluate_single_config(
            signal=sig, fs=fs, config_dict=dict(base_config_dict, m_grid=[4, 6, 8]),
            kernel_scale_rule="median_heuristic", kernel_param=1.0, m_ms_grid=[16, 32, 64, 128],
        )
        res_inv["id"] = int(rec_id)
        res_inv["d_med"] = d_med
        cohort_eval["PTBXL_Test_Invariant"].append(res_inv)

    print(f"  Evaluated {len(test_ids)} held-out PTB-XL test records.")

    # Evaluate EchoNext Test
    d_med_echonext = []
    for i, idx in enumerate(echonext_indices):
        sig, fs = load_echonext_record(idx, n_samples=750)
        d_med = compute_median_distance(sig)
        d_med_echonext.append(d_med)

        # Baseline
        res_base = evaluate_single_config(
            signal=sig, fs=fs, config_dict=dict(base_config_dict, m_grid=[4, 6, 8]),
            kernel_scale_rule="fixed", kernel_param=1.0, m_ms_grid=None,
        )
        res_base["id"] = int(idx)
        res_base["d_med"] = d_med
        cohort_eval["EchoNext_Fixed"].append(res_base)

        # Invariant
        res_inv = evaluate_single_config(
            signal=sig, fs=fs, config_dict=dict(base_config_dict, m_grid=[4, 6, 8]),
            kernel_scale_rule="median_heuristic", kernel_param=1.0, m_ms_grid=[16, 32, 64, 128],
        )
        res_inv["id"] = int(idx)
        res_inv["d_med"] = d_med
        cohort_eval["EchoNext_Invariant"].append(res_inv)

    print(f"  Evaluated {len(echonext_indices)} EchoNext external test records.")

    # Compile Final Comparative Metrics
    comparison_table = []
    for cohort_key, records in cohort_eval.items():
        df_c = pd.DataFrame(records)
        comparison_table.append({
            "cohort_condition": cohort_key,
            "mean_d_med": float(df_c["d_med"].mean()),
            "mean_density": float(df_c["graph_density"].mean()),
            "std_density": float(df_c["graph_density"].std()),
            "mean_clique_groups": float(df_c["n_clique_groups"].mean()),
            "mean_cc_groups": float(df_c["n_cc_groups"].mean()),
            "mean_overlap_fraction": float(df_c["overlap_fraction"].mean()),
        })

    df_comp = pd.DataFrame(comparison_table)
    print("\n" + "=" * 80)
    print("CROSS-COHORT GENERALIZATION RESULTS: FIXED BASELINE VS INVARIANT RULE")
    print("=" * 80)
    print(df_comp.to_string(index=False))

    # Save to disk
    out_csv = os.path.join(output_dir, "scale_invariance_comparison.csv")
    df_comp.to_csv(out_csv, index=False)

    full_output = {
        "training_sweep_summary": train_summary,
        "frozen_rule": {
            "name": chosen_rule_name,
            "kernel_scale_rule": "median_heuristic",
            "gamma": 1.0,
            "m_ms_grid": [16, 32, 64, 128],
        },
        "cohort_comparison": comparison_table,
        "raw_train_d_med_mean": float(np.mean(d_med_train)),
        "raw_echonext_d_med_mean": float(np.mean(d_med_echonext)),
    }

    out_json = os.path.join(output_dir, "scale_invariance_summary.json")
    with open(out_json, "w") as f:
        json.dump(full_output, f, indent=2)

    print(f"\nSaved scale invariance summary to: {out_json}")


if __name__ == "__main__":
    main()
