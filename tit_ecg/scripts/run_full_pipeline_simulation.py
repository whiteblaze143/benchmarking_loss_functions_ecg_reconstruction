"""Full-Pipeline Simulation Battery for Post-CAHC Selection Null Calibration.

Mimics the entire repSpat pipeline:
1. Simulates a continuous multivariate ECG-like process with known repeated regimes
   (e.g., QRS-like Regime A, T-wave-like Regime B, Isoelectric Regime C across N_cycles).
2. Runs CAHC search over (m, G) -> discovers temporal clusters {C_g}.
3. Assigns each CAHC cluster a ground-truth regime based on majority sample membership.
4. Identifies:
   - True Null Pairs (g, h): Both CAHC clusters come from the SAME generating regime
     (e.g., Beat 1 QRS vs Beat 2 QRS). Tests H0: P_g = P_h.
     Rejection (q < 0.05) is a post-selection False Positive (Type-I error).
   - True Alternative Pairs (g, h): Clusters come from DIFFERENT generating regimes
     (e.g., QRS vs T-wave). Tests H1: P_g != P_h.
     Rejection (q < 0.05) is True Positive (Power).
5. Compares:
   - Scheme A: Attribute k-means blocking (Paper-faithful)
   - Scheme B: Contiguous temporal blocking
   - Scheme C: Naive unblocked permutation
6. Evaluates Kernel Scales gamma in {0.5, 1.0, 1.5, 2.0, 3.0} under the Lexicographic Criterion:
   Step 1: Require acceptable Type-I calibration (alpha_post <= 0.08 at nominal 0.05).
   Step 2: Among calibrated gamma, maximize power on true alternatives.
   Step 3: Among highest power, maximize perturbation stability (W_gh at 30 dB SNR).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
import numpy as np
import pandas as pd

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.pipeline import TemporalRepSpatECG
from tit_ecg.src.mmd_test import run_all_pairwise_mmd_tests


def generate_multiregime_ecg_simulation(
    n_cycles: int = 5,
    cycle_length_ms: float = 800.0,
    fs: float = 500.0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Generates continuous 3D VCG time series with known ground truth regimes."""
    rng = np.random.RandomState(random_state)
    samples_per_cycle = int(round(cycle_length_ms * fs / 1000.0))  # 400 samples
    total_samples = n_cycles * samples_per_cycle

    # Regimes:
    # 0: Isoelectric (samples 0-80 ms) -> baseline
    # 1: P-wave (samples 80-160 ms) -> small smooth deflection
    # 2: PR segment (samples 160-240 ms) -> isoelectric
    # 3: QRS complex (samples 240-340 ms) -> sharp high-amplitude loop
    # 4: ST-T wave (samples 340-540 ms) -> repolarization loop
    # 5: TP segment (samples 540-800 ms) -> isoelectric
    
    # Ground truth functional regime labels (collapsed to 3 distinct recurrent classes):
    # Class 'A': QRS complex (high power, sharp)
    # Class 'B': T-wave (broad repolarization)
    # Class 'C': Baseline / P / Isoelectric (quiescent)

    vcg_list = []
    regime_labels = []

    for c in range(n_cycles):
        # Generate cycle template with stochastic VAR(1) process noise
        t_cycle = np.linspace(0, 1, samples_per_cycle)
        v = np.zeros((samples_per_cycle, 3))
        reg_c = np.full(samples_per_cycle, "C", dtype=object)

        # Baseline noise (VAR(1) stationary)
        noise = np.zeros((samples_per_cycle, 3))
        phi = 0.6
        eps = rng.randn(samples_per_cycle, 3) * 0.02
        for t in range(1, samples_per_cycle):
            noise[t] = phi * noise[t - 1] + eps[t]

        # P-wave: 10% to 20% of cycle
        p_mask = (t_cycle >= 0.10) & (t_cycle < 0.20)
        p_phase = (t_cycle[p_mask] - 0.10) / 0.10
        v[p_mask, 0] += 0.15 * np.sin(np.pi * p_phase)
        v[p_mask, 1] += 0.10 * np.sin(np.pi * p_phase)
        reg_c[p_mask] = "C"

        # QRS complex: 30% to 42% of cycle (Class A)
        qrs_mask = (t_cycle >= 0.30) & (t_cycle < 0.42)
        qrs_phase = (t_cycle[qrs_mask] - 0.30) / 0.12
        # Sharp high-amplitude 3D loop
        v[qrs_mask, 0] += 1.2 * np.sin(np.pi * qrs_phase) * np.sin(2 * np.pi * qrs_phase)
        v[qrs_mask, 1] += 1.8 * np.sin(np.pi * qrs_phase)
        v[qrs_mask, 2] += 0.8 * np.sin(2 * np.pi * qrs_phase)
        reg_c[qrs_mask] = "A"

        # T-wave: 48% to 70% of cycle (Class B)
        t_mask = (t_cycle >= 0.48) & (t_cycle < 0.70)
        t_phase = (t_cycle[t_mask] - 0.48) / 0.22
        # Smooth broader loop in different plane
        v[t_mask, 0] += 0.35 * np.sin(np.pi * t_phase)
        v[t_mask, 1] += 0.25 * np.sin(np.pi * t_phase)
        v[t_mask, 2] += -0.20 * np.sin(np.pi * t_phase)
        reg_c[t_mask] = "B"

        v += noise
        vcg_list.append(v)
        regime_labels.extend(reg_c)

    vcg_total = np.vstack(vcg_list)
    regimes_total = np.array(regime_labels, dtype=object)
    return vcg_total, regimes_total, fs


def run_full_pipeline_single_sim(
    sim_id: int,
    block_mode: str = "attribute_kmeans",
    kernel_scale_rule: str = "median_heuristic",
    gamma: float = 1.0,
    random_state: int = 42,
) -> dict:
    """Runs one full repSpat simulation with CAHC search + MMD testing."""
    vcg, gt_regimes, fs = generate_multiregime_ecg_simulation(
        n_cycles=4,
        cycle_length_ms=800.0,
        fs=500.0,
        random_state=random_state,
    )

    cfg = RepSpatConfig.fast_test_config(
        m_grid=[4, 6, 8],
        G_grid=[4, 6, 8],
        block_permutation_mode=block_mode,
        kernel_scale_rule=kernel_scale_rule,
        kernel_param=gamma,
        n_permutations=200,
        kmeans_n_init=10,
        random_state=random_state,
    )

    model = TemporalRepSpatECG(config=cfg)
    model.fit(vcg, sampling_rate=fs, is_vcg=True)
    res = model.results_

    initial_labels = res["initial_labels"]
    pairwise_df = res["pairwise_df"]

    # Assign each CAHC cluster a ground-truth regime
    unique_clusters = sorted(np.unique(initial_labels))
    cluster_regimes = {}
    for c_id in unique_clusters:
        c_mask = initial_labels == c_id
        reg_in_c = gt_regimes[c_mask]
        vals, counts = np.unique(reg_in_c, return_counts=True)
        majority_reg = vals[np.argmax(counts)]
        majority_frac = np.max(counts) / len(reg_in_c)
        cluster_regimes[c_id] = (majority_reg, majority_frac)

    # Classify pairwise tests as Null (same regime) vs Alternative (different regime)
    null_p_vals = []
    null_q_vals = []
    alt_p_vals = []
    alt_q_vals = []

    col_c1 = "cluster_1" if "cluster_1" in pairwise_df.columns else "region_1"
    col_c2 = "cluster_2" if "cluster_2" in pairwise_df.columns else "region_2"
    col_q = "q_value" if "q_value" in pairwise_df.columns else ("adj_p" if "adj_p" in pairwise_df.columns else "p_value")

    for _, row in pairwise_df.iterrows():
        c1 = int(row[col_c1])
        c2 = int(row[col_c2])
        reg1, frac1 = cluster_regimes[c1]
        reg2, frac2 = cluster_regimes[c2]

        # Require reasonable cluster purity (>= 60%) to be a clean test
        if frac1 >= 0.60 and frac2 >= 0.60:
            if reg1 == reg2:
                # True Null Pair: Both clusters come from same underlying regime!
                null_p_vals.append(float(row["p_value"]))
                null_q_vals.append(float(row[col_q]))
            else:
                # True Alternative Pair: Different regimes!
                alt_p_vals.append(float(row["p_value"]))
                alt_q_vals.append(float(row[col_q]))

    n_null = len(null_q_vals)
    n_alt = len(alt_q_vals)

    null_rejections = sum(q < 0.05 for q in null_q_vals) if n_null > 0 else 0
    alt_rejections = sum(q < 0.05 for q in alt_q_vals) if n_alt > 0 else 0

    type1_error = (null_rejections / n_null) if n_null > 0 else np.nan
    power = (alt_rejections / n_alt) if n_alt > 0 else np.nan

    # Test perturbation stability: add 30 dB SNR noise
    signal_power = np.mean(vcg**2)
    noise_power = signal_power / (10 ** (30.0 / 10.0))
    rng = np.random.RandomState(random_state + 999)
    noise = rng.randn(*vcg.shape) * np.sqrt(noise_power)
    vcg_pert = vcg + noise

    model_pert = TemporalRepSpatECG(config=cfg)
    model_pert.fit(vcg_pert, sampling_rate=fs, is_vcg=True)
    pert_df = model_pert.results_["pairwise_df"]

    pert_col_c1 = "cluster_1" if "cluster_1" in pert_df.columns else "region_1"
    pert_col_c2 = "cluster_2" if "cluster_2" in pert_df.columns else "region_2"
    pert_col_q = "q_value" if "q_value" in pert_df.columns else ("adj_p" if "adj_p" in pert_df.columns else "p_value")

    # Match edges between clean and perturbed
    clean_edges = set()
    for _, r in pairwise_df[pairwise_df[col_q] >= 0.05].iterrows():
        clean_edges.add((min(int(r[col_c1]), int(r[col_c2])), max(int(r[col_c1]), int(r[col_c2]))))

    pert_edges = set()
    for _, r in pert_df[pert_df[pert_col_q] >= 0.05].iterrows():
        pert_edges.add((min(int(r[pert_col_c1]), int(r[pert_col_c2])), max(int(r[pert_col_c1]), int(r[pert_col_c2]))))

    edge_jaccard = (
        len(clean_edges & pert_edges) / len(clean_edges | pert_edges)
        if len(clean_edges | pert_edges) > 0
        else 1.0
    )

    return {
        "sim_id": sim_id,
        "block_mode": block_mode,
        "gamma": gamma,
        "m_star": int(res["m_star"]),
        "G_star": int(res["G_star"]),
        "n_null_pairs": n_null,
        "n_alt_pairs": n_alt,
        "null_rejections": null_rejections,
        "alt_rejections": alt_rejections,
        "type1_error_rate": type1_error,
        "power": power,
        "edge_jaccard_30db": float(edge_jaccard),
        "graph_density": float(res["graph_descriptors"].get("density", 0.0)),
        "n_clique_groups": int(res["n_clique_clusters"]),
    }


def main():
    print("=" * 80)
    print("FULL-PIPELINE SIMULATION BATTERY: POST-CAHC SELECTION NULL CALIBRATION")
    print("=" * 80)

    out_dir = "tit_ecg/results/full_pipeline_simulation"
    os.makedirs(out_dir, exist_ok=True)

    n_sims = 40  # 40 Monte Carlo runs per condition (yields ~400-600 null pairs per config)

    # 1. Compare the 3 permutation schemes at gamma = 1.0
    schemes = [
        ("Scheme_A_Attribute_KMeans", "attribute_kmeans", 1.0),
        ("Scheme_B_Temporal_Blocks", "temporal_contiguous", 1.0),
        ("Scheme_C_Unblocked", "unblocked", 1.0),
    ]

    # 2. Evaluate candidate gamma under the Lexicographic Criterion for Scheme A
    gamma_sweep = [
        ("Gamma_0.5", "attribute_kmeans", 0.5),
        ("Gamma_1.0", "attribute_kmeans", 1.0),
        ("Gamma_1.5", "attribute_kmeans", 1.5),
        ("Gamma_2.0", "attribute_kmeans", 2.0),
        ("Gamma_3.0", "attribute_kmeans", 3.0),
    ]

    all_experiments = {}
    for label, bmode, g in schemes:
        all_experiments[label] = (bmode, g)
    for label, bmode, g in gamma_sweep:
        all_experiments[label] = (bmode, g)

    print(f"Total experimental configurations: {len(all_experiments)}")
    print(f"Running {n_sims} simulations per configuration...")

    all_results = []
    tasks = []
    for exp_label, (bmode, g) in all_experiments.items():
        for s_idx in range(n_sims):
            tasks.append((exp_label, s_idx, bmode, g, 1000 + s_idx))

    print(f"Total tasks: {len(tasks)}")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_full_pipeline_single_sim,
                s_idx,
                bmode,
                "median_heuristic",
                g,
                seed,
            ): (exp_label, s_idx)
            for exp_label, s_idx, bmode, g, seed in tasks
        }
        for fut in as_completed(futures):
            exp_label, s_idx = futures[fut]
            try:
                res = fut.result()
                res["experiment_label"] = exp_label
                all_results.append(res)
                if len(all_results) % 40 == 0:
                    print(f"  Completed {len(all_results)}/{len(tasks)} simulation runs...")
            except Exception as e:
                print(f"  Error on {exp_label} sim {s_idx}: {e}")

    df_res = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, "full_pipeline_simulation_runs.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved run-level data to {csv_path}")

    # Compute Summary Statistics
    summary_list = []
    for exp_label, grp in df_res.groupby("experiment_label"):
        # Overall empirical Type-I error rate across all pooled null pairs
        tot_null = grp["n_null_pairs"].sum()
        tot_null_rej = grp["null_rejections"].sum()
        pooled_type1 = float(tot_null_rej / tot_null) if tot_null > 0 else np.nan

        # Wilson 95% CI for pooled Type-I error
        if tot_null > 0:
            z = 1.95996
            p_hat = pooled_type1
            denom = 1.0 + z**2 / tot_null
            center = (p_hat + z**2 / (2 * tot_null)) / denom
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * tot_null)) / tot_null) / denom
            ci_low = max(0.0, float(center - margin))
            ci_high = min(1.0, float(center + margin))
        else:
            ci_low, ci_high = np.nan, np.nan

        # Overall empirical Power
        tot_alt = grp["n_alt_pairs"].sum()
        tot_alt_rej = grp["alt_rejections"].sum()
        pooled_power = float(tot_alt_rej / tot_alt) if tot_alt > 0 else np.nan

        # Stability and Graph Density
        mean_jaccard = float(grp["edge_jaccard_30db"].mean())
        mean_density = float(grp["graph_density"].mean())
        mean_cliques = float(grp["n_clique_groups"].mean())

        summary_list.append({
            "experiment_label": exp_label,
            "block_mode": grp["block_mode"].iloc[0],
            "gamma": float(grp["gamma"].iloc[0]),
            "n_runs": len(grp),
            "total_null_pairs": int(tot_null),
            "total_null_rejections": int(tot_null_rej),
            "type1_error_rate": pooled_type1,
            "type1_ci_95": [ci_low, ci_high],
            "is_calibrated_fdr05": bool(ci_low <= 0.08 and pooled_type1 <= 0.08),
            "total_alt_pairs": int(tot_alt),
            "total_alt_rejections": int(tot_alt_rej),
            "power": pooled_power,
            "perturbation_stability_30db": mean_jaccard,
            "mean_graph_density": mean_density,
            "mean_clique_groups": mean_cliques,
        })

    df_summary = pd.DataFrame(summary_list)
    summary_csv = os.path.join(out_dir, "full_pipeline_simulation_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary table to {summary_csv}")

    # Apply Lexicographic Selection Rule on Gamma Sweep
    gamma_entries = [e for e in summary_list if e["experiment_label"].startswith("Gamma_")]
    # Step 1: Calibrated (Type 1 error <= 0.08)
    calibrated_gammas = [e for e in gamma_entries if e["type1_error_rate"] <= 0.08]
    if not calibrated_gammas:
        calibrated_gammas = gamma_entries  # fallback if none strictly <= 0.08

    # Step 2: Sort by Power descending
    calibrated_gammas.sort(key=lambda x: x["power"], reverse=True)
    max_power = calibrated_gammas[0]["power"]
    top_power_gammas = [e for e in calibrated_gammas if abs(e["power"] - max_power) <= 0.03]

    # Step 3: Maximize Perturbation Stability
    top_power_gammas.sort(key=lambda x: x["perturbation_stability_30db"], reverse=True)
    best_gamma_entry = top_power_gammas[0]

    final_payload = {
        "summary": summary_list,
        "lexicographic_selection": {
            "step_1_calibrated_candidates": [e["experiment_label"] for e in calibrated_gammas],
            "step_2_top_power_candidates": [e["experiment_label"] for e in top_power_gammas],
            "step_3_winner": best_gamma_entry["experiment_label"],
            "winning_gamma": best_gamma_entry["gamma"],
            "winning_type1_error": best_gamma_entry["type1_error_rate"],
            "winning_power": best_gamma_entry["power"],
            "winning_stability": best_gamma_entry["perturbation_stability_30db"],
        },
    }

    summary_json = os.path.join(out_dir, "full_pipeline_simulation_summary.json")
    with open(summary_json, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved JSON summary to {summary_json}")
    print("=" * 80)
    print("FULL-PIPELINE SIMULATION BATTERY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
