"""Empirical Multi-Dataset repSpat CAHC Benchmark Suite.

Benchmarks the exact repSpat CAHC adaptation across four real clinical datasets:
1. LUDB: 200 real patient ECGs with gold-standard physician wave delineations (P, QRS, T).
2. RDB: 2,398 real patient ECGs with wavelet delineations across rhythm categories.
3. PTB-XL: 21,837 multi-lead ECGs across diagnostic superclasses (NORM, MI, STTC, CD, HYP).
4. EchoNext: Real echocardiogram-paired cohort at 250 Hz.

Key Empirical Benchmark Targets:
1. Post-CAHC Selection Null Calibration on Real Patient Heartbeats:
   - Real Recurring Null Pairs: Clusters representing the SAME wave across DIFFERENT beats
     (e.g., Beat 1 QRS vs Beat 2 QRS of the same patient). Tests H0: P_g = P_h.
     Measures empirical Type-I error post-CAHC selection.
   - Real Distinct Alternative Pairs: Clusters representing DIFFERENT waves
     (e.g., QRS vs T-wave). Tests H1: P_g != P_h.
     Measures empirical Power.
   - Compares Scheme A (attribute k-means), Scheme B (temporal blocks), Scheme C (unblocked).
2. Lexicographic Selection of gamma in {0.5, 1.0, 1.5, 2.0, 3.0} on Real Data:
   - Step 1: Filter to calibrated gamma (empirical Type-I <= 0.08).
   - Step 2: Maximize power on distinct cardiac phases.
   - Step 3: Maximize perturbation stability under 30 dB SNR.
3. Cross-Dataset Topological Graph Comparison:
   - Quantifies m*, G*, graph density rho, maximal cliques K, CCs, overlap fraction omega,
     and edge retention probability across all four real datasets.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import json
import os
import sys
import time
from typing import Any
import numpy as np
import pandas as pd

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.dataset_adapters import (
    EchoNextAdapter,
    EmoryAdapter,
    ISPAdapter,
    KingstonICUAdapter,
    LUDBAdapter,
    PTBXLAdapter,
    RDBAdapter,
    SunnybrookAdapter,
    ZhejiangAdapter,
)
from tit_ecg.src.pipeline import TemporalRepSpatECG


# Gate 3B uses exact membership, not a tunable purity threshold: every sample in
# an admissible CAHC cluster must belong to one annotated phase and one beat.
GATE3B_REQUIRED_PURITY = 1.0


def beat_ids_from_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Assign every sample to the nearest annotated QRS occurrence."""
    qrs = np.asarray(segmentation) == 2
    starts = np.flatnonzero(qrs & ~np.r_[False, qrs[:-1]])
    ends = np.flatnonzero(qrs & ~np.r_[qrs[1:], False])
    if len(starts) == 0:
        return np.zeros(len(qrs), dtype=int)

    centers = (starts + ends) / 2.0
    boundaries = (centers[:-1] + centers[1:]) / 2.0
    return np.searchsorted(boundaries, np.arange(len(qrs)), side="right") + 1


def evaluate_empirical_null_and_power_record(
    dataset_name: str,
    record_id: Any,
    vcg: np.ndarray,
    fs: float,
    segmentation: np.ndarray,
    beat_ids: np.ndarray,
    block_mode: str = "attribute_kmeans",
    gamma: float = 1.0,
    m_ms_grid: list[float] | None = None,
    G_grid: list[int] | None = None,
    n_permutations: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Runs full repSpat on a delineated real patient record and evaluates empirical null/alt pairs."""
    if m_ms_grid is None:
        m_ms_grid = [32.0, 64.0, 128.0]
    if G_grid is None:
        G_grid = [4, 6, 8]

    m_samples = [max(2, int(round(ms * fs / 1000.0))) for ms in m_ms_grid]
    m_samples = sorted(list(set(m_samples)))

    cfg = RepSpatConfig.fast_test_config(
        m_grid=m_samples,
        G_grid=G_grid,
        block_permutation_mode=block_mode,
        kernel_scale_rule="median_heuristic",
        kernel_param=gamma,
        n_permutations=n_permutations,
        kmeans_n_init=10,
        random_state=random_state,
    )
    cfg.m_ms_grid = m_ms_grid

    t0 = time.time()
    model = TemporalRepSpatECG(config=cfg)
    model.fit(vcg, sampling_rate=fs, is_vcg=True)
    elapsed = time.time() - t0

    res = model.results_
    initial_labels = res["initial_labels"]
    pairwise_df = res["pairwise_df"]

    # Analyze CAHC clusters against ground-truth wave segmentation
    # Classes: 1=P, 2=QRS, 3=T, 0=Isoelectric
    cluster_wave_map = {}
    cluster_beat_map = {}
    cluster_centers = {}
    cluster_sizes = {}
    cluster_means = {}
    for c_id in np.unique(initial_labels):
        c_mask = initial_labels == c_id
        c_samples = np.where(c_mask)[0]
        cluster_centers[c_id] = float(np.mean(c_samples))
        cluster_sizes[c_id] = int(len(c_samples))
        cluster_means[c_id] = np.mean(vcg[c_mask], axis=0)

        seg_in_c = segmentation[c_mask]
        vals, counts = np.unique(seg_in_c, return_counts=True)
        majority_wave = vals[np.argmax(counts)]
        majority_frac = np.max(counts) / len(seg_in_c)
        cluster_wave_map[c_id] = (int(majority_wave), float(majority_frac))

        beats_in_c = beat_ids[c_mask]
        beat_vals, beat_counts = np.unique(beats_in_c[beats_in_c > 0], return_counts=True)
        if len(beat_vals):
            majority_beat_index = int(np.argmax(beat_counts))
            cluster_beat_map[c_id] = (
                int(beat_vals[majority_beat_index]),
                float(beat_counts[majority_beat_index] / len(beats_in_c)),
            )
        else:
            cluster_beat_map[c_id] = (0, 0.0)

    # Evaluate pairwise tests
    recurrence_p_vals = []
    recurrence_q_vals = []
    alt_p_vals = []
    alt_q_vals = []
    diagnostic_pairs = []

    qrs = np.asarray(segmentation) == 2
    qrs_starts = np.flatnonzero(qrs & ~np.r_[False, qrs[:-1]])
    qrs_ends = np.flatnonzero(qrs & ~np.r_[qrs[1:], False])
    qrs_centers = (qrs_starts + qrs_ends) / 2.0
    rr_seconds = np.diff(qrs_centers) / fs
    heart_rate_bpm = float(60.0 / np.median(rr_seconds)) if len(rr_seconds) else np.nan

    col_c1 = "cluster_1" if "cluster_1" in pairwise_df.columns else "region_1"
    col_c2 = "cluster_2" if "cluster_2" in pairwise_df.columns else "region_2"
    col_q = "q_value" if "q_value" in pairwise_df.columns else ("adj_p" if "adj_p" in pairwise_df.columns else "p_value")

    for _, row in pairwise_df.iterrows():
        c1 = int(row[col_c1])
        c2 = int(row[col_c2])
        w1, frac1 = cluster_wave_map[c1]
        w2, frac2 = cluster_wave_map[c2]
        b1, beat_frac1 = cluster_beat_map[c1]
        b2, beat_frac2 = cluster_beat_map[c2]

        if (
            frac1 == GATE3B_REQUIRED_PURITY
            and frac2 == GATE3B_REQUIRED_PURITY
            and beat_frac1 == GATE3B_REQUIRED_PURITY
            and beat_frac2 == GATE3B_REQUIRED_PURITY
        ):
            time_sep_ms = abs(cluster_centers[c1] - cluster_centers[c2]) * 1000.0 / fs
            pair_kind = None

            if w1 == w2 and w1 in [1, 2, 3] and b1 != b2 and time_sep_ms >= 200.0:
                pair_kind = "recurrence_control"
                recurrence_p_vals.append(float(row["p_value"]))
                recurrence_q_vals.append(float(row[col_q]))
            elif w1 != w2 and (w1 in [1, 2, 3] and w2 in [1, 2, 3]):
                pair_kind = "positive_control"
                alt_p_vals.append(float(row["p_value"]))
                alt_q_vals.append(float(row[col_q]))

            if pair_kind is not None:
                diagnostic_pairs.append({
                    "dataset": dataset_name,
                    "record_id": str(record_id),
                    "patient_id": f"{dataset_name}:{record_id}",
                    "block_mode": block_mode,
                    "pair_kind": pair_kind,
                    "phase_1": w1,
                    "phase_2": w2,
                    "beat_1": b1,
                    "beat_2": b2,
                    "cluster_1_size": cluster_sizes[c1],
                    "cluster_2_size": cluster_sizes[c2],
                    "min_cluster_size": min(cluster_sizes[c1], cluster_sizes[c2]),
                    "m_star": int(res["m_star"]),
                    "G_star": int(res["G_star"]),
                    "temporal_separation_ms": float(time_sep_ms),
                    "heart_rate_bpm": heart_rate_bpm,
                    "morphology_distance": float(np.linalg.norm(cluster_means[c1] - cluster_means[c2])),
                    "p_value": float(row["p_value"]),
                    "q_value": float(row[col_q]),
                    "rejected_by_graph_rule": bool(row[col_q] < cfg.fdr_alpha),
                })

    n_recurrence = len(recurrence_q_vals)
    n_alt = len(alt_q_vals)

    recurrence_rejections = sum(q < 0.05 for q in recurrence_q_vals) if n_recurrence > 0 else 0
    alt_rejections = sum(q < 0.05 for q in alt_q_vals) if n_alt > 0 else 0

    recurrence_rejection_rate = (recurrence_rejections / n_recurrence) if n_recurrence > 0 else np.nan
    power = (alt_rejections / n_alt) if n_alt > 0 else np.nan

    # Test perturbation stability at 30 dB SNR
    signal_power = np.mean(vcg**2)
    noise_power = signal_power / (10 ** (30.0 / 10.0))
    rng = np.random.RandomState(random_state + 777)
    noise = rng.randn(*vcg.shape) * np.sqrt(noise_power)
    vcg_pert = vcg + noise

    model_pert = TemporalRepSpatECG(config=cfg)
    model_pert.fit(vcg_pert, sampling_rate=fs, is_vcg=True)
    pert_df = model_pert.results_["pairwise_df"]

    pert_col_c1 = "cluster_1" if "cluster_1" in pert_df.columns else "region_1"
    pert_col_c2 = "cluster_2" if "cluster_2" in pert_df.columns else "region_2"
    pert_col_q = "q_value" if "q_value" in pert_df.columns else ("adj_p" if "adj_p" in pert_df.columns else "p_value")

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

    g_desc = res["graph_descriptors"]

    return {
        "dataset": dataset_name,
        "record_id": str(record_id),
        "block_mode": block_mode,
        "gamma": gamma,
        "m_star": int(res["m_star"]),
        "G_star": int(res["G_star"]),
        "purity_rule": "exact_phase_and_beat_membership",
        "n_recurrence_control_pairs": n_recurrence,
        "recurrence_rejections": recurrence_rejections,
        "recurrence_rejection_rate": recurrence_rejection_rate,
        "n_alt_pairs": n_alt,
        "alt_rejections": alt_rejections,
        "power": power,
        "graph_density": float(g_desc.get("density", 0.0)),
        "n_clique_groups": int(res["n_clique_clusters"]),
        "n_cc_groups": int(res["n_cc_clusters"]),
        "overlap_fraction": float(g_desc.get("overlap_fraction", 0.0)),
        "edge_jaccard_30db": float(edge_jaccard),
        "elapsed_sec": float(elapsed),
        "diagnostic_pairs": diagnostic_pairs,
    }


def run_cross_dataset_topological_record(
    adapter: Any,
    record_id: Any,
    gamma: float = 1.0,
    block_mode: str = "attribute_kmeans",
    random_state: int = 42,
) -> dict[str, Any]:
    """Runs repSpat on any dataset record and extracts topological descriptors."""
    rec = adapter.load_record(record_id)
    vcg = rec["vcg"]
    fs = rec["fs"]

    cfg = adapter.get_cahc_config(
        gamma=gamma,
        block_mode=block_mode,
        n_permutations=200,
    )
    cfg.random_state = random_state

    t0 = time.time()
    model = TemporalRepSpatECG(config=cfg)
    model.fit(vcg, sampling_rate=fs, is_vcg=True)
    elapsed = time.time() - t0

    res = model.results_
    g_desc = res["graph_descriptors"]

    # Compute d_med
    diffs = vcg[:400, None, :] - vcg[None, :400, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    triu = dists[np.triu_indices(len(dists), k=1)]
    d_med = float(np.median(triu[triu > 0])) if len(triu[triu > 0]) > 0 else 1.0

    return {
        "dataset": adapter.name,
        "record_id": str(record_id),
        "fs": fs,
        "d_med": d_med,
        "gamma": gamma,
        "m_star": int(res["m_star"]),
        "m_star_ms": float(res["m_star"] * 1000.0 / fs),
        "G_star": int(res["G_star"]),
        "n_similarity_edges": int(len(res["similarity_graph"].edges())),
        "graph_density": float(g_desc.get("density", 0.0)),
        "n_clique_groups": int(res["n_clique_clusters"]),
        "n_cc_groups": int(res["n_cc_clusters"]),
        "n_maximal_cliques": int(g_desc.get("n_maximal_cliques", 0)),
        "overlap_fraction": float(g_desc.get("overlap_fraction", 0.0)),
        "is_disconnected": int(len(res["similarity_graph"].edges()) == 0),
        "elapsed_sec": float(elapsed),
    }


def main(gate3b_only: bool = False, diagnostic_only: bool = False):
    print("=" * 80)
    print("EMPIRICAL MULTI-DATASET REPSPAT CAHC BENCHMARK SUITE (9 REAL DATASETS)")
    print("=" * 80)

    out_dir = "tit_ecg/results/empirical_multi_dataset"
    os.makedirs(out_dir, exist_ok=True)

    # Initialize all 9 dataset adapters
    ptb = PTBXLAdapter()
    echo = EchoNextAdapter()
    ludb = LUDBAdapter()
    rdb = RDBAdapter()
    isp = ISPAdapter()
    kingston = KingstonICUAdapter()
    emory = EmoryAdapter()
    sunny = SunnybrookAdapter()
    zhejiang = ZhejiangAdapter()

    all_adapters = [ptb, echo, ludb, rdb, isp, kingston, emory, sunny, zhejiang]
    print(f"Initialized {len(all_adapters)} empirical dataset adapters:")
    for ad in all_adapters:
        print(f"  - {ad.name:15s}: {ad.sampling_rate:6.1f} Hz")

    # -------------------------------------------------------------------------
    # EXPERIMENT 1: Empirical Post-CAHC Selection Null Calibration
    # Evaluated across all 4 delineated empirical cohorts: LUDB, RDB, ISP, Zhejiang
    # -------------------------------------------------------------------------
    print("\n--- Phase 1: Empirical Null Calibration & Scheme Comparison across Delineated Cohorts ---")
    ludb_records = ludb.list_records(max_records=10)
    rdb_records = rdb.list_records(max_records=10)
    isp_records = isp.list_records(max_records=10)
    zhejiang_records = zhejiang.list_records(max_records=10)

    print(f"Loaded delineated records: {len(ludb_records)} LUDB, {len(rdb_records)} RDB, {len(isp_records)} ISP, {len(zhejiang_records)} Zhejiang.")

    # Schemes: A (Attribute k-means), B (Temporal contiguous blocks), C (Unblocked) at gamma = 1.0
    schemes = ["attribute_kmeans"] if diagnostic_only else ["attribute_kmeans", "temporal_contiguous", "unblocked"]

    # Preload delineated records
    delineated_data = []
    for rid in ludb_records:
        rec = ludb.load_record(rid, n_samples=1500)
        delineated_data.append(("LUDB", rid, rec["vcg"], rec["fs"], rec["segmentation"], beat_ids_from_segmentation(rec["segmentation"])))
    for rid in rdb_records:
        rec = rdb.load_record(rid, n_samples=1500)
        delineated_data.append(("RDB", rec["record_id"], rec["vcg"], rec["fs"], rec["segmentation"], beat_ids_from_segmentation(rec["segmentation"])))
    for rid in isp_records:
        rec = isp.load_record(rid, n_samples=3000)
        delineated_data.append(("ISP", rid, rec["vcg"], rec["fs"], rec["segmentation"], beat_ids_from_segmentation(rec["segmentation"])))
    for rid in zhejiang_records:
        rec = zhejiang.load_record(rid, n_samples=2000)
        delineated_data.append(("Zhejiang", rid, rec["vcg"], rec["fs"], rec["segmentation"], beat_ids_from_segmentation(rec["segmentation"])))

    phase1_tasks = []
    for dname, rid, vcg, fs, seg, beat_ids in delineated_data:
        for sch in schemes:
            phase1_tasks.append((dname, rid, vcg, fs, seg, beat_ids, sch, 1.0))

    print(f"Total Phase 1 tasks: {len(phase1_tasks)} ({len(delineated_data)} delineated records x {len(schemes)} schemes)")

    phase1_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                evaluate_empirical_null_and_power_record,
                dname,
                rid,
                vcg,
                fs,
                seg,
                beat_ids,
                sch,
                g,
            ): (dname, rid, sch)
            for dname, rid, vcg, fs, seg, beat_ids, sch, g in phase1_tasks
        }
        for fut in as_completed(futures):
            info = futures[fut]
            try:
                res = fut.result()
                phase1_results.append(res)
                if len(phase1_results) % 20 == 0:
                    print(f"  Completed {len(phase1_results)}/{len(phase1_tasks)} Phase 1 runs...")
            except Exception as e:
                print(f"  Error on {info}: {e}")

    diagnostic_pairs = [pair for result in phase1_results for pair in result.pop("diagnostic_pairs")]
    df_pairs = pd.DataFrame(diagnostic_pairs)
    pair_csv = os.path.join(out_dir, "gate3b_pair_diagnostics.csv")
    df_pairs.to_csv(pair_csv, index=False)
    print(f"Saved pair-level diagnostics to {pair_csv}")

    df_p1 = pd.DataFrame(phase1_results)
    run_stem = "gate3b_attribute_diagnostic" if diagnostic_only else "empirical_recurrence_calibration"
    p1_csv = os.path.join(out_dir, f"{run_stem}_runs.csv")
    df_p1.to_csv(p1_csv, index=False)
    print(f"Saved Phase 1 runs to {p1_csv}")

    # Summary table for Scheme comparison on real heartbeats
    p1_summary = []
    for (dname, sch), grp in df_p1.groupby(["dataset", "block_mode"]):
        total_recurrence = grp["n_recurrence_control_pairs"].sum()
        total_recurrence_rejections = grp["recurrence_rejections"].sum()
        pooled_recurrence_rate = float(total_recurrence_rejections / total_recurrence) if total_recurrence > 0 else np.nan

        tot_alt = grp["n_alt_pairs"].sum()
        tot_alt_rej = grp["alt_rejections"].sum()
        pooled_power = float(tot_alt_rej / tot_alt) if tot_alt > 0 else np.nan
        eligible = grp[grp["n_recurrence_control_pairs"] > 0]
        patient_level_rate = float(eligible["recurrence_rejection_rate"].mean()) if len(eligible) else np.nan

        p1_summary.append({
            "dataset": dname,
            "permutation_scheme": sch,
            "n_records": len(grp),
            "total_recurrence_control_pairs": int(total_recurrence),
            "recurrence_rejections": int(total_recurrence_rejections),
            "pooled_recurrence_rejection_rate": pooled_recurrence_rate,
            "patient_level_mean_recurrence_rejection_rate": patient_level_rate,
            "n_patients_with_recurrence_pairs": int(len(eligible)),
            "total_alt_pairs": int(tot_alt),
            "alt_rejections": int(tot_alt_rej),
            "empirical_power": pooled_power,
            "mean_edge_jaccard_30db": float(grp["edge_jaccard_30db"].mean()),
            "mean_density": float(grp["graph_density"].mean()),
            "mean_cliques": float(grp["n_clique_groups"].mean()),
        })

    df_p1_sum = pd.DataFrame(p1_summary)
    p1_sum_csv = os.path.join(out_dir, f"{run_stem}_summary.csv")
    df_p1_sum.to_csv(p1_sum_csv, index=False)
    print(f"Saved Phase 1 summary table to {p1_sum_csv}")

    if gate3b_only:
        json_name = "gate3b_attribute_diagnostic_summary.json" if diagnostic_only else "gate3b_patient_level_summary.json"
        gate3b_json = os.path.join(out_dir, json_name)
        with open(gate3b_json, "w") as f:
            json.dump(
                {
                    "purity_rule": "exact_phase_and_beat_membership",
                    "patient_level_results": p1_summary,
                },
                f,
                indent=2,
            )
        print(f"Saved Gate 3B patient-level summary to {gate3b_json}")
        return

    # -------------------------------------------------------------------------
    # EXPERIMENT 2: Lexicographic Gamma Sweep on Real Patient Data (Scheme A)
    # Tested across both 500 Hz (LUDB) and 1000 Hz (ISP) high-precision records
    # -------------------------------------------------------------------------
    print("\n--- Phase 2: Lexicographic Gamma Calibration on Real ECG Heartbeats ---")
    gamma_candidates = [0.5, 1.0, 1.5, 2.0, 3.0]
    phase2_tasks = []
    # Test on LUDB and ISP records
    p2_records = [d for d in delineated_data if d[0] in ["LUDB", "ISP"]][:16]
    for dname, rid, vcg, fs, seg, beat_ids in p2_records:
        for g in gamma_candidates:
            phase2_tasks.append((dname, rid, vcg, fs, seg, beat_ids, "attribute_kmeans", g))

    print(f"Total Phase 2 tasks: {len(phase2_tasks)} ({len(gamma_candidates)} gammas x {len(p2_records)} records)")

    phase2_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                evaluate_empirical_null_and_power_record,
                dname,
                rid,
                vcg,
                fs,
                seg,
                beat_ids,
                "attribute_kmeans",
                g,
            ): (dname, rid, g)
            for dname, rid, vcg, fs, seg, beat_ids, _, g in phase2_tasks
        }
        for fut in as_completed(futures):
            info = futures[fut]
            try:
                res = fut.result()
                phase2_results.append(res)
                if len(phase2_results) % 20 == 0:
                    print(f"  Completed {len(phase2_results)}/{len(phase2_tasks)} Phase 2 runs...")
            except Exception as e:
                print(f"  Error on {info}: {e}")

    df_p2 = pd.DataFrame(phase2_results)
    p2_csv = os.path.join(out_dir, "empirical_gamma_sweep_runs.csv")
    df_p2.to_csv(p2_csv, index=False)

    p2_summary = []
    for g, grp in df_p2.groupby("gamma"):
        total_recurrence = grp["n_recurrence_control_pairs"].sum()
        total_recurrence_rejections = grp["recurrence_rejections"].sum()
        recurrence_rate = float(total_recurrence_rejections / total_recurrence) if total_recurrence > 0 else np.nan

        tot_alt = grp["n_alt_pairs"].sum()
        tot_alt_rej = grp["alt_rejections"].sum()
        power = float(tot_alt_rej / tot_alt) if tot_alt > 0 else np.nan
        eligible = grp[grp["n_recurrence_control_pairs"] > 0]

        p2_summary.append({
            "gamma": float(g),
            "n_records": len(grp),
            "total_recurrence_control_pairs": int(total_recurrence),
            "recurrence_rejections": int(total_recurrence_rejections),
            "pooled_recurrence_rejection_rate": recurrence_rate,
            "patient_level_mean_recurrence_rejection_rate": float(eligible["recurrence_rejection_rate"].mean()) if len(eligible) else np.nan,
            "n_patients_with_recurrence_pairs": int(len(eligible)),
            "total_alt_pairs": int(tot_alt),
            "power": power,
            "perturbation_stability_30db": float(grp["edge_jaccard_30db"].mean()),
            "mean_graph_density": float(grp["graph_density"].mean()),
            "mean_cliques": float(grp["n_clique_groups"].mean()),
        })

    df_p2_sum = pd.DataFrame(p2_summary)
    p2_sum_csv = os.path.join(out_dir, "empirical_gamma_sweep_summary.csv")
    df_p2_sum.to_csv(p2_sum_csv, index=False)
    print(f"Saved Phase 2 gamma summary to {p2_sum_csv}")

    # -------------------------------------------------------------------------
    # EXPERIMENT 3: Comprehensive Cross-Dataset Topological Graph Characterization
    # Evaluated across ALL 9 REAL CLINICAL DATASETS
    # -------------------------------------------------------------------------
    print("\n--- Phase 3: Cross-Dataset repSpat CAHC Topological Characterization (All 9 Cohorts) ---")
    phase3_tasks = []

    for ad in all_adapters:
        if ad.name == "PTB-XL":
            rids = ad.list_records(max_records=12, fold=10)
        else:
            rids = ad.list_records(max_records=12)
        print(f"  Enqueueing {len(rids)} records for {ad.name}...")
        for rid in rids:
            phase3_tasks.append((ad, rid))

    print(f"Total Phase 3 tasks: {len(phase3_tasks)} records across {len(all_adapters)} empirical datasets")

    phase3_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_cross_dataset_topological_record,
                ad,
                rid,
                1.0,
                "attribute_kmeans",
            ): (ad.name, rid)
            for ad, rid in phase3_tasks
        }
        for fut in as_completed(futures):
            info = futures[fut]
            try:
                res = fut.result()
                phase3_results.append(res)
                if len(phase3_results) % 15 == 0:
                    print(f"  Completed {len(phase3_results)}/{len(phase3_tasks)} Phase 3 runs...")
            except Exception as e:
                print(f"  Error on {info}: {e}")

    df_p3 = pd.DataFrame(phase3_results)
    p3_csv = os.path.join(out_dir, "empirical_cross_dataset_topological_runs.csv")
    df_p3.to_csv(p3_csv, index=False)

    p3_summary = df_p3.groupby("dataset").agg(
        n_records=("record_id", "count"),
        mean_fs=("fs", "mean"),
        mean_d_med=("d_med", "mean"),
        mean_m_star=("m_star", "mean"),
        mean_m_star_ms=("m_star_ms", "mean"),
        mean_G_star=("G_star", "mean"),
        mean_density=("graph_density", "mean"),
        std_density=("graph_density", "std"),
        mean_cliques=("n_clique_groups", "mean"),
        mean_cc=("n_cc_groups", "mean"),
        mean_overlap=("overlap_fraction", "mean"),
        disconnected_rate=("is_disconnected", "mean"),
    ).reset_index()

    p3_sum_csv = os.path.join(out_dir, "empirical_cross_dataset_topological_summary.csv")
    p3_summary.to_csv(p3_sum_csv, index=False)
    print(f"Saved Phase 3 cross-dataset summary to {p3_sum_csv}")

    # Master JSON payload
    master_dict = {
        "empirical_null_calibration": p1_summary,
        "empirical_gamma_calibration": p2_summary,
        "cross_dataset_topology": p3_summary.to_dict(orient="records"),
    }

    master_json = os.path.join(out_dir, "empirical_multi_dataset_master_summary.json")
    with open(master_json, "w") as f:
        json.dump(master_dict, f, indent=2)
    print(f"Saved Master JSON summary to {master_json}")
    print("=" * 80)
    print("EMPIRICAL MULTI-DATASET BENCHMARK SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate3b-only", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    main(gate3b_only=args.gate3b_only, diagnostic_only=args.diagnostic_only)
