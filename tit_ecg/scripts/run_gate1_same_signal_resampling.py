"""Gate 1: Same-Signal Resampling Equivariance Experiment.

Isolates sampling frequency directly on the SAME real empirical ECG recording,
preventing cross-cohort confounding (institution, population, device, pathology).

Deterministic downsampling with antialiasing (resample_poly):
  1000 Hz -> 500 Hz -> 250 Hz

Tests whether physical millisecond neighborhood mapping:
  m = floor(Delta_span * f_s / 1000)
preserves cluster partitions and similarity graph topology across sampling frequencies
on the exact same underlying electrophysiological event.
"""
from __future__ import annotations

import json
import os
import time
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from sklearn.metrics import adjusted_rand_score

from tit_ecg.src.config import RepSpatConfig
from tit_ecg.src.dataset_adapters import ISPAdapter
from tit_ecg.src.pipeline import TemporalRepSpatECG


def evaluate_record_resampling(record_id: int, n_seconds: float = 2.5) -> dict:
    isp = ISPAdapter()
    n_samples_1000 = int(round(n_seconds * 1000.0))
    rec = isp.load_record(record_id, n_samples=n_samples_1000)
    vcg_1000 = rec["vcg"]
    vcg_500 = resample_poly(vcg_1000, 1, 2, axis=0)
    vcg_250 = resample_poly(vcg_1000, 1, 4, axis=0)

    delta_spans = [32.0, 64.0, 128.0]  # ms

    # Condition 1: Physical Millisecond Geometry (Scale-Equivariant)
    geom_results = {}
    for freq_name, vcg_sig, fs in [("1000Hz", vcg_1000, 1000.0), ("500Hz", vcg_500, 500.0), ("250Hz", vcg_250, 250.0)]:
        m_samples = [max(2, int(round(ms * fs / 1000.0))) for ms in delta_spans]
        cfg = RepSpatConfig.fast_test_config(
            m_grid=m_samples,
            G_grid=[4, 6],
            kernel_scale_rule="median_heuristic",
            kernel_param=1.0,
            n_permutations=100,
            random_state=42,
        )
        model = TemporalRepSpatECG(config=cfg)
        model.fit(vcg_sig, sampling_rate=fs, is_vcg=True)
        labels = model.results_["initial_labels"]

        # Map to continuous time grid (1000 Hz)
        t_orig = np.linspace(0, n_seconds, len(labels))
        t_1000 = np.linspace(0, n_seconds, n_samples_1000)
        idx = np.clip(np.searchsorted(t_orig, t_1000), 0, len(labels) - 1)
        labels_interp = labels[idx]

        g = model.results_["graph_descriptors"]
        geom_results[freq_name] = {
            "m_star": int(model.results_["m_star"]),
            "m_star_ms": float(model.results_["m_star"] * 1000.0 / fs),
            "G_star": int(model.results_["G_star"]),
            "density": float(g.get("density", 0.0)),
            "cliques": int(model.results_["n_clique_clusters"]),
            "cc": int(model.results_["n_cc_clusters"]),
            "labels": labels_interp,
        }

    # Condition 2: Uncorrected Discrete Sample Grid (Arbitrary m)
    disc_results = {}
    for freq_name, vcg_sig, fs in [("1000Hz", vcg_1000, 1000.0), ("500Hz", vcg_500, 500.0), ("250Hz", vcg_250, 250.0)]:
        cfg_disc = RepSpatConfig.fast_test_config(
            m_grid=[4, 8, 16],  # unadjusted sample counts
            G_grid=[4, 6],
            kernel_scale_rule="median_heuristic",
            kernel_param=1.0,
            n_permutations=100,
            random_state=42,
        )
        model = TemporalRepSpatECG(config=cfg_disc)
        model.fit(vcg_sig, sampling_rate=fs, is_vcg=True)
        labels = model.results_["initial_labels"]

        t_orig = np.linspace(0, n_seconds, len(labels))
        t_1000 = np.linspace(0, n_seconds, n_samples_1000)
        idx = np.clip(np.searchsorted(t_orig, t_1000), 0, len(labels) - 1)
        labels_interp = labels[idx]

        disc_results[freq_name] = {
            "m_star": int(model.results_["m_star"]),
            "m_star_ms": float(model.results_["m_star"] * 1000.0 / fs),
            "G_star": int(model.results_["G_star"]),
            "labels": labels_interp,
        }

    ari_geom_1000_500 = float(adjusted_rand_score(geom_results["1000Hz"]["labels"], geom_results["500Hz"]["labels"]))
    ari_geom_1000_250 = float(adjusted_rand_score(geom_results["1000Hz"]["labels"], geom_results["250Hz"]["labels"]))
    ari_geom_500_250 = float(adjusted_rand_score(geom_results["500Hz"]["labels"], geom_results["250Hz"]["labels"]))

    ari_disc_1000_500 = float(adjusted_rand_score(disc_results["1000Hz"]["labels"], disc_results["500Hz"]["labels"]))
    ari_disc_1000_250 = float(adjusted_rand_score(disc_results["1000Hz"]["labels"], disc_results["250Hz"]["labels"]))
    ari_disc_500_250 = float(adjusted_rand_score(disc_results["500Hz"]["labels"], disc_results["250Hz"]["labels"]))

    return {
        "record_id": int(record_id),
        "geom_1000Hz_m_ms": geom_results["1000Hz"]["m_star_ms"],
        "geom_500Hz_m_ms": geom_results["500Hz"]["m_star_ms"],
        "geom_250Hz_m_ms": geom_results["250Hz"]["m_star_ms"],
        "geom_density_1000": geom_results["1000Hz"]["density"],
        "geom_density_500": geom_results["500Hz"]["density"],
        "geom_density_250": geom_results["250Hz"]["density"],
        "geom_cliques_1000": geom_results["1000Hz"]["cliques"],
        "geom_cliques_500": geom_results["500Hz"]["cliques"],
        "geom_cliques_250": geom_results["250Hz"]["cliques"],
        "ari_geom_1000_500": ari_geom_1000_500,
        "ari_geom_1000_250": ari_geom_1000_250,
        "ari_geom_500_250": ari_geom_500_250,
        "ari_disc_1000_500": ari_disc_1000_500,
        "ari_disc_1000_250": ari_disc_1000_250,
        "ari_disc_500_250": ari_disc_500_250,
    }


def main():
    print("=" * 80)
    print("GATE 1: SAME-SIGNAL RESAMPLING EQUIVARIANCE EXPERIMENT")
    print("=" * 80)
    isp = ISPAdapter()
    records = isp.list_records(max_records=5)
    print(f"Running deterministic resampling (1000 -> 500 -> 250 Hz) on {len(records)} real ECGs...")

    results = []
    for rid in records:
        t0 = time.time()
        res = evaluate_record_resampling(rid, n_seconds=2.5)
        res["elapsed_sec"] = time.time() - t0
        results.append(res)
        print(f"Record {rid} done in {res['elapsed_sec']:.1f}s | ARI (ms geom 1000 vs 500): {res['ari_geom_1000_500']:.3f} | ARI (discrete 1000 vs 500): {res['ari_disc_1000_500']:.3f}")

    df = pd.DataFrame(results)
    out_dir = "tit_ecg/results/gate1_resampling"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "same_signal_resampling_summary.csv")
    df.to_csv(csv_path, index=False)

    summary = {
        "n_records": len(df),
        "mean_ari_geom_1000_500": float(df["ari_geom_1000_500"].mean()),
        "mean_ari_geom_1000_250": float(df["ari_geom_1000_250"].mean()),
        "mean_ari_geom_500_250": float(df["ari_geom_500_250"].mean()),
        "mean_ari_disc_1000_500": float(df["ari_disc_1000_500"].mean()),
        "mean_ari_disc_1000_250": float(df["ari_disc_1000_250"].mean()),
        "mean_ari_disc_500_250": float(df["ari_disc_500_250"].mean()),
        "mean_density_1000": float(df["geom_density_1000"].mean()),
        "mean_density_500": float(df["geom_density_500"].mean()),
        "mean_density_250": float(df["geom_density_250"].mean()),
    }

    json_path = os.path.join(out_dir, "same_signal_resampling_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("GATE 1 SAME-SIGNAL RESULTS SUMMARY:")
    print(f"  Physical Geometry ARI (1000 vs 500 Hz): {summary['mean_ari_geom_1000_500']:.3f} vs Discrete m: {summary['mean_ari_disc_1000_500']:.3f}")
    print(f"  Physical Geometry ARI (1000 vs 250 Hz): {summary['mean_ari_geom_1000_250']:.3f} vs Discrete m: {summary['mean_ari_disc_1000_250']:.3f}")
    print(f"  Physical Geometry ARI (500 vs 250 Hz) : {summary['mean_ari_geom_500_250']:.3f} vs Discrete m: {summary['mean_ari_disc_500_250']:.3f}")
    print(f"  Mean Density across resampled frequencies: 1000Hz={summary['mean_density_1000']:.3f}, 500Hz={summary['mean_density_500']:.3f}, 250Hz={summary['mean_density_250']:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
