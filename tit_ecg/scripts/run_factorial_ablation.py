"""Clean 2x2 Factorial Ablation of repSpat Geometry vs. Kernel Scale.

Isolates the confounded effects of:
- Geometry: Original discrete m in {4, 6, 8} vs. ms-based m in {32, 64, 128} ms
- Kernel Scale: Fixed c = 1.0 vs. Relative c = gamma * d_med (gamma = 1.0)

Design:
| Condition | Geometry               | Kernel                  | Purpose                          |
|-----------|------------------------|-------------------------|----------------------------------|
| C1        | Original discrete m    | fixed c = 1.0           | reference                        |
| C2        | ms-based m             | fixed c = 1.0           | isolate temporal-geometry effect |
| C3        | Original discrete m    | c = gamma * d_med       | isolate kernel-scale effect      |
| C4        | ms-based m             | c = gamma * d_med       | combined adaptation              |

Evaluated across both sampling rates:
- PTB-XL (500 Hz, 16 balanced held-out records)
- EchoNext (250 Hz, 16 held-out records)
"""
from __future__ import annotations

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
    data = get_echonext_data()
    raw = data["waveforms"][idx, 0]  # [2500, 12]
    mean_val = data["mean"]
    std_val = data["std"]
    wave_uv = np.asarray(raw[:n_samples], dtype=float) * std_val + mean_val
    wave_mv = wave_uv / 1000.0
    return wave_mv, 250.0


def run_single_ablation(
    cohort: str,
    record_id: str | int,
    signal: np.ndarray,
    fs: float,
    condition_name: str,
    geometry_type: str,
    kernel_type: str,
) -> dict:
    """Runs a single condition on a single record."""
    cfg = RepSpatConfig.fast_test_config(
        G_grid=[4, 6, 8],
        n_permutations=200,
        kmeans_n_init=10,
    )
    
    # Configure Geometry
    if geometry_type == "discrete_m":
        cfg.m_grid = [4, 6, 8]
        cfg.m_ms_grid = None
    elif geometry_type == "ms_based":
        cfg.m_grid = [4, 6, 8]  # fallback
        cfg.m_ms_grid = [32.0, 64.0, 128.0]
    else:
        raise ValueError(f"Unknown geometry_type: {geometry_type}")

    # Configure Kernel Scale
    if kernel_type == "fixed_c1":
        cfg.kernel_scale_rule = "fixed"
        cfg.kernel_param = 1.0
    elif kernel_type == "relative_dmed":
        cfg.kernel_scale_rule = "median_heuristic"
        cfg.kernel_param = 1.0
    else:
        raise ValueError(f"Unknown kernel_type: {kernel_type}")

    t0 = time.time()
    model = TemporalRepSpatECG(config=cfg)
    model.fit(signal, sampling_rate=fs)
    elapsed = time.time() - t0

    res = model.results_
    g_desc = res["graph_descriptors"]

    return {
        "cohort": cohort,
        "record_id": str(record_id),
        "fs": fs,
        "condition": condition_name,
        "geometry": geometry_type,
        "kernel": kernel_type,
        "m_star": int(res["m_star"]),
        "G_star": int(res["G_star"]),
        "graph_density": float(g_desc.get("density", 0.0)),
        "n_similarity_edges": int(len(res["similarity_graph"].edges())),
        "n_clique_groups": int(res["n_clique_clusters"]),
        "n_cc_groups": int(res["n_cc_clusters"]),
        "n_maximal_cliques": int(g_desc.get("n_maximal_cliques", 0)),
        "overlap_fraction": float(g_desc.get("overlap_fraction", 0.0)),
        "is_disconnected": int(len(res["similarity_graph"].edges()) == 0),
        "elapsed_sec": float(elapsed),
    }


def main():
    print("=" * 80)
    print("RUNNING 2x2 FACTORIAL ABLATION EXPERIMENT (GEOMETRY vs. KERNEL SCALE)")
    print("=" * 80)

    out_dir = "tit_ecg/results/factorial_ablation"
    os.makedirs(out_dir, exist_ok=True)

    # 16 PTB-XL records from Fold 10
    meta_df = pd.read_csv("data/ptb_xl/ptbxl_database.csv", index_col="ecg_id")
    ptb_ids = meta_df[meta_df["strat_fold"] == 10].index.values[:16]

    # 16 EchoNext test records
    echo_indices = [10, 42, 105, 210, 315, 420, 525, 630, 735, 840, 945, 1050, 1155, 1260, 1365, 1470]

    conditions = [
        ("C1_reference", "discrete_m", "fixed_c1"),
        ("C2_geom_only", "ms_based", "fixed_c1"),
        ("C3_kernel_only", "discrete_m", "relative_dmed"),
        ("C4_combined", "ms_based", "relative_dmed"),
    ]

    tasks = []
    # Pre-load signals
    print("Pre-loading records...")
    ptb_signals = {}
    for eid in ptb_ids:
        sig, fs = load_ptbxl_record(eid)
        ptb_signals[eid] = (sig, fs)

    echo_signals = {}
    for idx in echo_indices:
        sig, fs = load_echonext_record(idx)
        echo_signals[idx] = (sig, fs)

    print(f"Loaded {len(ptb_signals)} PTB-XL records and {len(echo_signals)} EchoNext records.")

    for cond_name, geom, kern in conditions:
        for eid, (sig, fs) in ptb_signals.items():
            tasks.append(("PTB-XL", eid, sig, fs, cond_name, geom, kern))
        for idx, (sig, fs) in echo_signals.items():
            tasks.append(("EchoNext", idx, sig, fs, cond_name, geom, kern))

    print(f"Total tasks: {len(tasks)} ({len(conditions)} conditions x 32 records)")

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                run_single_ablation,
                cohort,
                rid,
                sig,
                fs,
                cname,
                geom,
                kern,
            ): (cohort, rid, cname)
            for cohort, rid, sig, fs, cname, geom, kern in tasks
        }
        for fut in as_completed(futures):
            c_info = futures[fut]
            try:
                res = fut.result()
                results.append(res)
                if len(results) % 16 == 0:
                    print(f"  Completed {len(results)}/{len(tasks)} tasks...")
            except Exception as e:
                print(f"  Error on {c_info}: {e}")

    df_res = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "factorial_ablation_records.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"Saved records to {csv_path}")

    # Compute Factorial Summary Table
    grouped = df_res.groupby(["cohort", "condition", "geometry", "kernel"]).agg(
        mean_density=("graph_density", "mean"),
        std_density=("graph_density", "std"),
        mean_cliques=("n_clique_groups", "mean"),
        mean_cc=("n_cc_groups", "mean"),
        disconnected_rate=("is_disconnected", "mean"),
        mean_m_star=("m_star", "mean"),
        mean_G_star=("G_star", "mean"),
        mean_edges=("n_similarity_edges", "mean"),
    ).reset_index()

    summary_csv = os.path.join(out_dir, "factorial_ablation_summary.csv")
    grouped.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    # Main effects and interactions
    summary_dict = {
        "conditions": conditions,
        "n_ptbxl": len(ptb_ids),
        "n_echonext": len(echo_indices),
        "summary_table": grouped.to_dict(orient="records"),
    }

    # Marginal effect of geometry holding kernel fixed
    for kern in ["fixed_c1", "relative_dmed"]:
        for cohort in ["PTB-XL", "EchoNext"]:
            sub = df_res[(df_res["kernel"] == kern) & (df_res["cohort"] == cohort)]
            dens_discrete = sub[sub["geometry"] == "discrete_m"]["graph_density"].mean()
            dens_ms = sub[sub["geometry"] == "ms_based"]["graph_density"].mean()
            diff = dens_ms - dens_discrete
            summary_dict[f"marginal_geom_effect_{cohort}_{kern}"] = {
                "discrete_m_density": float(dens_discrete),
                "ms_based_density": float(dens_ms),
                "delta_density": float(diff),
            }

    # Marginal effect of kernel holding geometry fixed
    for geom in ["discrete_m", "ms_based"]:
        for cohort in ["PTB-XL", "EchoNext"]:
            sub = df_res[(df_res["geometry"] == geom) & (df_res["cohort"] == cohort)]
            dens_fixed = sub[sub["kernel"] == "fixed_c1"]["graph_density"].mean()
            dens_rel = sub[sub["kernel"] == "relative_dmed"]["graph_density"].mean()
            diff = dens_rel - dens_fixed
            summary_dict[f"marginal_kernel_effect_{cohort}_{geom}"] = {
                "fixed_c1_density": float(dens_fixed),
                "relative_dmed_density": float(dens_rel),
                "delta_density": float(diff),
            }

    json_path = os.path.join(out_dir, "factorial_ablation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved JSON summary to {json_path}")
    print("=" * 80)
    print("FACTORIAL ABLATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
