"""Multi-Fold PTB-XL Queue for Exact repSpat ECG/VCG Adaptation.

Evaluates Temporal repSpat across PTB-XL official stratified folds:
- Fold 10: Official Held-Out Test Set (~2,198 records)
- Fold 9:  Validation Set (~2,183 records)
- Fold 1-8: Training Set (~17,418 records)

Critical Design:
- repSpat is an unsupervised, per-recording statistical inference procedure.
- Evaluating across folds tests cross-population and cross-pathology stability
  (NORM, MI, STTC, CD, HYP), rather than model weight generalization.
- Multi-worker concurrent execution ensures high throughput.
"""
from __future__ import annotations

import argparse
import ast
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run Exact repSpat on PTB-XL Folds")
    parser.add_argument("--folds", type=int, nargs="+", default=[10, 9, 1], help="PTB-XL folds to evaluate (e.g. 10 9 1)")
    parser.add_argument("--samples-per-class", type=int, default=10, help="Number of records per diagnostic superclass per fold")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of time samples per record (1000 = 2.0s at 500Hz)")
    parser.add_argument("--n-permutations", type=int, default=500, help="Number of block permutations B")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--output-dir", type=str, default="tit_ecg/results/ptbxl_multifold", help="Output directory")
    return parser.parse_args()


def process_record_worker(record_meta: dict, n_samples: int, config_dict: dict) -> dict | None:
    """Worker function to process a single PTB-XL patient record."""
    try:
        t0 = time.time()
        record_path = record_meta["filename_hr"]  # e.g. records500/00000/00001_hr
        full_path = os.path.join("data/ptb_xl", record_path)

        # Read 12-lead signal
        signals, fields = wfdb.rdsamp(full_path)
        ecg_sub = signals[:n_samples]

        # Reconstruct config
        cfg = RepSpatConfig(**config_dict)
        model = TemporalRepSpatECG(config=cfg)
        model.fit(ecg_sub, sampling_rate=float(fields["fs"]))

        res = model.results_
        elapsed = time.time() - t0

        G_sim = res["similarity_graph"]
        density = float(G_sim.number_of_edges() / (res["G_star"] * (res["G_star"] - 1) / 2)) if res["G_star"] > 1 else 0.0

        return {
            "ecg_id": int(record_meta["ecg_id"]),
            "fold": int(record_meta["strat_fold"]),
            "diagnostic_class": str(record_meta["diagnostic_superclass"]),
            "age": float(record_meta.get("age", np.nan)),
            "sex": int(record_meta.get("sex", -1)),
            "m_star": int(res["m_star"]),
            "G_star": int(res["G_star"]),
            "n_similarity_edges": int(G_sim.number_of_edges()),
            "graph_density": density,
            "n_cc_groups": int(res["n_cc_clusters"]),
            "n_clique_groups": int(res["n_clique_clusters"]),
            "has_clique_overlap": bool(res["clique_audit"]["has_overlap"]),
            "elapsed_sec": float(elapsed),
        }
    except Exception as e:
        return {"ecg_id": int(record_meta.get("ecg_id", -1)), "error": str(e)}


def map_diagnostic_superclass(scp_codes_str: str, scp_df: pd.DataFrame) -> str:
    """Maps scp_codes dictionary string to primary diagnostic superclass (NORM, MI, STTC, CD, HYP)."""
    try:
        codes = ast.literal_eval(scp_codes_str)
        superclasses = []
        for code in codes.keys():
            if code in scp_df.index:
                sclass = scp_df.loc[code, "diagnostic_class"]
                if pd.notna(sclass) and sclass != "":
                    superclasses.append(sclass)
        if "NORM" in superclasses and len(superclasses) == 1:
            return "NORM"
        for priority in ["MI", "CD", "HYP", "STTC", "NORM"]:
            if priority in superclasses:
                return priority
        return superclasses[0] if superclasses else "OTHER"
    except Exception:
        return "OTHER"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("PTB-XL MULTI-FOLD repSpat EVALUATION QUEUE")
    print(f"Target Folds: {args.folds}")
    print(f"Samples per diagnostic class: {args.samples_per_class}")
    print(f"Permutations: B={args.n_permutations}")
    print(f"Workers: {args.num_workers}")
    print("=" * 80)

    # Load database metadata
    db_df = pd.read_csv("data/ptb_xl/ptbxl_database.csv")
    scp_df = pd.read_csv("data/ptb_xl/scp_statements.csv", index_col=0)

    # Assign diagnostic superclasses
    db_df["diagnostic_superclass"] = db_df["scp_codes"].apply(lambda s: map_diagnostic_superclass(s, scp_df))

    cfg_dict = {
        "m_grid": [4, 6, 8],
        "G_grid": [4, 6, 8],
        "n_permutations": args.n_permutations,
        "kmeans_n_init": 10,
        "strict_block_exceed": True,
        "p_value_correction": True,
        "random_state": 42,
    }

    all_results = []
    classes = ["NORM", "MI", "STTC", "CD", "HYP"]

    for fold in args.folds:
        print(f"\n>>> Processing Fold {fold}...")
        fold_df = db_df[db_df["strat_fold"] == fold]

        # Sample balanced records across diagnostic classes
        selected_records = []
        for c in classes:
            sub = fold_df[fold_df["diagnostic_superclass"] == c]
            sample_n = min(args.samples_per_class, len(sub))
            if sample_n > 0:
                selected_records.append(sub.sample(n=sample_n, random_state=42))

        if not selected_records:
            continue
        batch_df = pd.concat(selected_records).sample(frac=1.0, random_state=42)  # shuffle
        print(f"  Selected {len(batch_df)} stratified patient records for Fold {fold}")

        records_meta = batch_df.to_dict(orient="records")
        fold_records_results = []

        start_time = time.time()
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_id = {
                executor.submit(process_record_worker, meta, args.n_samples, cfg_dict): meta["ecg_id"]
                for meta in records_meta
            }

            for future in as_completed(future_to_id):
                res = future.result()
                if res and "error" not in res:
                    fold_records_results.append(res)
                    print(f"    [Fold {fold}] ECG {res['ecg_id']} ({res['diagnostic_class']}): "
                          f"m*={res['m_star']}, G*={res['G_star']}, CC={res['n_cc_groups']}, Clique={res['n_clique_groups']}, time={res['elapsed_sec']:.1f}s")
                elif res:
                    print(f"    [Fold {fold}] Error on ECG {res.get('ecg_id')}: {res.get('error')}")

        fold_elapsed = time.time() - start_time
        print(f"  Fold {fold} finished in {fold_elapsed:.1f}s ({len(fold_records_results)} records successful)")

        # Save fold summary
        fold_csv = os.path.join(args.output_dir, f"ptbxl_fold_{fold}_summary.csv")
        pd.DataFrame(fold_records_results).to_csv(fold_csv, index=False)
        all_results.extend(fold_records_results)

    # Save overall summary
    master_df = pd.DataFrame(all_results)
    master_csv = os.path.join(args.output_dir, "ptbxl_multifold_master_summary.csv")
    master_json = os.path.join(args.output_dir, "ptbxl_multifold_master_summary.json")
    master_df.to_csv(master_csv, index=False)

    summary_stats = {
        "total_records": len(master_df),
        "folds_evaluated": args.folds,
        "mean_graph_density": float(master_df["graph_density"].mean()) if len(master_df) else 0.0,
        "mean_cc_groups": float(master_df["n_cc_groups"].mean()) if len(master_df) else 0.0,
        "mean_clique_groups": float(master_df["n_clique_groups"].mean()) if len(master_df) else 0.0,
        "fraction_with_clique_overlap": float(master_df["has_clique_overlap"].mean()) if len(master_df) else 0.0,
        "per_class_cliques": master_df.groupby("diagnostic_class")["n_clique_groups"].mean().to_dict() if len(master_df) else {},
    }
    with open(master_json, "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 80)
    print("PTB-XL MULTI-FOLD BENCHMARK COMPLETED")
    print(f"Master CSV saved -> {master_csv}")
    print(f"Master JSON saved -> {master_json}")
    print(f"Mean Graph Density: {summary_stats['mean_graph_density']:.3f}")
    print(f"Mean Clique Groups: {summary_stats['mean_clique_groups']:.2f} vs CC Groups: {summary_stats['mean_cc_groups']:.2f}")
    print(f"Clique Overlap Occurrence: {summary_stats['fraction_with_clique_overlap'] * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
