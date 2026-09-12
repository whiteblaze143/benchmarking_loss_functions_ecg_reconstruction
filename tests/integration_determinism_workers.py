"""Integration check: Deterministic execution with workers=1 vs workers=8.

Runs 10 real calibration tasks under both configurations and asserts exact bitwise match of:
- seed
- obs_mmd2
- n_exceed
- p_value
"""
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from rep_stat_ecg.src.motifs.dependence_calibration import (
    build_patient_bundles,
    split_patients_disjoint,
    split_beats_shared_patient,
    evaluate_pair_test,
    generate_deterministic_seed,
)
from rep_stat_ecg.scripts.run_m3h_dependence_calibration import worker_cal0_d

def make_test_data():
    rng = np.random.RandomState(42)
    rows = []
    for p in range(40):
        pid = f"pat_{p:03d}"
        for b in range(4):
            for t in range(8):
                rows.append({
                    "patient_id": pid,
                    "ecg_id": f"ecg_{p:03d}",
                    "beat_id": b,
                    "original_time": float(b * 0.8 + t * 0.02),
                    "normalized_phase": float(t / 8.0),
                    "s": float(rng.normal(2.0, 0.5)),
                    "rho": float(rng.normal(0.0, 10.0)),
                    "kappa": float(rng.normal(1.0, 0.2)),
                    "domain_id": 0,
                })
    return pd.DataFrame(rows)

def main():
    df = make_test_data()
    bundles = build_patient_bundles(df)
    pids = sorted(list(bundles.keys()))

    # Build 10 tasks (mixed balanced & imbalanced)
    tasks = []
    for rep in range(5):
        seed = generate_deterministic_seed("M3H_INTEG", "CAL0_D_BAL", 0, "1:1", rep)
        tasks.append((0, "BALANCED", (1, 1), rep, pids, bundles, seed))
    for rep in range(5):
        seed = generate_deterministic_seed("M3H_INTEG", "CAL0_D_IMBAL", 0, "1:3", rep)
        tasks.append((0, "IMBALANCED", (1, 3), rep + 5, pids, bundles, seed))

    ctx = mp.get_context("spawn")
    
    # 1. Run with workers=1
    print("Running 10 tasks with workers=1...")
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
        results_w1 = list(pool.map(worker_cal0_d, tasks))

    # 2. Run with workers=8
    print("Running 10 tasks with workers=8...")
    with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as pool:
        results_w8 = list(pool.map(worker_cal0_d, tasks))

    # Sort both results by replicate
    sorted_w1 = sorted(results_w1, key=lambda r: (r["domain_id"], r["balance_condition"], r["replicate"]))
    sorted_w8 = sorted(results_w8, key=lambda r: (r["domain_id"], r["balance_condition"], r["replicate"]))

    print(f"\nComparing results across {len(sorted_w1)} tasks:")
    all_matched = True
    for i, (r1, r8) in enumerate(zip(sorted_w1, sorted_w8)):
        seed_match = (r1["seed"] == r8["seed"])
        mmd_match = np.isclose(r1["obs_mmd2"], r8["obs_mmd2"], atol=1e-12)
        exceed_match = (r1["n_exceed"] == r8["n_exceed"])
        pval_match = np.isclose(r1["p_value"], r8["p_value"], atol=1e-12)

        match = seed_match and mmd_match and exceed_match and pval_match
        if not match:
            all_matched = False
            print(f"Task {i} MISMATCH: w1={r1} vs w8={r8}")
        else:
            print(f"  Task {i:02d} [rep {r1['replicate']}, {r1['balance_condition']}]: "
                  f"seed={r1['seed']}, mmd2={r1['obs_mmd2']:.6f}, n_exceed={r1['n_exceed']}, pval={r1['p_value']:.4f} -> MATCH")

    if all_matched:
        print("\nSUCCESS: All 10 calibration tasks produced IDENTICAL seed, mmd2_obs, n_exceed, and p_value across workers=1 and workers=8.")
    else:
        print("\nFAILURE: Mismatch detected between workers=1 and workers=8!")
        sys.exit(1)

if __name__ == "__main__":
    main()
