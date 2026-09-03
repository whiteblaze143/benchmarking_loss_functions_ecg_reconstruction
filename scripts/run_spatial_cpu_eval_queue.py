#!/usr/bin/env python3
"""Queue runner to evaluate all 30 Spatial Architecture Grid Lead I models on CPU using 6 cores.

Orchestration pattern:
1. Target discovery matching '3-Epoch Spatial Architecture Grid' in results/lead1_all_models_comprehensive_metrics.csv
2. Pre-flight disk space safety check (>= 4.0 GB free)
3. Materialize checkpoint on-demand via scripts/onelead_checkpoint_store.py
4. Run evaluate_spatial_per_lead.py strictly on CPU (6 threads, batch_size 32)
5. Evict materialized .pt immediately after SQLite commit to maintain free disk space
6. Live-update results/lead1_all_models_comprehensive_metrics.csv after each model
"""

from __future__ import annotations
import datetime as dt
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MASTER_CSV = _ROOT / "results/lead1_all_models_comprehensive_metrics.csv"
COMPACT_DB = _ROOT / "results/convergence_per_lead_evaluation_v1/compact.sqlite"
RDB_DB = _ROOT / "results/onelead_rdb_semiseg_screened_v1/compact.sqlite"
CACHE_DIR = _ROOT / "checkpoints/onelead_cache"
STATE_FILE = _ROOT / "refine-logs/queue_spatial_eval_cpu/state.json"
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_free_disk_gb(path: Path) -> float:
    stat = shutil.disk_usage(path)
    return stat.free / (1024 ** 3)


def get_evaluated_models() -> set[str]:
    if not COMPACT_DB.is_file():
        return set()
    with sqlite3.connect(COMPACT_DB) as con:
        cur = con.cursor()
        cur.execute("SELECT model_id FROM evaluations WHERE total_samples >= 2100")
        return {r[0] for r in cur.fetchall()}


def get_rdb_evaluated_models() -> set[str]:
    if not RDB_DB.is_file():
        return set()
    try:
        with sqlite3.connect(f"file:{RDB_DB}?mode=ro", uri=True) as con:
            cur = con.cursor()
            cur.execute("SELECT model_id FROM evaluations WHERE stage='full' AND status='complete'")
            return {r[0] for r in cur.fetchall()}
    except Exception:
        return set()


def update_master_csv():
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Updating master CSV: results/lead1_all_models_comprehensive_metrics.csv...")
    cmd = [sys.executable, str(_ROOT / "scripts/build_lead1_comprehensive_csv.py")]
    res = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True)
    if res.returncode == 0:
        print("  ✓ Master CSV successfully regenerated.")
    else:
        print(f"  ✗ Warning updating master CSV: {res.stderr}")


def main():
    print("=" * 70)
    print("Spatial Architecture Grid CPU Evaluation Queue (PTB-XL + RDB)")
    print(f"Started at: {dt.datetime.now().isoformat()}")
    print("=" * 70)

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if not MASTER_CSV.is_file():
        raise FileNotFoundError(f"Master CSV missing: {MASTER_CSV}")

    # Discover target models: exact 30 Spatial Architecture Grid models
    df = pd.read_csv(MASTER_CSV)
    target_models = df[df.study_track == "3-Epoch Spatial Architecture Grid"]["model_id"].tolist()
    print(f"Total target Spatial Lead I models in master CSV: {len(target_models)}")

    while True:
        evaluated_ptbxl = get_evaluated_models()
        evaluated_rdb = get_rdb_evaluated_models()
        pending = [m for m in target_models if m not in evaluated_ptbxl or m not in evaluated_rdb]
        print(f"\n[Queue Status] PTB-XL Complete: {len([m for m in target_models if m in evaluated_ptbxl])}/{len(target_models)} | RDB Complete: {len([m for m in target_models if m in evaluated_rdb])}/{len(target_models)} | Pending: {len(pending)}")
        if not pending:
            print("All 30 Spatial Lead I models successfully evaluated on both PTB-XL and RDB!")
            break

        state = {
            "total": len(target_models),
            "completed_ptbxl": len([m for m in target_models if m in evaluated_ptbxl]),
            "completed_rdb": len([m for m in target_models if m in evaluated_rdb]),
            "pending": len(pending),
            "current_model": None,
            "history": [],
            "updated_at": dt.datetime.now().isoformat(),
        }
        STATE_FILE.write_text(json.dumps(state, indent=2))

        for idx, model_id in enumerate(pending, 1):
            free_gb = get_free_disk_gb(_ROOT)
            print(f"\n[{idx}/{len(pending)}] Model: {model_id} (Disk Free: {free_gb:.1f} GB)")

            if free_gb < 4.0:
                print(f"  ✗ Disk space below 4.0 GB ({free_gb:.1f} GB). Evicting cache...")
                for f in CACHE_DIR.glob("*.pt"):
                    f.unlink(missing_ok=True)
                free_gb = get_free_disk_gb(_ROOT)
                if free_gb < 3.5:
                    print(f"  CRITICAL: Insufficient disk space ({free_gb:.1f} GB). Aborting queue.")
                    return

            ckpt_path = CACHE_DIR / f"{model_id}.pt"

            # Step 1: Materialize checkpoint if not cached
            if not ckpt_path.is_file():
                print(f"  -> Materializing {model_id} via onelead_checkpoint_store...")
                mat_cmd = [sys.executable, str(_ROOT / "scripts/onelead_checkpoint_store.py"), "materialize", model_id]
                mat_res = subprocess.run(mat_cmd, cwd=_ROOT, capture_output=True, text=True)
                if mat_res.returncode != 0:
                    print(f"  ✗ Failed to materialize {model_id}: {mat_res.stderr}")
                    continue

            if not ckpt_path.is_file():
                print(f"  ✗ Materialized checkpoint not found at {ckpt_path}")
                continue

            # Step 2a: Run PTB-XL evaluation if needed
            if model_id not in evaluated_ptbxl:
                t0 = time.time()
                print(f"  -> Running CPU evaluation on PTB-XL validation fold (6 threads, batch_size 32)...")
                eval_cmd = [
                    sys.executable,
                    str(_ROOT / "scripts/evaluate_spatial_per_lead.py"),
                    "--checkpoint-path", str(ckpt_path),
                    "--model-id", model_id,
                    "--threads", "6",
                    "--batch-size", "32",
                ]
                eval_res = subprocess.run(eval_cmd, cwd=_ROOT, capture_output=True, text=True)
                dur = time.time() - t0

                if eval_res.returncode == 0:
                    print(f"  ✓ PTB-XL Evaluated {model_id} in {dur:.1f}s ({dur/60:.1f} min)")
                    state["history"].append({"model_id": model_id, "benchmark": "ptbxl", "duration_s": dur, "status": "success", "at": dt.datetime.now().isoformat()})
                else:
                    print(f"  ✗ PTB-XL evaluation failed for {model_id} (code {eval_res.returncode}):")
                    print(eval_res.stderr)
                    state["history"].append({"model_id": model_id, "benchmark": "ptbxl", "duration_s": dur, "status": "error", "error": eval_res.stderr, "at": dt.datetime.now().isoformat()})

            # Step 2b: Run RDB external generalization if needed
            if model_id not in evaluated_rdb:
                t0_rdb = time.time()
                print(f"  -> Running CPU RDB external generalization on 360 records (6 threads)...")
                rdb_cmd = [
                    sys.executable,
                    str(_ROOT / "scripts/evaluate_onelead_rdb_semiseg_blinded.py"),
                    "--phase", "full-all",
                    "--model-id", model_id,
                    "--torch-threads", "6",
                    "--reconstruction-batch-size", "8",
                    "--delineation-batch-size", "32",
                ]
                rdb_res = subprocess.run(rdb_cmd, cwd=_ROOT, capture_output=True, text=True)
                dur_rdb = time.time() - t0_rdb

                if rdb_res.returncode == 0:
                    print(f"  ✓ RDB Evaluated {model_id} in {dur_rdb:.1f}s ({dur_rdb/60:.1f} min)")
                    state["history"].append({"model_id": model_id, "benchmark": "rdb", "duration_s": dur_rdb, "status": "success", "at": dt.datetime.now().isoformat()})
                else:
                    print(f"  ✗ RDB evaluation warning for {model_id} (code {rdb_res.returncode}):")
                    print(rdb_res.stderr[:200] if rdb_res.stderr else "unknown error")
                    state["history"].append({"model_id": model_id, "benchmark": "rdb", "duration_s": dur_rdb, "status": "error", "error": rdb_res.stderr, "at": dt.datetime.now().isoformat()})

            # Step 3: Evict downloaded checkpoint to preserve disk headroom
            if ckpt_path.is_file():
                ckpt_path.unlink(missing_ok=True)
                print(f"  -> Evicted {ckpt_path.name} from cache.")

            # Step 4: Update state and master CSV
            evaluated_ptbxl = get_evaluated_models()
            evaluated_rdb = get_rdb_evaluated_models()
            state["completed_ptbxl"] = len([m for m in target_models if m in evaluated_ptbxl])
            state["completed_rdb"] = len([m for m in target_models if m in evaluated_rdb])
            state["pending"] = len([m for m in target_models if m not in evaluated_ptbxl or m not in evaluated_rdb])
            state["current_model"] = model_id
            state["updated_at"] = dt.datetime.now().isoformat()
            STATE_FILE.write_text(json.dumps(state, indent=2))

            # Update master CSV after each model
            update_master_csv()

    print("\n" + "=" * 70)
    print("Spatial Architecture Grid CPU Evaluation Queue Completed.")
    print("Final Master CSV Update...")
    update_master_csv()
    print("=" * 70)


if __name__ == "__main__":
    main()
