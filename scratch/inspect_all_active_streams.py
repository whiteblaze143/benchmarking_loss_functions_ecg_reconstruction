import sqlite3, json, os, datetime
from pathlib import Path
import numpy as np

# 1. Inspect wavelet_ssl_1110000 (GPU Queue)
print("="*85)
print("1. GPU QUEUE: WAVELET SSL 1110000 (Lead-I & Lead-II Screening)")
print("="*85)
gpu_db = Path("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
if gpu_db.is_file():
    conn = sqlite3.connect(gpu_db)
    completed_rows = conn.execute("SELECT id, summary_json, completed_at FROM jobs WHERE status='completed'").fetchall()
    running_rows = conn.execute("SELECT id, started_at FROM jobs WHERE status='running'").fetchall()
    pending_count = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
    conn.close()

    l0_models = {}
    l1_models = {}
    for jid, s_str, c_at in completed_rows:
        if not s_str: continue
        s = json.loads(s_str)
        arch = jid.replace("_1110000_s42_l0", "").replace("_1110000_s42_l1", "")
        if "_l0" in jid:
            l0_models[arch] = s
        elif "_l1" in jid:
            l1_models[arch] = s

    print(f"Total GPU Jobs Completed: {len(completed_rows)}/120")
    print(f"  - Lead-I (_l0):  {len(l0_models)}/60 completed")
    print(f"  - Lead-II (_l1): {len(l1_models)}/60 completed")
    print(f"  - Running:       {[r[0] for r in running_rows]}")
    print(f"  - Pending:       {pending_count}")

    # Paired comparisons for models with BOTH Lead-I and Lead-II complete
    paired_archs = sorted(list(set(l0_models.keys()) & set(l1_models.keys())))
    print(f"\nPaired Completed Models (Both Lead-I & Lead-II): {len(paired_archs)} models")
    
    if paired_archs:
        print(f"{'Architecture':<45} | {'r (L1)':<7} | {'r (L2)':<7} | {'Mean r':<7} | {'Dr':<7} | {'P (L1)':<6} | {'P (L2)':<6} | {'T (L1)':<6} | {'T (L2)':<6}")
        print("-" * 115)
        
        table = []
        for arch in paired_archs:
            s0 = l0_models[arch]
            s1 = l1_models[arch]
            r0 = s0.get("val_missing_pearson", 0.0)
            r1 = s1.get("val_missing_pearson", 0.0)
            r_mean = (r0 + r1) / 2.0
            r_diff = r1 - r0
            p0 = s0.get("P_iou", 0.0)
            p1 = s1.get("P_iou", 0.0)
            t0 = s0.get("T_iou", 0.0)
            t1 = s1.get("T_iou", 0.0)
            table.append((arch, r0, r1, r_mean, r_diff, p0, p1, t0, t1))
            
        # Sort by Mean r
        table.sort(key=lambda x: x[3], reverse=True)
        for row in table[:15]:
            print(f"{row[0]:<45} | {row[1]:.4f} | {row[2]:.4f} | {row[3]:.4f} | {row[4]:+.4f} | {row[5]:.3f} | {row[6]:.3f} | {row[7]:.3f} | {row[8]:.3f}")

# 2. Inspect queue_3arch (CPU Factorial Queue)
print("\n" + "="*85)
print("2. CPU QUEUE: 3-ARCHITECTURE FACTORIAL QUEUE (344 Factorial Masks)")
print("="*85)
cpu_state_file = Path("refine-logs/queue_3arch/queue_state.json")
if cpu_state_file.is_file():
    with open(cpu_state_file) as f:
        cpu_state = json.load(f)
    print(f"Total CPU jobs: {cpu_state.get('total_jobs')}")
    print(f"Completed: {cpu_state.get('completed_count')}")
    print(f"Failed: {cpu_state.get('failed_count')}")
    print(f"Pending: {cpu_state.get('pending_count')}")
    print(f"Running: {cpu_state.get('running_count')}")
    print(f"Progress: {cpu_state.get('progress_pct', 0):.1f}%")

