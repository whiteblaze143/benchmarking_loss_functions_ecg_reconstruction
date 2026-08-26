import sqlite3, json, os
from pathlib import Path
import numpy as np

# 1. GPU Wavelet SSL Screening (1110000)
conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
rows = conn.execute("SELECT id, summary_json, status FROM jobs WHERE status='completed'").fetchall()
running = conn.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
pending = conn.execute("SELECT count(*) FROM jobs WHERE status='pending'").fetchone()[0]
conn.close()

l0_dict = {}
l1_dict = {}
for jid, s_str, st in rows:
    if not s_str: continue
    s = json.loads(s_str)
    arch = jid.replace("_1110000_s42_l0", "").replace("_1110000_s42_l1", "")
    if "_l0" in jid:
        l0_dict[arch] = s
    elif "_l1" in jid:
        l1_dict[arch] = s

paired_archs = sorted(list(set(l0_dict.keys()) & set(l1_dict.keys())))

print(f"=========================================================================================")
print(f"1. GPU STREAM: WAVELET SSL 1110000 (Lead-I & Lead-II Screening)")
print(f"=========================================================================================")
print(f"Total Completed: {len(rows)}/120 (Lead-I: {len(l0_dict)}/60, Lead-II: {len(l1_dict)}/60)")
print(f"Currently Running: {[r[0] for r in running]}")
print(f"Pending in Queue:  {pending}")
print(f"Paired Complete Models (Matched Across Leads): {len(paired_archs)}\n")

# Aggregate physiological shifts
if paired_archs:
    r0_list = [l0_dict[a].get("val_missing_pearson", 0.0) for a in paired_archs]
    r1_list = [l1_dict[a].get("val_missing_pearson", 0.0) for a in paired_archs]
    p0_list = [l0_dict[a].get("P_iou", 0.0) for a in paired_archs]
    p1_list = [l1_dict[a].get("P_iou", 0.0) for a in paired_archs]
    t0_list = [l0_dict[a].get("T_iou", 0.0) for a in paired_archs]
    t1_list = [l1_dict[a].get("T_iou", 0.0) for a in paired_archs]
    q0_list = [l0_dict[a].get("QRS_iou", 0.0) for a in paired_archs]
    q1_list = [l1_dict[a].get("QRS_iou", 0.0) for a in paired_archs]

    print("=== PAIRED PHYSIOLOGICAL DELTAS (Lead-II vs Lead-I across 30 matched models) ===")
    print(f"Global Pearson r:   Lead-I = {np.mean(r0_list):.4f} -> Lead-II = {np.mean(r1_list):.4f} (Delta = {np.mean(r1_list)-np.mean(r0_list):+.4f})")
    print(f"P-Wave IoU:         Lead-I = {np.mean(p0_list):.4f} -> Lead-II = {np.mean(p1_list):.4f} (Delta = {np.mean(p1_list)-np.mean(p0_list):+.4f})")
    print(f"T-Wave IoU:         Lead-I = {np.mean(t0_list):.4f} -> Lead-II = {np.mean(t1_list):.4f} (Delta = {np.mean(t1_list)-np.mean(t0_list):+.4f})")
    print(f"QRS IoU:            Lead-I = {np.mean(q0_list):.4f} -> Lead-II = {np.mean(q1_list):.4f} (Delta = {np.mean(q1_list)-np.mean(q0_list):+.4f})")

    # Group comparison: Gated-Add vs Cross-Attn
    gated_models = [a for a in paired_archs if "gated_add" in a]
    cross_models = [a for a in paired_archs if "cross_attn" in a]
    ueg_models = [a for a in paired_archs if "ueg" in a]

    print("\n=== ARCHITECTURAL MECHANISM BREAKDOWN (Mean over Paired Models) ===")
    if gated_models:
        g_r = np.mean([(l0_dict[a]["val_missing_pearson"] + l1_dict[a]["val_missing_pearson"])/2 for a in gated_models])
        g_p = np.mean([(l0_dict[a]["P_iou"] + l1_dict[a]["P_iou"])/2 for a in gated_models])
        g_t = np.mean([(l0_dict[a]["T_iou"] + l1_dict[a]["T_iou"])/2 for a in gated_models])
        print(f"Gated-Add Fusion (N={len(gated_models):02d}):   Mean r = {g_r:.4f} | Mean P_IoU = {g_p:.3f} | Mean T_IoU = {g_t:.3f}")

    if ueg_models:
        u_r = np.mean([(l0_dict[a]["val_missing_pearson"] + l1_dict[a]["val_missing_pearson"])/2 for a in ueg_models])
        u_p = np.mean([(l0_dict[a]["P_iou"] + l1_dict[a]["P_iou"])/2 for a in ueg_models])
        u_t = np.mean([(l0_dict[a]["T_iou"] + l1_dict[a]["T_iou"])/2 for a in ueg_models])
        print(f"UEG Representations (N={len(ueg_models):02d}):  Mean r = {u_r:.4f} | Mean P_IoU = {u_p:.3f} | Mean T_IoU = {u_t:.3f}")

    if cross_models:
        c_r = np.mean([(l0_dict[a]["val_missing_pearson"] + l1_dict[a]["val_missing_pearson"])/2 for a in cross_models])
        c_p = np.mean([(l0_dict[a]["P_iou"] + l1_dict[a]["P_iou"])/2 for a in cross_models])
        c_t = np.mean([(l0_dict[a]["T_iou"] + l1_dict[a]["T_iou"])/2 for a in cross_models])
        print(f"Cross-Attention (N={len(cross_models):02d}):     Mean r = {c_r:.4f} | Mean P_IoU = {c_p:.3f} | Mean T_IoU = {c_t:.3f}")

# 2. Inspect CPU 3-arch factorial queue state
print(f"\n=========================================================================================")
print(f"2. CPU STREAM: 3-ARCHITECTURE FACTORIAL QUEUE (344-Job Matrix)")
print(f"=========================================================================================")
with open("refine-logs/queue_3arch/queue_state.json") as f:
    cpu_data = json.load(f)
cpu_jobs = cpu_data.get("jobs", [])
cpu_counts = {}
for jinfo in cpu_jobs:
    st = jinfo.get("status", "unknown")
    cpu_counts[st] = cpu_counts.get(st, 0) + 1

print(f"Total jobs: {len(cpu_jobs)}")
print(f"Status breakdown: {cpu_counts}")
print(f"Completion rate: {cpu_counts.get('completed', 0)}/{len(cpu_jobs)} ({cpu_counts.get('completed', 0)/len(cpu_jobs)*100:.1f}%)")

