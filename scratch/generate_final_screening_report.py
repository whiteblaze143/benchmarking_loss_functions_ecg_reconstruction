import sqlite3, json, numpy as np
from pathlib import Path

FROZEN_CONVERGENCE_PANEL = [
    "conv_control",
    "R7_morlet_mag_ueg_real",
    "R5_morlet_mag_ueg_phase_wyatt",
    "ssl_log_magnitude_phase_sin_local_gated_add",
    "ssl_log_magnitude_real_both_gated_add",
    "tf_sc16_cy4",
    "tf_sc16_cy8",
    "del_wave_ce",
    "C1_E1_morlet_mag_morlet_phase",
    "A0_raw",
    "A0_wave_noSSL_gated_add",
    "ssl_magnitude_phase_both_cross_attn",
]

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
rows = conn.execute("SELECT id, summary_json FROM jobs WHERE status='completed'").fetchall()
conn.close()

l0, l1 = {}, {}
for jid, s_str in rows:
    if not s_str: continue
    s = json.loads(s_str)
    arch = jid.replace("_1110000_s42_l0", "").replace("_1110000_s42_l1", "")
    if "_l0" in jid: l0[arch] = s
    elif "_l1" in jid: l1[arch] = s

paired_archs = sorted(list(set(l0.keys()) & set(l1.keys())))

print(f"Total Completed Screening Models: {len(rows)}/120 (Lead-I: {len(l0)}/60, Lead-II: {len(l1)}/60)")
print(f"Matched Paired Models: {len(paired_archs)}/60\n")

# Compute Statistics for Frozen Panel
print("="*105)
print("  FROZEN 12-MODEL CONVERGENCE PANEL SCREENING CHARACTERIZATION (Descriptive Only)")
print("="*105)
print(f"{'#':<3} | {'Architecture':<42} | {'r (L1)':<7} | {'r (L2)':<7} | {'Mean r':<7} | {'r_min':<7} | {'P (L1)':<6} | {'P (L2)':<6} | {'T (L1)':<6} | {'T (L2)':<6} | {'Dr':<7}")
print("-" * 105)

for idx, arch in enumerate(FROZEN_CONVERGENCE_PANEL):
    s0 = l0[arch]
    s1 = l1[arch]
    r0 = s0.get("val_missing_pearson", 0.0)
    r1 = s1.get("val_missing_pearson", 0.0)
    p0 = s0.get("P_iou", 0.0)
    p1 = s1.get("P_iou", 0.0)
    t0 = s0.get("T_iou", 0.0)
    t1 = s1.get("T_iou", 0.0)
    mean_r = (r0 + r1) / 2.0
    r_min = min(r0, r1)
    delta_r = r1 - r0
    print(f"{idx+1:02d}  | {arch:<42} | {r0:.4f} | {r1:.4f} | {mean_r:.4f} | {r_min:.4f} | {p0:.3f} | {p1:.3f} | {t0:.3f} | {t1:.3f} | {delta_r:+.4f}")

# Compute Global Statistics across ALL matched models
r0_all = [l0[a]["val_missing_pearson"] for a in paired_archs]
r1_all = [l1[a]["val_missing_pearson"] for a in paired_archs]
p0_all = [l0[a]["P_iou"] for a in paired_archs]
p1_all = [l1[a]["P_iou"] for a in paired_archs]
t0_all = [l0[a]["T_iou"] for a in paired_archs]
t1_all = [l1[a]["T_iou"] for a in paired_archs]

print("\n" + "="*80)
print(f"  GLOBAL PHYSIOLOGICAL DELTAS ACROSS ALL {len(paired_archs)} MATCHED SCREENING MODELS")
print("="*80)
print(f"Global Pearson r:   Lead-I = {np.mean(r0_all):.4f} -> Lead-II = {np.mean(r1_all):.4f} (Mean Delta = {np.mean(r1_all)-np.mean(r0_all):+.4f})")
print(f"P-Wave IoU:         Lead-I = {np.mean(p0_all):.4f} -> Lead-II = {np.mean(p1_all):.4f} (Mean Delta = {np.mean(p1_all)-np.mean(p0_all):+.4f})")
print(f"T-Wave IoU:         Lead-I = {np.mean(t0_all):.4f} -> Lead-II = {np.mean(t1_all):.4f} (Mean Delta = {np.mean(t1_all)-np.mean(t0_all):+.4f})")

