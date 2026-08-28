import sqlite3, json

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
cursor = conn.cursor()

print("=== CHECKING FROZEN 12-MODEL PANEL STATUS IN SCREENING MATRIX ===")
all_completed = True
for idx, arch in enumerate(FROZEN_CONVERGENCE_PANEL):
    id_l0 = f"{arch}_1110000_s42_l0"
    id_l1 = f"{arch}_1110000_s42_l1"
    
    st_l0 = cursor.execute("SELECT status, summary_json FROM jobs WHERE id=?", (id_l0,)).fetchone()
    st_l1 = cursor.execute("SELECT status, summary_json FROM jobs WHERE id=?", (id_l1,)).fetchone()
    
    s0_status = st_l0[0] if st_l0 else "missing"
    s1_status = st_l1[0] if st_l1 else "missing"
    
    r0 = json.loads(st_l0[1]).get("val_missing_pearson") if (st_l0 and st_l0[1]) else None
    r1 = json.loads(st_l1[1]).get("val_missing_pearson") if (st_l1 and st_l1[1]) else None
    
    ok = (s0_status == "completed" and s1_status == "completed")
    if not ok: all_completed = False
    
    print(f"[{idx+1:02d}/12] {arch:<42} | L1: {s0_status:<10} (r={r0}) | L2: {s1_status:<10} (r={r1}) | OK: {ok}")

conn.close()
print(f"\nAll 12 Frozen Panel Members Completed in Screening Matrix: {all_completed}")
