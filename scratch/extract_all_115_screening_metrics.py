import sqlite3, json
from pathlib import Path
import numpy as np

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
cursor = conn.cursor()
rows = cursor.execute("""
    SELECT id, summary_json 
    FROM jobs 
    WHERE status='completed' AND summary_json IS NOT NULL
    ORDER BY id
""").fetchall()
conn.close()

l0_list = []
l1_list = []

for jid, s_str in rows:
    s = json.loads(s_str)
    arch = jid.replace("_1110000_s42_l0", "").replace("_1110000_s42_l1", "")
    lead = 0 if "_l0" in jid else 1
    
    r = s.get("val_missing_pearson", 0.0)
    r05 = s.get("val_missing_pearson_p05", 0.0)
    p_iou = s.get("P_iou", 0.0)
    t_iou = s.get("T_iou", 0.0)
    qrs_iou = s.get("QRS_iou", 0.0)
    bf1 = s.get("boundary_f1_smoke", 0.0)
    loss = s.get("val_recon_loss", s.get("val_loss", 0.0))
    
    rec = {
        "id": jid,
        "arch": arch,
        "lead": lead,
        "r": r,
        "r05": r05,
        "p_iou": p_iou,
        "t_iou": t_iou,
        "qrs_iou": qrs_iou,
        "bf1": bf1,
        "loss": loss
    }
    if lead == 0:
        l0_list.append(rec)
    else:
        l1_list.append(rec)

l0_list.sort(key=lambda x: x["r"], reverse=True)
l1_list.sort(key=lambda x: x["r"], reverse=True)

print(f"Loaded: Lead-I = {len(l0_list)} jobs, Lead-II = {len(l1_list)} jobs. Total = {len(l0_list)+len(l1_list)}")

# Write formatted markdown tables to an artifact file
out_path = Path("scratch/all_115_screening_jobs.json")
with open(out_path, "w") as f:
    json.dump({"l0": l0_list, "l1": l1_list}, f, indent=2)

print("Saved raw records to scratch/all_115_screening_jobs.json")
