import sqlite3, json, pandas as pd
import numpy as np

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
rows = conn.execute("SELECT id, status, summary_json FROM jobs WHERE id LIKE '%_l0' ORDER BY id").fetchall()
conn.close()

data = []
for row in rows:
    model_id, status, summary_str = row
    if status != "completed" or not summary_str:
        continue
    s = json.loads(summary_str)
    
    # Categorize model
    name = model_id.replace("_1110000_s42_l0", "")
    category = "Other"
    if "conv_control" in name:
        category = "CNN Baseline"
    elif name.startswith("A0_") or name.startswith("E1_") or name.startswith("C1_"):
        category = "Core Architecture Matrix"
    elif name.startswith("R1_") or name.startswith("R2_") or name.startswith("R3_") or name.startswith("R4_") or name.startswith("R5_") or name.startswith("R6_") or name.startswith("R7_") or name.startswith("R8_") or name.startswith("R9_"):
        category = "Representation Sweep (R1-R9)"
    elif name.startswith("ssl_"):
        category = "Wavelet SSL / Contrastive"
    elif name.startswith("tf_") or name.startswith("enc_") or name.startswith("dec_") or name.startswith("dim_") or name.startswith("bank_") or name.startswith("fus_") or name.startswith("view_") or name.startswith("head_"):
        category = "Structural Ablation"
    elif name.startswith("del_") or name.startswith("loss_"):
        category = "Loss & Delineation Aux"
    elif name.startswith("byol_") or name.startswith("cons_"):
        category = "Self-Supervision Regularizer"

    data.append({
        "model_id": model_id,
        "clean_name": name,
        "category": category,
        "val_pearson": s.get("val_missing_pearson"),
        "val_p05": s.get("val_missing_pearson_p05"),
        "val_recon_loss": s.get("val_recon_loss"),
        "boundary_f1": s.get("boundary_f1_smoke"),
        "boundary_ppv": s.get("boundary_ppv_smoke"),
        "boundary_sens": s.get("boundary_sensitivity_smoke"),
        "macro_f1_wave": s.get("macro_f1_wave"),
        "miou_wave": s.get("miou_wave"),
        "P_iou": s.get("P_iou"),
        "QRS_iou": s.get("QRS_iou"),
        "T_iou": s.get("T_iou"),
        "train_recon": s.get("train_recon"),
        "train_ssl": s.get("train_ssl"),
        "train_total": s.get("train_total"),
    })

df = pd.DataFrame(data)
df.to_csv("scratch/all_57_lead1_models.csv", index=False)
print(f"Exported {len(df)} models to scratch/all_57_lead1_models.csv")

# Print grouped by category sorted by val_pearson
for cat, group in df.groupby("category"):
    print(f"\n==========================================")
    print(f" CATEGORY: {cat} (Count: {len(group)})")
    print(f"==========================================")
    cols = ["clean_name", "val_pearson", "val_p05", "val_recon_loss", "boundary_f1", "macro_f1_wave", "miou_wave", "QRS_iou", "P_iou", "T_iou"]
    print(group[cols].sort_values("val_pearson", ascending=False).to_string(index=False))

print(f"\n==========================================")
print(f" TOP 15 OVERALL (out of {len(df)})")
print(f"==========================================")
cols = ["clean_name", "category", "val_pearson", "val_p05", "val_recon_loss", "boundary_f1", "macro_f1_wave", "QRS_iou", "P_iou", "T_iou"]
print(df[cols].sort_values("val_pearson", ascending=False).head(15).to_string(index=False))

print(f"\n==========================================")
print(f" BOTTOM 10 OVERALL (out of {len(df)})")
print(f"==========================================")
print(df[cols].sort_values("val_pearson", ascending=True).head(10).to_string(index=False))
