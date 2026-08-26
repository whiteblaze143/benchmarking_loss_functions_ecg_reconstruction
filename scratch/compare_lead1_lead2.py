import sqlite3, json, pandas as pd

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
rows = conn.execute("SELECT id, status, summary_json FROM jobs WHERE status='completed'").fetchall()
conn.close()

l0_dict = {}
l1_dict = {}

for r in rows:
    model_id = r[0]
    s = json.loads(r[2]) if r[2] else {}
    if "_l0" in model_id:
        clean = model_id.replace("_1110000_s42_l0", "")
        l0_dict[clean] = {
            "r": s.get("val_missing_pearson"),
            "r05": s.get("val_missing_pearson_p05"),
            "loss": s.get("val_recon_loss"),
            "bF1": s.get("boundary_f1_smoke"),
            "mIoU": s.get("miou_wave"),
            "P_iou": s.get("P_iou"),
            "QRS_iou": s.get("QRS_iou"),
            "T_iou": s.get("T_iou")
        }
    elif "_l1" in model_id:
        clean = model_id.replace("_1110000_s42_l1", "")
        l1_dict[clean] = {
            "r": s.get("val_missing_pearson"),
            "r05": s.get("val_missing_pearson_p05"),
            "loss": s.get("val_recon_loss"),
            "bF1": s.get("boundary_f1_smoke"),
            "mIoU": s.get("miou_wave"),
            "P_iou": s.get("P_iou"),
            "QRS_iou": s.get("QRS_iou"),
            "T_iou": s.get("T_iou")
        }

common_models = sorted(list(set(l0_dict.keys()) & set(l1_dict.keys())))
print(f"Common completed models in both Lead 1 & Lead 2: {len(common_models)}")

comp_data = []
for m in common_models:
    m0 = l0_dict[m]
    m1 = l1_dict[m]
    comp_data.append({
        "name": m,
        "L1_r": m0["r"],
        "L2_r": m1["r"],
        "diff_r": m1["r"] - m0["r"],
        "L1_loss": m0["loss"],
        "L2_loss": m1["loss"],
        "diff_loss": m1["loss"] - m0["loss"],
        "L1_bF1": m0["bF1"],
        "L2_bF1": m1["bF1"],
        "diff_bF1": m1["bF1"] - m0["bF1"],
        "L1_mIoU": m0["mIoU"],
        "L2_mIoU": m1["mIoU"],
        "diff_mIoU": m1["mIoU"] - m0["mIoU"],
        "L1_P_iou": m0["P_iou"],
        "L2_P_iou": m1["P_iou"],
        "diff_P_iou": m1["P_iou"] - m0["P_iou"],
        "L1_T_iou": m0["T_iou"],
        "L2_T_iou": m1["T_iou"],
        "diff_T_iou": m1["T_iou"] - m0["T_iou"],
    })

df_comp = pd.DataFrame(comp_data)
print("\n--- Summary Comparison Statistics (Lead 2 minus Lead 1) ---")
print(df_comp[["diff_r", "diff_loss", "diff_bF1", "diff_mIoU", "diff_P_iou", "diff_T_iou"]].describe().to_string())

print("\n--- Model-by-Model Lead 1 vs Lead 2 Comparison (Sorted by L2 Pearson) ---")
cols = ["name", "L1_r", "L2_r", "diff_r", "L1_loss", "L2_loss", "L1_bF1", "L2_bF1", "L1_P_iou", "L2_P_iou", "L1_T_iou", "L2_T_iou"]
print(df_comp[cols].sort_values("L2_r", ascending=False).to_string(index=False))
