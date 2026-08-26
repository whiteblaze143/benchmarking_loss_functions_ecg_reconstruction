import sqlite3
import json
import pandas as pd

conn = sqlite3.connect("refine-logs/wavelet_ssl_1110000/full/queue.sqlite")
rows = conn.execute("SELECT id, summary_json FROM jobs WHERE status='completed'").fetchall()
conn.close()

data = []
for row in rows:
    model_id = row[0]
    if "_l0" not in model_id: continue # Only Lead 1s
    summary = json.loads(row[1]) if row[1] else {}
    data.append({
        "model_id": model_id,
        "val_pearson": summary.get("val_missing_pearson"),
        "val_recon_loss": summary.get("val_recon_loss"),
        "val_p05": summary.get("val_missing_pearson_p05"),
        "boundary_f1": summary.get("boundary_f1_smoke")
    })

if not data:
    print("No Lead 1 models found with completed status.")
else:
    df = pd.DataFrame(data).dropna(subset=["val_pearson"])
    print(f"Total Completed Lead 1s: {len(df)}")
    print("\nTop 10 Lead-1 Models by Validation Pearson Correlation:")
    print(df.sort_values("val_pearson", ascending=False).head(10).to_string(index=False))
