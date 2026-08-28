import json, os
from pathlib import Path

runs_dir = Path("refine-logs/convergence_10e/runs")
if not runs_dir.is_dir():
    print("No convergence runs directory yet.")
    exit()

runs = sorted(list(runs_dir.iterdir()))
print(f"Total Convergence Runs in Directory: {len(runs)}")

for r in runs:
    metrics_file = r / "metrics.jsonl"
    if not metrics_file.is_file():
        print(f"[{r.name:<45}] No metrics.jsonl yet (starting)")
        continue
    
    epochs = []
    with open(metrics_file) as f:
        for line in f:
            if line.strip():
                epochs.append(json.loads(line))
    
    if not epochs:
        print(f"[{r.name:<45}] Empty metrics.jsonl")
        continue
    
    print(f"\n[{r.name:<45}] Epochs Completed: {len(epochs)}/10")
    print(f"{'Epoch':<6} | {'Val Pearson r':<14} | {'Tail r05':<10} | {'P_iou':<8} | {'T_iou':<8} | {'QRS_iou':<8} | {'Recon Loss':<10} | {'Train Loss':<10}")
    print("-" * 90)
    for ep in epochs:
        e_num = ep.get("epoch", len(epochs))
        r_val = ep.get("val_missing_pearson", 0.0)
        r05 = ep.get("val_missing_pearson_p05", 0.0)
        p = ep.get("P_iou", 0.0)
        t = ep.get("T_iou", 0.0)
        q = ep.get("QRS_iou", 0.0)
        rloss = ep.get("val_recon_loss", 0.0)
        tloss = ep.get("train_total", 0.0)
        print(f"{e_num:<6} | {r_val:<14.4f} | {r05:<10.4f} | {p:<8.3f} | {t:<8.3f} | {q:<8.3f} | {rloss:<10.4f} | {tloss:<10.4f}")

