import torch
import numpy as np
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import continuous_setoperator_models as csm
from run_stage1b_corrective_queue import prepare_training_dataset, prepare_validation_dataset
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

device = "cuda" if torch.cuda.is_available() else "cpu"
train_tensors = prepare_training_dataset(target_count=4096)
norm_stats = train_tensors[6]
informative_segments = [33, 86, 89, 90, 100, 118, 125, 126, 187, 237, 241, 261, 262, 263, 274]
val_data = prepare_validation_dataset(informative_segments, norm_stats)

ckpt = torch.load("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator/checkpoints/A001_best.pt", map_location=device, weights_only=True)
m = csm.ContinuousSetOperatorA001().to(device)
m.load_state_dict(ckpt["state_dict"])
m.eval()

all_z = []
all_y = []
for fi in informative_segments:
    seg = val_data[fi]
    resp = seg["responses"].to(device)
    B = len(resp)
    ops = torch.zeros(B, 2, 8, device=device); ops[:, 0, 0] = 1.0; ops[:, 1, 1] = 1.0
    mask = torch.zeros(B, 2, dtype=torch.bool, device=device)
    with torch.no_grad():
        out = m(responses=resp, operators=ops, mask=mask)
        z = out["logits"].cpu().numpy()
    all_z.append(z)
    all_y.append(seg["targets"])
    print(f"Segment {fi:3d} (Pt {seg['patient_id']}): z in [{np.min(z):.3f}, {np.max(z):.3f}], AF={np.sum(seg['targets']==1)}/{len(z)}, AUC={roc_auc_score(seg['targets'], z) if len(np.unique(seg['targets'])) > 1 else 'N/A'}")

all_z = np.concatenate(all_z)
all_y = np.concatenate(all_y)
print("\nA001 Overall:")
print(f"  z mean: {np.mean(all_z):.4f}, std: {np.std(all_z):.4f}, min: {np.min(all_z):.4f}, max: {np.max(all_z):.4f}")
print(f"  Overall raw logits AUROC: {roc_auc_score(all_y, all_z):.4f}")

lr = LogisticRegression()
lr.fit(all_z.reshape(-1, 1), all_y)
print(f"  Logistic regression slope: {lr.coef_[0][0]:.4f}, intercept: {lr.intercept_[0]:.4f}")
p_cal = lr.predict_proba(all_z.reshape(-1, 1))[:, 1]
print(f"  Calibrated prob AUROC: {roc_auc_score(all_y, p_cal):.4f}")
