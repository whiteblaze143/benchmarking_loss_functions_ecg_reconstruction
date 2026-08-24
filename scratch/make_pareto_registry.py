import json
from pathlib import Path

# The 5 models we care about, now all using s201 since that's what's currently in checkpoints/
PARETO_MODELS = [
    "f_1000000_s201",  # Baseline M0
    "f_1000002_s201",  # Anchor + MMD
    "f_1000101_s201",  # Anchor + ED + MMD
    "f_1000100_s201",  # Anchor + ED
    "f_1100001_s201",  # Anchor + Lead + MMD
]

root = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
registry_path = root / "results/pareto_models_registry.json"

models = []
for mid in PARETO_MODELS:
    ckpt_path = root / "checkpoints" / f"factorial_{mid.replace('f_', '')}.pt"
    if not ckpt_path.exists():
        print(f"WARNING: Checkpoint missing for {mid} at {ckpt_path}")
        continue
    
    spec = {
        "id": mid,
        "family": "unet",
        "kind": "unet",
        "checkpoint": f"checkpoints/factorial_{mid.replace('f_', '')}.pt",
        "observed_leads": [0, 1, 6, 7, 8, 9, 10, 11],
        "factorial_mask": mid.split("_")[1]
    }
    models.append(spec)

registry = {
    "schema_version": 3,
    "ptbxl_csv": "data/ptb_xl/ptbxl_database.csv",
    "scp_statements_csv": "data/ptb_xl/scp_statements.csv",
    "ecgfounder_repo": "ecg_fm_integration/ecgfounder_repo",
    "ecgfounder_checkpoint": "ecg_fm_integration/ecgfounder_repo/checkpoint/12_lead_ECGFounder.pth",
    "ecgfounder_labels_csv": "ecg_fm_integration/ecgfounder_repo/csv/ptbxl_label.csv",
    "ecgfounder_tasks": "ecg_fm_integration/ecgfounder_repo/tasks.txt",
    "five_superclass_backbone": "ecg_fm_integration/checkpoints/mimic_iv_ecg_physionet_pretrained.pt",
    "five_superclass_checkpoint": "checkpoints/factorial_v4/parity/ecgfm_five_superclass.pt",
    "nstdb_dir": "data/mit-bih-noise-stress-test-database-1.0.0",
    "fitbit_noise_dir": "data/fitbit_noise",
    "split": "test",
    "models": models
}

with open(registry_path, "w") as f:
    json.dump(registry, f, indent=2)

print(f"Wrote {len(models)} models to {registry_path}")
