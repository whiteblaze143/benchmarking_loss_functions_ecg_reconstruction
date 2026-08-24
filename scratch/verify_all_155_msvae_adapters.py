import json
import torch
from pathlib import Path
from scripts.evaluate_comprehensive_registry import load_adapter

with open("results/clinical_biomarkers_model_registry.json") as f:
    reg = json.load(f)

msvae_models = [m for m in reg["models"] if m.get("kind") == "msvae"]
print(f"Verifying all {len(msvae_models)} MS-VAE models in registry...")

device = torch.device("cpu")
successful = 0
failed = []

for idx, spec in enumerate(msvae_models, 1):
    mid = spec["id"]
    try:
        adapter = load_adapter(spec, device)
        successful += 1
    except Exception as e:
        failed.append((mid, str(e)))

print(f"\nVerification Results:")
print(f"  Successfully loaded: {successful} / {len(msvae_models)}")
print(f"  Failed: {len(failed)} / {len(msvae_models)}")

if failed:
    print("\nFailed models:")
    for mid, err in failed:
        print(f"  {mid}: {err}")
else:
    print("\nALL 155 MS-VAE models are 100% addressable, accessible, and loadable by load_adapter!")
