import torch
from pathlib import Path
import json
import sys
from scripts.evaluate_comprehensive_registry import load_adapter

project_root = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
sys.path.insert(0, str(project_root))

with open(project_root / "results/pareto_models_registry.json") as f:
    registry = json.load(f)
    
spec = registry["models"][0]
adapter = load_adapter(spec, torch.device("cpu"))
data = torch.load(project_root / "data/ptb_xl/tensors/test/0.pt", weights_only=True).unsqueeze(0)
recon = adapter.reconstruct(data)
print(recon.shape)
