import torch
import numpy as np
from pathlib import Path
from evaluator_physical_operator import FastResponseComputer, build_canonical_and_derived_operator_bank

resp_fit_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/response_fit_final.npz")
device = torch.device("cpu")
computer = FastResponseComputer(resp_fit_path, device)

rng = np.random.default_rng(42)
test_beats = rng.normal(size=(10, 256, 8)).astype(np.float64)

q_bank = build_canonical_and_derived_operator_bank()
e = np.eye(8, dtype=np.float32)

r16 = computer.compute_record_responses(test_beats, q_bank)
re = computer.compute_record_responses(test_beats, e)

diff = np.max(np.abs(r16[:8] - re))
print(f"Diff on random beats: {diff}")
