import torch
import numpy as np
from pathlib import Path
from evaluator_physical_operator import FastResponseComputer, build_canonical_and_derived_operator_bank

d_beats = np.load("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/fold9_authentic_beats.npz", allow_pickle=True)
beats = list(d_beats["beats"])
test_beats = beats[0][:10]

resp_fit_path = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/paper07_operator/final_representations/response_fit_final.npz")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
computer = FastResponseComputer(resp_fit_path, device)

q_bank = build_canonical_and_derived_operator_bank()
e = np.eye(8, dtype=np.float32)

r16 = computer.compute_record_responses(test_beats, q_bank)
r8 = computer.compute_record_responses(test_beats, q_bank[:8])
re = computer.compute_record_responses(test_beats, e)

print("diff r16[:8] vs r8:", np.max(np.abs(r16[:8] - r8)))
print("diff r8 vs re:", np.max(np.abs(r8 - re)))
print("diff r16[:8] vs re:", np.max(np.abs(r16[:8] - re)))
