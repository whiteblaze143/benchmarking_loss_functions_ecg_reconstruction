import sys
from pathlib import Path
import numpy as np
import torch

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts/paper07"))
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from evaluator_physical_operator import FastResponseComputer, build_canonical_and_derived_operator_bank, verify_operator_physical_consistency
from repecg.paper07_operator.model import OperatorSetModel

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = CODEX_DIR / "paper07_operator/final_representations"
CHECKPOINTS_SO = CODEX_DIR / "ptbxl_three_way_v1/checkpoints/setoperator"

def run_test(enable_tf32=False, load_models=False):
    print(f"\n--- Testing with enable_tf32={enable_tf32}, load_models={load_models} ---")
    if enable_tf32:
        torch.backends.cudnn.enabled = False
        torch.set_float32_matmul_precision("high")
    else:
        torch.backends.cudnn.enabled = True
        torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_models:
        chk_path = CHECKPOINTS_SO / "basis_s42.pt"
        chk = torch.load(chk_path, map_location="cpu", weights_only=False)
        m = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous")
        m.load_state_dict(chk["state_dict"] if "state_dict" in chk else chk)
        m.to(device)
        m.eval()

    val_path = FINAL_REPS_DIR / "representation_final_select_waveforms.npz"
    d_val = np.load(val_path)
    operators_16 = d_val["operators"]

    beats_npz_path = FINAL_REPS_DIR / "fold9_authentic_beats.npz"
    d_beats = np.load(beats_npz_path, allow_pickle=True)
    record_beats_list = list(d_beats["beats"])

    resp_fit_path = FINAL_REPS_DIR / "response_fit_final.npz"
    computer = FastResponseComputer(resp_fit_path, device)

    test_beats_mv = record_beats_list[0]
    q_bank_canonical = build_canonical_and_derived_operator_bank()

    e = np.eye(8, dtype=np.float32)
    r16 = computer.compute_record_responses(test_beats_mv, q_bank_canonical)
    r8 = computer.compute_record_responses(test_beats_mv, e)
    diff = np.max(np.abs(r16[:8] - r8))
    print(f"Direct r16[:8] vs r8 diff: {diff:.6e}")

    try:
        res = verify_operator_physical_consistency(computer, test_beats_mv, q_bank_canonical)
        print("verify_operator_physical_consistency: PASSED")
    except AssertionError as err:
        print(f"verify_operator_physical_consistency: FAILED ({err})")

if __name__ == "__main__":
    run_test(enable_tf32=False, load_models=False)
    run_test(enable_tf32=True, load_models=False)
    run_test(enable_tf32=True, load_models=True)
