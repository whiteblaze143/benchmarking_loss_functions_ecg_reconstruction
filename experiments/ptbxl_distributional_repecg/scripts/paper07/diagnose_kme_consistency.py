import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parents[4]
EXPERIMENT_DIR = REPO_DIR / "experiments/ptbxl_distributional_repecg"
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts/paper07"))
sys.path.insert(0, str(EXPERIMENT_DIR / "src"))

from evaluator_physical_operator import (
    FastResponseComputer,
    build_canonical_and_derived_operator_bank,
    verify_operator_physical_consistency,
)
from repecg.paper07_operator.measurements import normalize_operator, response_atoms
from repecg.common.kernels import imq_kernel

CODEX_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")
FINAL_REPS_DIR = CODEX_DIR / "paper07_operator/final_representations"

def reference_kme_float64(beats_mv: np.ndarray, q: np.ndarray, fit_data: dict) -> np.ndarray:
    """Independent, purely float64 reference implementation of Phase-KME for a single operator q."""
    v_scale = float(fit_data["voltage_scale"])
    w_mean = fit_data["whitening_mean"].astype(np.float64)        # (2,)
    w_comp = fit_data["whitening_components"].astype(np.float64)  # (2, 2)
    w_scale = fit_data["whitening_scales"].astype(np.float64)     # (2,)
    landmarks = fit_data["landmarks"].astype(np.float64)          # (128, 2)
    inv_root = fit_data["inverse_root"].astype(np.float64)        # (128, 128)
    c2 = float(fit_data["c2"])

    # 1. Atoms: (n_beats, 256, 2)
    atoms = response_atoms(beats_mv.astype(np.float64), q, v_scale)
    n_beats = len(beats_mv)

    # 2. Cells: (16 phase bins, n_beats * 16 samples, 2)
    cells = atoms.reshape(n_beats, 16, 16, 2).transpose(1, 0, 2, 3).reshape(16, n_beats * 16, 2)

    # 3. Whitening: (16, N, 2)
    white = (cells - w_mean) @ w_comp * w_scale

    # 4. Kernel / Nystrom mapping per bin
    # white[b]: (N, 2), landmarks: (128, 2)
    # imq_kernel(white[b], landmarks): (N, 128)
    # mean over N samples -> (128,) @ inv_root -> (128,)
    resps = np.empty((16, 128), dtype=np.float64)
    for b in range(16):
        K_val = imq_kernel(white[b], landmarks, c2=c2)  # (N, 128)
        phi = K_val @ inv_root                          # (N, 128)
        resps[b] = phi.mean(axis=0)                     # (128,)

    return resps.astype(np.float32)


def main():
    print("=" * 80)
    print("PHASE-KME DETERMINISTIC CONSISTENCY & ROOT-CAUSE INVESTIGATION")
    print("=" * 80)

    # Load patient 0 authentic beats
    beats_path = FINAL_REPS_DIR / "fold9_authentic_beats.npz"
    d_beats = np.load(beats_path, allow_pickle=True)
    beats_mv = d_beats["beats"][0]  # (11, 256, 8)
    print(f"Patient 0 authentic beats shape: {beats_mv.shape}, dtype: {beats_mv.dtype}")

    fit_path = FINAL_REPS_DIR / "response_fit_final.npz"
    fit_data = dict(np.load(fit_path))

    # Initialize FastResponseComputer on GPU and CPU
    device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    computer_gpu = FastResponseComputer(fit_path, device_gpu)
    computer_cpu = FastResponseComputer(fit_path, torch.device("cpu"))

    # Operators
    e = np.eye(8, dtype=np.float32)
    q_bank_16 = build_canonical_and_derived_operator_bank()
    print(f"e shape: {e.shape}, q_bank_16 shape: {q_bank_16.shape}")

    # Stage 1: Compare float64 Reference vs GPU / CPU FastResponseComputer for individual canonical leads
    print("\n--- STAGE 1: Individual Operators (Float64 Reference vs GPU vs CPU) ---")
    diffs_ref_gpu = []
    diffs_ref_cpu = []
    diffs_gpu_cpu = []

    for i in range(8):
        q = e[i]
        ref = reference_kme_float64(beats_mv, q, fit_data)
        gpu = computer_gpu.compute_record_responses(beats_mv, q[None, :])[0]
        cpu = computer_cpu.compute_record_responses(beats_mv, q[None, :])[0]

        d_rg = np.max(np.abs(ref - gpu))
        d_rc = np.max(np.abs(ref - cpu))
        d_gc = np.max(np.abs(gpu - cpu))

        diffs_ref_gpu.append(d_rg)
        diffs_ref_cpu.append(d_rc)
        diffs_gpu_cpu.append(d_gc)
        print(f"Lead {i}: Ref vs GPU diff: {d_rg:.2e} | Ref vs CPU diff: {d_rc:.2e} | GPU vs CPU diff: {d_gc:.2e}")

    print(f"Max Ref vs GPU: {max(diffs_ref_gpu):.2e}, Max Ref vs CPU: {max(diffs_ref_cpu):.2e}")

    # Stage 2: Batch Size Invariance (Single vs 8-bank vs 16-bank vs Duplicated vs Shuffled)
    print("\n--- STAGE 2: Batch Size & Permutation Invariance on GPU ---")
    resp_16_gpu = computer_gpu.compute_record_responses(beats_mv, q_bank_16)  # (16, 16, 128)
    resp_8_gpu = computer_gpu.compute_record_responses(beats_mv, e)           # (8, 16, 128)

    # Individual single-op evaluations stacked
    resp_singles_gpu = np.stack([
        computer_gpu.compute_record_responses(beats_mv, e[i:i+1])[0] for i in range(8)
    ], axis=0)  # (8, 16, 128)

    # Shuffled 16-bank
    rng = np.random.default_rng(1234)
    perm = rng.permutation(16)
    inv_perm = np.argsort(perm)
    q_shuffled = q_bank_16[perm]
    resp_shuffled_gpu = computer_gpu.compute_record_responses(beats_mv, q_shuffled)[inv_perm]

    # Duplicated 32-bank
    q_dup = np.repeat(q_bank_16, 2, axis=0)
    resp_dup_gpu = computer_gpu.compute_record_responses(beats_mv, q_dup)[::2]

    diff_16_vs_8 = np.max(np.abs(resp_16_gpu[:8] - resp_8_gpu))
    diff_8_vs_singles = np.max(np.abs(resp_8_gpu - resp_singles_gpu))
    diff_16_vs_singles = np.max(np.abs(resp_16_gpu[:8] - resp_singles_gpu))
    diff_16_vs_shuffled = np.max(np.abs(resp_16_gpu - resp_shuffled_gpu))
    diff_16_vs_dup = np.max(np.abs(resp_16_gpu - resp_dup_gpu))

    print(f"GPU: 16-bank[:8] vs 8-bank:       diff = {diff_16_vs_8:.6e}")
    print(f"GPU: 8-bank vs Singles:           diff = {diff_8_vs_singles:.6e}")
    print(f"GPU: 16-bank[:8] vs Singles:      diff = {diff_16_vs_singles:.6e}")
    print(f"GPU: 16-bank vs Shuffled-unwound: diff = {diff_16_vs_shuffled:.6e}")
    print(f"GPU: 16-bank vs Duplicated-bank:  diff = {diff_16_vs_dup:.6e}")

    # Stage 3: Repeat Stage 2 on CPU
    print("\n--- STAGE 3: Batch Size & Permutation Invariance on CPU ---")
    resp_16_cpu = computer_cpu.compute_record_responses(beats_mv, q_bank_16)
    resp_8_cpu = computer_cpu.compute_record_responses(beats_mv, e)
    diff_16_vs_8_cpu = np.max(np.abs(resp_16_cpu[:8] - resp_8_cpu))
    print(f"CPU: 16-bank[:8] vs 8-bank:       diff = {diff_16_vs_8_cpu:.6e}")

    # Stage 4: Compare with CACHED PRECOMPUTED REPRESENTATIONS
    print("\n--- STAGE 4: Comparison with Cached Precomputed Representations ---")
    val_path = FINAL_REPS_DIR / "representation_final_select_waveforms.npz"
    d_val = np.load(val_path)
    cached_ops_0 = d_val["operators"][0]       # (16, 8)
    cached_resps_0 = d_val["responses"][0]     # (16, 16, 128)

    diff_cached_ops = np.max(np.abs(cached_ops_0 - q_bank_16))
    diff_cached_vs_gpu16 = np.max(np.abs(cached_resps_0 - resp_16_gpu))
    diff_cached_vs_cpu16 = np.max(np.abs(cached_resps_0 - resp_16_cpu))
    diff_cached_vs_ref = np.max(np.abs(cached_resps_0[:8] - np.stack([reference_kme_float64(beats_mv, e[i], fit_data) for i in range(8)])))

    print(f"Cached Operators vs q_bank_16:            diff = {diff_cached_ops:.6e}")
    print(f"Cached Responses[0] vs GPU 16-bank:        diff = {diff_cached_vs_gpu16:.6e}")
    print(f"Cached Responses[0] vs CPU 16-bank:        diff = {diff_cached_vs_cpu16:.6e}")
    print(f"Cached Responses[0][:8] vs Float64 Ref:    diff = {diff_cached_vs_ref:.6e}")

    # Stage 5: Dissection of Intermediate Tensors if diff > 1e-5
    print("\n--- STAGE 5: Step-by-Step Intermediate Dissection ---")
    atoms_16 = [response_atoms(beats_mv.astype(np.float64), q, computer_gpu.v_scale) for q in q_bank_16]
    atoms_8 = [response_atoms(beats_mv.astype(np.float64), q, computer_gpu.v_scale) for q in e]

    diff_atoms = np.max(np.abs(np.stack(atoms_16[:8]) - np.stack(atoms_8)))
    print(f"Step 5.1: Raw Atoms diff (first 8 ops): {diff_atoms:.6e}")

    cells_16 = np.stack(atoms_16).reshape(16, len(beats_mv), 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(16, 16, -1, 2)
    cells_8 = np.stack(atoms_8).reshape(8, len(beats_mv), 16, 16, 2).transpose(0, 2, 1, 3, 4).reshape(8, 16, -1, 2)
    diff_cells = np.max(np.abs(cells_16[:8] - cells_8))
    print(f"Step 5.2: Reshaped Cells diff: {diff_cells:.6e}")

    t_cells_16 = torch.as_tensor(cells_16, device=device_gpu, dtype=torch.float32)
    t_cells_8 = torch.as_tensor(cells_8, device=device_gpu, dtype=torch.float32)
    white_16 = (t_cells_16 - computer_gpu.w_mean) @ computer_gpu.w_comp * computer_gpu.w_scale
    white_8 = (t_cells_8 - computer_gpu.w_mean) @ computer_gpu.w_comp * computer_gpu.w_scale
    diff_white = float((white_16[:8] - white_8).abs().max())
    print(f"Step 5.3: Whitened Coordinates diff: {diff_white:.6e}")

    d2_16 = torch.cdist(white_16, computer_gpu.landmarks.expand(16, 16, -1, -1)).square() + computer_gpu.c2
    d2_8 = torch.cdist(white_8, computer_gpu.landmarks.expand(8, 16, -1, -1)).square() + computer_gpu.c2
    diff_d2 = float((d2_16[:8] - d2_8).abs().max())
    print(f"Step 5.4: Squared Distances (d2) diff: {diff_d2:.6e}")

    feat_16 = torch.rsqrt(d2_16) @ computer_gpu.inv_root
    feat_8 = torch.rsqrt(d2_8) @ computer_gpu.inv_root
    diff_feat = float((feat_16[:8] - feat_8).abs().max())
    print(f"Step 5.5: Nystrom Features diff: {diff_feat:.6e}")

    final_16 = feat_16.mean(dim=2).cpu().numpy().astype(np.float32)
    final_8 = feat_8.mean(dim=2).cpu().numpy().astype(np.float32)
    diff_final = float(np.max(np.abs(final_16[:8] - final_8)))
    print(f"Step 5.6: Final Mean Features diff: {diff_final:.6e}")

    # Check: What was the 0.0312 diff previously observed comparing?!
    print("\n--- STAGE 6: Investigating the exact line that produced 0.0312 in the log ---")
    # In evaluate_ptbxl_three_way.py, line 204 was:
    # test_beats_mv = record_beats_list[0][:10]
    # consistency_result = verify_operator_physical_consistency(computer, test_beats_mv, operators_16[0])
    test_10_beats = beats_mv[:10]
    r_16_ten = computer_gpu.compute_record_responses(test_10_beats, q_bank_16)
    r_8_ten = computer_gpu.compute_record_responses(test_10_beats, e)
    print(f"With 10 beats: r_16[:8] vs r_8 diff = {np.max(np.abs(r_16_ten[:8] - r_8_ten)):.6e}")
    print(f"With 11 beats vs 10 beats: r_16(11)[:8] vs r_16(10)[:8] diff = {np.max(np.abs(resp_16_gpu[:8] - r_16_ten[:8])):.6e}")
    print(f"r_16_ten[:8] vs cached_resps_0[:8] diff = {np.max(np.abs(r_16_ten[:8] - cached_resps_0[:8])):.6e}")

    print("\n--- STAGE 7: Calling verify_operator_physical_consistency directly ---")
    consistency_res = verify_operator_physical_consistency(computer_gpu, beats_mv, q_bank_16)
    print("verify_operator_physical_consistency result:", consistency_res)

    summary = {
        "diff_16_vs_8_gpu": float(diff_16_vs_8),
        "diff_16_vs_8_cpu": float(diff_16_vs_8_cpu),
        "diff_ref_vs_gpu": float(max(diffs_ref_gpu)),
        "diff_cached_vs_gpu16": float(diff_cached_vs_gpu16),
        "diff_11beats_vs_10beats": float(np.max(np.abs(resp_16_gpu[:8] - r_16_ten[:8]))),
        "consistency_result": consistency_res,
    }
    out_json = EXPERIMENT_DIR / "scripts/paper07/diagnostic_kme_consistency_results.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved diagnostic summary to {out_json}")
    print("=" * 80)

if __name__ == "__main__":
    main()
