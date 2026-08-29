#!/usr/bin/env python3
import os
import sys
import torch
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.train_mcma_3lead import MCMAModel
from unified_latents.engineering.models.fm_v2 import MCMAModel_FM_V2, STMEM_Prior

def run_smoke_tests():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running smoke tests on {device}...\n")
    
    # Dummy input
    B, C, T = 4, 3, 5120
    x = torch.randn(B, C, T, device=device)
    
    print("--- Test A: F0 Reproduction ---")
    model_f0 = MCMAModel(in_channels=3, out_channels=12).to(device)
    model_f0.eval()
    with torch.no_grad():
        out_f0 = model_f0(x)
    print("Test A Passed: F0 instantiated and processed successfully.")
    
    print("\n--- Test B: Zero-Injection Identity (F2) ---")
    model_f2 = MCMAModel_FM_V2(in_channels=3, fm_class=STMEM_Prior, use_film=True, use_residual=True).to(device)
    
    # We must copy F0's base model weights so they are identical before testing the residual
    model_f2.base_model.load_state_dict(model_f0.state_dict())
    model_f2.eval()
    
    with torch.no_grad():
        out_f2 = model_f2(x)
        
    diff = torch.max(torch.abs(out_f2 - out_f0)).item()
    if diff < 1e-6:
        print(f"Test B Passed: Max diff {diff:.2e} < 1e-6. F2 exactly matches F0 at init.")
    else:
        print(f"Test B FAILED: Max diff {diff:.2e} > 1e-6!")
        sys.exit(1)
        
    print("\n--- Test C & D: FM Variance ---")
    # Using the mock encode from STMEM_Prior
    with torch.no_grad():
        fm_out = model_f2.fm.encode(x, lead_id=1)
        z = fm_out["global"]
        H = fm_out["temporal"]
        
    z_var = torch.var(z, dim=0).mean().item()
    H_var = torch.var(H, dim=1).mean().item()
    
    if z_var > 0:
        print(f"Test C Passed: Batch variance of z_FM > 0 ({z_var:.4f})")
    else:
        print("Test C FAILED: Batch variance of z_FM is 0")
        sys.exit(1)
        
    if H_var > 0:
        print(f"Test D Passed: Temporal variance of H_FM > 0 ({H_var:.4f})")
    else:
        print("Test D FAILED: Temporal variance of H_FM is 0")
        sys.exit(1)

    print("\n--- Test E: Frozen State Verification ---")
    is_frozen = all(not p.requires_grad for n, p in model_f2.fm.named_parameters() if 'dummy_param' not in n)
    is_eval = not model_f2.fm.training
    if is_frozen and is_eval:
        print("Test E Passed: FM requires_grad=False and training=False.")
    else:
        print(f"Test E FAILED: requires_grad={not is_frozen}, training={not is_eval}")
        sys.exit(1)
        
    print("\n--- Test F: Shuffled Prior Invariant ---")
    with torch.no_grad():
        out_f2_shuffled = model_f2(x, shuffle_z_fm=True)
    diff_shuffle = torch.max(torch.abs(out_f2 - out_f2_shuffled)).item()
    if diff_shuffle < 1e-6:
        print(f"Test F Passed: Max diff {diff_shuffle:.2e} < 1e-6. Shuffling FM has no effect at init.")
    else:
        print(f"Test F FAILED: Shuffling FM altered output by {diff_shuffle:.2e} at init.")
        sys.exit(1)
        
    print("\nALL SMOKE TESTS PASSED!")

if __name__ == "__main__":
    run_smoke_tests()
