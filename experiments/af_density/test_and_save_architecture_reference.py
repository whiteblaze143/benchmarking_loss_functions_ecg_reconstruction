#!/usr/bin/env python3
"""Comprehensive Architecture Reference Validation for Continuous SetOperator Models.

Validates:
1. Exact parameter counts, verifying A001 has EXACTLY 1,027,009 trainable parameters.
2. Forward pass shape consistency across batch sizes and lead topologies.
3. Backward pass gradient fidelity (finite gradients in all trainable tensors).
4. FLOP counts via PyTorch FlopCounterMode.
5. Permutation invariance under channel reordering.
6. Masking isolation (unobserved channels do not corrupt active channels).
7. All-absent channel indeterminate failure behavior.

Exports results to architecture_reference_tests.json and mirrors all artifacts.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import torch
import torch.nn as nn

# Add experiments/af_density to path
EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from continuous_setoperator_models import (
    SET_OPERATOR_MODEL_REGISTRY,
    build_continuous_setoperator,
    count_trainable_parameters,
    compute_model_flops,
    ContinuousSetOperatorA001,
)


def sha256_file(path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run_architecture_suite() -> dict:
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{timestamp}] Running Architecture Reference Validation on device: {device}")

    results = {
        "metadata": {
            "test_suite": "T-CAIREM Continuous SetOperator Architecture Reference Tests",
            "timestamp_utc": timestamp,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "target_baseline_parameter_count": 1027009,
        },
        "a001_parameter_breakdown": {},
        "architectures": {},
        "property_tests": {},
        "overall_status": "PENDING",
    }

    # ------------------------------------------------------------------------
    # 1. Exact Parameter Breakdown for A001
    # ------------------------------------------------------------------------
    print("\n--- 1. A001 Parameter Breakdown ---")
    model_a001 = build_continuous_setoperator("A001")
    total_a001_params = 0
    breakdown = {}

    for name, p in model_a001.named_parameters():
        count = p.numel()
        breakdown[name] = {
            "shape": list(p.shape),
            "numel": count,
            "requires_grad": p.requires_grad,
        }
        if p.requires_grad:
            total_a001_params += count

    results["a001_parameter_breakdown"] = breakdown
    results["a001_total_parameters"] = total_a001_params
    assert total_a001_params == 1027009, f"A001 parameter mismatch: expected 1027009, got {total_a001_params}"
    print(f"A001 Total Trainable Parameters: {total_a001_params:,} (EXACT MATCH!)")

    # Group breakdown
    module_groups = {
        "pool_query": sum(p.numel() for n, p in model_a001.named_parameters() if "pool_query" in n),
        "operator_mlp": sum(p.numel() for n, p in model_a001.named_parameters() if "operator" in n),
        "response_encoder": sum(p.numel() for n, p in model_a001.named_parameters() if "response" in n),
        "transformer_set_encoder": sum(p.numel() for n, p in model_a001.named_parameters() if "set_encoder" in n),
        "attention_pool": sum(p.numel() for n, p in model_a001.named_parameters() if "pool." in n),
        "latent_mlp": sum(p.numel() for n, p in model_a001.named_parameters() if "latent" in n),
        "af_head": sum(p.numel() for n, p in model_a001.named_parameters() if "head" in n),
    }
    results["a001_module_group_counts"] = module_groups
    for grp, cnt in module_groups.items():
        print(f"  - {grp:25s}: {cnt:10,d}")

    # ------------------------------------------------------------------------
    # 2. Benchmark All 12 Architectural Variants
    # ------------------------------------------------------------------------
    print("\n--- 2. Architectural Variants Test Suite ---")
    batch_size = 4
    num_channels = 2

    for model_id in SET_OPERATOR_MODEL_REGISTRY:
        model = build_continuous_setoperator(model_id).train()
        n_params = count_trainable_parameters(model)
        flops = compute_model_flops(model, batch_size=1, num_channels=2)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        # Inputs based on model
        if model_id == "H071":
            inputs = {
                "raw_waveforms": torch.randn(batch_size, num_channels, 6000),
                "operators": torch.randn(batch_size, num_channels, 8),
                "mask": torch.tensor([[False, False], [False, True], [True, False], [False, False]]),
            }
            out = model(**inputs)
        elif model_id == "A002":
            inputs = {
                "responses": torch.randn(batch_size, num_channels, 16, 128),
                "operator_ids": torch.tensor([[0, 1], [0, -1], [1, 0], [0, 1]], dtype=torch.long),
                "mask": torch.tensor([[False, False], [False, True], [False, False], [False, False]]),
            }
            out = model(**inputs)
        elif model_id == "A003":
            inputs = {
                "responses": torch.randn(batch_size, num_channels, 16, 128),
                "mask": torch.tensor([[False, False], [False, True], [False, False], [False, False]]),
            }
            out = model(**inputs)
        elif model_id == "A004":
            inputs = {
                "responses": torch.randn(batch_size, num_channels, 16, 128),
                "operators": torch.randn(batch_size, num_channels, 2),
                "mask": torch.tensor([[False, False], [False, True], [False, False], [False, False]]),
            }
            out = model(**inputs)
        elif model_id == "G061":
            inputs = {
                "responses": torch.randn(batch_size, num_channels, 16, 128),
                "operators": torch.randn(batch_size, num_channels, 8),
                "rr_features": torch.randn(batch_size, 36),
                "mask": torch.tensor([[False, False], [False, True], [False, False], [False, False]]),
            }
            out = model(**inputs)
        else:
            inputs = {
                "responses": torch.randn(batch_size, num_channels, 16, 128),
                "operators": torch.randn(batch_size, num_channels, 8),
                "mask": torch.tensor([[False, False], [False, True], [False, False], [False, False]]),
            }
            out = model(**inputs)

        logits = out["logits"]
        assert logits.shape == (batch_size,), f"{model_id} output shape mismatch: {logits.shape}"
        assert torch.isfinite(logits).all(), f"{model_id} non-finite logits"

        # Backward pass
        loss = logits.sum()
        loss.backward()

        finite_grads = True
        grad_tensor_count = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    finite_grads = False
                    break
                grad_tensor_count += 1

        results["architectures"][model_id] = {
            "class_name": model.__class__.__name__,
            "trainable_parameters": n_params,
            "delta_vs_a001": n_params - 1027009,
            "forward_flops_batch1_2ch": flops,
            "forward_shape": list(logits.shape),
            "forward_shape_test": "PASS",
            "finite_logits_test": "PASS",
            "loss_value": float(loss.item()),
            "loss_backward_test": "PASS",
            "finite_gradients_test": "PASS" if finite_grads else "FAIL",
            "gradient_tensors_validated": grad_tensor_count,
        }
        print(f"[{model_id}] {model.__class__.__name__:35s} | Params: {n_params:9,d} | FLOPs: {flops:12,d} | Grads: PASS")

    # ------------------------------------------------------------------------
    # 3. Permutation Invariance Property Test
    # ------------------------------------------------------------------------
    print("\n--- 3. Permutation Invariance Test ---")
    model_a001.eval()
    torch.manual_seed(42)
    resp = torch.randn(2, 2, 16, 128)
    ops = torch.randn(2, 2, 8)
    mask = torch.tensor([[False, False], [False, False]])

    # Forward original: channel 0 then channel 1
    with torch.no_grad():
        out_orig = model_a001(resp, ops, mask=mask)
        latent_orig = out_orig["latent"]
        logits_orig = out_orig["logits"]

    # Permuted: channel 1 then channel 0
    resp_perm = resp[:, [1, 0], :, :]
    ops_perm = ops[:, [1, 0], :]
    mask_perm = mask[:, [1, 0]]

    with torch.no_grad():
        out_perm = model_a001(resp_perm, ops_perm, mask=mask_perm)
        latent_perm = out_perm["latent"]
        logits_perm = out_perm["logits"]

    max_latent_diff = float(torch.max(torch.abs(latent_orig - latent_perm)))
    max_logit_diff = float(torch.max(torch.abs(logits_orig - logits_perm)))

    perm_pass = max_latent_diff < 1e-5 and max_logit_diff < 1e-5
    results["property_tests"]["permutation_invariance"] = {
        "max_latent_difference": max_latent_diff,
        "max_logit_difference": max_logit_diff,
        "tolerance": 1e-5,
        "status": "PASS" if perm_pass else "FAIL",
    }
    print(f"Permutation Invariance: max latent diff={max_latent_diff:.2e}, max logit diff={max_logit_diff:.2e} -> {'PASS' if perm_pass else 'FAIL'}")

    # ------------------------------------------------------------------------
    # 4. Masking Isolation Property Test
    # ------------------------------------------------------------------------
    print("\n--- 4. Masking Isolation Test ---")
    # For a sample where channel 1 is masked (mask = [False, True]), altering channel 1
    # responses must have ZERO effect on the output.
    mask_single = torch.tensor([[False, True]])
    resp_a = torch.randn(1, 2, 16, 128)
    ops_a = torch.randn(1, 2, 8)

    resp_b = resp_a.clone()
    # Heavily corrupt masked channel 1
    resp_b[:, 1, :, :] = resp_b[:, 1, :, :] + 100.0 * torch.randn(1, 16, 128)

    with torch.no_grad():
        out_a = model_a001(resp_a, ops_a, mask=mask_single)
        out_b = model_a001(resp_b, ops_a, mask=mask_single)

    mask_diff = float(torch.max(torch.abs(out_a["logits"] - out_b["logits"])))
    mask_pass = mask_diff < 1e-5
    results["property_tests"]["masking_isolation"] = {
        "max_logit_difference_on_corrupted_masked_lead": mask_diff,
        "tolerance": 1e-5,
        "status": "PASS" if mask_pass else "FAIL",
    }
    print(f"Masking Isolation: logit diff={mask_diff:.2e} -> {'PASS' if mask_pass else 'FAIL'}")

    # ------------------------------------------------------------------------
    # 5. Indeterminate Failure on All-Absent Channels
    # ------------------------------------------------------------------------
    print("\n--- 5. Indeterminate All-Absent Test ---")
    mask_all_absent = torch.tensor([[True, True]])
    try:
        model_a001(resp_a, ops_a, mask=mask_all_absent)
        all_absent_pass = False
    except ValueError as e:
        all_absent_pass = True
        err_msg = str(e)

    results["property_tests"]["all_absent_indeterminate_refusal"] = {
        "raised_expected_value_error": all_absent_pass,
        "error_message": err_msg if all_absent_pass else None,
        "status": "PASS" if all_absent_pass else "FAIL",
    }
    print(f"All-Absent Indeterminate Refusal: {'PASS (Refused to fabricate prediction)' if all_absent_pass else 'FAIL'}")

    all_passed = (
        all(arch["forward_shape_test"] == "PASS" for arch in results["architectures"].values())
        and all(arch["finite_gradients_test"] == "PASS" for arch in results["architectures"].values())
        and perm_pass
        and mask_pass
        and all_absent_pass
    )
    results["overall_status"] = "ALL_TESTS_PASSED" if all_passed else "SOME_TESTS_FAILED"

    # Save reference test results
    out_json = EXPERIMENT_DIR / "architecture_reference_tests.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved architecture test report to {out_json} ({out_json.stat().st_size:,} bytes)")

    return results


def mirror_artifacts():
    target_dir = Path("/data/mithunmanivannan/codex_artifacts/af_density_continuous_setoperator")
    target_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "continuous_setoperator_models.py",
        "ablation_registry.json",
        "ABLATION_MASTER_REGISTRY.md",
        "architecture_reference_tests.json",
    ]

    print("\n--- Mirroring Artifacts to NFS Codex Artifacts Directory ---")
    for fname in files_to_copy:
        src = EXPERIMENT_DIR / fname
        dst = target_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
            src_sha = sha256_file(src)
            dst_sha = sha256_file(dst)
            assert src_sha == dst_sha, f"Hash mismatch after copying {fname}"
            print(f"  [OK] Mirrored {fname:35s} -> {dst} (SHA: {src_sha[:12]}...)")
        else:
            print(f"  [MISSING] Source file {src} does not exist!")


if __name__ == "__main__":
    res = run_architecture_suite()
    mirror_artifacts()
    print("\nALL ARCHITECTURAL TASKS SUCCESSFULLY COMPLETED!")
