#!/usr/bin/env python3
"""Generates the updated architecture_reference_tests.json including I082-v1, I082-v2, H071-v2, and all Stage 1 models."""
import os
import sys
import json
import datetime
from pathlib import Path
import torch

# Ensure cuDNN issue is avoided and math attention is used
torch.backends.cudnn.enabled = False

sys.path.insert(0, str(Path(__file__).resolve().parent))
import continuous_setoperator_models as m

def run_tests():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running architecture reference tests on {device}...")

    # 1. Parameter breakdown for A001
    a001 = m.build_continuous_setoperator("A001")
    a001_breakdown = {}
    for name, p in a001.named_parameters():
        a001_breakdown[name] = {
            "shape": list(p.shape),
            "numel": p.numel(),
            "requires_grad": p.requires_grad,
        }

    # 2. Benchmarks for all registered models
    benchmarks = {}
    models_to_test = [
        "A001", "A002", "A003", "A004",
        "D032", "D032-V1", "D037", "D037-V1",
        "E047", "F053", "F057", "G061",
        "H071", "H071-V2",
        "I082", "I082-V1", "I082-V2"
    ]

    for model_id in models_to_test:
        model = m.build_continuous_setoperator(model_id).to(device)
        model.train()
        params = m.count_trainable_parameters(model)
        
        # Test FLOPs
        flops = m.compute_model_flops(model, batch_size=1, num_channels=2, device=device)
        
        # Forward and backward test
        B = 4
        ops = torch.randn(B, 2, 8, device=device)
        resp = torch.randn(B, 2, 16, 128, device=device)
        raw = torch.randn(B, 2, 6000, device=device)
        rr = torch.randn(B, 36, device=device)
        cat_ids = torch.tensor([[0, 1]], dtype=torch.long, device=device).expand(B, -1)
        ops_2d = torch.randn(B, 2, 2, device=device)
        mask = torch.zeros(B, 2, dtype=torch.bool, device=device)

        model.zero_grad(set_to_none=True)
        if model_id == "A002":
            out = model(resp, cat_ids, mask=mask)
        elif model_id == "A003":
            out = model(resp, mask=mask)
        elif model_id == "A004":
            out = model(resp, ops_2d, mask=mask)
        elif model_id == "G061":
            out = model(resp, ops, rr, mask=mask)
        elif model_id in ["H071", "H071-V2"]:
            out = model(raw, ops, mask=mask)
        elif model_id == "I082-V2":
            out = model(raw, resp, ops, mask=mask)
        else:
            out = model(resp, ops, mask=mask)

        logits = out["logits"]
        loss = logits.sum()
        loss.backward()

        grad_finite = all(
            p.grad is not None and torch.isfinite(p.grad).all().item()
            for p in model.parameters()
            if p.requires_grad
        )

        benchmarks[model_id] = {
            "model_id": model_id,
            "class_name": model.__class__.__name__,
            "trainable_parameters": params,
            "forward_shape": list(logits.shape),
            "forward_pass_finite": torch.isfinite(logits).all().item(),
            "backward_gradients_finite": grad_finite,
            "flops_2lead": flops,
        }
        print(f"Model {model_id:8s}: {params:10,d} params | {flops:12,d} FLOPs | Finite Grad: {grad_finite}")

    # 3. Special test for I082-v2 frozen raw branch
    i082_v2 = m.build_continuous_setoperator("I082-V2")
    i082_v2.freeze_raw_branch()
    frozen_params = m.count_trainable_parameters(i082_v2)
    benchmarks["I082-V2"]["trainable_params_frozen_raw"] = frozen_params
    print(f"I082-V2 (frozen raw branch): {frozen_params:10,d} trainable params")

    # 4. Invariance & Refusal Tests for A001
    a001 = a001.to(device).eval()
    with torch.no_grad():
        # Permutation invariance
        r_nominal = torch.randn(1, 2, 16, 128, device=device)
        q_nominal = torch.tensor([[[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0]]], dtype=torch.float32, device=device)
        out_nominal = a001(r_nominal, q_nominal)

        r_swapped = torch.flip(r_nominal, dims=[1])
        q_swapped = torch.flip(q_nominal, dims=[1])
        out_swapped = a001(r_swapped, q_swapped)

        perm_latent_diff = float((out_nominal["latent"] - out_swapped["latent"]).abs().max())
        perm_logit_diff = float((out_nominal["logits"] - out_swapped["logits"]).abs().max())

        # Masking isolation
        mask_ch1 = torch.tensor([[False, True]], dtype=torch.bool, device=device)
        r_corrupt = r_nominal.clone()
        r_corrupt[:, 1] = r_corrupt[:, 1] * 100.0 + 999.0
        out_clean = a001(r_nominal, q_nominal, mask=mask_ch1)
        out_corrupt = a001(r_corrupt, q_nominal, mask=mask_ch1)
        mask_logit_diff = float((out_clean["logits"] - out_corrupt["logits"]).abs().max())

        # Indeterminate refusal
        mask_all = torch.tensor([[True, True]], dtype=torch.bool, device=device)
        all_absent_refusal = False
        try:
            _ = a001(r_nominal, q_nominal, mask=mask_all)
        except ValueError:
            all_absent_refusal = True

    invariance_tests = {
        "permutation_invariance": {
            "status": "PASS" if perm_logit_diff < 1e-5 else "FAIL",
            "max_latent_abs_diff": perm_latent_diff,
            "max_logit_abs_diff": perm_logit_diff,
        },
        "masking_isolation": {
            "status": "PASS" if mask_logit_diff < 1e-5 else "FAIL",
            "max_logit_abs_diff_under_100x_noise": mask_logit_diff,
        },
        "all_absent_indeterminate_refusal": {
            "status": "PASS" if all_absent_refusal else "FAIL",
            "refusal_raised_value_error": all_absent_refusal,
        }
    }

    results = {
        "metadata": {
            "test_suite": "T-CAIREM Continuous SetOperator Architecture Reference Tests (Stage 1B Corrected Suite)",
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "target_baseline_parameter_count": 1027009,
        },
        "a001_parameter_breakdown": a001_breakdown,
        "all_model_benchmarks": benchmarks,
        "invariance_tests": invariance_tests,
    }

    out_path = Path(__file__).resolve().parent / "architecture_reference_tests.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Successfully saved {out_path} ({len(json.dumps(results))} bytes).")

if __name__ == "__main__":
    run_tests()
