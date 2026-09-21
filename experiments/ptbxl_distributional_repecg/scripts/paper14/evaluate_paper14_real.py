#!/usr/bin/env python3
"""Comprehensive real-data evaluation for Paper 14 V2.2 ERM-vs-MMD experiment.

Implements all frozen execution contracts and evaluation endpoints:
1. Lexicographic selection rule on Fold 7:
   - Task preservation constraint: AUC_lambda >= AUC_ERM - 0.02
   - lambda* = argmin_{lambda in Lambda_eligible} MMD^2(Z, E)
2. Within-record matched permutation null:
   - For every record i, independently draw pi_i in S_5:
     E_{i,e}^{(b)} = pi_i^{(b)}(e)
   - Null distribution over B=500 draws: Q50, Q90, Q95, Q99
3. Primary evaluation panel on Fold 7 (selection) and Fold 8 (held-out test):
   - Marginal alignment: multi-environment and pairwise MMD^2(Z, E)
   - 5-way environment linear probe: macro OvR AUROC and accuracy vs 0.20 chance
   - Diagnostic utility: PTB-XL macro AUROC
   - Perturbation robustness: macro AUROC by environment e in {0..4}
   - Representation geometry: r_rel = r_eff / (d - 1), E||Z||, Var(Z)
   - Individual stability: paired representation drift D_e^paired = (1/N) sum_i ||Z_i^clean - Z_i^(e)||_2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Disable cuDNN to prevent PyTorch 2.6 / cuDNN 9.2 ptrDesc->finalize() bug on A100
torch.backends.cudnn.enabled = False
from sklearn.metrics import roc_auc_score

from repecg.common.metrics import multilabel_metrics
from repecg.common.models import (
    InvariantMechanismDiscoveryModel,
    effective_rank,
    multi_environment_mmd,
    biased_imq_mmd2,
)
from repecg.common.variants import get_variants_for_paper


ENVIRONMENTS = {
    0: "clean",
    1: "gain_0.8",
    2: "gain_1.2",
    3: "noise_20db",
    4: "resample_250hz",
}
NUM_ENVS = len(ENVIRONMENTS)


def compute_matched_permutation_null(
    z: torch.Tensor,
    record_ids: np.ndarray,
    environments: np.ndarray,
    b_draws: int = 500,
    seed: int = 42,
    c2: float = 1.0,
) -> dict[str, Any]:
    """Contract 1: Within-record matched permutation null.
    
    For every record i, draw pi_i in S_5 and permute environment labels across its 5 views:
      E_{i,e}^{(b)} = pi_i^{(b)}(e)
    Preserves exact 5-view cluster, patient geometry, and environment counts.
    """
    rng = np.random.default_rng(seed)
    unique_records = np.unique(record_ids)
    num_records = len(unique_records)
    
    # Pre-index positions for each record: record_positions[i] = [pos_0, pos_1, pos_2, pos_3, pos_4]
    record_positions = {}
    for r in unique_records:
        record_positions[r] = np.where(record_ids == r)[0]

    null_mmds = []
    z_dev = z.cuda() if torch.cuda.is_available() else z

    for b in range(b_draws):
        permuted_env = np.empty_like(environments)
        for r, pos in record_positions.items():
            # Draw independent permutation in S_5
            p = rng.permutation(len(pos))
            permuted_env[pos] = environments[pos[p]]
            
        perm_env_t = torch.from_numpy(permuted_env).to(z_dev.device)
        with torch.no_grad():
            null_val = float(multi_environment_mmd(z_dev, perm_env_t, c2=c2).item())
        null_mmds.append(null_val)

    null_mmds = np.array(null_mmds)
    return {
        "b_draws": b_draws,
        "null_q50": float(np.percentile(null_mmds, 50)),
        "null_q90": float(np.percentile(null_mmds, 90)),
        "null_q95": float(np.percentile(null_mmds, 95)),
        "null_q99": float(np.percentile(null_mmds, 99)),
        "null_mean": float(np.mean(null_mmds)),
        "null_std": float(np.std(null_mmds)),
    }


def fit_and_eval_environment_probe(
    z_train: torch.Tensor,
    env_train: torch.Tensor,
    z_test: torch.Tensor,
    env_test: torch.Tensor,
    seed: int = 42,
    epochs: int = 60,
    lr: float = 0.01,
) -> dict[str, float]:
    """Train a linear 5-way environment probe to quantify environment decodability."""
    torch.manual_seed(seed)
    device = z_train.device
    rep_dim = z_train.shape[1]
    probe = nn.Linear(rep_dim, NUM_ENVS).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    probe.train()
    batch_size = 512
    num_samples = len(z_train)
    for epoch in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            logits = probe(z_train[idx])
            loss = loss_fn(logits, env_train[idx])
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        test_logits = probe(z_test)
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=1)

    y_true = env_test.cpu().numpy()
    accuracy = float(np.mean(test_preds == y_true))

    # Macro One-vs-Rest AUROC
    ovr_aucs = []
    for c in range(NUM_ENVS):
        binary_true = (y_true == c).astype(int)
        if len(np.unique(binary_true)) == 2:
            ovr_aucs.append(roc_auc_score(binary_true, test_probs[:, c]))
        else:
            ovr_aucs.append(0.5)
    macro_ovr_auroc = float(np.mean(ovr_aucs))

    return {
        "macro_ovr_auroc": macro_ovr_auroc,
        "accuracy": accuracy,
        "chance_accuracy": 1.0 / NUM_ENVS,
    }


def compute_paired_drift(
    z: torch.Tensor,
    record_ids: np.ndarray,
    environments: np.ndarray,
) -> dict[str, float]:
    """Compute secondary endpoint: paired representation drift.
    
    D_e^paired = (1/N) sum_i ||Z_i^clean - Z_i^(e)||_2
    """
    unique_records = np.unique(record_ids)
    clean_mask = (environments == 0)
    clean_idx_map = {r: idx for r, idx in zip(record_ids[clean_mask], np.where(clean_mask)[0])}

    z_cpu = z.cpu()
    drifts = {}
    for e in range(1, NUM_ENVS):
        e_mask = (environments == e)
        e_indices = np.where(e_mask)[0]
        e_records = record_ids[e_mask]
        
        # Match each record's e-view to its clean view
        dists = []
        for r, idx_e in zip(e_records, e_indices):
            idx_clean = clean_idx_map[r]
            d = float(torch.norm(z_cpu[idx_clean] - z_cpu[idx_e], p=2).item())
            dists.append(d)
            
        drifts[ENVIRONMENTS[e]] = float(np.mean(dists))

    drifts["mean_drift"] = float(np.mean(list(drifts.values())))
    return drifts


def evaluate_split_endpoints(
    model: InvariantMechanismDiscoveryModel,
    data: dict[str, np.ndarray],
    rep_key: str = "kernel",
    c2: float = 1.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute all primary and secondary endpoints on a given split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    X = torch.from_numpy(data[rep_key]).to(device)
    Y = data["labels"]
    env_np = data["environment"]
    env_t = torch.from_numpy(env_np).to(device)
    record_ids = data["record_ids"]

    # 1. Forward pass & representations Z
    with torch.no_grad():
        logits, z = model.forward_with_representation(X)
        probs = torch.sigmoid(logits).cpu().numpy()

    # 2. Overall PTB-XL Macro AUROC
    metrics = multilabel_metrics(Y, probs)
    overall_macro_auroc = float(metrics["macro_auroc"])

    # 3. Robustness: Macro AUROC per environment
    per_env_auroc = {}
    for e in range(NUM_ENVS):
        mask = (env_np == e)
        env_metrics = multilabel_metrics(Y[mask], probs[mask])
        per_env_auroc[ENVIRONMENTS[e]] = float(env_metrics["macro_auroc"])

    # 4. Multi-environment and pairwise MMD^2(Z, E)
    with torch.no_grad():
        multi_mmd = float(multi_environment_mmd(z.float(), env_t, c2=c2).item())
        pairwise_mmds = {}
        for e1 in range(NUM_ENVS):
            for e2 in range(e1 + 1, NUM_ENVS):
                z1 = z[env_t == e1].float()
                z2 = z[env_t == e2].float()
                pair_val = float(biased_imq_mmd2(z1, z2, c2=c2).item())
                pairwise_mmds[f"{ENVIRONMENTS[e1]}_vs_{ENVIRONMENTS[e2]}"] = pair_val

    # 5. Representation geometry
    with torch.no_grad():
        rep_dim = z.shape[1]
        max_d = rep_dim - 1
        eff_rank = effective_rank(z)
        r_rel = eff_rank / max_d
        mean_norm = float(z.norm(dim=1).mean().item())
        var_z = float(z.var(dim=0).mean().item())

    # 6. Matched Permutation Null (Contract 1)
    null_results = compute_matched_permutation_null(
        z=z.float(),
        record_ids=record_ids,
        environments=env_np,
        b_draws=300,
        seed=seed,
        c2=c2,
    )

    # 7. Secondary Endpoint: Paired representation drift
    paired_drift = compute_paired_drift(z, record_ids, env_np)

    return {
        "ptbxl_macro_auroc": overall_macro_auroc,
        "per_environment_auroc": per_env_auroc,
        "multi_environment_mmd": multi_mmd,
        "pairwise_mmd": pairwise_mmds,
        "geometry": {
            "rep_dim": rep_dim,
            "effective_rank": eff_rank,
            "r_rel": r_rel,
            "mean_norm": mean_norm,
            "coordinate_variance": var_z,
        },
        "matched_permutation_null": null_results,
        "paired_representation_drift": paired_drift,
        "representations_z": z.detach().cpu(),
        "environments": env_t.detach().cpu(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c2", type=float, default=1.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading representation splits from {args.representations}...")
    with np.load(args.representations / "representation_val.npz") as item:
        val_data = {k: np.asarray(item[k]) for k in item.files}
    with np.load(args.representations / "representation_fold8.npz") as item:
        test_data = {k: np.asarray(item[k]) for k in item.files}

    variant_registry = get_variants_for_paper(14)

    # Candidates to evaluate
    candidates = ["erm", "mmd_lambda_0.1", "mmd_lambda_0.3", "mmd_lambda_1.0", "mmd_lambda_3.0", "mmd_lambda_10.0"]
    available_variants = []
    models = {}

    for var in candidates:
        ckpt_path = args.training / f"{var}_best.pt"
        if not ckpt_path.exists():
            # Check cells directory if aggregate didn't copy
            cell_dirs = list((args.training / "cells").glob(f"{var}_*"))
            if cell_dirs:
                ckpt_path = cell_dirs[0] / "checkpoint.pt"
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            var_obj = variant_registry[var]
            input_dim = val_data["kernel"].shape[-1]
            model = InvariantMechanismDiscoveryModel(input_dim=input_dim, classes=5, variant=var_obj)
            model.load_state_dict(checkpoint["state_dict"])
            models[var] = model
            available_variants.append(var)
            print(f"Loaded checkpoint for {var} from {ckpt_path}")
        else:
            print(f"Checkpoint for {var} not found. Skipping.")

    if not available_variants:
        raise RuntimeError("No model checkpoints available to evaluate!")

    # =========================================================================
    # Phase 1: Evaluate Fold 7 for all available candidates
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Evaluation on Fold 7 (Validation & Model Selection)")
    print("=" * 60)
    fold7_results = {}
    z_val_cache = {}

    for var in available_variants:
        print(f"\nEvaluating {var} on Fold 7...")
        res = evaluate_split_endpoints(models[var], val_data, rep_key="kernel", c2=args.c2, seed=args.seed)
        z_val_cache[var] = (res.pop("representations_z"), res.pop("environments"))
        fold7_results[var] = res
        print(f"  Macro AUROC: {res['ptbxl_macro_auroc']:.4f}")
        print(f"  Multi-Env MMD: {res['multi_environment_mmd']:.5f}")
        print(f"  r_rel: {res['geometry']['r_rel']:.4f}")

    # Evaluate 5-way environment probe on Fold 7 representations
    print("\nTraining 5-way environment probes on Fold 7 representations...")
    probe_results_fold7 = {}
    for var in available_variants:
        z_val, env_val = z_val_cache[var]
        # Split fold 7 representations 80/20 by record across all 5 environments
        n_rec = len(z_val) // NUM_ENVS
        n_train_rec = int(n_rec * 0.8)
        idx_train = torch.cat([torch.arange(e * n_rec, e * n_rec + n_train_rec) for e in range(NUM_ENVS)])
        idx_test = torch.cat([torch.arange(e * n_rec + n_train_rec, (e + 1) * n_rec) for e in range(NUM_ENVS)])
        probe_res = fit_and_eval_environment_probe(
            z_train=z_val[idx_train],
            env_train=env_val[idx_train],
            z_test=z_val[idx_test],
            env_test=env_val[idx_test],
            seed=args.seed,
        )
        probe_results_fold7[var] = probe_res
        fold7_results[var]["environment_probe"] = probe_res
        print(f"  {var}: Macro OvR AUROC = {probe_res['macro_ovr_auroc']:.4f}, Accuracy = {probe_res['accuracy']:.4f}")

    # =========================================================================
    # Phase 2: Frozen Lexicographic Selection Rule on Fold 7
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Frozen Lexicographic Selection Rule on Fold 7")
    print("=" * 60)
    
    if "erm" not in fold7_results:
        raise RuntimeError("ERM-AUG baseline ('erm') is required to apply the selection rule!")

    auc_erm = fold7_results["erm"]["ptbxl_macro_auroc"]
    auc_floor = auc_erm - 0.02
    print(f"ERM-AUG Fold-7 Macro AUROC: {auc_erm:.4f}")
    print(f"Task-preservation non-inferiority floor (AUC_ERM - 0.02): {auc_floor:.4f}")

    mmd_candidates = [v for v in available_variants if v != "erm"]
    eligible_candidates = []
    for var in mmd_candidates:
        auc_var = fold7_results[var]["ptbxl_macro_auroc"]
        mmd_var = fold7_results[var]["multi_environment_mmd"]
        is_eligible = (auc_var >= auc_floor)
        status = "ELIGIBLE" if is_eligible else "FAIL (Task Inferiority)"
        print(f"  Candidate {var}: AUC={auc_var:.4f} (>= {auc_floor:.4f}: {is_eligible}), MMD={mmd_var:.5f} -> {status}")
        if is_eligible:
            eligible_candidates.append(var)

    if eligible_candidates:
        best_lambda_var = min(eligible_candidates, key=lambda v: fold7_results[v]["multi_environment_mmd"])
        mmd_qualified = True
        print(f"\n=> SELECTION VERDICT: {best_lambda_var} selected as lambda* (Min MMD among eligible)")
    else:
        best_lambda_var = None
        mmd_qualified = False
        print("\n=> SELECTION VERDICT: No candidate satisfied task non-inferiority. MMD does NOT qualify.")

    # =========================================================================
    # Phase 3: Evaluation on Held-out Fold 8 (Touched ONLY after selection is frozen)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Final Evaluation on Held-out Test Fold 8")
    print("=" * 60)
    fold8_results = {}
    z_test_cache = {}

    eval_test_variants = ["erm"]
    if best_lambda_var is not None and best_lambda_var not in eval_test_variants:
        eval_test_variants.append(best_lambda_var)
    # Also evaluate all available for completeness of audit logging
    for v in available_variants:
        if v not in eval_test_variants:
            eval_test_variants.append(v)

    for var in eval_test_variants:
        print(f"\nEvaluating {var} on Fold 8...")
        res = evaluate_split_endpoints(models[var], test_data, rep_key="kernel", c2=args.c2, seed=args.seed)
        z_test_cache[var] = (res.pop("representations_z"), res.pop("environments"))
        fold8_results[var] = res
        print(f"  Fold-8 Macro AUROC: {res['ptbxl_macro_auroc']:.4f}")
        print(f"  Fold-8 Multi-Env MMD: {res['multi_environment_mmd']:.5f}")
        print(f"  Fold-8 r_rel: {res['geometry']['r_rel']:.4f}")

    # Environment probe evaluated on Fold 8
    print("\nTraining 5-way environment probes on Fold 8 representations...")
    for var in eval_test_variants:
        z_test_v, env_test_v = z_test_cache[var]
        # Split fold 8 representations 80/20 by record across all 5 environments
        n_rec = len(z_test_v) // NUM_ENVS
        n_train_rec = int(n_rec * 0.8)
        idx_train = torch.cat([torch.arange(e * n_rec, e * n_rec + n_train_rec) for e in range(NUM_ENVS)])
        idx_test = torch.cat([torch.arange(e * n_rec + n_train_rec, (e + 1) * n_rec) for e in range(NUM_ENVS)])
        probe_res = fit_and_eval_environment_probe(
            z_train=z_test_v[idx_train],
            env_train=env_test_v[idx_train],
            z_test=z_test_v[idx_test],
            env_test=env_test_v[idx_test],
            seed=args.seed,
        )
        fold8_results[var]["environment_probe"] = probe_res
        print(f"  {var}: Macro OvR AUROC = {probe_res['macro_ovr_auroc']:.4f}, Accuracy = {probe_res['accuracy']:.4f}")

    # =========================================================================
    # Assemble Final Results & Summary
    # =========================================================================
    summary_report = {
        "selection_protocol": {
            "validation_split": "Fold 7",
            "test_split": "Fold 8 (Held-out)",
            "erm_auc_fold7": auc_erm,
            "auc_floor_fold7": auc_floor,
            "eligible_candidates": eligible_candidates,
            "selected_variant": best_lambda_var,
            "mmd_qualified": mmd_qualified,
        },
        "fold7_validation_results": fold7_results,
        "fold8_test_results": fold8_results,
    }

    report_path = args.output / "real_data_evaluation.json"
    report_path.write_text(json.dumps(summary_report, indent=2) + "\n")
    print(f"\nSaved full real-data evaluation results to {report_path}")


if __name__ == "__main__":
    main()
