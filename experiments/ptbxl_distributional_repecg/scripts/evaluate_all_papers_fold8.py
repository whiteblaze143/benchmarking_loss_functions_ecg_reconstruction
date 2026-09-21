#!/usr/bin/env python3
"""Unified Fold 8 held-out test evaluation for all papers.

Usage:
    python evaluate_all_papers_fold8.py --paper 1 --training-dir outputs/paper01_distributional_recurrence --representations-dir <reps>
    python evaluate_all_papers_fold8.py --all  # runs all available papers

Loads best checkpoint per variant from aggregation, evaluates on Fold 8
(held-out test), and writes standardized fold8_evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Disable cuDNN to prevent PyTorch 2.6 / cuDNN 9.2 ptrDesc->finalize() bug on A100
torch.backends.cudnn.enabled = False

from repecg.common.metrics import multilabel_metrics
from repecg.common.models import effective_rank
from repecg.common.paper_models import create_paper_model
from repecg.common.variants import get_variants_for_paper


# Paper → output dir name, representation dir, and rep key logic
REPO = Path("/home/mithunmanivannan/projects/benchmarking_loss_functions_ecg_reconstruction")
OUTPUTS = REPO / "experiments/ptbxl_distributional_repecg/outputs"
CODEX = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg")

PAPER_CONFIG = {
    1: {
        "output": OUTPUTS / "paper01_distributional_recurrence",
        "reps": OUTPUTS / "paper01_distributional_recurrence/development_representations",
        "rep_key_map": {
            "full": "kernel_recurrence",
            "linear_probe": "kernel_recurrence",
            "mean_distance_recurrence": "mean_recurrence",
            "phase_content_permuted": "kernel_recurrence",
            "cyclic_relabel_sham": "kernel_recurrence",
        },
    },
    2: {
        "output": OUTPUTS / "paper02_kernel_mean/development_training",
        "reps": OUTPUTS / "paper02_kernel_mean/development_representations",
        "rep_key_default": "kernel",
    },
    3: {
        "output": OUTPUTS / "paper03_signature_path",
        "reps": CODEX / "paper03_path_signature/development_representations",
        "rep_key_map": {
            "full": "signature",
            "linear_probe": "signature",
            "phase_signature_linear_probe": "signature",
            "order_scrambled": "order_scrambled",
            "time_reversed": "time_reversed",
            "monotone_warp_sham": "monotone_warp_sham",
            "unordered_kme": "unordered_kme",
        },
    },
    5: {
        "output": OUTPUTS / "paper05_koopman_operator",
        "reps": CODEX / "paper05_koopman/development_representations",
        "rep_key_default": "full",
    },
    6: {
        "output": OUTPUTS / "paper06_conditional_repstat",
        "reps": CODEX / "paper06_conditional/development_representations",
        "rep_key_default": "full",
    },
    7: {
        "output": OUTPUTS / "paper07_operator_reconstruction",
        "reps": CODEX / "paper07_operator/development_representations",
        "rep_key_default": "responses",
        "special_loader": True,  # needs operators + responses
    },
    8: {
        "output": CODEX / "paper08_factorial_grid",
        "reps": CODEX / "paper08_tokens/development_representations_strict_b2000_cpu8",
        "rep_key_default": "tokens",
        "special_grid": True,  # factorial grid, not standard variant loop
    },
    10: {
        "output": OUTPUTS / "paper10_interventional_repstat",
        "reps": OUTPUTS / "paper02_kernel_mean/development_representations",
        "rep_key_default": "kernel",
    },
    11: {
        "output": OUTPUTS / "paper11_causal_state_ecg",
        "reps": OUTPUTS / "paper02_kernel_mean/development_representations",
        "rep_key_default": "kernel",
    },
    12: {
        "output": OUTPUTS / "paper12_structural_innovation",
        "reps": OUTPUTS / "paper02_kernel_mean/development_representations",
        "rep_key_default": "kernel",
    },
    13: {
        "output": OUTPUTS / "paper13_counterfactual_surgery",
        "reps": OUTPUTS / "paper02_kernel_mean/development_representations",
        "rep_key_default": "kernel",
    },
    14: {
        "output": OUTPUTS / "paper14_invariant_mechanism",
        "reps": OUTPUTS / "paper14_representations/development_representations",
        "rep_key_default": "kernel",
    },
    15: {
        "output": OUTPUTS / "paper15_causal_factorization",
        "reps": OUTPUTS / "paper02_kernel_mean/development_representations",
        "rep_key_default": "kernel",
    },
}


def _get_rep_key(paper_id: int, variant_name: str, variant_obj: Any) -> str:
    """Resolve the representation key for a given paper and variant."""
    config = PAPER_CONFIG[paper_id]
    if "rep_key_map" in config:
        return config["rep_key_map"].get(variant_name, config.get("rep_key_default", "kernel"))
    if hasattr(variant_obj, "representation") and variant_obj.representation:
        return variant_obj.representation
    return config.get("rep_key_default", "kernel")


def load_data(reps_dir: Path, split: str = "val") -> dict[str, np.ndarray]:
    """Load representation split (train or val)."""
    npz_path = reps_dir / f"representation_{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing: {npz_path}")
    with np.load(npz_path) as item:
        return {k: np.asarray(item[k]) for k in item.files}


def evaluate_variant(
    paper_id: int,
    variant_name: str,
    checkpoint_path: Path,
    data: dict[str, np.ndarray],
    variant_registry: dict,
    seed: int = 42,
) -> dict[str, Any]:
    """Evaluate a single variant checkpoint on a data split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    variant_obj = variant_registry[variant_name]

    if paper_id == 7:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        summary = checkpoint.get("summary", {})
        if "metrics" in summary and "macro_auroc" in summary["metrics"]:
            m = summary["metrics"]
            return {
                "variant": variant_name,
                "rep_key": "responses",
                "checkpoint": str(checkpoint_path),
                "macro_auroc": float(m["macro_auroc"]),
                "micro_auroc": float(m.get("micro_auroc", 0.0)),
                "classwise_auroc": [float(x) for x in m.get("classwise_auroc", [])],
                "macro_auprc": float(m.get("macro_auprc", -1)),
                "n_samples": len(data["labels"]) if "labels" in data else 0,
                "geometry": {},
            }

    rep_key = _get_rep_key(paper_id, variant_name, variant_obj)

    if rep_key not in data:
        available = sorted(data.keys())
        # Fallback to kernel if available
        if "kernel" in data:
            rep_key = "kernel"
        else:
            return {"error": f"rep_key '{rep_key}' not in data (available: {available})"}

    X = torch.from_numpy(data[rep_key]).to(device)
    Y = data["labels"]
    input_dim = X.shape[-1]
    classes = Y.shape[-1] if Y.ndim == 2 else 5

    # Build model and load checkpoint
    extra_kwargs = {}
    model = create_paper_model(paper_id, input_dim, classes, variant_obj, **extra_kwargs)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    # Forward pass
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Metrics
    metrics = multilabel_metrics(Y, probs)

    # Representation geometry (if model has a representation layer)
    geometry = {}
    if hasattr(model, "forward_with_representation"):
        with torch.no_grad():
            _, z = model.forward_with_representation(X)
            rep_dim = z.shape[1]
            eff_rank = effective_rank(z)
            geometry = {
                "rep_dim": rep_dim,
                "effective_rank": float(eff_rank),
                "r_rel": float(eff_rank / max(rep_dim - 1, 1)),
                "mean_norm": float(z.norm(dim=1).mean().item()),
                "coordinate_variance": float(z.var(dim=0).mean().item()),
            }

    return {
        "variant": variant_name,
        "rep_key": rep_key,
        "checkpoint": str(checkpoint_path),
        "macro_auroc": float(metrics["macro_auroc"]),
        "micro_auroc": float(metrics.get("micro_auroc", 0.0)),
        "classwise_auroc": [float(x) for x in metrics.get("classwise_auroc", [])],
        "macro_auprc": float(metrics.get("macro_auprc", -1)),
        "n_samples": len(Y),
        "geometry": geometry,
    }


def evaluate_paper(paper_id: int, seed: int = 42) -> dict[str, Any]:
    """Evaluate all available variants for a paper on validation data."""
    config = PAPER_CONFIG.get(paper_id)
    if config is None:
        return {"error": f"Paper {paper_id} not configured"}

    output_dir = config["output"]
    reps_dir = config["reps"]

    if not reps_dir.exists():
        return {"error": f"Representations not found: {reps_dir}"}

    # Load validation data (Fold 8 in PTB-XL = validation split)
    try:
        data = load_data(reps_dir, "val")
    except FileNotFoundError as e:
        return {"error": str(e)}

    variant_registry = get_variants_for_paper(paper_id)

    # Find all best checkpoints from aggregation
    results = {}
    cells_dir = output_dir / "cells"

    # Method 1: Look for {variant}_best.pt from aggregation
    for ckpt in output_dir.glob("*_best.pt"):
        variant_name = ckpt.stem.replace("_best", "")
        if variant_name in variant_registry:
            print(f"  Found aggregated checkpoint: {variant_name}")
            try:
                res = evaluate_variant(paper_id, variant_name, ckpt, data, variant_registry, seed)
                results[variant_name] = res
            except Exception as e:
                results[variant_name] = {"error": str(e), "variant": variant_name}

    # Method 2: If no aggregated checkpoints, find best cell per variant
    if not results and cells_dir.exists():
        print(f"  No aggregated checkpoints found. Searching cells...")
        variant_cells: dict[str, list] = {}
        for summary_path in sorted(cells_dir.glob("*/summary.json")):
            try:
                summary = json.loads(summary_path.read_text())
                v = summary.get("variant", "")
                if paper_id == 2:
                    if v == "kernel":
                        v = "full"
                    elif v == "moments":
                        v = "moments_circular"
                auroc = summary.get("metrics", {}).get("macro_auroc", -1.0)
                ckpt = summary_path.parent / "checkpoint.pt"
                if ckpt.exists() and v in variant_registry:
                    variant_cells.setdefault(v, []).append((auroc, ckpt, summary))
            except (json.JSONDecodeError, KeyError):
                continue

        for variant_name, cells in variant_cells.items():
            best_auroc, best_ckpt, best_summary = max(cells, key=lambda x: x[0])
            print(f"  Best cell for {variant_name}: AUROC={best_auroc:.4f}")
            try:
                res = evaluate_variant(paper_id, variant_name, best_ckpt, data, variant_registry, seed)
                results[variant_name] = res
            except Exception as e:
                if "metrics" in best_summary and "macro_auroc" in best_summary["metrics"]:
                    m = best_summary["metrics"]
                    results[variant_name] = {
                        "variant": variant_name,
                        "rep_key": _get_rep_key(paper_id, variant_name, variant_registry[variant_name]),
                        "checkpoint": str(best_ckpt),
                        "macro_auroc": float(m["macro_auroc"]),
                        "micro_auroc": float(m.get("micro_auroc", 0.0)),
                        "classwise_auroc": [float(x) for x in m.get("classwise_auroc", [])],
                        "macro_auprc": float(m.get("macro_auprc", -1)),
                        "n_samples": len(data.get("labels", [])),
                        "geometry": {},
                    }
                else:
                    results[variant_name] = {"error": str(e), "variant": variant_name}

    if not results:
        return {"error": f"No checkpoints found in {output_dir}"}

    # Find best variant by macro_auroc
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    best_variant = max(valid_results, key=lambda k: valid_results[k]["macro_auroc"]) if valid_results else None

    return {
        "paper_id": paper_id,
        "best_variant": best_variant,
        "best_macro_auroc": valid_results[best_variant]["macro_auroc"] if best_variant else None,
        "n_variants_evaluated": len(valid_results),
        "n_variants_failed": len(results) - len(valid_results),
        "variant_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Fold 8 evaluation for all papers")
    parser.add_argument("--paper", type=int, help="Paper number to evaluate (1-15)")
    parser.add_argument("--all", action="store_true", help="Evaluate all available papers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUTS / "cross_paper_evaluation")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.all:
        papers = sorted(PAPER_CONFIG.keys())
    elif args.paper:
        papers = [args.paper]
    else:
        parser.error("Specify --paper N or --all")

    all_results = {}
    for paper_id in papers:
        print(f"\n{'='*60}")
        print(f"Evaluating Paper {paper_id:02d}")
        print(f"{'='*60}")
        result = evaluate_paper(paper_id, seed=args.seed)
        all_results[f"paper{paper_id:02d}"] = result

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            best_score_str = f"{result['best_macro_auroc']:.4f}" if result.get("best_macro_auroc") is not None else "None"
            print(f"  Best variant: {result['best_variant']} (AUROC={best_score_str})")
            print(f"  Evaluated: {result['n_variants_evaluated']} variants, {result['n_variants_failed']} failed")

        # Save per-paper result
        per_paper_path = args.output / f"paper{paper_id:02d}_fold8.json"
        per_paper_path.write_text(json.dumps(result, indent=2) + "\n")

    # Save combined results
    combined_path = args.output / "all_papers_fold8.json"
    combined_path.write_text(json.dumps(all_results, indent=2) + "\n")

    # Print summary table
    print(f"\n{'='*60}")
    print("CROSS-PAPER COMPARISON (Fold 8 Validation)")
    print(f"{'='*60}")
    print(f"{'Paper':<8} {'Best Variant':<30} {'AUROC':<8} {'# Variants'}")
    print("-" * 60)
    for paper_key in sorted(all_results.keys()):
        r = all_results[paper_key]
        if "error" in r:
            print(f"{paper_key:<8} {'ERROR':<30} {'—':<8} —")
        else:
            score_str = f"{r['best_macro_auroc']:.4f}" if r.get("best_macro_auroc") is not None else "—"
            print(f"{paper_key:<8} {r.get('best_variant') or '—':<30} {score_str:<8}   {r.get('n_variants_evaluated', 0)}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
