#!/usr/bin/env python3
"""Stage 2C-1: frozen geometry audit of the continuous operator MLP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from repecg.paper07_operator import canonical_operators, derived_limb_operators, normalize_operator
from repecg.paper07_operator.model import OperatorSetModel


def _bank(seed: int, count: int, seen: np.ndarray) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    qs, families = [], []
    def add(q, family):
        q = normalize_operator(q)
        if np.min(np.linalg.norm(seen - q, axis=1)) < 1e-8 or np.min(np.linalg.norm(seen + q, axis=1)) < 1e-8:
            return
        if qs and min(np.min(np.linalg.norm(np.asarray(qs) - q, axis=1)), np.min(np.linalg.norm(np.asarray(qs) + q, axis=1))) < 1e-8:
            return
        qs.append(q); families.append(family)
    canonical = canonical_operators()
    for i, q0 in enumerate(canonical):
        for j in range(8):
            tangent = rng.normal(size=8); tangent -= tangent.dot(q0) * q0; tangent /= np.linalg.norm(tangent)
            theta = np.deg2rad((j + 1) * 2.0)
            add(np.cos(theta) * q0 + np.sin(theta) * tangent, "local_perturbation")
    for i in range(8):
        for j in range(i + 1, 8):
            for alpha in (0.2, 0.4, 0.6, 0.8): add((1 - alpha) * canonical[i] + alpha * canonical[j], "geodesic_interpolant")
    for q in derived_limb_operators(): add(q, "derived_limb")
    for q in (canonical[4] - canonical[3], canonical[7] - canonical[6]): add(q, "oblique_derived")
    while len(qs) < count:
        add(rng.normal(size=8), "heldout_oblique")
    return np.asarray(qs[:count], dtype=np.float64), families[:count]


def _encode(model, q: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.inference_mode(): return model.encode_operator(torch.as_tensor(q, dtype=torch.float32, device=device), None).float().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-banks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = OperatorSetModel(response_dim=128, classes=5, operator_mode="continuous"); model.load_state_dict(payload["state_dict"], strict=True); model = model.to(device).eval()
    with np.load(args.operator_banks, allow_pickle=False) as item: seen = np.asarray(item["seen"], dtype=np.float64)
    q, families = _bank(args.seed, args.count, seen)
    e = _encode(model, q, device)
    d_q = squareform(pdist(q, metric="cosine")); d_q = np.arccos(np.clip(1.0 - d_q, -1.0, 1.0))
    d_e = squareform(pdist(e, metric="euclidean"))
    tri = np.triu_indices(len(q), 1)
    rho = float(spearmanr(d_q[tri], d_e[tri]).statistic)
    neighbors = {}
    for k in (5, 10, 20):
        overlaps = []
        for i in range(len(q)):
            nq = set(np.argsort(d_q[i])[1:k + 1]); ne = set(np.argsort(d_e[i])[1:k + 1]); overlaps.append(len(nq & ne) / max(1, len(nq | ne)))
        neighbors[str(k)] = {"mean_jaccard": float(np.mean(overlaps)), "median_jaccard": float(np.median(overlaps))}
    ratios = d_e[tri] / np.maximum(d_q[tri], 1e-12)
    # Exact local Jacobian spectrum on the tangent space of the unit sphere.
    singular = []
    q_tensor = torch.as_tensor(q, dtype=torch.float32, device=device)
    for row in q_tensor:
        row = row.detach().requires_grad_(True)
        jac = torch.autograd.functional.jacobian(lambda x: model.encode_operator(x[None], None)[0], row, create_graph=False).detach().cpu().numpy()
        tangent = np.linalg.svd(np.eye(8) - np.outer(row.detach().cpu().numpy(), row.detach().cpu().numpy()), full_matrices=False)[0][:, :7]
        values = np.linalg.svd(jac @ tangent, compute_uv=False)[:7]
        singular.append(values)
    singular = np.asarray(singular)
    condition = singular[:, 0] / np.maximum(singular[:, -1], 1e-12)
    result = {
        "schema_version": "p07_operator_geometry_audit_v1", "status": "complete", "seed": args.seed, "count": int(len(q)),
        "checkpoint": {"path": str(args.checkpoint), "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()},
        "bank": {"families": {name: families.count(name) for name in sorted(set(families))}, "seen_bank_path": str(args.operator_banks)},
        "global_spearman_cosine_vs_embedding": rho, "neighbor_jaccard": neighbors,
        "pairwise_local_distortion": {"median": float(np.median(ratios)), "p05": float(np.quantile(ratios, .05)), "p95": float(np.quantile(ratios, .95)), "min": float(np.min(ratios)), "max": float(np.max(ratios))},
        "jacobian_tangent_singular_values": {"sigma_min_median": float(np.median(singular[:, -1])), "sigma_max_median": float(np.median(singular[:, 0])), "condition_median": float(np.median(condition)), "condition_p95": float(np.quantile(condition, .95))},
    }
    args.output.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output / "operator_bank.npz", q=q, embedding=e, family=np.asarray(families)); (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__": main()
