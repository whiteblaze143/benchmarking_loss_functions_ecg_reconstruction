#!/usr/bin/env python3
"""Stage 2B: frozen one- versus two-layer SetTransformer audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from run_stagewise_linear_probe_audit import CONFIGURATIONS, LEADS, _load_checkpoint, _macro_auc, _patient_split


STAGES = ("E3_combined_token", "E4a_layer1", "E4b_layer2", "E5_attention_pooled", "E6_latent")


def _features(model, operators, responses):
    batch, count = operators.shape[:2]
    q = model.encode_operator(operators, None)
    response = model.response(responses.reshape(batch * count, 16, 128)).reshape(batch, count, 128)
    token = torch.cat((q, response), dim=-1)
    layer1 = model.set_encoder.layers[0](token)
    layer2 = model.set_encoder.layers[1](layer1)
    query = model.pool_query.expand(batch, -1, -1)
    pooled, _ = model.pool(query, layer2, layer2)
    latent = model.latent(pooled[:, 0])
    return {"E3_combined_token": token.mean(dim=1), "E4a_layer1": layer1.mean(dim=1), "E4b_layer2": layer2.mean(dim=1), "E5_attention_pooled": pooled[:, 0], "E6_latent": latent}


def _extract(model, response_array, ops, device, batch):
    chunks = {stage: [] for stage in STAGES}
    with torch.inference_mode():
        for start in range(0, len(response_array), batch):
            stop = min(start + batch, len(response_array))
            r = torch.as_tensor(np.asarray(response_array[start:stop]), device=device)
            q = torch.as_tensor(ops, device=device).expand(stop - start, -1, -1)
            values = _features(model, q, r)
            for stage, value in values.items(): chunks[stage].append(value.float().cpu().numpy())
    return {stage: np.concatenate(values).astype(np.float32) for stage, values in chunks.items()}


def _fit(train_x, train_y, select_x, select_y):
    choices = []
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0):
        scale = StandardScaler().fit(train_x)
        probe = Ridge(alpha=alpha).fit(scale.transform(train_x), train_y)
        choices.append((alpha, _macro_auc(select_y, probe.predict(scale.transform(select_x)))))
    alpha = max(choices, key=lambda x: (x[1], -x[0]))[0]
    scale = StandardScaler().fit(np.concatenate((train_x, select_x)))
    probe = Ridge(alpha=alpha).fit(scale.transform(np.concatenate((train_x, select_x))), np.concatenate((train_y, select_y)))
    return scale, probe, float(alpha), [{"alpha": float(a), "selection_macro_auroc": float(s)} for a, s in choices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError(args.output)
    manifest = json.loads((args.contexts / "manifest.json").read_text())
    if manifest.get("status") != "complete": raise ValueError("incomplete contexts")
    torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = {key: np.load(args.contexts / f"train_{key}.npy", mmap_mode="r") for key in ("responses", "labels", "patient_ids")}
    validation = {key: np.load(args.contexts / f"validation_{key}.npy", mmap_mode="r") for key in ("responses", "labels", "patient_ids")}
    operators = np.load(args.contexts / "operators.npy", allow_pickle=False)
    fit_mask, select_mask = _patient_split(train["patient_ids"], args.seed)
    model, metadata = _load_checkpoint(args.checkpoint); model = model.to(device).eval()
    q8_indices = list(CONFIGURATIONS["Q8"])
    train_features = _extract(model, train["responses"][:, q8_indices], operators[q8_indices], device, args.batch)
    probes = {}; result = {"schema_version": "p07_transformer_layer_audit_v1", "status": "complete", "checkpoint": metadata, "protocol": {"probe_fit": "correct Q8 only, folds 1-7", "selection": "20% patient-disjoint", "test_fold": 8, "interventions": ["correct", "zero", "wrong"], "stages": list(STAGES)}, "results": {}}
    for stage in STAGES:
        scale, probe, alpha, candidates = _fit(train_features[stage][fit_mask], train["labels"][fit_mask], train_features[stage][select_mask], train["labels"][select_mask]); probes[stage] = (scale, probe); result["results"][stage] = {"dimension": int(train_features[stage].shape[1]), "selected_alpha": alpha, "selection_candidates": candidates, "fold8": {}}
    predictions = {"labels": validation["labels"], "patient_ids": validation["patient_ids"]}
    for configuration, indices in CONFIGURATIONS.items():
        base = operators[list(indices)]; response = validation["responses"][:, list(indices)]
        for intervention, ops in (("correct", base), ("zero", np.zeros_like(base)), ("wrong", np.roll(base, 1, axis=0))):
            extracted = _extract(model, response, ops, device, args.batch)
            for stage in STAGES:
                scale, probe = probes[stage]; scores = probe.predict(scale.transform(extracted[stage])).astype(np.float32)
                predictions[f"{configuration}__{intervention}__{stage}"] = scores
                result["results"][stage]["fold8"][f"{configuration}__{intervention}"] = {"macro_auroc": _macro_auc(validation["labels"], scores)}
    args.output.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.output / "fold8_predictions.npz", **predictions); (args.output / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); print(json.dumps({"status": "complete", "output": str(args.output)}))


if __name__ == "__main__": main()
