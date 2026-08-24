#!/usr/bin/env python3
"""Deterministic one-record overfit probe for ECG AliTok/AIM adapters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.bootstrap_paths import setup_import_paths

setup_import_paths()

from unified_latents.engineering.experimental.alitok_vae_exp import build_alitok_vae_1d
from unified_latents.engineering.utils.common import TensorFolderDataset, mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/ptb_xl/tensors/train")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--architecture",
        choices=["ecg_aim_v1", "stage1_causal"],
        default="ecg_aim_v1",
    )
    parser.add_argument("--clustering-vq", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def metrics(pred: torch.Tensor, target: torch.Tensor, missing: list[int]) -> dict[str, float]:
    pred = pred[:, missing].float()
    target = target[:, missing].float()
    mse = torch.mean((pred - target) ** 2)
    pred_centered = pred - pred.mean(dim=-1, keepdim=True)
    target_centered = target - target.mean(dim=-1, keepdim=True)
    corr = (
        (pred_centered * target_centered).sum(dim=-1)
        / (
            pred_centered.square().sum(dim=-1).sqrt()
            * target_centered.square().sum(dim=-1).sqrt()
        ).clamp_min(1e-8)
    ).mean()
    return {"mse": float(mse), "corr": float(corr)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(1337)
    device = torch.device("cuda")
    target = TensorFolderDataset(args.data_dir)[0][0].unsqueeze(0).to(device)
    observed = [0, 1, 7]
    missing = [idx for idx in range(12) if idx not in observed]
    masked = mask_unobserved_leads(target, observed)
    lead_indices = make_lead_indices(observed, 1, device)
    aim = args.architecture == "ecg_aim_v1"
    model = build_alitok_vae_1d(
        architecture=args.architecture,
        patch_size=25 if aim else 10,
        encoder_width=256 if aim else 768,
        decoder_width=256 if aim else 1024,
        encoder_depth=6 if aim else 12,
        decoder_depth=4 if aim else 24,
        encoder_heads=8 if aim else 12,
        decoder_heads=8 if aim else 16,
        clustering_vq=args.clustering_vq,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.eval()
    with torch.inference_mode():
        initial_out = model(masked, target=target, lead_indices=lead_indices)
        initial = metrics(initial_out["y_pred"], target, missing)
        initial["perplexity"] = float(initial_out["codebook_perplexity"])

    losses: list[float] = []
    model.train()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(masked, target=target, lead_indices=lead_indices)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(out["loss"]))

    model.eval()
    with torch.inference_mode():
        final_out = model(masked, target=target, lead_indices=lead_indices)
        final = metrics(final_out["y_pred"], target, missing)
        final["perplexity"] = float(final_out["codebook_perplexity"])

    print(
        json.dumps(
            {
                "architecture": args.architecture,
                "clustering_vq": args.clustering_vq,
                "lr": args.lr,
                "steps": args.steps,
                "initial": initial,
                "final": final,
                "first_loss": losses[0],
                "last_loss": losses[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
