#!/usr/bin/env python3
"""Fail-closed launcher for the matched three-arm A0 x frozen-NEF screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_1lead_wavelet_ssl_mtl import build_model, forward_model
from unified_latents.engineering.experimental.nef_latent_graft import FrozenNEFPreDecoderGraft

A0_CONFIG = ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0/config.json"
A0_CHECKPOINT = ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0/best.pt"
AEVE = Path("/data/mithunmanivannan/nef-net-aim/02_nef_core/pretrained/nefcore_ae_ve.pt")
FULL = Path("/data/mithunmanivannan/nef-net-aim/02_nef_core/pretrained/nefcore_full.pt")
EXPECTED = {
    A0_CHECKPOINT: "fdf6e9977ac77631725ea070b2087ef08024a707dd09dfcbaf792e82827a5acf",
    AEVE: "affdea859c2a3ce90b4970c84999844880f8ae4b3e9ee87f1029c79362de6b38",
    FULL: "d5d9197e2f3d58a711e2b8ad198f8766650591d8cdf4963d6a532127d31ef054",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def verify_inputs() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"frozen artifact mismatch: {path}")
    if not (ROOT / "data/ptb_xl/tensors/train").is_dir():
        raise FileNotFoundError("PTB-XL folds 1-8 tensor directory is missing")
    if not (ROOT / "data/ptb_xl/tensors/val").is_dir():
        raise FileNotFoundError("PTB-XL fold-9 tensor directory is missing")
    if not (ROOT / "data/ptb_xl/tensors/test").is_dir():
        raise FileNotFoundError("sealed PTB-XL fold-10 tensor directory is missing")


def identity_gate() -> dict:
    config = argparse.Namespace(**json.loads(A0_CONFIG.read_text()))
    parent = build_model(config)
    payload = torch.load(A0_CHECKPOINT, map_location="cpu", weights_only=False)
    parent.load_state_dict(payload["model_state_dict"], strict=True)
    parent.eval()
    records = sorted((ROOT / "data/ptb_xl/tensors/val").glob("*.pt"), key=lambda p: int(p.stem))[:2]
    if len(records) != 2:
        raise RuntimeError("identity gate requires two real fold-9 records")
    results = {}
    with torch.inference_mode():
        waveforms = torch.stack([torch.load(p, map_location="cpu", weights_only=True) for p in records])
        baseline = forward_model(parent, waveforms, [0], compute_delineation=True, compute_ssl=False)
        for mode, checkpoint in (("AE_VE", AEVE), ("FULL", FULL)):
            grafted = copy.deepcopy(parent)
            grafted.pre_decoder_adapter = FrozenNEFPreDecoderGraft(
                mode, checkpoint, grafted.width, grafted.num_patches
            )
            grafted.eval()
            observed = forward_model(grafted, waveforms, [0], compute_delineation=True, compute_ssl=False)
            errors = {
                "reconstruction_max_abs": float((baseline["y_pred"] - observed["y_pred"]).abs().max()),
                "segmentation_max_abs": float((baseline["seg_logits"] - observed["seg_logits"]).abs().max()),
                "post_decoder_grid_max_abs": float((baseline["grid_features"] - observed["grid_features"]).abs().max()),
                "pre_decoder_injection_max_abs": grafted.pre_decoder_adapter.last_identity_grid_error,
            }
            if any(value >= 1e-7 for value in errors.values()):
                raise RuntimeError(f"alpha=0 identity gate failed for {mode}: {errors}")
            results[mode] = errors
    return {"records": [str(p) for p in records], "threshold": 1e-7, "results": results}


def command(cell: str, mode: str, checkpoint: Path | None, output_root: Path) -> list[str]:
    cfg = json.loads(A0_CONFIG.read_text())
    args = [
        sys.executable, str(ROOT / "scripts/train_1lead_wavelet_ssl_mtl.py"),
        "--run-name", cell, "--output-dir", str(output_root / cell),
        "--data-dir", str(ROOT / "data/ptb_xl/tensors"),
        "--data-manifest", str(ROOT / "refine-logs/ptbxl_tensor_content_manifest.json"),
        "--delineation-dir", str(ROOT / cfg["delineation_dir"]),
        "--init-checkpoint", str(A0_CHECKPOINT), "--init-strict",
        "--epochs", "3", "--seed", str(cfg["seed"]),
        "--batch-size", str(cfg["batch_size"]),
        "--delineation-batch-size", str(cfg["delineation_batch_size"]),
        "--num-workers", str(cfg["num_workers"]),
        "--delineation-every", str(cfg["delineation_every"]),
        "--factorial-mask", cfg["factorial_mask"], "--observed-leads", "0",
        "--patch-size", str(cfg["patch_size"]), "--width", str(cfg["width"]),
        "--encoder-depth", str(cfg["encoder_depth"]),
        "--decoder-depth", str(cfg["decoder_depth"]), "--heads", str(cfg["heads"]),
        "--random-mask-ratio", str(cfg["random_mask_ratio"]),
        "--temporal-mask-ratio", str(cfg["temporal_mask_ratio"]),
        "--consistency-weight", str(cfg["consistency_weight"]),
        "--lead-conditioning-mode", cfg["lead_conditioning_mode"],
        "--spatial-gain-init", str(cfg["spatial_gain_init"]),
        "--geometry-control", cfg["geometry_control"], "--no-use-wavelet-branch",
        "--ssl-mode", cfg["ssl_mode"], "--ssl-weight", str(cfg["ssl_weight"]),
        "--delineation-hidden", str(cfg["delineation_hidden"]),
        "--delineation-kernel", str(cfg["delineation_kernel"]),
        "--seg-ce-weight", str(cfg["seg_ce_weight"]), "--dice-weight", str(cfg["dice_weight"]),
        "--boundary-weight", str(cfg["boundary_weight"]),
        "--fiducial-weight", str(cfg["fiducial_weight"]), "--no-fiducial-head",
        "--mask-type-mode", cfg["mask_type_mode"],
        "--lr", str(cfg["lr"]), "--max-lr", str(cfg["max_lr"]),
        "--pct-start", str(cfg["pct_start"]), "--weight-decay", str(cfg["weight_decay"]),
        "--grad-clip", str(cfg["grad_clip"]),
        "--reconstruction-weight", str(cfg["reconstruction_weight"]),
        "--checkpoint-policy", "all", "--rolling-resume", "--save-every-epoch",
        "--require-cuda",
    ]
    if mode != "none":
        args += ["--nef-core-mode", mode, "--nef-core-checkpoint", str(checkpoint)]
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "refine-logs/a0_nef_hybrid_stage1")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument(
        "--exec-cell",
        choices=[
            "H0_A0_BASE_CONTINUED",
            "H1_A0_PLUS_FROZEN_AEVE_E15",
            "H2_A0_PLUS_FROZEN_FULL_E13",
        ],
        help="Replace this process with one training cell; avoids retaining orchestrator RAM.",
    )
    args = parser.parse_args()
    verify_inputs()
    cells = (
        ("H0_A0_BASE_CONTINUED", "none", None),
        ("H1_A0_PLUS_FROZEN_AEVE_E15", "AE_VE", AEVE),
        ("H2_A0_PLUS_FROZEN_FULL_E13", "FULL", FULL),
    )
    if args.exec_cell:
        selected = next(row for row in cells if row[0] == args.exec_cell)
        cmd = command(*selected, args.output_root)
        print("EXEC_CELL " + json.dumps(cmd), flush=True)
        os.execv(cmd[0], cmd)
    gate = identity_gate()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "identity_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(gate, sort_keys=True), flush=True)
    if args.identity_only:
        return
    for cell, mode, checkpoint in cells:
        cmd = command(cell, mode, checkpoint, args.output_root)
        print("LAUNCH_CELL " + json.dumps(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
