#!/usr/bin/env python3
"""Fail-closed launcher for native, jointly trainable NEF components in A0."""

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

from scripts.train_1lead_wavelet_ssl_mtl import build_model, forward_model, seed_all
from unified_latents.engineering.experimental.native_nef_ecg_aim import NativeNEFECGAIM
from unified_latents.engineering.utils.common import mask_unobserved_leads
from unified_latents.engineering.utils.regimes import make_lead_indices

A0_CONFIG = ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0/config.json"
A0_CHECKPOINT = ROOT / "refine-logs/convergence_10e/runs/conv15e_A0_raw_s42_l0/best.pt"
FULL = Path("/data/mithunmanivannan/nef-net-aim/02_nef_core/pretrained/nefcore_full.pt")
DEFAULT_OUTPUT = Path("/data/mithunmanivannan/nef-net-aim/08_runs/a0_nef_native_stage1")
EXPECTED = {
    A0_CHECKPOINT: "fdf6e9977ac77631725ea070b2087ef08024a707dd09dfcbaf792e82827a5acf",
    FULL: "d5d9197e2f3d58a711e2b8ad198f8766650591d8cdf4963d6a532127d31ef054",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def tensor_digest(module: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value.update(name.encode())
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def verify_inputs() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"frozen artifact mismatch: {path}")
    for split in ("train", "val", "test"):
        if not (ROOT / "data/ptb_xl/tensors" / split).is_dir():
            raise FileNotFoundError(f"PTB-XL {split} tensor directory is missing")


def load_parent() -> torch.nn.Module:
    config = argparse.Namespace(**json.loads(A0_CONFIG.read_text()))
    parent = build_model(config)
    payload = torch.load(A0_CHECKPOINT, map_location="cpu", weights_only=False)
    parent.load_state_dict(payload["model_state_dict"], strict=True)
    return parent


def attach(parent: torch.nn.Module, mode: str) -> torch.nn.Module:
    model = copy.deepcopy(parent)
    seed_all(42)
    model.native_nef_adapter = NativeNEFECGAIM(
        mode, FULL, model.width, model.num_patches, fusion_init=0.1
    )
    return model


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach() - b.detach()).abs().max())


def preflight() -> dict:
    verify_inputs()
    parent = load_parent().eval()
    c1 = attach(parent, "AE_VE").eval()
    c2 = attach(parent, "FULL").eval()

    shared = {}
    for name in ("angle_embed", "view_encoder", "angle_projection", "view_projection"):
        left = tensor_digest(getattr(c1.native_nef_adapter, name))
        right = tensor_digest(getattr(c2.native_nef_adapter, name))
        shared[name] = {"c1": left, "c2": right, "exact": left == right}
    if not all(row["exact"] for row in shared.values()):
        raise RuntimeError("C1/C2 shared initialization is not identical")

    records = sorted(
        (ROOT / "data/ptb_xl/tensors/val").glob("*.pt"), key=lambda p: int(p.stem)
    )[:32]
    if len(records) != 32:
        raise RuntimeError("native preflight requires 32 real fold-9 records")
    waveforms = torch.stack(
        [torch.load(path, map_location="cpu", weights_only=True) for path in records]
    )
    with torch.inference_mode():
        identity_waveforms = waveforms[:2]
        baseline = forward_model(parent, identity_waveforms, [0], compute_delineation=True, compute_ssl=False)
        identity = {}
        for name, model in (("C1", c1), ("C2", c2)):
            model.native_nef_adapter.contribution_enabled = False
            observed = forward_model(model, identity_waveforms, [0], compute_delineation=True, compute_ssl=False)
            errors = {
                "reconstruction_max_abs": _max_abs(baseline["y_pred"], observed["y_pred"]),
                "segmentation_max_abs": _max_abs(baseline["seg_logits"], observed["seg_logits"]),
                "grid_max_abs": _max_abs(baseline["grid_features"], observed["grid_features"]),
            }
            if any(value >= 1e-6 for value in errors.values()):
                raise RuntimeError(f"disabled native identity gate failed for {name}: {errors}")
            identity[name] = errors
            model.native_nef_adapter.contribution_enabled = True

    device = torch.device("cuda")
    # Match the production trainer's required A100/cuDNN-9 execution contract.
    torch.backends.cudnn.enabled = False
    gradient_checks = {}
    for name, model in (("C1", c1), ("C2", c2)):
        model = model.to(device).train()
        sample = waveforms.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = forward_model(model, sample, [0], compute_delineation=True, compute_ssl=False)
        if tuple(output["y_pred"].shape) != (32, 12, 5000):
            raise RuntimeError(f"{name} output shape mismatch: {tuple(output['y_pred'].shape)}")
        loss = output["y_pred"].square().mean() + output["seg_logits"].square().mean()
        loss.backward()
        groups = {
            "a0": model.patch_projection[0].weight,
            "angle": model.native_nef_adapter.angle_embed.harmonic_proj[0].weight,
            "view": model.native_nef_adapter.view_encoder.stem[0].weight,
        }
        if name == "C2":
            groups["geovt"] = model.native_nef_adapter.view_transformer.layers["layer_0"]["conv1"].weight
        norms = {
            key: 0.0 if parameter.grad is None else float(parameter.grad.detach().norm())
            for key, parameter in groups.items()
        }
        if not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in norms.values()):
            raise RuntimeError(f"{name} missing/nonfinite gradients: {norms}")

        model.eval()
        masked = mask_unobserved_leads(sample, [0]).contiguous()
        indices = make_lead_indices([0], sample.shape[0], device)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                first = model(masked, y_full=sample, lead_indices=indices, compute_delineation=False, compute_ssl=False)
            altered_target = sample.clone()
            altered_target[:, 1:12] = altered_target[:, 1:12] * 7.0 + 3.0
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                second = model(masked, y_full=altered_target, lead_indices=indices, compute_delineation=False, compute_ssl=False)
        target_independence = _max_abs(first["y_pred"], second["y_pred"])
        if target_independence != 0.0:
            raise RuntimeError(f"{name} prediction depends on hidden target: {target_independence}")
        gradient_checks[name] = {
            "gradient_norms": norms,
            "target_independence_max_abs": target_independence,
            "output_shape": list(output["y_pred"].shape),
        }
        del model, output, loss
        torch.cuda.empty_cache()

    return {
        "status": "PASS",
        "records": [str(path) for path in records],
        "identity_threshold": 1e-6,
        "identity": identity,
        "shared_initialization": shared,
        "gradient_checks": gradient_checks,
        "nef_reconstruction_head_present": False,
        "full_checkpoint_sha256": EXPECTED[FULL],
    }


def command(cell: str, mode: str, output_root: Path) -> list[str]:
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
        args += [
            "--native-nef-mode", mode,
            "--native-nef-checkpoint", str(FULL),
            "--native-nef-fusion-init", "0.1",
        ]
    return args


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--exec-cell",
        choices=["C0_A0_NATIVE_CONTROL", "C1_A0_NATIVE_AE_VE", "C2_A0_NATIVE_FULL"],
    )
    args = parser.parse_args()
    verify_inputs()
    cells = (
        ("C0_A0_NATIVE_CONTROL", "none"),
        ("C1_A0_NATIVE_AE_VE", "AE_VE"),
        ("C2_A0_NATIVE_FULL", "FULL"),
    )
    if args.exec_cell:
        cell, mode = next(row for row in cells if row[0] == args.exec_cell)
        cmd = command(cell, mode, args.output_root)
        print("EXEC_CELL " + json.dumps(cmd), flush=True)
        os.execv(cmd[0], cmd)

    report = preflight()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "native_preflight.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if args.preflight_only:
        return
    for cell, mode in cells:
        subprocess.run(command(cell, mode, args.output_root), cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
