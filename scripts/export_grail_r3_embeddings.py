#!/usr/bin/env python3
"""Export leakage-safe frozen GRAIL representations for the R3 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch.backends.cudnn.enabled = False

from grail_ecg.src.data.ptbxl_dataset import PTBXLDataset
from grail_ecg.src.geometry.lead_geometry import INDEPENDENT_8_LEADS, get_angles_tensor
from grail_ecg.src.models.baselines import FactorialGRAILEncoder, SupervisedUpperBoundUB


MODEL_CONFIGS = {
    "ub": None,
    "b0": (False, False, False),
    "b1": (False, False, False),
    "model_001": (False, False, True),
    "model_101": (True, False, True),
    "model_m": (True, True, True),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(name: str, concept: dict, device: torch.device):
    ckpt_path = ROOT / "checkpoints" / "grail_v2" / f"{name}_best.pt"
    if name == "ub":
        model = SupervisedUpperBoundUB(num_leads=8, hidden_dim=128, num_classes=25)
    else:
        g, s, v = MODEL_CONFIGS[name]
        model = FactorialGRAILEncoder(
            use_geometry=g, use_slots=s, use_view_aux=v, num_leads=8,
            num_tokens_per_lead=32, hidden_dim=128, num_slots=6, slot_dim=16,
            anchor_counts_per_domain=(concept["anchor_counts_per_domain"] if s else None),
            total_anchor_classes=25,
        )
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().requires_grad_(False).to(device)
    return model, ckpt_path


@torch.inference_mode()
def extract(model, loader, device, use_geometry: bool):
    zs, slots_out = [], []
    metadata = {k: [] for k in (
        "ecg_id", "patient_id", "strat_fold", "age", "sex",
        "anchor_labels", "probe_labels",
    )}
    angles = get_angles_tensor(INDEPENDENT_8_LEADS).to(device) if use_geometry else None
    for batch in tqdm(loader, desc="frozen embedding export"):
        x = batch["ecg_8l"].to(device, non_blocking=True)
        if hasattr(model, "use_slots"):
            slots, z, _ = model(x, custom_angles=angles)
            if slots is not None:
                slots_out.append(slots.cpu().numpy())
        else:
            b, leads, length = x.shape
            h = model.temporal_encoder(x.reshape(b * leads, 1, length))
            z = h.reshape(b, leads, model.hidden_dim, 32).mean(dim=(1, 3))
        zs.append(z.cpu().numpy())
        for key in metadata:
            metadata[key].append(batch[key].numpy())
    return (
        np.concatenate(zs),
        np.concatenate(slots_out) if slots_out else None,
        {k: np.concatenate(v) for k, v in metadata.items()},
    )


def write_export(out: Path, name: str, z: np.ndarray, slots, meta: dict, concept: dict,
                 checkpoint: Path):
    model_dir = out / name
    model_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"z": z.astype(np.float32), **meta}
    if slots is not None:
        arrays["slots"] = slots.astype(np.float32)
    np.savez_compressed(model_dir / "embeddings_folds1_8.npz", **arrays)

    frame = pd.DataFrame({
        "ecg_id": meta["ecg_id"], "patient_id": meta["patient_id"],
        "strat_fold": meta["strat_fold"], "age": meta["age"], "sex": meta["sex"],
    })
    for i in range(z.shape[1]):
        frame[f"z_{i}"] = z[:, i]
    for i, code in enumerate(concept["all_anchor_codes"]):
        frame[f"anchor__{code}"] = meta["anchor_labels"][:, i].astype(np.int8)
    tier = {c: "P1" for c in concept["tier_p1_codes"]}
    tier.update({c: "P2" for c in concept["tier_p2_codes"]})
    tier.update({c: "P3" for c in concept["tier_p3_codes"]})
    for i, code in enumerate(concept["all_probe_codes"]):
        frame[f"heldout__{tier[code]}__{code}"] = meta["probe_labels"][:, i].astype(np.int8)
    if slots is not None:
        flat = slots.reshape(len(slots), -1)
        for i in range(flat.shape[1]):
            frame[f"slot_{i // 16}_{i % 16}"] = flat[:, i]
    frame.to_parquet(model_dir / "embeddings_folds1_8.parquet", index=False)

    manifest = {
        "model": name, "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint), "encoder_frozen": True,
        "fit_folds": [1, 2, 3, 4, 5, 6, 7], "evaluation_fold": 8,
        "n_ecgs": int(len(z)), "latent_dim": int(z.shape[1]),
        "slot_shape": list(slots.shape[1:]) if slots is not None else None,
        "contains_patient_id": True, "contains_labels": True,
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS))
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "results/grail_v2/r3_linear_probe_audit/embeddings")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("R3 production embedding export requires CUDA")
    concept_path = ROOT / "configs/ptbxl_concept_tiers.yaml"
    concept = yaml.safe_load(concept_path.read_text())
    ds = PTBXLDataset(ROOT / "data/ptb_xl", range(1, 9), concept_path, return_12l=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    for name in args.models:
        if name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown R3 model: {name}")
        target = args.output_dir / name / "manifest.json"
        if target.exists():
            print(f"{name}: complete export exists, skipping")
            continue
        print(f"\nR3 export {name} on {device}")
        model, checkpoint = build_model(name, concept, device)
        cfg = MODEL_CONFIGS[name]
        z, slots, meta = extract(model, loader, device, bool(cfg and cfg[0]))
        write_export(args.output_dir, name, z, slots, meta, concept, checkpoint)
        print(f"{name}: wrote {len(z)} frozen representations")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
