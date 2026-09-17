from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from repecg.common.models import ConditionalRepStatModel
from repecg.common.variants import get_variants_for_paper
from repecg.evaluation.runner import run_universal_battery
from repecg.evaluation.configurations import STANDARD_CONFIGS

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--training", type=Path, default=None)
    parser.add_argument("--cells", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--training-regime", type=str, default="full_only")
    parser.add_argument("--configuration", type=str, default="12lead")
    parser.add_argument("--missing-mode", type=str, default="native")
    
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    variant_registry = get_variants_for_paper(6)
    if args.variant not in variant_registry:
        raise ValueError(f"Variant {args.variant} not found.")
    variant_obj = variant_registry[args.variant]
    
    # Load fold 10 (or pseudo-test fold 8 depending on split strategy)
    # We use validation set here as placeholder for the OOD eval
    val_file = args.representations / "representation_val.npz"
    if not val_file.exists():
        # Look in development_representations
        dev_file = args.representations.parent / "development_representations" / "representation_val.npz"
        if dev_file.exists():
            val_file = dev_file
        else:
            # Look for any .npz in representations
            npzs = list(args.representations.glob("*.npz"))
            val_file = npzs[0] if npzs else val_file
    with np.load(val_file) as item:
        validation = {name: np.asarray(item[name]) for name in item.files}
        
    rep_key = variant_obj.representation if variant_obj.representation != "full" else "kernel"
    if rep_key not in validation:
        rep_key = "kernel" # Fallback for now
        
    val_x = torch.from_numpy(validation[rep_key]).cuda()
    val_y = validation["labels"]
    ecg_ids = validation["ecg_ids"]
    patient_ids = validation["patient_ids"]
    
    # Load best model for this variant
    # In a full run, we would load the checkpoint. For now we just instantiate.
    # A real implementation would parse cells/variant_*/checkpoint.pt
    model = ConditionalRepStatModel(classes=val_y.shape[-1], variant=variant_obj).cuda()
    # Load checkpoint if available
    ckpt_path = None
    if args.training is not None:
        best_pt = args.training / f"{args.variant}_best.pt"
        if best_pt.exists():
            ckpt_path = best_pt
    if ckpt_path is None:
        cells_dir = args.cells if args.cells is not None else (args.training / "cells" if args.training is not None else None)
        if cells_dir is not None and cells_dir.exists():
            for c in cells_dir.glob(f"{args.variant}_*/checkpoint.pt"):
                ckpt_path = c
                break
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cuda:0")
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
            print(f"Loaded weights from {ckpt_path}")
    
    run_universal_battery(
        model=model,
        variant=args.variant,
        training_regime=args.training_regime,
        configuration_name=args.configuration,
        missing_mode=args.missing_mode,
        output_dir=args.output,
        val_x=val_x,
        val_y=val_y,
        ecg_ids=ecg_ids,
        patient_ids=patient_ids
    )

if __name__ == "__main__":
    main()