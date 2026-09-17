import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .configurations import STANDARD_CONFIGS, ECGConfiguration
from .missingness import apply_missingness
from repecg.common.metrics import multilabel_metrics

def run_universal_battery(
    model: torch.nn.Module,
    variant: str,
    training_regime: str,
    configuration_name: str,
    missing_mode: str,
    output_dir: Path,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    ecg_ids: np.ndarray,
    patient_ids: np.ndarray,
):
    """
    Executes the Universal Evaluation Protocol on a given model.
    """
    config = STANDARD_CONFIGS.get(configuration_name)
    if not config:
        config = ECGConfiguration(name=configuration_name, missing_mode=missing_mode)
    
    # We will expand the full suite (E0-E15) here in future steps.
    # For now, it runs the specified configuration and outputs to CSV.
    
    torch.backends.cudnn.enabled = False
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # 1. Apply Missingness (Z0, ZM, Native)
        lead_names = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
        
        # Determine operator depending on model (operator model vs standard)
        # For standard models, the input is `val_x`.
        # Note: val_x typically represents a distance operator or phase features.
        
        # Stub execution: just forward the model
        if val_x.ndim == 3 and val_x.shape[2] == 12:
            x_mod, mask = apply_missingness(val_x, config, lead_names)
            # if ZM, concat mask
            if missing_mode == "ZM":
                x_mod = torch.cat([x_mod, mask], dim=-1)
            logits = model(x_mod)
        else:
            # If the input isn't directly 12 leads (e.g. operators), just pass it
            logits = model(val_x)
            
        probs = torch.sigmoid(logits).float().cpu().numpy()
        
    metrics = multilabel_metrics(val_y, probs)
    
    # Save the result
    out_file = output_dir / f"eval_{configuration_name}_{missing_mode}.csv"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([metrics])
    df["variant"] = variant
    df["training_regime"] = training_regime
    df["configuration"] = configuration_name
    df["missing_mode"] = missing_mode
    
    df.to_csv(out_file, index=False)
    print(f"Evaluated {configuration_name} (Mode: {missing_mode}) -> {out_file}")
