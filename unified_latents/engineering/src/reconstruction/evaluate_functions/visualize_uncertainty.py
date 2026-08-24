"""Visualize uncertainty predictions for heteroscedastic model."""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.dataset import ConditionalECGDataset, get_dataloaders, load_metadata, split_patients
from src.models.builder import build_model_from_config, load_config as load_model_config
from src.models.utils import set_global_seed

def load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model = build_model_from_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def plot_sample(
    t: np.ndarray,
    target: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    lead_name: str,
    save_path: Path
):
    plt.figure(figsize=(12, 4))
    plt.plot(t, target, 'k-', label='Ground Truth', linewidth=1.5, alpha=0.7)
    plt.plot(t, mu, 'b-', label='Prediction', linewidth=1.5)
    plt.fill_between(t, mu - 2*sigma, mu + 2*sigma, color='b', alpha=0.2, label='Uncertainty (2$\sigma$)')
    plt.title(f"Reconstruction with Uncertainty - {lead_name}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude (mV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--features-csv", default="sunnybrook_features.csv")
    parser.add_argument("--reconstructed-dir", default="data/reconstructed")
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    config = load_model_config(Path(args.config))
    metadata_df = load_metadata(Path(args.features_csv))
    splits = split_patients(metadata_df, test_ratio=0.2, seed=42)
    
    val_dataset = ConditionalECGDataset(
        Path(args.reconstructed_dir),
        metadata_df,
        "val",
        splits,
    )
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Load Model
    model = load_model(Path(args.config), Path(args.checkpoint), device)

    # Inference
    count = 0
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    # Adjust lead names based on output channels (usually 9: aVF, aVR, aVL, V1-V6)
    # Dataset default target leads are 3:12 (9 leads)
    target_lead_names = ["aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if count >= args.num_samples:
                break
            
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)
            metadata = {k: v.to(device) for k, v in batch["metadata"].items()}
            
            outputs = model(inputs, metadata)
            
            if isinstance(outputs, tuple):
                mu, logvar = outputs
                sigma = torch.exp(0.5 * logvar)
                
                mu_np = mu.cpu().numpy()[0]
                sigma_np = sigma.cpu().numpy()[0]
                target_np = targets.cpu().numpy()[0]
                
                # Plot each lead
                sample_dir = output_dir / f"sample_{i}"
                sample_dir.mkdir(exist_ok=True)
                
                t = np.arange(mu_np.shape[1])
                
                for lead_idx in range(mu_np.shape[0]):
                    lead_name = target_lead_names[lead_idx] if lead_idx < len(target_lead_names) else f"Lead_{lead_idx}"
                    plot_sample(
                        t, 
                        target_np[lead_idx], 
                        mu_np[lead_idx], 
                        sigma_np[lead_idx], 
                        lead_name, 
                        sample_dir / f"{lead_name}.png"
                    )
                
                count += 1
            else:
                print("Model does not output uncertainty.")
                break

if __name__ == "__main__":
    main()
