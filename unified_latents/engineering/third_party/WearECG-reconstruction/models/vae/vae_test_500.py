import sys
import numpy as np
import torch
import argparse
import logging
import os

from tqdm import tqdm

project_root_dir = "/wujidata/xdl/run/ECG360"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from models.vae.vae_model_500 import VAE_Decoder, VAE_Encoder
from utils.io_utils import load_yaml_config
from data.build_dataloader import build_dataloader_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/PTBXL/ptbxl_cond_500.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/vae_500/",
        help="directory to save checkpoints",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="synthesis",
        help="Infilling, Forecasting or Synthesis.",
    )
    parser.add_argument(
        "--synthesis_channels",
        type=lambda x: list(map(int, x.split(","))),
        default=list(range(1, 12)),
        help="List of synthesis channels (default is [1, 2, 3, ..., 11]).",
    )
    return parser.parse_args()


def generate_samples(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()

    samples = []
    with torch.no_grad():
        for (X, _) in tqdm(dataloader):
            X = X.to(device)
            z, mean, log_var = encoder(X)
            recons = decoder(z)
            samples.append(recons.cpu())

    return torch.cat(samples, dim=0)


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    args.save_dir = os.path.join(
        args.save_dir, config["dataloader"]["train_dataset"]["params"]["name"]
    )

    # Setup logging
    logger = logging.getLogger("vae_test")
    logger.setLevel("INFO")
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # Build dataloader
    dataloader_info = build_dataloader_sample(config, args)
    dataloader = dataloader_info["dataloader"]
    # Load models
    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    # Load checkpoint
    vae_path = "results/vae_500/PTBXL/checkpoints/VAE-10.pth"
    checkpoint = torch.load(vae_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    logger.info(f"Loaded model from {vae_path}")

    # Generate samples
    os.makedirs(args.save_dir, exist_ok=True)
    samples = generate_samples(encoder, decoder, dataloader, device)
    samples = np.array(samples)

    # Save samples
    save_path = os.path.join(args.save_dir, "samples", "overall_fake_data.npy")
    np.save(save_path, samples)
    logger.info(f"Saved samples to {save_path}")
