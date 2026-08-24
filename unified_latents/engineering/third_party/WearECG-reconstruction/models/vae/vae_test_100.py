import sys
import numpy as np
import torch
import argparse
import logging
import os

from tqdm import tqdm

project_root_dir = "/data/0shared/guanxinyan/reconstruction"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from models.vae.vae_model_100 import VAE_Decoder, VAE_Encoder
from utils.io_utils import load_yaml_config
from data.build_dataloader import build_dataloader_sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/MIMIC/mimic_cond.yaml",
        help="path to config file",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./results/vae_100/",
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
    indices = []
    with torch.no_grad():
        offset = 0
        for batch in tqdm(dataloader):
            if isinstance(batch, (list, tuple)):
                X = batch[0]
            else:
                X = batch
            X = X.to(device)
            z, mean, log_var = encoder(X)
            recons = decoder(z)
            samples.append(recons.cpu())
            batch_size = X.shape[0]
            indices.extend(range(offset, offset + batch_size))
            offset += batch_size
    return torch.cat(samples, dim=0), indices


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
    vae_path = "results/vae_100/MIMIC/checkpoints0/VAE-10.pth"
    checkpoint = torch.load(vae_path, map_location=device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    logger.info(f"Loaded model from {vae_path}")

    # Generate samples
    os.makedirs(args.save_dir, exist_ok=True)

    samples, indices = generate_samples(encoder, decoder, dataloader, device)
    samples = np.array(samples)

    # Save samples
    save_path = os.path.join(args.save_dir, "samples", "overall_fake_data0.npy")
    np.save(save_path, samples)
    logger.info(f"Saved samples to {save_path}")

    # Save meta info for generated samples
    meta_info = dataloader_info["dataset"].meta_info
    meta_for_samples = [meta_info[i] for i in indices]
    meta_save_path = save_path.replace('.npy', '_meta.json')
    import json
    with open(meta_save_path, 'w') as f:
        json.dump(meta_for_samples, f, ensure_ascii=False, indent=4)
    logger.info(f"Saved meta info to {meta_save_path}")

    # Save subset samples，比如前500个
    
    # subset_samples = samples[:500]
    # save_path_subset = os.path.join(args.save_dir, "samples", "subset_fake_data_500.npy")
    # np.save(save_path_subset, subset_samples)
    # logger.info(f"Saved subset samples (500) to {save_path_subset}")