import sys
import torch
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


project_root_dir = "/wujidata/xdl/run/ECG360"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from models.vae.vae_model_500 import VAE_Decoder, VAE_Encoder, loss_function
from utils.io_utils import load_yaml_config, seed_everything
from data.build_dataloader import build_dataloader_train, build_dataloader_sample
from evaluation.ECGFounder import ft_ECGFounder


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
    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=True,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    return parser.parse_args()


def train_loop(
    dataloader, encoder, decoder, loss_fn, optimizer, scheduler, kld_weight=1e-4, perceptual_weight=1e-2, ecg_founder_model=None, device=None
):
    size = len(dataloader.dataset)
    encoder.train()
    decoder.train()
    if ecg_founder_model:
        ecg_founder_model.eval()

    for batch, (X, target, mask) in enumerate(dataloader):
        X, target = X.to(device), target.to(device)
        z, mean, log_var = encoder(X)
        recons = decoder(z)

        # ecg_founder_model=None  # for no perceptual loss

        loss = loss_fn(recons, target, mean, log_var, kld_weight, perceptual_weight, ecg_founder_model, device)

        loss["loss"].backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        if batch % 50 == 0:
            loss_vals, current = loss, (batch + 1) * len(X)
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"loss: {loss_vals['loss']:>7f}  recons_loss: {loss_vals['recons_loss']:>7f}  KLd_loss: {loss_vals['KLD_loss']:>7f}  perceptual_loss: {loss_vals['perceptual_loss']:>7f}  lr: {current_lr:.2e}  [{current:>5d}/{size:>5d}]"
            )


if __name__ == "__main__":
    args = parse_args()
    config = load_yaml_config(args.config)
    seed_everything(args.seed, args.cudnn_deterministic)

    args.save_dir = os.path.join(
        args.save_dir, config["dataloader"]["train_dataset"]["params"]["name"]
    )
    save_weights_path = os.path.join(
        args.save_dir,
        "checkpoints",
    )
    os.makedirs(save_weights_path, exist_ok=True)

    logger = logging.getLogger("vae")
    logger.setLevel("INFO")
    fh = logging.FileHandler(
        os.path.join(
            args.save_dir,
            "train.log",
        ),
        encoding="utf-8",
    )
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Update batch size from config
    H_ = {
        "lr": config["solver"]["vae"]["lr"],
        "batch_size": config["solver"]["vae"]["batch_size"],
        "epochs": config["solver"]["vae"]["epochs"],
        "kld_weight": config["solver"]["vae"]["kld_weight"],
        "perceptual_weight": config["solver"]["vae"]["perceptual_weight"],
    }
    logger.info(H_)

    is_save = True

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    dataloader_info = build_dataloader_train(config, args)
    train_dataloader = dataloader_info["dataloader"]

    encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)

    ecg_founder_model = None
    if H_["perceptual_weight"] > 0:
        logger.info("Initializing ECGFounder for perceptual loss...")
        ecg_founder_model = ft_ECGFounder(device=device)
        ecg_founder_model.eval()
        # 保存ECGFounder模型结构到文件
        model_structure = str(ecg_founder_model)
        structure_path = os.path.join(args.save_dir, "ecg_founder_structure.txt")
        with open(structure_path, "w", encoding="utf-8") as f:
            f.write(model_structure)
        logger.info("ECGFounder initialized.")

    loss_fn = loss_function
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=H_["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        epochs=H_["epochs"],
        steps_per_epoch=len(train_dataloader),
        pct_start=0.3
    )

    for epoch in range(H_["epochs"]):
        logger.info(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(
            train_dataloader, encoder, decoder, loss_fn, optimizer, scheduler, H_["kld_weight"], H_["perceptual_weight"], ecg_founder_model, device
        )
        if is_save and (epoch + 1) % 10 == 0:
            model_states = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            save_path = os.path.join(save_weights_path, f"VAE-{epoch+1}.pth")
            torch.save(model_states, save_path)
    logger.info("done!")
