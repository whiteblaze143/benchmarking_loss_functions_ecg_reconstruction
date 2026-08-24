"""Train the exact public WearECG VAE baseline.

This trainer is intentionally aligned to the authors' public code path:
- train `VAE_Encoder` and `VAE_Decoder` directly
- feed full ECG tensors after converting to `(B, L, 12)`
- optimize plain full-waveform MSE + KL
- use AdamW + OneCycleLR

Local additions are limited to dataset loading, checkpoint/output handling, and
optional validation through the shared evaluator.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from scripts.bootstrap_paths import setup_import_paths
setup_import_paths()

from unified_latents.engineering.utils.common import (
    TensorFolderDataset,
    cleanup_partial_checkpoints,
    mask_unobserved_leads,
    prune_epoch_checkpoints,
    write_best_summary,
    write_run_artifacts,
    write_json,
    write_warm_start_summary,
)
from unified_latents.engineering.eval.eval_reconstruction import evaluate_reconstruction
from unified_latents.engineering.utils.regimes import (
    LEAD_NAMES,
    format_lead_set,
    make_lead_indices,
    resolve_obs_leads,
)
from unified_latents.engineering.models.vae import VAE_Decoder, VAE_Encoder, WearECGVAE, loss_function


_WANDB_WARNED_PREFIXES: set[str] = set()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true")
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def safe_wandb_log(payload: dict[str, float], *, step: int | None = None, prefix: str = "") -> bool:
    if wandb.run is None:
        return False
    try:
        wandb.log(payload, step=step)
        return True
    except Exception as exc:
        warn_key = prefix or "default"
        if warn_key not in _WANDB_WARNED_PREFIXES:
            print(f"[W&B] log failed during {warn_key}; continuing with local artifacts ({exc})")
            _WANDB_WARNED_PREFIXES.add(warn_key)
        return False


def build_run_metadata(args) -> dict[str, object]:
    metadata = {
        "family": "engineering",
        "experiment_family": "engineering",
        "model_family": "wearecg_vae",
        "baseline_semantics": "wear_ecg_public_exact_modules",
        "primary_selector": "lowest_val_mse_then_lowest_val_mae_then_lowest_val_rmse",
        "regime": args.regime,
        "obs_leads": [LEAD_NAMES[idx] for idx in args.obs_lead_indices],
        "obs_lead_indices": args.obs_lead_indices,
        "num_observed_leads": len(args.obs_lead_indices),
        "lead_regime": f"{len(args.obs_lead_indices)}lead",
        "split": args.split,
        "seed": int(args.seed),
        "beta_kl": args.beta_kl,
        "latent_channels": 4,
        "has_regressor_path": True,
        "has_teacher_path": False,
        "has_diffusion_rollout": False,
        "has_direct_prior": False,
        "initialization_type": "scratch" if not args.resume else "full_warm_start",
    }
    if args.run_tag:
        metadata["run_tag"] = args.run_tag
    return metadata


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=5e-5)
    parser.add_argument("--target_len", type=int, default=5000)
    parser.add_argument("--latent_channels", type=int, default=4)
    parser.add_argument("--beta_kl", type=float, default=1e-4)
    parser.add_argument("--loss_factor", type=float, default=1.5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--regime", type=str, choices=["current", "wearecg", "historical"], default="wearecg")
    parser.add_argument("--obs_leads", type=str, default=None)
    parser.add_argument("--split", type=str, choices=["val"], default="val")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_training_state", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to reach target batch size.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_tag", type=str, default=None)
    add_bool_arg(parser, "fast_eval", False)
    args = parser.parse_args()
    args.obs_lead_indices = resolve_obs_leads(args.regime, args.obs_leads)
    return args


def get_baseline_selector(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        -metrics.get("val/mse_reg", float("inf")),
        -metrics.get("val/mae_reg", float("inf")),
        -metrics.get("val/rmse_reg", float("inf")),
    )


def build_eval_wrapper(encoder, decoder, args):
    """Robust wrapper to match full interface of evaluate_reconstruction."""
    class Wrapper(torch.nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, x_seq, **kwargs):
            # evaluate_reconstruction provides (B, 12, L)
            # Encoder wants (B, L, 12) for its internal transpose
            z, _, _ = self.encoder(x_seq.transpose(1, 2))
            recons = self.decoder(z) # (B, L, 12)
            # evaluator wants a dict for losses
            return {
                "decoder_loss": torch.tensor(0.0, device=x_seq.device),
                "kl_loss": torch.tensor(0.0, device=x_seq.device),
                "y_pred": recons # Not strictly needed if impute_from_regressor exists
            }
            
        def impute_from_regressor(self, x_masked, **kwargs):
            # x_masked is (B, 12, L)
            z, _, _ = self.encoder(x_masked.transpose(1, 2))
            y_pred_l12 = self.decoder(z) # (B, L, 12)
            return {
                "y_pred": y_pred_l12.transpose(1, 2), # (B, 12, L)
                "z_latent": z,
                "available": True
            }
            
        def impute_from_teacher(self, x_masked, **kwargs):
            # Not available in pure VAE baseline
            return None

    return Wrapper(encoder, decoder)


def train(args):
    set_global_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Baseline] Training on {device}")

    lead_tag = format_lead_set(args.obs_lead_indices)
    run_suffix = f"_{args.run_tag}" if args.run_tag else ""
    wandb_enabled = not args.debug
    if wandb_enabled:
        try:
            wandb.init(
                project="cNVAE-ECG-Scientific-Production",
                name=f"ul_engineering_wearecg_exact_{lead_tag}_bs{args.batch_size}_lf{args.loss_factor}{run_suffix}",
                config=vars(args),
            )
        except Exception as exc:
            print(f"[W&B] init failed; continuing with local artifacts ({exc})")

    base_dir = "data/ptb_xl/tensors"
    train_dataset = TensorFolderDataset(f"{base_dir}/train")
    val_dataset = TensorFolderDataset(f"{base_dir}/{args.split}")
    if args.debug:
        train_dataset = torch.utils.data.Subset(train_dataset, range(64))
        val_dataset = torch.utils.data.Subset(val_dataset, range(64))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(args.seed))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=loader_generator,
    )

    save_dir = (
        f"/home/mithunmanivannan/checkpoints/ul_ecg/"
        f"engineering_wearecg_exact_{lead_tag}_bs{args.batch_size}_lf{args.loss_factor}{run_suffix}"
    )
    os.makedirs(save_dir, exist_ok=True)
    cleanup_partial_checkpoints(save_dir)

    encoder = VAE_Encoder().to(device, dtype=torch.float32)
    decoder = VAE_Decoder().to(device, dtype=torch.float32)

    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.accum_steps,
        pct_start=0.2,
    )

    start_epoch = 0
    best_selector = None
    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt and "scheduler_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        best_selector = tuple(ckpt.get("best_selector_tuple", [])) or None
        write_warm_start_summary(
            save_dir,
            {
                "checkpoint_path": args.resume,
                "loaded_tensor_count": len(encoder.state_dict()) + len(decoder.state_dict()),
                "skipped_tensor_count": 0,
                "loaded_prefixes": ["encoder", "decoder"],
                "skipped_prefixes_sample": [],
                "loaded_keys_sample": (["encoder." + k for k in sorted(encoder.state_dict().keys())[:10]] + ["decoder." + k for k in sorted(decoder.state_dict().keys())[:10]]),
                "skipped_keys_sample": [],
                "initialization_type": "full_warm_start",
            },
        )

    metadata = build_run_metadata(args)
    write_json(os.path.join(save_dir, "run_metadata.json"), metadata)
    step_counter = start_epoch * len(train_loader)
    opt_step = 0

    encoder.train()
    decoder.train()
    for epoch in range(start_epoch, args.epochs):
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for i, (x, y, _meta) in enumerate(pbar):
            x = x.to(device, dtype=torch.float32, non_blocking=True) # (B, 12, L)
            y = y.to(device, dtype=torch.float32, non_blocking=True) # (B, 12, L)
            
            x_masked = mask_unobserved_leads(x, args.obs_lead_indices)
            
            # Encoder transposes (B, L, 12) -> (B, 12, L)
            # So pass (B, L, 12)
            x_enc_in = x_masked.transpose(1, 2)
            y_target = y.transpose(1, 2)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                z, mean, log_var = encoder(x_enc_in)
                recons = decoder(z) # (B, L, 12)
                losses = loss_function(recons, y_target, mean, log_var, args.beta_kl, 0.0, None, device)
                loss = losses["loss"] / args.accum_steps

            loss.backward()

            if (i + 1) % args.accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
                if not torch.isfinite(grad_norm):
                    print(f"Non-finite grad_norm at step {step_counter}. Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                else:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

            decoder_loss_val = float(losses["recons_loss"].item())
            kl_loss_val = float(losses["KLD_loss"].item())
            finite_losses = math.isfinite(decoder_loss_val) and math.isfinite(kl_loss_val)
            if not finite_losses:
                print(f"Non-finite component loss at step {step_counter}. Skipping batch.")
                step_counter += 1
                continue

            if wandb_enabled:
                safe_wandb_log(
                    {
                        "train/decoder_loss": decoder_loss_val,
                        "train/kl_loss": kl_loss_val,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "step": step_counter,
                        "opt_step": opt_step,
                    },
                    step=step_counter,
                    prefix="train",
                )
            pbar.set_postfix({"Dec": f"{decoder_loss_val:.2f}", "KL": f"{kl_loss_val:.4f}"})
            step_counter += 1

        if (epoch + 1) % 1 == 0 or args.debug:
            eval_model = build_eval_wrapper(encoder, decoder, args).to(device, dtype=torch.float32)
            metrics = evaluate_reconstruction(
                eval_model,
                val_loader,
                device,
                args.obs_lead_indices,
                split="val",
                step=step_counter,
                model_family="wearecg_vae",
                log_to_wandb=False,
                fast_eval=args.fast_eval,
            )
            write_run_artifacts(save_dir, metadata, metrics)
            current_selector = get_baseline_selector(metrics)
            is_best = best_selector is None or current_selector > best_selector
            if is_best:
                best_selector = current_selector

            ckpt_path = os.path.join(save_dir, f"ul_ecp_ep{epoch + 1}.pt")
            tmp_ckpt_path = ckpt_path + ".tmp"
            save_dict = {
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_selector_tuple": list(best_selector) if best_selector is not None else None,
                "current_selector": list(current_selector),
                "model_family": "wearecg_vae",
                "obs_leads": args.obs_lead_indices,
                "regime": args.regime,
                "seed": int(args.seed),
                "beta_kl": args.beta_kl,
                "latent_channels": 4,
                "target_len": args.target_len,
            }
            if args.save_training_state:
                save_dict["optimizer_state_dict"] = optimizer.state_dict()
                save_dict["scheduler_state_dict"] = scheduler.state_dict()

            prune_epoch_checkpoints(save_dir, keep_latest=2)
            torch.save(save_dict, tmp_ckpt_path)
            os.replace(tmp_ckpt_path, ckpt_path)
            print(f"Saved: {ckpt_path}")

            if is_best:
                best_ckpt_path = os.path.join(save_dir, "ul_ecp_best.pt")
                best_tmp_ckpt_path = best_ckpt_path + ".tmp"
                torch.save(save_dict, best_tmp_ckpt_path)
                os.replace(best_tmp_ckpt_path, best_ckpt_path)
                write_best_summary(save_dir, current_selector, metrics, epoch + 1)
                print(f"Updated best checkpoint: {best_ckpt_path} | selector={current_selector}")

            prune_epoch_checkpoints(save_dir, keep_latest=2)

            if wandb_enabled:
                safe_wandb_log(metrics, step=step_counter, prefix="val")

    if wandb_enabled:
        try:
            wandb.finish()
        except Exception as exc:
            print(f"[W&B] finish failed: {exc}")


if __name__ == "__main__":
    train(get_args())
