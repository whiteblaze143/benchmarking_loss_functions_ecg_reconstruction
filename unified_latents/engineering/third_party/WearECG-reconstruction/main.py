import os
import torch
import argparse
import numpy as np

from engine.logger import Logger
from engine.solver import Trainer
from data.build_dataloader import build_dataloader_train, build_dataloader_sample
from models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from utils.io_utils import (
    load_yaml_config,
    seed_everything,
    merge_opts_to_config,
    instantiate_from_config,
)

# huggingface offline mode
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training Script")
    parser.add_argument("--name", type=str, default=None)

    parser.add_argument(
        "--config_file", type=str, default=None, help="path of config file"
    )
    parser.add_argument(
        "--output", type=str, default="OUTPUT", help="directory to save the results"
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="use tensorboard for logging"
    )

    # args for random

    parser.add_argument(
        "--cudnn_deterministic",
        action="store_true",
        default=True,
        help="set cudnn.deterministic True",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="seed for initializing training."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU id to use. If given, only the specific gpu will be"
        " used, and ddp will be disabled",
    )

    # args for training
    parser.add_argument(
        "--train", action="store_true", default=False, help="Train or Test."
    )
    parser.add_argument(
        "--condition_type",
        type=int,
        default=0,
        choices=[0, 1],
        help="Uncondition or Condition.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="infill",
        help="Infilling, Forecasting or Synthesis.",
    )
    parser.add_argument("--milestone", type=int, default=10)

    parser.add_argument(
        "--missing_ratio", type=float, default=0.0, help="Ratio of Missing Values."
    )
    parser.add_argument(
        "--pred_len", type=int, default=0, help="Length of Predictions."
    )
    parser.add_argument(
        "--synthesis_channels",
        type=lambda x: list(map(int, x.split(","))),
        default=list(range(1, 12)),
        help="List of synthesis channels (default is [1, 2, 3, ..., 11]).",
    )

    # args for modify config
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f"{args.name}")

    return args


def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    config = load_yaml_config(args.config_file)
    config = merge_opts_to_config(config, args.opts)

    print(config)

    logger = Logger(args)
    logger.save_config(config)

    if args.train:
        model = instantiate_from_config(config["model"]).cuda()
        dataloader_info = build_dataloader_train(config, args)
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=dataloader_info,
            logger=logger,
        )
        trainer.train()

    elif args.condition_type == 0:  # Uncondition Sampling
        model = instantiate_from_config(config["model"]).cuda()
        dataloader_info = build_dataloader_train(config, args)
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=dataloader_info,
            logger=logger,
        )
        trainer.load(args.milestone)
        dataset = dataloader_info["dataset"]
        samples = trainer.sample(
            num=len(dataset),
            size_every=2001,
            shape=[dataset.window, dataset.var_num],
            subset_save_threshold=50000,
            save_dir=os.path.join(args.save_dir, "samples"),
        )
        # if dataset.auto_norm:
        #     samples = unnormalize_to_zero_to_one(samples)
        #     samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        # np.save(os.path.join(args.save_dir, f"ddpm_fake_{args.name}.npy"), samples)

    elif args.condition_type == 1 and args.mode in ["infill", "predict", "synthesis"]:
        model = instantiate_from_config(config["model"]).cuda()
        test_dataloader_info = build_dataloader_sample(config, args)
        trainer = Trainer(
            config=config,
            args=args,
            model=model,
            dataloader=test_dataloader_info,
            logger=logger,
        )

        trainer.load(args.milestone)
        dataloader, dataset = (
            test_dataloader_info["dataloader"],
            test_dataloader_info["dataset"],
        )
        coef = config["dataloader"]["test_dataset"]["coefficient"]
        learning_rate = config["dataloader"]["test_dataset"]["learning_rate"]
        sampling_steps = config["dataloader"]["test_dataset"]["sampling_steps"]
        samples, *_ = trainer.sample_shift(
            dataloader,
            [dataset.window, dataset.channels],
            sampling_steps,
            subset_save_threshold=50000,
            save_dir=os.path.join(args.save_dir, "samples"),
        )
        # if dataset.auto_norm:
        # samples = unnormalize_to_zero_to_one(samples)
        # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        # np.save(
        #     os.path.join(args.save_dir, f"ddpm_{args.mode}_{args.name}.npy"), samples
        # )


if __name__ == "__main__":
    main()
