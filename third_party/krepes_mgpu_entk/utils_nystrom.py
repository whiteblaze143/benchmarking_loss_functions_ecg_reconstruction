import logging
import pathlib
import sys
import time
import psutil
import os
import torch
import torch.nn as nn
import torchvision.models as models
from utility.model import ResMLP

def humanize_units(size, unit="B"):
    for prefix in ["", "Ki", "Mi", "Gi", "Ti", "Pi"]:
        if size < 1024.0 or prefix == "Pi":
            break
        size /= 1024.0
    return f"{size:.1f}{prefix}"

def init_torch(allow_tf32=False, benchmark=False, deterministic=True, verbose=False):

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

    if verbose:
        logging.info(f"{torch.backends.cuda.matmul.allow_tf32 = }")
        logging.info(f"{torch.backends.cudnn.allow_tf32 = }")
        logging.info(f"{torch.backends.cudnn.benchmark = }")
        logging.info(f"{torch.backends.cudnn.deterministic = }")

def init_logging(handle, logdir):
    if logdir is not None:
        logdir = pathlib.Path(logdir)
        logdir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        filename = logdir / f"{handle}-{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(filename=filename),
                logging.StreamHandler(sys.stdout),
            ],
        )
        logging.info(f"Logging to {filename}")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.DEBUG,
            stream=sys.stdout,
        )

def load_model(name, **kwargs):

    rng_state = torch.get_rng_state()
    torch.manual_seed(438)
    num_classes = 1

    if "resnet" in name:
        if name == "resnet-18_init":
            model = models.resnet18()
            model.fc = nn.Linear(512, num_classes)
        elif name == "resnet-18_pretrained":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(512, num_classes)
        elif name == "resnet-34_pretrained":
            torch.manual_seed(438)
            model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            model.fc = nn.Linear(512, num_classes)
        elif name == "resnet-50_init":
            model = models.resnet50()
            model.fc = nn.Linear(2048, num_classes)
        elif name == "resnet-50_pretrained":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = nn.Linear(2048, num_classes)
        else:
            raise ValueError(f"Model '{name}' not recognized.")
    elif name == "resmlp":
        logging.info(f"Loading custom ResMLP with args: {kwargs}")
        model = ResMLP(
            in_features=kwargs.get("in_features"),
            width=kwargs.get("mlp_width"),
            num_blocks=kwargs.get("num_blocks"),
            num_classes=num_classes
        )
    
    else:
        raise ValueError(f"Model architecture '{name}' not recognized.")

    torch.set_rng_state(rng_state)
    return model


def num_classes_of(dataset_name):

    name_base = dataset_name.split('_')
    if name_base == "cifar10":
        return 10
    elif name_base == "CIFAR-100":
        return 100
    elif name_base == "SVHN":
        return 10
    elif name_base == "FashionMNIST":
        return 10
    elif name_base == "imagenet":
        return 1000
    elif dataset_name == "adult":
        return 1

    else:
        logging.warning(f"Dataset '{dataset_name}' not in num_classes_of, defaulting to 1000 classes.")
        return 1000

def save_ntk(ntk, savedir, handle):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    path = savedir / f"{handle}_nystrom-ntk_{timestamp}.pt"
    torch.save(ntk, path)
    logging.info(f"Saved Nystrom NTK to {path}")


def log_memory(checkpoint_name: str):
    """Logs the RSS memory of the current process."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
   
    rss_gb = mem_info.rss / (1024**3)
    print(f"[{checkpoint_name}] Memory Usage: {rss_gb:.2f} GB")

def save_ntk_ch(ntk, savedir, handle):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    # timestamp = int(time.time())
    # path = savedir / f"{handle}_nystrom-ntk_{timestamp}.pt"
    # torch.save(ntk, path)
    # logging.info(f"Saved Nystrom NTK to {path}")
    path = savedir / f"{handle}.pt" 
    torch.save(ntk, path)
    logging.info(f"Saved/Checkpointed NTK to {path}")