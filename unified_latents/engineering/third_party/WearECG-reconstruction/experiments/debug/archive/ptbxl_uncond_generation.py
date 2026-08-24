import os
import torch
import numpy as np
import sys
import os

project_root_dir = "/data1/1shared/jinjiarui/run/Diffusion-TS"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from engine.solver import Trainer
from utils.metric_utils import visualization
from data.build_dataloader import build_dataloader_train
from utils.io_utils import load_yaml_config, instantiate_from_config
from models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one


class Args_Example:
    def __init__(self) -> None:
        self.config_path = "./config/ptbxl.yaml"
        self.save_dir = "./results/ptbxl_exp"
        self.gpu = 0
        os.makedirs(self.save_dir, exist_ok=True)


args = Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

dl_info = build_dataloader_train(configs, args)
model = instantiate_from_config(configs["model"]).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

# trainer.train()
# trainer.load(args.milestone)

dataset = dl_info["dataset"]
seq_length, feature_dim = dataset.window, dataset.var_num
ori_data = np.load(os.path.join(dataset.dir, f"norm_truth_{seq_length}_train.npy"))

print(f"original data scale: {np.min(ori_data)} {np.max(ori_data)}")

# ori_data = np.load(os.path.join(dataset.dir, f"{dataset_name}_norm_truth_{seq_length}_train.npy"))  # Uncomment the line if dataset other than Sine is used.
fake_data = np.load("results/ptbxl_exp/ddpm_fake_ptbxl.npy")
# fake_data = trainer.sample(
#     num=len(dataset), size_every=256, shape=[seq_length, feature_dim]
# )
# np.save(os.path.join(args.save_dir, f"ddpm_fake_ptbxl.npy"), fake_data)
print(f"generated data scale: {np.min(fake_data)} {np.max(fake_data)}")

visualization(
    ori_data=ori_data,
    generated_data=fake_data,
    analysis="pca",
    compare=ori_data.shape[0],
    save_dir="figures/ptbxl",
)

visualization(
    ori_data=ori_data,
    generated_data=fake_data,
    analysis="tsne",
    compare=ori_data.shape[0],
    save_dir="figures/ptbxl",
)

visualization(
    ori_data=ori_data,
    generated_data=fake_data,
    analysis="kernel",
    compare=ori_data.shape[0],
    save_dir="figures/ptbxl",
)
