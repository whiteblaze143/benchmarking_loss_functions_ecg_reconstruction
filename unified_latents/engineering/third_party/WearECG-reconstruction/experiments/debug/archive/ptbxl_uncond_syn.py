import os
from matplotlib import pyplot as plt
import pandas as pd
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
from data.build_dataloader import build_dataloader_sample


class Args_Example:
    def __init__(self) -> None:
        self.gpu = 0
        self.config_path = "./config/ptbxl.yaml"
        self.save_dir = "./results/ptbxl_uncond_syn_exp"
        self.mode = "synthesis"
        self.synthesis_channels = list(range(1, 12))
        self.milestone = 10
        os.makedirs(self.save_dir, exist_ok=True)


args = Args_Example()
configs = load_yaml_config(args.config_path)
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

print(f"args.mode: {args.mode}")

dl_info = build_dataloader_sample(configs, args)
model = instantiate_from_config(configs["model"]).to(device)
trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

trainer.load(args.milestone)

dataloader, dataset = dl_info["dataloader"], dl_info["dataset"]
coef = configs["dataloader"]["test_dataset"]["coefficient"]
stepsize = configs["dataloader"]["test_dataset"]["step_size"]
sampling_steps = configs["dataloader"]["test_dataset"]["sampling_steps"]
seq_length, feature_dim = dataset.window, dataset.var_num

# samples, ori_data, masks = trainer.restore(
#     dataloader, [seq_length, feature_dim], coef, stepsize, sampling_steps
# )

# np.save(os.path.join(dataset.dir, f"synthesis_{seq_length}_test.npy"), samples)
samples = np.load(os.path.join(dataset.dir, f"synthesis_{seq_length}_test.npy"))

ori_data = np.load(os.path.join(dataset.dir, f"norm_truth_{seq_length}_test.npy"))
# ori_data = np.load(os.path.join(dataset.dir, f"{dataset_name}_norm_truth_{seq_length}_test.npy"))  # Uncomment the line if dataset other than Sine is used.
masks = np.load(os.path.join(dataset.dir, f"masking_{seq_length}.npy"))
sample_num, seq_len, feat_dim = ori_data.shape
observed = ori_data * masks

# bs, time, channel = ori_data.shape
index = 30

# 创建画板
fig, axes = plt.subplots(nrows=12, ncols=1, figsize=(10, 15), constrained_layout=True)
fig.suptitle(f"ECG Comparison for Sample Index {index}", fontsize=16)

# 遍历每个导联进行绘制
for i in range(ori_data.shape[2]):  # 遍历 channel
    ax = axes[i]
    ax.plot(
        ori_data[index, :, i],
        color="red",
        label="Original Data",
        alpha=0.7,  # 设置透明度
    )
    ax.plot(
        samples[index, :, i],
        color="blue",
        label="Generated Data",
        alpha=0.7,  # 设置透明度
    )
    ax.set_title(f"Lead {i + 1}")  # 标题：导联编号
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    ax.grid(True)

# 保存图形
fig_save_dir = os.path.join(args.save_dir, "figs")
os.makedirs(fig_save_dir, exist_ok=True)
plt.savefig(os.path.join(fig_save_dir, f"{index}.png"), dpi=400)  # 设置 DPI
