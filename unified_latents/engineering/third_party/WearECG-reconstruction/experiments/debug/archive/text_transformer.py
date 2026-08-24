import sys
import os

project_root_dir = "/data1/1shared/jinjiarui/run/Diffusion-TS"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

import torch
from models.interpretable_diffusion.transformer import Transformer


if __name__ == "__main__":
    ecg = torch.randn((2, 1000, 12))
    t = torch.randint(0, 10, size=(2,))
    text = ["NMSL", "NMSL"]
    model = Transformer(
        n_feat=12,
        n_channel=1000,
        n_layer_enc=4,
        n_layer_dec=2,
        n_heads=4,
        attn_pdrop=0,
        resid_pdrop=0,
        mlp_hidden_times=4,
        n_embd=96,
        conv_params=[1, 0],
        max_len=1000,
        use_text=True,
        text_encoder_url="ncbi/MedCPT-Query-Encoder",
    )

    trend, season_error = model(ecg, t, report=text)
    model_out = trend + season_error
    print(model_out.shape)
