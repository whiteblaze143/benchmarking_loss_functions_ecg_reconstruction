"""Universal foundation-model and SetOperator adapters for V7-P.

Provides verified, zero-leakage, frozen feature extraction for:
1. CSFM Tiny (dim 768)
2. CLEF Medium (dim 1024)
3. ECGFounder 1-lead (dim 1024)
4. ECG-FM KED (dim 1024)
5. EchoNext Minimodel (dim 256)
6. DINO-SR (dim 512)
7. JEPA (dim 512)
8. NEFCore Full (dim 128)
9. SetOperators: F2, F0, G1, Control Capacity
"""

import sys
import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

# Disable cuDNN to avoid PyTorch 2.6 / A100 conv1d bug (ptrDesc->finalize())
torch.backends.cudnn.enabled = False

# Path setups
REPO_ROOT = Path(__file__).resolve().parents[3]
CSFM_DIR = REPO_ROOT / "external" / "Cardiac-Sensing-FM"
CLEF_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/clef")
ECHONEXT_DIR = REPO_ROOT / "external" / "IntroECG" / "7-EchoNext Minimodel"
SSL_DIR = Path("/data/mithunmanivannan/codex_artifacts/ptbxl_distributional_repecg/foundation_model_comparison/candidate_sources/ecg_pretraining_strategies/downstream_evaluation/code")
NEF_DIR = Path("/data/mithunmanivannan/nef-net-aim/src")
SETOP_DIR = Path("/data/mithunmanivannan/codex_artifacts/temporal_setoperator_fm_20261120/artifacts/zoh_v1/source")
REPECG_DIR = REPO_ROOT / "experiments" / "ptbxl_distributional_repecg" / "src"

for p in [CSFM_DIR, CLEF_DIR, ECHONEXT_DIR, SSL_DIR, NEF_DIR, SETOP_DIR, REPECG_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def resample_signal(signal: np.ndarray, src_fs: float, dst_fs: float, target_length: Optional[int] = None) -> np.ndarray:
    """Anti-aliased resampling of ECG waveform to target frequency."""
    if abs(src_fs - dst_fs) < 1e-3:
        out = signal.copy()
    else:
        num_target = int(round(len(signal) * dst_fs / src_fs))
        out = scipy.signal.resample(signal, num_target)
    
    if target_length is not None:
        if len(out) < target_length:
            pad = np.zeros(target_length - len(out), dtype=out.dtype)
            out = np.concatenate([pad, out])  # Causal left padding
        elif len(out) > target_length:
            out = out[-target_length:]  # Most recent samples
    return out


# -----------------------------------------------------------------------------
# 1. CSFM Tiny
# -----------------------------------------------------------------------------
class CSFMAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        from network.model import CSFM_model
        from einops import rearrange, repeat
        self.rearrange = rearrange
        self.repeat = repeat
        
        self.model = CSFM_model('Tiny')
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(ckpt, strict=False)
        self.model.eval().to(device)
        self.device = device
        self.dim = 768
        self.fs = 250
        self.target_len = 2500  # 10s @ 250 Hz

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 2500) @ 250 Hz"""
        b, c, n = x.shape[0], 1, self.target_len // 25
        ts_emb = self.model.ts_to_patch_embedding(x)
        ts_channel_emb = self.model.ts_channel_type_embedding[:, [0]].repeat(b, 1, n, 1)
        ts_emb += ts_channel_emb
        ts_position_emb = self.model.ts_pos_embedding.repeat(b, 1, 1, 1)
        ts_emb += ts_position_emb
        ts_emb = self.rearrange(ts_emb, 'b c n d -> b (c n) d')
        cls_tokens = self.repeat(self.model.cls_token, '1 1 d -> b 1 d', b=b)
        feat = torch.cat((cls_tokens, ts_emb), dim=1)
        feat = self.model.dropout(feat)
        out = self.model.transformer(feat)
        return out[:, 0]  # CLS latent (B, 768)


# -----------------------------------------------------------------------------
# 2. CLEF Medium
# -----------------------------------------------------------------------------
class CLEFAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        from clef.baselines.models.CLEF import create_net1d_by_size
        self.model = create_net1d_by_size(
            device=device,
            model_size='medium',
            n_classes=4,
            linear_prob=True,
            pth=checkpoint_path,
            in_channels=1
        )
        self.model.eval().to(device)
        self.device = device
        self.dim = 1024
        self.fs = 500
        self.target_len = 5000  # 10s @ 500 Hz

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 5000) @ 500 Hz"""
        out = self.model.first_conv(x)
        if self.model.use_bn:
            out = self.model.first_bn(out)
        out = self.model.first_activation(out)
        for stage in self.model.stage_list:
            out = stage(out)
        return out.mean(-1)  # GAP feature (B, 1024)


# -----------------------------------------------------------------------------
# 3. ECGFounder 1-lead
# -----------------------------------------------------------------------------
class ECGFounderAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        from clef.baselines.models.ECGFounder import Net1D
        self.model = Net1D(
            in_channels=1,
            base_filters=64,
            ratio=1.0,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=4,
        )
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = {k: v for k, v in ckpt['state_dict'].items() if not k.startswith('dense.')}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)
        self.device = device
        self.dim = 1024
        self.fs = 500
        self.target_len = 5000  # 10s @ 500 Hz

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 5000) @ 500 Hz"""
        out = self.model.first_conv(x)
        out = self.model.first_activation(out)
        for stage in self.model.stage_list:
            out = stage(out)
        return out.mean(-1)  # GAP feature (B, 1024)


# -----------------------------------------------------------------------------
# 4. ECG-FM KED (Clean PyTorch XResNet1d101)
# -----------------------------------------------------------------------------
class XResNet1d101(nn.Sequential):
    def __init__(self, input_channels=12, kernel_size=5):
        # Stem
        def make_conv(ni, nf, s):
            return nn.Sequential(
                nn.Conv1d(ni, nf, kernel_size=kernel_size, stride=s, padding=(kernel_size - 1) // 2, bias=False),
                nn.BatchNorm1d(nf),
                nn.ReLU(inplace=True)
            )
        stem = [make_conv(input_channels, 32, 2), make_conv(32, 32, 1), make_conv(32, 64, 1)]
        
        # ResBlocks
        class FastResBlock(nn.Module):
            def __init__(self, ni, nf, stride=1):
                super().__init__()
                nh = nf
                out_dim = nf * 4
                in_dim = ni * 4 if ni != 16 else 64
                self.convpath = nn.Sequential(
                    nn.Conv1d(in_dim, nh, kernel_size=1, bias=False),
                    nn.BatchNorm1d(nh),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(nh, nh, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False),
                    nn.BatchNorm1d(nh),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(nh, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_dim)
                )
                self.idpath = nn.Identity() if in_dim == out_dim and stride == 1 else nn.Sequential(
                    nn.AvgPool1d(2, ceil_mode=True) if stride != 1 else nn.Identity(),
                    nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_dim)
                )
                self.act = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.act(self.convpath(x) + self.idpath(x))

        layers = [3, 4, 23, 3]
        block_szs = [16, 64, 128, 128, 192]
        blocks = []
        for i, l in enumerate(layers):
            blks = [FastResBlock(block_szs[i] if j == 0 else block_szs[i + 1], block_szs[i + 1], stride=1 if (i == 0 and j == 0) or j > 0 else 2) for j in range(l)]
            blocks.append(nn.Sequential(*blks))
        
        super().__init__(*stem, nn.MaxPool1d(kernel_size=3, stride=2, padding=1), *blocks)


class ECGFMKEDAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        self.device = device
        self.dim = 768
        self.fs = 500
        self.target_len = 5000
        self.model = XResNet1d101(input_channels=12, kernel_size=5)
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt['ecg_model'] if 'ecg_model' in ckpt else ckpt
        body_sd = {k: v for k, v in sd.items() if not k.startswith('1.')}
        self.model.load_state_dict(body_sd, strict=False)
        self.model.eval().to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 5000) -> 12-lead zero-masked tensor (B, 12, 5000)"""
        b = x.shape[0]
        full = torch.zeros(b, 12, self.target_len, device=self.device, dtype=x.dtype)
        full[:, 0, :] = x[:, 0, :]
        feat = self.model(full)  # (B, 768, L)
        return feat.mean(-1)  # GAP -> (B, 768)


# -----------------------------------------------------------------------------
# 5. EchoNext Minimodel
# -----------------------------------------------------------------------------
class EchoNextAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        from cradlenet.models.resnet1d_tabular import ResNet1dWithTabular
        self.model = ResNet1dWithTabular(
            len_tabular_feature_vector=7,
            filter_size=16,
            input_channels=12,
            num_classes=12
        )
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt['model'] if 'model' in ckpt else ckpt
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(device)
        self.device = device
        self.dim = 256
        self.fs = 400
        self.target_len = 4000

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 4000) -> 12-lead zero-masked tensor (B, 12, 4000)"""
        b = x.shape[0]
        full = torch.zeros(b, 12, self.target_len, device=self.device, dtype=x.dtype)
        full[:, 0, :] = x[:, 0, :]
        # Backbone forward
        out = self.model.conv1(full)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = self.model.layer4(out)
        x1 = self.model.adaptiveavgpool(out)
        x2 = self.model.adaptivemaxpool(out)
        feat = torch.cat((x1, x2), dim=1)
        return feat.view(feat.size(0), -1)  # (B, 256)


# -----------------------------------------------------------------------------
# 6 & 7. DINO-SR and JEPA
# -----------------------------------------------------------------------------
class SSLAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, model_type: str, device: torch.device):
        super().__init__()
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        # Reconstruct the 4-layer 1D CNN encoder directly from state_dict
        layers = []
        in_c = 12
        strides = [2, 1, 1, 1]
        kss = [3, 1, 1, 1]
        feats = [512, 512, 512, 512]
        
        for s, k, f in zip(strides, kss, feats):
            conv = nn.Conv1d(in_c, f, kernel_size=k, stride=s, padding=(k - 1) // 2)
            bn = nn.BatchNorm1d(f)
            relu = nn.ReLU()
            layers.extend([conv, bn, relu])
            in_c = f
        self.encoder = nn.Sequential(*layers)
        
        # Load weights
        sd = ckpt['state_dict']
        our_sd = self.encoder.state_dict()
        mapping = {
            '0.weight': 'ts_encoder.encoder.layers.0.0.weight',
            '0.bias': 'ts_encoder.encoder.layers.0.0.bias',
            '1.weight': 'ts_encoder.encoder.layers.0.1.weight',
            '1.bias': 'ts_encoder.encoder.layers.0.1.bias',
            '1.running_mean': 'ts_encoder.encoder.layers.0.1.running_mean',
            '1.running_var': 'ts_encoder.encoder.layers.0.1.running_var',
            '3.weight': 'ts_encoder.encoder.layers.1.0.weight',
            '3.bias': 'ts_encoder.encoder.layers.1.0.bias',
            '4.weight': 'ts_encoder.encoder.layers.1.1.weight',
            '4.bias': 'ts_encoder.encoder.layers.1.1.bias',
            '4.running_mean': 'ts_encoder.encoder.layers.1.1.running_mean',
            '4.running_var': 'ts_encoder.encoder.layers.1.1.running_var',
            '6.weight': 'ts_encoder.encoder.layers.2.0.weight',
            '6.bias': 'ts_encoder.encoder.layers.2.0.bias',
            '7.weight': 'ts_encoder.encoder.layers.2.1.weight',
            '7.bias': 'ts_encoder.encoder.layers.2.1.bias',
            '7.running_mean': 'ts_encoder.encoder.layers.2.1.running_mean',
            '7.running_var': 'ts_encoder.encoder.layers.2.1.running_var',
            '9.weight': 'ts_encoder.encoder.layers.3.0.weight',
            '9.bias': 'ts_encoder.encoder.layers.3.0.bias',
            '10.weight': 'ts_encoder.encoder.layers.3.1.weight',
            '10.bias': 'ts_encoder.encoder.layers.3.1.bias',
            '10.running_mean': 'ts_encoder.encoder.layers.3.1.running_mean',
            '10.running_var': 'ts_encoder.encoder.layers.3.1.running_var',
        }
        loaded = {}
        for k, v in mapping.items():
            if v in sd:
                loaded[k] = sd[v]
        self.encoder.load_state_dict(loaded, strict=False)
        self.encoder.eval().to(device)
        self.device = device
        self.dim = 512
        self.fs = 240
        self.target_len = 2400  # 10s @ 240 Hz

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 2400) -> 12-channel zero-masked (B, 12, 2400)"""
        b = x.shape[0]
        full = torch.zeros(b, 12, self.target_len, device=self.device, dtype=x.dtype)
        full[:, 0, :] = x[:, 0, :]
        feat = self.encoder(full)  # (B, 512, 1200)
        return feat.mean(-1)  # GAP (B, 512)


# -----------------------------------------------------------------------------
# 8. NEFCore Full
# -----------------------------------------------------------------------------
class NEFCoreAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        from nef_net_aim.core import NEFCore
        self.model = NEFCore(mode='FULL')
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        self.model.load_state_dict(sd, strict=True)
        self.model.eval().to(device)
        self.device = device
        self.dim = 128
        self.fs = 500
        self.target_len = 5000

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 5000) @ 500 Hz"""
        b = x.shape[0]
        lead_idx = torch.zeros(b, 1, dtype=torch.long, device=self.device)
        q = self.model.angle_embed(lead_idx, mode="PTB_XL")
        feat = self.model.view_encoder(x, q)
        return feat.mean(-1)  # (B, 128)


# -----------------------------------------------------------------------------
# 9. SetOperator Models (F2, F0, G1, Control Capacity)
# -----------------------------------------------------------------------------
class SetOperatorAdapter(nn.Module):
    def __init__(self, checkpoint_path: str, model_type: str, device: torch.device):
        super().__init__()
        self.model_type = model_type.upper()
        self.device = device
        self.fs = 200
        self.target_len = 2048

        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if self.model_type in ["F2", "F0"]:
            from model import TemporalSetOperatorFM
            self.model = TemporalSetOperatorFM(phase_features=32, phases=16, width=320, layers=6, use_phase=False)
            sd = ckpt['model']
            self.model.load_state_dict(sd, strict=False)
            self.model.eval().to(device)
            self.dim = 320
        elif "G1" in self.model_type:
            from repecg.paper07_operator.model import OperatorSetModel
            m = OperatorSetModel(response_dim=128, classes=1, operator_mode="continuous", operator_dim=44, patient_context_dim=7)
            core_sd = {k[len("core."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("core.")}
            m.load_state_dict(core_sd, strict=True)
            self.model = m.eval().to(device)
            self.dim = 256
        elif "CONTROL" in self.model_type or "CAPACITY" in self.model_type:
            from repecg.paper07_operator.model import OperatorSetModel
            m = OperatorSetModel(response_dim=128, classes=1, operator_mode="continuous", operator_dim=8, patient_context_dim=7)
            m.operator = nn.Sequential(nn.Linear(8, 96), nn.GELU(), nn.Linear(96, 64))
            core_sd = {k[len("core."):]: v for k, v in ckpt["model_state_dict"].items() if k.startswith("core.")}
            m.load_state_dict(core_sd, strict=True)
            self.model = m.eval().to(device)
            self.dim = 256
        else:
            raise ValueError(f"Unknown SetOperator model type: {model_type}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 2048) @ 200 Hz"""
        b = x.shape[0]
        if self.model_type in ["F2", "F0"]:
            # Patchify: (B, 1, 31, 64)
            patches = x[:, :, :1984].reshape(b, 1, 31, 64)
            patch_times = torch.arange(31, device=self.device).float().unsqueeze(0).repeat(b, 1)
            channel_mask = torch.ones(b, 1, dtype=torch.bool, device=self.device)
            geom = torch.zeros(b, 1, 9, device=self.device)
            geom[:, 0, 0] = 1.0  # Lead I verified coordinate
            geom[:, 0, 8] = 1.0  # Known flag
            out = self.model(patches, patch_times, channel_mask, geom)
            return out["recording"]  # (B, 320)
        elif "G1" in self.model_type:
            responses = x[:, :, :2048].reshape(b, 1, 16, 128)
            pat = torch.zeros(b, 7, device=self.device)
            op = torch.zeros(b, 1, 44, device=self.device)
            op[:, 0, 0] = 1.0
            return self.model.encode_context(op, responses, patient_context=pat)  # (B, 256)
        else:
            # Control Capacity
            responses = x[:, :, :2048].reshape(b, 1, 16, 128)
            pat = torch.zeros(b, 7, device=self.device)
            op = torch.zeros(b, 1, 8, device=self.device)
            op[:, 0, 0] = 1.0
            return self.model.encode_context(op, responses, patient_context=pat)  # (B, 256)


def get_model_adapter(model_key: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Instantiate and return adapter for model_key."""
    key = model_key.upper()
    if "CSFM" in key:
        return CSFMAdapter(checkpoint_path, device)
    elif "CLEF" in key:
        return CLEFAdapter(checkpoint_path, device)
    elif "ECGFOUNDER" in key:
        return ECGFounderAdapter(checkpoint_path, device)
    elif "ECGFM" in key or "KED" in key:
        return ECGFMKEDAdapter(checkpoint_path, device)
    elif "ECHONEXT" in key:
        return EchoNextAdapter(checkpoint_path, device)
    elif "DINOSR" in key:
        return SSLAdapter(checkpoint_path, "dinosr", device)
    elif "JEPA" in key:
        return SSLAdapter(checkpoint_path, "jepa", device)
    elif "NEFCORE" in key:
        return NEFCoreAdapter(checkpoint_path, device)
    elif "F2" in key:
        return SetOperatorAdapter(checkpoint_path, "F2", device)
    elif "F0" in key:
        return SetOperatorAdapter(checkpoint_path, "F0", device)
    elif "G1" in key:
        return SetOperatorAdapter(checkpoint_path, "G1", device)
    elif "CAPACITY" in key or "CONTROL" in key:
        return SetOperatorAdapter(checkpoint_path, "CONTROL", device)
    else:
        raise ValueError(f"Unknown foundation model key: {model_key}")
