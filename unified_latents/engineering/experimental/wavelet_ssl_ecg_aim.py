#!/usr/bin/env python3
"""
Wavelet-SSL + delineation extension for the current 1-lead ECG-AIM.

Drop into:
  unified_latents/engineering/experimental/wavelet_ssl_ecg_aim.py

This is an ABSTRACT-INSPIRED implementation of the CinC 2026 method
"Self-Supervised Learning for ECG Representation Using Physiology-Informed
Wavelet Transforms". The exact custom ECG-inspired mother-wavelet equation is
not provided in the abstract. The built-in analytic Morlet bank is therefore
an engineering control, NOT an exact reproduction. Exact/custom-wavelet mode
is hard-gated on a documented external asset.

Current ECG-AIM parity defaults are width=768, heads=12 because the active
build_alitok_vae_1d defaults override the class's 384/8 constructor defaults.
"""
from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unified_latents.engineering.experimental.aim_1_lead import (  # noqa: E402
    AliTokECGAIM,
    AliTokECGAIMSpatial,
)


class AnalyticMorletFilterBank(nn.Module):
    """Differentiable analytic Morlet-like FFT filter bank; smoke-use only."""
    def __init__(
        self, sample_rate_hz=500.0, target_len=5000, n_scales=32,
        min_freq_hz=0.5, max_freq_hz=45.0, cycles=6.0
    ):
        super().__init__()
        if not (0 < min_freq_hz < max_freq_hz < sample_rate_hz / 2):
            raise ValueError("Bad frequency range.")
        self.sample_rate_hz = float(sample_rate_hz)
        self.target_len = int(target_len)
        self.n_scales = int(n_scales)
        self.cycles = float(cycles)
        centers = torch.logspace(
            math.log10(min_freq_hz), math.log10(max_freq_hz), n_scales
        )
        self.register_buffer("center_frequencies_hz", centers, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.target_len:
            raise ValueError(f"Expected [B,{self.target_len}], got {tuple(x.shape)}")
        x = x.float()
        f = torch.fft.fftfreq(
            self.target_len, d=1 / self.sample_rate_hz, device=x.device
        ).to(x.dtype)
        c = self.center_frequencies_hz.to(x.device, x.dtype)[:, None]
        sigma = (c / self.cycles).clamp_min(0.15)
        h = torch.exp(-0.5 * ((f[None] - c) / sigma) ** 2)
        h = h * (f[None] > 0).to(h.dtype)
        h = h / torch.sqrt(h.square().sum(-1, keepdim=True).clamp_min(1e-8))
        X = torch.fft.fft(x, dim=-1)
        return torch.fft.ifft(X[:, None] * h[None], dim=-1)


class ECGAdmissibleMorletFilterBank(nn.Module):
    r"""Zero-mean analytic Morlet bank with ECG pseudo-frequency sampling.

    Uses the corrected mother wavelet
    ``pi**(-1/4) * (exp(i*w0*t) - exp(-w0**2/2)) * exp(-t**2/2)``.
    The correction term enforces admissibility. The two-Gaussian frequency
    response is evaluated on the FFT grid, restricted to positive frequencies,
    and L2-normalized per filter.

    Literature: Buessow, arXiv:0706.0099 (FFT Morlet CWT); Barmase et al.,
    arXiv:1311.6460 (Morlet CWT for ECG/QRS). This is a reproducible
    literature-grounded engineering bank, not an exact reproduction of the
    unpublished custom wavelet described by the motivating CinC abstract.
    """

    def __init__(
        self, sample_rate_hz=500.0, target_len=5000, n_scales=32,
        min_freq_hz=0.5, max_freq_hz=45.0, cycles=6.0,
    ):
        super().__init__()
        if not (0 < min_freq_hz < max_freq_hz < sample_rate_hz / 2):
            raise ValueError("Bad frequency range.")
        if int(target_len) < 2 or int(n_scales) < 1:
            raise ValueError("target_len and n_scales must be positive.")
        if float(cycles) <= 0:
            raise ValueError("cycles must be positive.")
        self.sample_rate_hz = float(sample_rate_hz)
        self.target_len = int(target_len)
        self.n_scales = int(n_scales)
        self.cycles = float(cycles)
        centers = torch.logspace(
            math.log10(min_freq_hz), math.log10(max_freq_hz), n_scales,
        )
        self.register_buffer("center_frequencies_hz", centers, persistent=True)

    def frequency_response(self, *, device=None, dtype=torch.float32) -> torch.Tensor:
        """Return fixed analysis responses with shape ``[scales, samples]``."""
        device = device or self.center_frequencies_hz.device
        f = torch.fft.fftfreq(
            self.target_len, d=1 / self.sample_rate_hz, device=device,
        ).to(dtype)
        centers = self.center_frequencies_hz.to(device=device, dtype=dtype)[:, None]
        w0 = torch.as_tensor(self.cycles, device=device, dtype=dtype)
        scaled_omega = f[None] * w0 / centers
        correction = torch.exp(-0.5 * w0.square())
        response = (
            torch.exp(-0.5 * (scaled_omega - w0).square())
            - correction * torch.exp(-0.5 * scaled_omega.square())
        )
        response = response * (f[None] > 0).to(dtype)
        return response / torch.sqrt(
            response.square().sum(-1, keepdim=True).clamp_min(1e-12)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[-1] != self.target_len:
            raise ValueError(f"Expected [B,{self.target_len}], got {tuple(x.shape)}")
        x = x.float()
        response = self.frequency_response(device=x.device, dtype=x.dtype)
        return torch.fft.ifft(
            torch.fft.fft(x, dim=-1)[:, None] * response[None], dim=-1,
        )


class FixedComplexKernelBank(nn.Module):
    """Asset-gated exact/custom wavelet hook. Never invents a custom wavelet.

    .pt schema:
      real: [F,K] required
      imag: [F,K] optional
      metadata: dict recommended
    """
    def __init__(self, asset_path: str | Path, target_len=5000):
        super().__init__()
        p = Path(asset_path)
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing custom-wavelet asset {p}. Exact CinC mode is BLOCKED."
            )
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if not isinstance(obj, dict) or "real" not in obj:
            raise ValueError("Asset must be dict with key 'real'.")
        real = torch.as_tensor(obj["real"], dtype=torch.float32)
        imag = torch.as_tensor(obj.get("imag", torch.zeros_like(real)), dtype=torch.float32)
        if real.ndim != 2 or imag.shape != real.shape:
            raise ValueError("Expected real/imag [F,K].")
        if real.shape[-1] > target_len:
            raise ValueError("Kernel longer than target signal.")
        self.target_len = int(target_len)
        self.metadata = dict(obj.get("metadata", {}))
        self.register_buffer("real_kernel", real, persistent=True)
        self.register_buffer("imag_kernel", imag, persistent=True)

    @property
    def n_scales(self): return int(self.real_kernel.shape[0])

    def forward(self, x):
        if x.ndim != 2 or x.shape[-1] != self.target_len:
            raise ValueError("Bad custom-bank input shape.")
        r, i = self.real_kernel.float(), self.imag_kernel.float()
        k = r.shape[-1]
        left = (self.target_len - k) // 2
        right = self.target_len - k - left
        z = torch.complex(F.pad(r, (left, right)), F.pad(i, (left, right))).to(x.device)
        H = torch.fft.fft(torch.fft.ifftshift(z, dim=-1), dim=-1)
        return torch.fft.ifft(torch.fft.fft(x.float(), dim=-1)[:, None] * H[None], dim=-1)


class WaveletViewBuilder(nn.Module):
    VALID = {"magnitude","log_magnitude","power","phase","phase_sin","phase_cos","real","imag"}
    def __init__(self, bank, target_len=5000, num_patches=200):
        super().__init__()
        if target_len % num_patches:
            raise ValueError("target_len must divide num_patches exactly.")
        self.bank = bank
        self.target_len = target_len
        self.num_patches = num_patches
        self.pool = target_len // num_patches

    def forward(self, x, view):
        if view not in self.VALID:
            raise ValueError(f"Unknown wavelet view {view}")
        c = self.bank(x)
        if view == "magnitude": v = c.abs()
        elif view == "log_magnitude": v = torch.log1p(c.abs())
        elif view == "power": v = c.abs().square()
        elif view == "phase": v = torch.angle(c) / math.pi
        elif view == "phase_sin": v = torch.sin(torch.angle(c))
        elif view == "phase_cos": v = torch.cos(torch.angle(c))
        elif view == "real": v = c.real
        else: v = c.imag
        B, Fq, _ = v.shape
        v = v.float().reshape(B, Fq, self.num_patches, self.pool).mean(-1)
        v = (v - v.mean(-1, keepdim=True)) / v.std(-1, keepdim=True).clamp_min(1e-5)
        return v[:, None]  # [B,1,F,P]


class FactorizedTFBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.tn = nn.LayerNorm(dim)
        self.ta = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.fn = nn.LayerNorm(dim)
        self.fa = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mn = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim,4*dim), nn.GELU(), nn.Linear(4*dim,dim))

    def forward(self, x):
        B,Fq,P,D = x.shape
        t = x.reshape(B*Fq,P,D)
        z = self.tn(t); t = t + self.ta(z,z,z,need_weights=False)[0]
        x = t.reshape(B,Fq,P,D)
        f = x.permute(0,2,1,3).reshape(B*P,Fq,D)
        z = self.fn(f); f = f + self.fa(z,z,z,need_weights=False)[0]
        x = f.reshape(B,P,Fq,D).permute(0,2,1,3)
        return x + self.mlp(self.mn(x))


class FactorizedScalogramEncoder(nn.Module):
    """TimeSformer-like encoder; pools frequency but preserves 200 time tokens."""
    def __init__(self, n_freq, num_patches, model_dim=192, output_dim=768, depth=2, heads=6):
        super().__init__()
        if model_dim % heads: raise ValueError("wavelet_dim must divide wavelet_heads")
        self.n_freq, self.num_patches = n_freq, num_patches
        self.inp = nn.Linear(1, model_dim)
        s = model_dim ** -0.5
        self.fe = nn.Parameter(s*torch.randn(n_freq,model_dim))
        self.te = nn.Parameter(s*torch.randn(num_patches,model_dim))
        self.blocks = nn.ModuleList([FactorizedTFBlock(model_dim,heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(model_dim)
        self.out = nn.Linear(model_dim, output_dim)

    def forward(self, s):
        B,C,Fq,P = s.shape
        if C != 1 or Fq != self.n_freq or P != self.num_patches:
            raise ValueError(f"Bad scalogram {tuple(s.shape)}")
        x = self.inp(s.permute(0,2,3,1)) + self.fe[None,:,None] + self.te[None,None]
        for b in self.blocks: x = b(x)
        return self.out(self.norm(x).mean(1))


class ConvScalogramEncoder(nn.Module):
    """Cheap capacity/mechanism control."""
    def __init__(self, n_freq, num_patches, output_dim=768, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,hidden,3,padding=1), nn.GELU(),
            nn.Conv2d(hidden,hidden,3,padding=1), nn.GELU()
        )
        self.proj = nn.Linear(hidden,output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self,s):
        x = self.net(s).mean(2).transpose(1,2)
        return self.norm(self.proj(x))


class MLPProjector(nn.Module):
    def __init__(self, i,h,o):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(i),nn.Linear(i,h),nn.GELU(),nn.Linear(h,o))
    def forward(self,x): return self.net(x)


class WaveletFusion(nn.Module):
    MODES={"none","gated_add","concat_mlp","cross_attn"}
    def __init__(self,width,mode="gated_add",heads=8):
        super().__init__()
        if mode not in self.MODES: raise ValueError(mode)
        self.mode=mode
        if mode=="gated_add":
            self.gain=nn.Parameter(torch.tensor(0.0))
        elif mode=="concat_mlp":
            self.mlp=nn.Sequential(
                nn.LayerNorm(2*width),nn.Linear(2*width,width),nn.GELU(),nn.Linear(width,width)
            )
            nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
        elif mode=="cross_attn":
            if width % heads: raise ValueError("fusion heads must divide width")
            self.qn=nn.LayerNorm(width); self.kn=nn.LayerNorm(width)
            self.attn=nn.MultiheadAttention(width,heads,batch_first=True)
            self.gain=nn.Parameter(torch.tensor(0.0))

    def forward(self,raw,wave):
        if self.mode=="none": return raw
        if self.mode=="gated_add": return raw + self.gain*wave
        if self.mode=="concat_mlp": return raw + self.mlp(torch.cat([raw,wave],-1))
        q,k=self.qn(raw),self.kn(wave)
        return raw + self.gain*self.attn(q,k,k,need_weights=False)[0]


class ECGDelineationHead(nn.Module):
    """One branch: 4 semantic classes + 6 supervised boundary heatmaps."""
    def __init__(self,width=768,patch_size=25,hidden=96,kernel=15,predict_fiducials=True):
        super().__init__()
        if kernel%2==0: raise ValueError("kernel must be odd")
        self.patch_size,self.hidden,self.num_fiducials=patch_size,hidden,6
        self.expand=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,hidden*patch_size),nn.GELU())
        p=kernel//2
        self.refine=nn.Sequential(
            nn.Conv1d(hidden,hidden,kernel,padding=p),nn.GELU(),
            nn.Conv1d(hidden,hidden,kernel,padding=p),nn.GELU()
        )
        self.seg=nn.Conv1d(hidden,4,1)
        self.fid=nn.Conv1d(hidden,self.num_fiducials,1) if predict_fiducials else None

    def forward(self,grid):
        B,L,P,W=grid.shape
        x=self.expand(grid).reshape(B,L,P,self.patch_size,self.hidden)
        x=x.reshape(B*L,P*self.patch_size,self.hidden).transpose(1,2).contiguous()
        # cuDNN 9.x can abort with ``ptrDesc->finalize()`` for this legal
        # high-batch Conv1d shape on the installed A100 stack.  Restrict the
        # workaround to this small head; ATen's CUDA convolution remains used.
        if x.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                x=self.refine(x)
                seg=self.seg(x).reshape(B,L,4,P*self.patch_size)
                fid=None if self.fid is None else self.fid(x).reshape(B,L,self.num_fiducials,P*self.patch_size)
        else:
            x=self.refine(x)
            seg=self.seg(x).reshape(B,L,4,P*self.patch_size)
            fid=None if self.fid is None else self.fid(x).reshape(B,L,self.num_fiducials,P*self.patch_size)
        return seg,fid


_PARENT = AliTokECGAIMSpatial if AliTokECGAIMSpatial is not None else AliTokECGAIM

class AliTokECGAIMWaveletMTL(_PARENT):
    """Current ECG-AIM + wavelet SSL + one delineation branch."""
    def __init__(
        self, *, target_len=5000, patch_size=25, width=768, encoder_depth=8,
        decoder_depth=4, heads=12, missing_lead_weight=1.0,
        random_mask_ratio=0.5, temporal_mask_ratio=0.25, consistency_weight=0.05,
        lead_conditioning_mode="learned", use_relative_geometry=False,
        use_spatial_film=False, spatial_gain_init=0.1, geometry_control="standard",
        use_wavelet_branch=True, wavelet_bank="morlet", custom_wavelet_asset=None,
        view_a_bank="inherit", view_b_bank="inherit",
        view_a_custom_wavelet_asset=None, view_b_custom_wavelet_asset=None,
        n_scales=32, min_freq_hz=0.5, max_freq_hz=45.0, morlet_cycles=6.0,
        view_a="magnitude", view_b="phase_sin", wavelet_encoder="timesformer",
        wavelet_dim=192, wavelet_depth=2, wavelet_heads=6, wavelet_conv_hidden=96,
        wavelet_fusion="gated_add", fusion_heads=8, inference_view="a",
        ssl_mode="both", ssl_projector_hidden=512, ssl_projector_dim=256,
        ssl_predictor_hidden=512, byol_tau=0.996, use_delineation_head=True,
        delineation_hidden=96, delineation_kernel=15, predict_fiducials=True,
        mask_type_mode="legacy"
    ):
        if ssl_mode not in {"none","global","local","both"}: raise ValueError(ssl_mode)
        if inference_view not in {"a","b","mean"}: raise ValueError(inference_view)
        if mask_type_mode not in {"legacy","all_masked"}: raise ValueError(mask_type_mode)

        kw=dict(
            target_len=target_len,patch_size=patch_size,width=width,
            encoder_depth=encoder_depth,decoder_depth=decoder_depth,heads=heads,
            missing_lead_weight=missing_lead_weight,random_mask_ratio=random_mask_ratio,
            temporal_mask_ratio=temporal_mask_ratio,consistency_weight=consistency_weight
        )
        if AliTokECGAIMSpatial is not None:
            kw.update(
                lead_conditioning_mode=lead_conditioning_mode,
                use_relative_geometry=use_relative_geometry,
                use_spatial_film=use_spatial_film,
                spatial_gain_init=spatial_gain_init, geometry_control=geometry_control
            )
        super().__init__(**kw)
        self.architecture="ecg_aim_wavelet_mtl_v1"
        self.use_wavelet_branch=bool(use_wavelet_branch)
        self.view_a,self.view_b=view_a,view_b
        self.ssl_mode,self.byol_tau=ssl_mode,float(byol_tau)
        self.inference_view=inference_view
        self.mask_type_mode=mask_type_mode

        if self.use_wavelet_branch:
            def make_bank(kind, asset):
                kind = wavelet_bank if kind == "inherit" else kind
                asset = custom_wavelet_asset if asset is None else asset
                if kind=="morlet":
                    return AnalyticMorletFilterBank(
                    target_len=target_len,n_scales=n_scales,min_freq_hz=min_freq_hz,
                    max_freq_hz=max_freq_hz,cycles=morlet_cycles
                    )
                if kind=="ecg_admissible_morlet":
                    return ECGAdmissibleMorletFilterBank(
                    target_len=target_len,n_scales=n_scales,min_freq_hz=min_freq_hz,
                    max_freq_hz=max_freq_hz,cycles=morlet_cycles
                    )
                if kind=="custom_asset":
                    if not asset: raise ValueError("custom wavelet asset required for selected view")
                    return FixedComplexKernelBank(asset,target_len)
                raise ValueError(kind)

            bank_a=make_bank(view_a_bank,view_a_custom_wavelet_asset)
            bank_b=make_bank(view_b_bank,view_b_custom_wavelet_asset)
            if bank_a.n_scales != bank_b.n_scales:
                raise ValueError("view A and B banks must have the same number of filters")
            nf=bank_a.n_scales
            self.views_a=WaveletViewBuilder(bank_a,target_len,target_len//patch_size)
            self.views_b=WaveletViewBuilder(bank_b,target_len,target_len//patch_size)
            if wavelet_encoder=="timesformer":
                self.wavelet_encoder=FactorizedScalogramEncoder(
                    nf,target_len//patch_size,wavelet_dim,width,wavelet_depth,wavelet_heads
                )
            elif wavelet_encoder=="conv":
                self.wavelet_encoder=ConvScalogramEncoder(
                    nf,target_len//patch_size,width,wavelet_conv_hidden
                )
            else: raise ValueError(wavelet_encoder)
            self.fusion=WaveletFusion(width,wavelet_fusion,fusion_heads)
            if ssl_mode!="none":
                self.online_proj=MLPProjector(width,ssl_projector_hidden,ssl_projector_dim)
                self.predictor=MLPProjector(ssl_projector_dim,ssl_predictor_hidden,ssl_projector_dim)
                self.target_encoder=copy.deepcopy(self.wavelet_encoder)
                self.target_proj=copy.deepcopy(self.online_proj)
                for m in (self.target_encoder,self.target_proj):
                    for p in m.parameters(): p.requires_grad=False
                    m.eval()
            else:
                self.online_proj=self.predictor=self.target_encoder=self.target_proj=None
        else:
            self.views_a=self.views_b=self.wavelet_encoder=None
            self.fusion=WaveletFusion(width,"none")
            self.online_proj=self.predictor=self.target_encoder=self.target_proj=None

        self.delineation_head = ECGDelineationHead(
            width,patch_size,delineation_hidden,delineation_kernel,predict_fiducials
        ) if use_delineation_head else None

    def train(self,mode=True):
        super().train(mode)
        if self.target_encoder is not None: self.target_encoder.eval()
        if self.target_proj is not None: self.target_proj.eval()
        return self

    @torch.no_grad()
    def update_byol_target(self,tau=None):
        if self.target_encoder is None: return
        tau=self.byol_tau if tau is None else float(tau)
        for a,b in zip(self.wavelet_encoder.parameters(),self.target_encoder.parameters()):
            b.data.mul_(tau).add_(a.data,alpha=1-tau)
        for a,b in zip(self.online_proj.parameters(),self.target_proj.parameters()):
            b.data.mul_(tau).add_(a.data,alpha=1-tau)

    def _points(self,mask,T):
        return mask.repeat_interleave(self.patch_size,-1)[...,:T]

    def _source(self,normalized,patch_mask):
        vis=(~self._points(patch_mask,normalized.shape[-1])).to(normalized.dtype)
        return (normalized*vis).sum(1)/vis.sum(1).clamp_min(1.0)

    def _fusion_source(self,signal,patch_mask):
        """Normalize and pool using visible samples only.

        The frozen parent normalizes its raw reconstruction path with the full
        observed lead.  Reusing that scale here would leak artificially hidden
        amplitudes into the wavelet fusion branch even after zero masking.
        """
        vis=(~self._points(patch_mask,signal.shape[-1])).to(signal.dtype)
        denom=vis.sum((1,2),keepdim=True).clamp_min(1.0)
        scale=torch.sqrt((signal.square()*vis).sum((1,2),keepdim=True)/denom).clamp_min(1e-3)
        return self._source(signal/scale,patch_mask)

    def _view(self,src,which):
        if which=="a": return self.views_a(src,self.view_a)
        if which=="b": return self.views_b(src,self.view_b)
        raise ValueError(which)

    def _wave(self,src,which):
        return self.wavelet_encoder(self._view(src,which))

    def _fusion_wave(self,src):
        if self.inference_view=="a": return self._wave(src,"a")
        if self.inference_view=="b": return self._wave(src,"b")
        return .5*(self._wave(src,"a")+self._wave(src,"b"))

    def _cos(self,a,b):
        return 2-2*F.cosine_similarity(a,b.detach(),dim=-1).mean()

    def _ssl(self,src):
        if self.ssl_mode=="none":
            z=src.new_zeros(()); return z,z,z
        va,vb=self._view(src,"a"),self._view(src,"b")
        oa,ob=self.wavelet_encoder(va),self.wavelet_encoder(vb)
        with torch.no_grad():
            ta,tb=self.target_encoder(va),self.target_encoder(vb)
        gl=src.new_zeros(()); ll=src.new_zeros(())
        if self.ssl_mode in {"global","both"}:
            pga=self.predictor(self.online_proj(oa.mean(1)))
            pgb=self.predictor(self.online_proj(ob.mean(1)))
            with torch.no_grad():
                tga=self.target_proj(ta.mean(1)); tgb=self.target_proj(tb.mean(1))
            gl=.5*(self._cos(pga,tgb)+self._cos(pgb,tga))
        if self.ssl_mode in {"local","both"}:
            pla=self.predictor(self.online_proj(oa)); plb=self.predictor(self.online_proj(ob))
            with torch.no_grad():
                tla=self.target_proj(ta); tlb=self.target_proj(tb)
            ll=.5*(self._cos(pla,tlb)+self._cos(plb,tla))
        return gl+ll,gl,ll

    def _decode_with_grid(self,normalized,inherited,artificial,fusion_source=None):
        patches=self._patchify(normalized)
        available=~(inherited|artificial)
        try: cond=self._lead_condition(inherited)
        except TypeError: cond=self.lead_embedding[None].expand(normalized.shape[0],-1,-1)
        tokens=self.patch_projection(patches)+cond[:,:,None]+self.time_embedding[None,None]
        tokens=tokens+self.mask_type_embedding[0][None,None,None]

        # HARD leakage rule: fusion sees artificial masks as zero/unavailable.
        if self.use_wavelet_branch:
            if fusion_source is None:
                raise ValueError("wavelet reconstruction requires a visible-only fusion source")
            src=fusion_source
            wave=self._fusion_wave(src)
            observed=(~inherited[:,:,0])
            fused_leads=[]
            for l in range(12):
                if bool(observed[:,l].any()):
                    fused=self.fusion(tokens[:,l],wave)
                    fused=torch.where(observed[:,l,None,None],fused,tokens[:,l])
                else:
                    fused=tokens[:,l]
                fused_leads.append(fused)
            tokens=torch.stack(fused_leads,dim=1)

        try: grid,memory=self._encode_grid(tokens,available,cond)
        except TypeError: grid,memory=self._encode_grid(tokens,available)

        mt=inherited if self.mask_type_mode=="legacy" else (inherited|artificial)
        grid=grid+torch.where(
            mt.unsqueeze(-1),self.mask_type_embedding[1][None,None,None],
            self.mask_type_embedding[0][None,None,None]
        )
        baseline=self._limb_prior(patches,available)
        grid=grid+self.baseline_projection(baseline)
        for block in self.decoder:
            try: grid=block(grid,lead_condition=cond)
            except TypeError: grid=block(grid)
        residual=self.patch_head(self.output_norm(grid))
        pred=baseline+self.residual_gain*residual
        return self._unpatchify(pred),memory,grid

    def forward(
        self,x,target=None,y_full=None,lead_indices=None,compute_delineation=True,
        compute_ssl=True,**_
    ):
        target=target if target is not None else y_full
        if target is None: target=x
        inherited=~self._lead_mask(x,lead_indices)
        artificial=self._artificial_mask(inherited)
        scale=self._scale(x,inherited)
        fusion_source=(
            self._fusion_source(x,inherited|artificial) if self.use_wavelet_branch else None
        )
        predn,memory,grid=self._decode_with_grid(
            x/scale,inherited,artificial,fusion_source=fusion_source
        )
        pred=predn*scale
        dec,art=self._masked_loss(pred,target,inherited,artificial)
        consistency=self._limb_consistency(pred)

        if self.use_wavelet_branch and compute_ssl:
            # SSL may use full genuinely observed source, but these tokens are
            # NOT routed into the reconstruction grid.
            ssl,sg,sl=self._ssl(self._source(x/scale,inherited))
        else:
            ssl=sg=sl=pred.new_zeros(())

        seg=fid=None
        if compute_delineation and self.delineation_head is not None:
            seg,fid=self.delineation_head(grid)
        base=dec+self.consistency_weight*consistency
        zero=base.new_zeros(())
        latent=memory.transpose(1,2)
        return {
            "loss":base,"decoder_loss":dec.detach(),"kl_loss":zero.detach(),
            "teacher_loss":zero.detach(),"align_loss":zero.detach(),
            "stft_loss":zero.detach(),"diff_loss":zero.detach(),"corr_loss":zero.detach(),
            "fm_perceptual_loss":zero.detach(),"prefix_aux_loss":art.detach(),
            "latent_align_loss":zero.detach(),"codebook_perplexity":zero.detach(),
            # Keep this live: the external multi-task trainer composes its own
            # reconstruction criterion and must still be able to optimize the
            # configured Einthoven/Goldberger consistency regularizer.
            "limb_consistency_loss":consistency,
            "wavelet_ssl_loss":ssl,"wavelet_ssl_global_loss":sg.detach(),
            "wavelet_ssl_local_loss":sl.detach(),"y_target":target,"y_pred":pred,
            "y_pred_reg":pred,"z_regressed":latent,"log_var_regressed":torch.zeros_like(latent),
            "grid_features":grid,"seg_logits":seg,"fiducial_logits":fid,
        }

    @torch.no_grad()
    def impute_from_regressor(self,x,lead_indices=None):
        inherited=~self._lead_mask(x,lead_indices); artificial=torch.zeros_like(inherited)
        scale=self._scale(x,inherited)
        fusion_source=(self._fusion_source(x,inherited) if self.use_wavelet_branch else None)
        predn,memory,grid=self._decode_with_grid(
            x/scale,inherited,artificial,fusion_source=fusion_source
        )
        pred=predn*scale; seg=fid=None
        if self.delineation_head is not None: seg,fid=self.delineation_head(grid)
        lat=memory.transpose(1,2)
        return {"available":True,"y_pred":pred,"z_latent":lat,"log_var":torch.zeros_like(lat),
                "seg_logits":seg,"fiducial_logits":fid}


def build_wavelet_ecg_aim(**kwargs): return AliTokECGAIMWaveletMTL(**kwargs)


def _run_self_tests_impl(device=None):
    """Synthetic tests; use a reduced model so CPU test is quick."""
    dev=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(7)
    kw=dict(target_len=500,patch_size=25,width=96,encoder_depth=2,decoder_depth=1,heads=4)
    m=AliTokECGAIMWaveletMTL(
        **kw,n_scales=8,min_freq_hz=1,max_freq_hz=40,wavelet_dim=48,
        wavelet_depth=1,wavelet_heads=4,delineation_hidden=32,delineation_kernel=9,
        use_wavelet_branch=True,ssl_mode="both"
    ).to(dev).train()
    y=torch.randn(2,12,500,device=dev)*.1
    x=torch.zeros_like(y); x[:,0]=y[:,0]
    li=torch.zeros(2,1,dtype=torch.long,device=dev)
    o=m(x,y_full=y,lead_indices=li)
    assert o["y_pred"].shape==(2,12,500)
    assert o["seg_logits"].shape==(2,12,4,500)
    assert o["fiducial_logits"].shape==(2,12,6,500)
    assert torch.isfinite(o["wavelet_ssl_loss"])
    (o["loss"]+.05*o["wavelet_ssl_loss"]).backward()
    assert any(p.grad is not None for p in m.wavelet_encoder.parameters())
    assert all(p.grad is None for p in m.target_encoder.parameters())
    before=next(m.target_encoder.parameters()).detach().clone()
    with torch.no_grad(): next(m.wavelet_encoder.parameters()).add_(.01)
    m.update_byol_target(.5)
    assert not torch.equal(before,next(m.target_encoder.parameters()).detach())
    bank=AnalyticMorletFilterBank(target_len=500,n_scales=8,min_freq_hz=1,max_freq_hz=40).to(dev)
    s=torch.randn(1,500,device=dev)
    assert torch.equal(bank(s),bank(s))

    # G4: a wavelet-disabled A0 must preserve the current spatial-parent
    # reconstruction path exactly after loading the same base weights.
    base=AliTokECGAIMSpatial(**kw).to(dev).eval()
    a0=AliTokECGAIMWaveletMTL(
        **kw,use_wavelet_branch=False,use_delineation_head=False,ssl_mode="none"
    ).to(dev).eval()
    missing,unexpected=a0.load_state_dict(base.state_dict(),strict=False)
    assert not missing and not unexpected,(missing,unexpected)
    with torch.no_grad():
        base_pred=base(x,y_full=y,lead_indices=li)["y_pred"]
        a0_pred=a0(x,y_full=y,lead_indices=li)["y_pred"]
    torch.testing.assert_close(a0_pred,base_pred,rtol=0,atol=0)

    # G5: changing samples hidden by an artificial mask must not alter the
    # wavelet features that are routed into reconstruction.
    inherited=torch.ones(1,12,m.num_patches,dtype=torch.bool,device=dev)
    inherited[:,0]=False
    artificial=torch.zeros_like(inherited); artificial[:,0,2]=True
    source=torch.randn(1,12,m.target_len,device=dev)
    changed=source.clone()
    start=2*m.patch_size; changed[:,0,start:start+m.patch_size]+=1000
    with torch.no_grad():
        fused_before=m._fusion_wave(m._fusion_source(source,inherited|artificial))
        fused_after=m._fusion_wave(m._fusion_source(changed,inherited|artificial))
    torch.testing.assert_close(fused_after,fused_before,rtol=0,atol=0)
    return {"device":str(dev),"shape":"PASS","finite_ssl":"PASS","wave_grad":"PASS",
            "target_no_grad":"PASS","ema":"PASS","deterministic_bank":"PASS",
            "a0_parity":"PASS","artificial_mask_leakage":"PASS"}


def run_self_tests(device=None):
    """Run the deterministic gate without contending for the host BLAS pool.

    The full-size training jobs deliberately use the regular backend settings.
    This small CPU witness uses one thread and disables oneDNN so it remains
    reliable while unrelated CPU/GPU experiments are active on the same host.
    """
    previous_threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.backends.mkldnn.flags(enabled=False):
            return _run_self_tests_impl(device)
    finally:
        torch.set_num_threads(previous_threads)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true"); p.add_argument("--device")
    a=p.parse_args()
    if a.self_test: print(json.dumps(run_self_tests(a.device),indent=2))
    else: p.print_help()
