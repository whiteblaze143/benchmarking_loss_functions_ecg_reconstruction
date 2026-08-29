import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.train_mcma_3lead import MCMAModel

class FoundationPrior(nn.Module):
    """Universal interface for frozen foundation models."""
    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dummy_param = nn.Parameter(torch.randn(1))
        
        # Preprocessing contract
        self.sampling_rate = 500
        self.input_length = 5000
        self.normalization = "standard"
        self.supports_native_single_lead = False
        self.preserves_lead_identity = False
        
    def _freeze(self):
        for param in self.parameters():
            if param is not self.dummy_param:
                param.requires_grad = False
                
    def preprocess(self, x, fs, lead_id):
        """
        Handle FM-specific resampling, amplitude normalization, 
        and spatial padding without mutating the 500 Hz base signal.
        """
        # Default pass-through
        return x
                
    def encode(self, x, lead_id, layer="middle"):
        """
        Extract features from the foundation model.
        Returns:
            dict: { "temporal": H, "global": z, "layer_id": {"temporal": ..., "global": ...} }
                where H is [B, Tf, D] and z is [B, D]
        """
        x_processed = self.preprocess(x, fs=500, lead_id=lead_id)
        
        B = x_processed.shape[0]
        device = x_processed.device
        
        # Stub implementation generating fake varied tensors for smoke tests
        H = torch.randn(B, self.seq_len, self.embed_dim, device=device)
        z = torch.randn(B, self.embed_dim, device=device)
        
        return {
            "temporal": H,
            "global": z,
            "layer_id": {
                "temporal": "middle",
                "global": "late"
            }
        }
        
    def forward(self, x, lead_id=None, layer="middle"):
        return self.encode(x, lead_id, layer)

class STMEM_Prior(FoundationPrior):
    def __init__(self, embed_dim=768, seq_len=196):
        super().__init__(embed_dim, seq_len)
        self.sampling_rate = 250
        self.supports_native_single_lead = True
        self.preserves_lead_identity = True
        self._freeze()
        
    def preprocess(self, x, fs, lead_id):
        # Stub for 500Hz -> 250Hz resampling
        if fs != self.sampling_rate:
            # simple linear downsample for stub purposes
            return F.interpolate(x, scale_factor=0.5, mode='linear', align_corners=False)
        return x

class MERL_Prior(FoundationPrior):
    def __init__(self, embed_dim=512, seq_len=196):
        super().__init__(embed_dim, seq_len)
        self._freeze()

class KED_Prior(FoundationPrior):
    def __init__(self, embed_dim=512, seq_len=196):
        super().__init__(embed_dim, seq_len)
        self._freeze()
        
class ECGFounder_Prior(FoundationPrior):
    def __init__(self, embed_dim=512, seq_len=196):
        super().__init__(embed_dim, seq_len)
        self._freeze()

class HuBERTECG_Prior(FoundationPrior):
    def __init__(self, embed_dim=768, seq_len=312):
        super().__init__(embed_dim, seq_len)
        self._freeze()

class ECGFM_Prior(FoundationPrior):
    def __init__(self, embed_dim=768, seq_len=312):
        super().__init__(embed_dim, seq_len)
        self._freeze()

class CSFM_Prior(FoundationPrior):
    def __init__(self, embed_dim=256, seq_len=200):
        super().__init__(embed_dim, seq_len)
        self.supports_native_single_lead = True
        self.preserves_lead_identity = True
        self._freeze()

class LeadConditioner(nn.Module):
    """
    Generates target-lead FiLM parameters from global z_FM.
    """
    def __init__(self, fm_dim, target_leads=12, spatial_dim=16):
        super().__init__()
        self.target_leads = target_leads
        self.spatial_dim = spatial_dim
        
        self.lead_emb = nn.Embedding(target_leads, 32)
        
        self.mlp = nn.Sequential(
            nn.Linear(fm_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, spatial_dim * 2)
        )
        
        # Zero-initialize the final projection so gamma=0, beta=0 at init
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        
    def forward(self, z_fm, spatial_features):
        B = z_fm.shape[0]
        device = z_fm.device
        
        gamma_list = []
        beta_list = []
        
        for l in range(self.target_leads):
            l_idx = torch.full((B,), l, dtype=torch.long, device=device)
            e_l = self.lead_emb(l_idx)
            c_l = torch.cat([z_fm, e_l], dim=-1)
            
            film_params = self.mlp(c_l)
            gamma, beta = film_params.chunk(2, dim=-1)
            gamma_list.append(gamma)
            beta_list.append(beta)
            
        avg_gamma = torch.stack(gamma_list, dim=0).mean(dim=0).unsqueeze(-1)
        avg_beta = torch.stack(beta_list, dim=0).mean(dim=0).unsqueeze(-1)
        
        return (1 + avg_gamma) * spatial_features + avg_beta

class TemporalAdapter(nn.Module):
    """
    Generates coarse residual correction from temporal H_FM.
    """
    def __init__(self, fm_dim, target_leads=12, fm_seq_len=200, seq_len=5000):
        super().__init__()
        self.target_leads = target_leads
        self.fm_seq_len = fm_seq_len
        self.seq_len = seq_len
        
        self.lead_emb = nn.Embedding(target_leads, 32)
        
        self.adapter_mlp = nn.Sequential(
            nn.Linear(fm_dim + 32, 128),
            nn.ReLU(),
            nn.Linear(128, fm_seq_len)
        )
        
        # Zero-initialize the final projection so Delta X = 0 at init
        nn.init.zeros_(self.adapter_mlp[-1].weight)
        nn.init.zeros_(self.adapter_mlp[-1].bias)
        
    def forward(self, H_fm):
        # H_fm: [B, Tf, D]
        # Pooling for now to simulate adapter projection
        H_pool = H_fm.mean(dim=1) # [B, D]
        
        B = H_pool.shape[0]
        device = H_pool.device
        
        delta_z_list = []
        for l in range(self.target_leads):
            l_idx = torch.full((B,), l, dtype=torch.long, device=device)
            e_l = self.lead_emb(l_idx)
            c_l = torch.cat([H_pool, e_l], dim=-1)
            
            d_z = self.adapter_mlp(c_l)
            delta_z_list.append(d_z.unsqueeze(1))
            
        delta_z = torch.cat(delta_z_list, dim=1)
        delta_x_fm = F.interpolate(delta_z, size=self.seq_len, mode='linear', align_corners=False)
        return delta_x_fm

class MCMAModel_FM_V2(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, fm_class=None, use_film=True, use_residual=True):
        super().__init__()
        self.base_model = MCMAModel(in_channels=in_channels, out_channels=out_channels)
        self.use_film = use_film
        self.use_residual = use_residual
        
        if fm_class is not None:
            self.fm = fm_class()
        else:
            self.fm = CSFM_Prior()
            
        spatial_dim = 16
        if self.use_film:
            self.lead_conditioner = LeadConditioner(
                fm_dim=self.fm.embed_dim,
                target_leads=out_channels,
                spatial_dim=spatial_dim
            )
            
        if self.use_residual:
            self.temporal_adapter = TemporalAdapter(
                fm_dim=self.fm.embed_dim,
                target_leads=out_channels,
                fm_seq_len=self.fm.seq_len,
                seq_len=5120
            )
        
        # Alpha parameter initialized to 0.1 so gradients flow immediately,
        # but zero-init in TemporalAdapter preserves X_hat = X_base at start.
        self.alpha = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, x, shuffle_z_fm=False):
        # 1. FM pathway
        fm_out = self.fm.encode(x, lead_id=None, layer="middle")
        z_fm = fm_out["global"]
        H_fm = fm_out["temporal"]
        
        if shuffle_z_fm:
            B = z_fm.shape[0]
            if B > 1:
                idx = torch.randperm(B, device=z_fm.device)
                z_fm = z_fm[idx]
                H_fm = H_fm[idx]
        
        # 2. Base model local pathway
        bm = self.base_model
        e0 = bm.down0(x)
        e1 = bm.down1(e0)
        e2 = bm.down2(e1)
        e3 = bm.down3(e2)
        e4 = bm.down4(e3)
        e5 = bm.down5(e4)
        
        d4 = bm.up4(e5)
        d4 = bm._match_size(d4, e4)
        d4 = bm.d_conv4(torch.cat([d4, e4], dim=1))
        
        d3 = bm.up3(d4)
        d3 = bm._match_size(d3, e3)
        d3 = bm.d_conv3(torch.cat([d3, e3], dim=1))
        
        d2 = bm.up2(d3)
        d2 = bm._match_size(d2, e2)
        d2 = bm.d_conv2(torch.cat([d2, e2], dim=1))
        
        d1 = bm.up1(d2)
        d1 = bm._match_size(d1, e1)
        d1 = bm.d_conv1(torch.cat([d1, e1], dim=1))
        
        d0 = bm.up0(d1)
        d0 = bm._match_size(d0, e0)
        d0 = bm.d_conv0(torch.cat([d0, e0], dim=1))
        
        # 3. Apply Residual/FiLM
        if self.use_film:
            d0 = self.lead_conditioner(z_fm, d0)
            
        out_base = bm.final_conv(d0)
        
        if self.use_residual:
            delta_x_fm = self.temporal_adapter(H_fm)
            out = out_base + self.alpha * delta_x_fm
        else:
            out = out_base
            
        return out
