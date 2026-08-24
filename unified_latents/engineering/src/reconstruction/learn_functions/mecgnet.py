import torch
import torch.nn as nn
import torch.nn.functional as F

class GConvMLPBlock(nn.Module):
    """
    MLP block using Grouped Conv2D followed by/preceded by a Linear layer
    to compensate for limited feature integration across groups (mEcgNet Section 2.2).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, dropout=0.1, mode='encoder'):
        super().__init__()
        self.mode = mode
        if mode == 'encoder':
            # Linear layer first to facilitate feature integration
            self.linear = nn.Linear(in_channels, in_channels)
            self.gconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        else:
            # GConv first
            self.gconv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
            self.linear = nn.Linear(out_channels, out_channels)
            
        self.norm = nn.BatchNorm2d(out_channels if mode == 'encoder' else out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.mode == 'encoder':
            # x: [B, C, H, W]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1) # [B, H, W, C]
            x = self.linear(x)
            x = x.permute(0, 3, 1, 2) # [B, C, H, W]
            x = self.gconv(x)
        else:
            x = self.gconv(x)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = self.linear(x)
            x = x.permute(0, 3, 1, 2)
            
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x

class EcgModule(nn.Module):
    """
    Base Modular Reconstruction block (mEcgNet Section 2.2).
    U-Net style architecture for 12x512 signal processing.
    Enhanced with latent conditioning (Z) from Fa-MAE.
    """
    def __init__(self, variant='L', in_ch=1, latent_dim=256):
        super().__init__()
        self.variant = variant
        self.latent_dim = latent_dim
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=(2, 4), stride=(2, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(2, 4), stride=(2, 2), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        if variant == 'S':
            # 3x128 -> 1x32 (approx)
            self.enc3 = GConvMLPBlock(128, 256, kernel_size=(3, 4), stride=(3, 4), padding=(0, 1), groups=64, mode='encoder')
        else:
            self.enc3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(3, 4), stride=(3, 4), padding=(0, 1)),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        # Latent Conditioning: Projects Z to match bottleneck channels
        self.z_proj = nn.Linear(latent_dim, 256)
            
        # Decoder
        if variant == 'S':
            self.dec1 = GConvMLPBlock(256, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), groups=4, mode='decoder')
        else:
            self.dec1 = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=(3, 4), stride=(3, 4), padding=(0, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
            
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 4), stride=(2, 2), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final block: Reconstruct 1 channel with height=12 (Leads)
        self.dec3 = nn.ConvTranspose2d(64, 1, kernel_size=(2, 4), stride=(2, 2), padding=(0, 1))

    def forward(self, x, z=None):
        # x: [B, 1, H, W] where H is usually 12
        # z: [B, latent_dim]
        B, C, H, W = x.shape
        if H == 1:
            x = x.repeat(1, 1, 12, 1)
            
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2) # [B, 256, 1, W_bottleneck]
        
        if z is not None:
            # Inject conditioning via addition at bottleneck
            z_feat = self.z_proj(z).unsqueeze(-1).unsqueeze(-1) # [B, 256, 1, 1]
            e3 = e3 + z_feat
        
        d1 = self.dec1(e3)
        d2 = self.dec2(d1)
        out = self.dec3(d2) # [B, 1, 12, W_out]
        
        # Ensure output length matches input length W
        if out.shape[3] != W:
            out = F.interpolate(out, size=(12, W), mode='bilinear', align_corners=False)
        
        return out.squeeze(1) # [B, 12, W]


class mEcgNet(nn.Module):
    """
    The full mEcgNet architecture: Frequency Partitioning + Cascaded Modules.
    """
    def __init__(self, fs=250, latent_dim=256):
        super().__init__()
        from util_functions.frequency import FrequencyPartitioner
        self.partitioner = FrequencyPartitioner(fs=fs)
        
        self.module_low = EcgModule(variant='S', latent_dim=latent_dim)
        self.module_mid = EcgModule(variant='S', latent_dim=latent_dim)
        self.module_high = EcgModule(variant='L', latent_dim=latent_dim)

    def forward(self, x, z=None):
        # x: [B, 1, L]
        # z: [B, latent_dim] - Optional conditioning from Fa-MAE
        
        # Partition
        low, mid, high = self.partitioner.decompose(x)
        
        # Cascade
        out_low = self.module_low(low.unsqueeze(1), z=z) # [B, 12, L]
        
        # 2nd module takes (mid + out_low)
        mid_rep = mid.unsqueeze(1).repeat(1, 1, 12, 1)
        out_mid = self.module_mid(mid_rep + out_low.unsqueeze(1), z=z)
        
        # 3rd module takes (high + out_mid)
        high_rep = high.unsqueeze(1).repeat(1, 1, 12, 1)
        out_final = self.module_high(high_rep + out_mid.unsqueeze(1), z=z)
        
        return out_final
