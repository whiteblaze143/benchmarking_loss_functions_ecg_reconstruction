
import torch
import torch.nn as nn
import numpy as np

class PatchEmbed(nn.Module):
    """ Time-Series Patch Embedding Layer """
    def __init__(self, img_size=5000, patch_size=100, in_chans=1, embed_dim=256):
        super().__init__()
        self.num_patches = img_size // patch_size
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, 1, L]
        x = self.proj(x)  # [B, E, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, E]
        return x

from util_functions.frequency import FrequencyPartitioner

class FaMAE(nn.Module):
    """
    Fa-MAE: Frequency-aware Masked Autoencoder for ECG.
    Input: [B, 1, L]
    Output: [B, 1, L], reconstruction + mask
    """
    def __init__(self, seq_len=5000, patch_size=50, embed_dim=256, depth=12, num_heads=8, decoder_depth=4, decoder_embed_dim=128, mask_ratio=0.75):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio

        # Frequency Analysis
        self.freq_analyzer = FrequencyPartitioner(fs=250)
        # Biases for masking: Higher bias = More likely to be masked (pushed to end of argsort)
        # We want to mask Mid (QRS) more (60%), and Low (P/T) less (40%).
        # Base random is [0, 1].
        # If we add 0.2 to Mid, its range is [0.2, 1.2].
        # If we subtract 0.2 from Low, its range is [-0.2, 0.8].
        # Patches with lower values are KEPT.
        # So Low patches (smaller values) are kept more often.
        self.freq_bias = {0: -0.2, 1: 0.2, 2: 0.0} # 0=Low, 1=Mid, 2=High

        # 1. Patch Embedding
        self.input_norm = nn.InstanceNorm1d(1, affine=False) # Standardize amplitude per segment
        self.patch_embed = PatchEmbed(img_size=seq_len, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # 3. Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Decoder Projection
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim))

        # 5. Decoder
        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_embed_dim, nhead=num_heads // 2, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # 6. Reconstruction Head
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # pos_embed initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def frequency_aware_masking(self, x, mask_ratio):
        """
        Masking strategy that biases towards keeping Low-Frequency (P/T wave) patches
        and masking Mid-Frequency (QRS) patches.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # 1. Base noise
        noise = torch.rand(N, L, device=x.device)  # [0, 1]
        
        # 2. Compute Band info
        # We need raw signal for this? 'x' here is embeddings.
        # We don't have raw signal in this function signature.
        # Modified forward_encoder to pass 'raw_patches' or dominant bands?
        # For now, fallback to random if we can't easily access.
        # BUT we can modify the pipeline to calculate bands BEFORE embedding.
        # See forward_encoder modification.
        
        return self.random_masking(x, mask_ratio) # Placeholder, overridden in forward with raw-aware logic?

    def random_masking_biased(self, x, mask_ratio, dominant_bands):
        """
        x: [N, L, D]
        dominant_bands: [N, L] indices (0, 1, 2)
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        
        # Add bias
        # 0 (Low) -> -0.2
        # 1 (Mid) -> +0.2
        # 2 (High) -> 0.0
        bias = torch.zeros_like(noise)
        bias[dominant_bands == 0] = self.freq_bias[0]
        bias[dominant_bands == 1] = self.freq_bias[1]
        bias[dominant_bands == 2] = self.freq_bias[2]
        
        noise = noise + bias
        
        # Sort (Smallest values are kept)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def interpolate_pos_encoding(self, x, w, h):
        """
        Interpolate pos_embed to match input size.
        """
        npatch = w
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N:
            return self.pos_embed
            
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=npatch,
            mode='linear',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def interpolate_decoder_pos_encoding(self, x, ids_restore):
        npatch = ids_restore.shape[1]
        N = self.decoder_pos_embed.shape[1] - 1
        
        if npatch == N:
            return self.decoder_pos_embed

        class_pos_embed = self.decoder_pos_embed[:, 0]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]
        
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=npatch,
            mode='linear',
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)
        
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_encoder(self, x):
        # x: [B, 1, L]
        
        # 0. Analyze Frequencies (on normalized input?)
        # Instance Norm is part of model, so we apply it first.
        x_norm = self.input_norm(x)
        
        dominant_bands = self.freq_analyzer.get_dominant_band_indices(x_norm, self.patch_size) # [B, N]
        
        # 1. Patch Embed
        x_embed = self.patch_embed(x_norm)  # [B, N, D]
        
        # interpolate pos embed
        pos_embed = self.interpolate_pos_encoding(x_embed, x_embed.shape[1], 0)
        x_embed = x_embed + pos_embed[:, 1:, :]

        # 2. Masking (Biased)
        x_masked, mask, ids_restore = self.random_masking_biased(x_embed, self.mask_ratio, dominant_bands)

        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_final = torch.cat((cls_tokens, x_masked), dim=1)

        # apply Transformer
        x_out = self.encoder(x_final)
        
        return x_out, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        pos_embed = self.interpolate_decoder_pos_encoding(x, ids_restore)
        x = x + pos_embed
        x = self.decoder(x)
        x = self.decoder_pred(x)
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, L] (BP Filtered)
        pred: [N, num_patches, patch_size]
        """
        # Ensure target is normalized to match pred space
        imgs_norm = self.input_norm(imgs)
        target = self.patchify(imgs_norm)
        
        # 1. MSE Loss (Time Domain)
        loss_mse = (pred - target) ** 2
        loss_mse = loss_mse.mean(dim=-1)
        loss_main = (loss_mse * mask).sum() / mask.sum()

        # 2. Frequency / Morphology Loss (on Reconstruction)
        pred_full = self.unpatchify(pred) # [N, 1, L]
        
        # FFT of Pred vs Target
        fft_pred = torch.fft.rfft(pred_full, dim=-1, norm='ortho')
        fft_target = torch.fft.rfft(imgs_norm, dim=-1, norm='ortho')
        
        # Magnitude Spectrums
        mag_pred = torch.abs(fft_pred)
        mag_target = torch.abs(fft_target)
        
        # Use precise bands from freq_analyzer
        freqs = torch.fft.rfftfreq(imgs.shape[-1], d=1/250.0).to(imgs.device)
        b = self.freq_analyzer.bands
        mask_low = (freqs >= b['low'][0]) & (freqs < b['low'][1])
        mask_mid = (freqs >= b['mid'][0]) & (freqs <= b['mid'][1])
        mask_high = (freqs > b['high'][0])
        
        # Weights (mEcgNet logic: Low=P/T, Mid=QRS)
        w_low = 1.1 
        w_mid = 0.5 
        w_high = 0.5 
        
        loss_freq_low = (mag_pred[:, :, mask_low] - mag_target[:, :, mask_low]).abs().mean()
        loss_freq_mid = (mag_pred[:, :, mask_mid] - mag_target[:, :, mask_mid]).abs().mean()
        loss_freq_high = (mag_pred[:, :, mask_high] - mag_target[:, :, mask_high]).abs().mean()
        
        loss_freq_total = w_low * loss_freq_low + w_mid * loss_freq_mid + w_high * loss_freq_high
        
        # Rescale Spectral Loss to match MSE magnitude (~0.3-0.8)
        # Normalizing by log or similar could work, but simple scaling is faster.
        # We target a 0.2 contribution as per PRD.
        total_loss = loss_main + 0.01 * loss_freq_total # Reduced from 0.2 to 0.01 due to spectral magnitude
        
        return total_loss

    def patchify(self, imgs):
        p = self.patch_size
        h = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p))
        return x

    def unpatchify(self, x):
        p = self.patch_size
        h = x.shape[1]
        imgs = x.reshape(shape=(x.shape[0], 1, h * p))
        return imgs

    def forward(self, imgs):
        # 1. Band-pass Filter (Section 3.1: 0.05 - 150 Hz)
        imgs = self.freq_analyzer.bandpass_filter(imgs)
        
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
