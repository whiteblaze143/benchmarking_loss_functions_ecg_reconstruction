import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import math


class SelfAttention(nn.Module):

    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, Seq_len, Dim)
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, sequence_length,
                         self.n_heads, self.d_head)

        # (B, S, D) -> (B, S, D * 3) -> 3 * (B, S, D) NOTE: nn.Linear multiplys last dimension of any given vector
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (B, S, D) -> (B, S, H, D/H) -> (B, H, S, D/H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, H, S, S)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, H, S, S) @ (B, H, S, D/H) -> (B, H, S, D/H)
        output = weight @ v

        # (B, H, S, D/H) -> (B, S, H, D/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (B, S, D)
        return output


class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Features, L)
        residue = x

        # x: (B, Features, L) -> x: (B, L, Features)
        x = x.transpose(-1, -2)

        # x: (B, L, Features) -> x: (B, Features, L)
        x = self.attention(x)
        x = x.transpose(-1, -2)

        x += residue
        return x


class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, In_channels, L)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)


class VAE_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(12, 128, kernel_size=3, padding=1),
                VAE_ResidualBlock(128, 128),
                VAE_ResidualBlock(128, 128),
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(128, 256),
                VAE_ResidualBlock(256, 256),
                nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(256, 512),
                VAE_ResidualBlock(512, 512),
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                # nn.Conv1d(512, 512, kernel_size=5, stride=5, padding=0),
                # VAE_ResidualBlock(512, 512),
                # VAE_ResidualBlock(512, 512),

                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                nn.GroupNorm(32, 512),
                nn.SiLU(),
                nn.Conv1d(512, 128, kernel_size=3, padding=1),
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
            ]
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, C)
        # noise: (B, C_Out, L/8)
        # output: (B, C_Out, L/8)
        # NOTE: apply noise after encoding
        x = x.transpose(1, 2)
        # x: (B, L, C) -> (B, C, L)
        # x = x.transpose(1, 2)
        # x: (B, C, L) -> (B, 8, L/8)
        for module in self.blocks:
            if getattr(module, "stride", None) == (2,):
                # Padding(left, right)
                x = F.pad(x, (0, 1))
            x = module(x)
            # print(x.shape)

        # (B, 8, L/8) -> 2 x (B, 4, L/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # z ~ N(0, 1) -> x ~ N(mean, variance)
        # x = mean + stdev * z
        # (B, 4, L/8) -> (B, 4, L/8)
        if noise is None:
            noise = torch.randn(stdev.shape, device=stdev.device)
        x = mean + stdev * noise

        # Scale the output by a constant (magic number)
        x *= 0.18215

        return x, mean, log_variance


class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Conv1d(64, 64, kernel_size=1, padding=0),
                nn.Conv1d(64, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                # (B, 512, L/8) -> (B, 512, L/4)
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                # (B, 512, L/4) -> (B, 512, L/2)
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                # (B, 256, L/2) -> (B, 256, L)
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),

                # nn.Upsample(scale_factor=5),
                # nn.Conv1d(256, 256, kernel_size=3, padding=1),
                # VAE_ResidualBlock(256, 256),
                # VAE_ResidualBlock(256, 256),
                # VAE_ResidualBlock(256, 256),

                nn.GroupNorm(32, 256),
                nn.SiLU(),
                # (B, 128, L) -> (B, 12, L)
                nn.Conv1d(256, 12, kernel_size=3, padding=1),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L/8)

        x /= 0.18215

        for module in self.blocks:
            x = module(x)
            # print(x.shape)

        # (B, 12, L) -> (B, L, 12)
        x = x.transpose(1, 2)
        return x


def compute_mmd(f1, f2, kernel='rbf', sigma=1.0):
    # f1, f2: (B, D)
    B = f1.size(0)

    def gaussian_kernel(x, y, sigma):
        x = x.unsqueeze(1)  # (B, 1, D)
        y = y.unsqueeze(0)  # (1, B, D)
        # (B, B)
        return torch.exp(-torch.sum((x - y)**2, dim=2) / (2 * sigma**2))

    K_XX = gaussian_kernel(f1, f1, sigma)
    K_YY = gaussian_kernel(f2, f2, sigma)
    K_XY = gaussian_kernel(f1, f2, sigma)

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    return mmd



def loss_function(recons, x, mu, log_var, kld_weight=1e-6, perceptual_weight=1e-1, ecg_founder_model=None, device=None) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :para recons: reconstruction vector
    :para x: original vector
    :para mu: mean of latent gaussian distribution
    :log_var: log of latent gaussian distribution variance
    :kld_weight: weight of kl-divergence term
    :perceptual_weight: weight of perceptual loss term
    :ecg_founder_model: pre-trained ECGFounder model for perceptual loss
    :device: torch device
    """
    # recons, x: (B, L, 12) -> number, batch wise average
    recons_loss = F.mse_loss(recons, x, reduction="mean")

    # q(z|x): distribution learned by encoder
    q_z_x = Normal(mu, log_var.mul(0.5).exp())
    # p(z): prior of z, intended to be standard Gaussian
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    # kld_loss: batch wise average
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()

    perceptual_loss = torch.tensor(0.0, device=device)
    if ecg_founder_model is not None and perceptual_weight > 0:
        recons_permuted = recons.transpose(1, 2)  # (B, 12, L)
        x_permuted = x.transpose(1, 2)       # (B, 12, L)
        with torch.no_grad():
            features_recons_list, _ = ecg_founder_model(recons_permuted)
            features_x_list, _ = ecg_founder_model(x_permuted)
        
        num_feature_layers = len(features_x_list)
        layer_weights = [1.0 / num_feature_layers] * num_feature_layers

        for i in range(num_feature_layers):
            f_recons = features_recons_list[i]
            f_x = features_x_list[i]
            perceptual_loss += layer_weights[i] * F.mse_loss(f_recons, f_x, reduction="mean")

    total_loss = recons_loss + kld_weight * kld_loss + perceptual_weight * perceptual_loss
        
    return {"loss": total_loss, "recons_loss": recons_loss.detach(), "KLD_loss": kld_loss.detach(), "perceptual_loss": perceptual_loss.detach() if isinstance(perceptual_loss, torch.Tensor) else torch.tensor(perceptual_loss)}


if __name__ == "__main__":
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()

    data = torch.randn(64, 5000, 12)
    mask = torch.ones(64, 5000, 12)
    mask[:, :, 0] = 0
    masked_data = data * (1 - mask)

    encoder_out = encoder(masked_data)[0]
    print(encoder_out.shape)

    decoder_out = decoder(encoder_out)
    print(decoder_out.shape)
