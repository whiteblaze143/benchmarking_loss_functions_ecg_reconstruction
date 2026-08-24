import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import math

#多头自注意力机制
class SelfAttention(nn.Module):

    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()
        #定义一个线性层，用来同时生成 Query、Key 和 Value
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        #输出线性层，把多个头的输出拼接后再线性变换一次，维持维度为 d_embed
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        #每个头的维度就是总维度除以头数，即 d_embed / n_heads
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: (Batch_size, Seq_len, Dim)
        # causal_mask 控制是否启用因果掩码（通常用于生成任务防止未来信息泄露
        # 记录输入形状，并准备中间 reshape 所需的形状信息
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

        # 计算注意力权重矩阵，维度是 (B, H, S, S)。

        # 每个 query 向量和所有 key 做点积
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # 创建一个上三角掩码（不包含主对角线），用于防止模型看到未来的信息
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        # Softmax 与缩放
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (B, H, S, S) @ (B, H, S, D/H) -> (B, H, S, D/H)加权求值输出
        output = weight @ v

        # (B, H, S, D/H) -> (B, S, H, D/H)多头合并与线性映射
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (B, S, D)
        return output

#注意力残差块
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
        # 第一个规范化层：Group Normalization
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        # 第一层一维卷积（Conv1d）
        self.conv_1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1)
        # 实现残差连接的维度匹配
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, In_channels, L)
        residue = x 

        # 规范化 + 激活 + 卷积
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
                nn.Conv1d(12, 128, kernel_size=3, padding=1),# 初始卷积，12通道→128通道，保持序列长度
                VAE_ResidualBlock(128, 128),# 残差块，128→128
                VAE_ResidualBlock(128, 128),
                nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=0),# 下采样，序列长度减半，通道128不变
                VAE_ResidualBlock(128, 256),# 通道变多，128→256
                VAE_ResidualBlock(256, 256),
                nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=0),# 再次下采样
                VAE_ResidualBlock(256, 512),# 通道变多，256→512
                VAE_ResidualBlock(512, 512),
                nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=0),# 下采样
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),

                # nn.Conv1d(512, 512, kernel_size=5, stride=5, padding=0),
                # VAE_ResidualBlock(512, 512),
                # VAE_ResidualBlock(512, 512),

                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),# 注意力模块，增强特征表达能力
                VAE_ResidualBlock(512, 512),
                nn.GroupNorm(32, 512),
                nn.SiLU(),
                nn.Conv1d(512, 8, kernel_size=3, padding=1),# 降通道到8
                nn.Conv1d(8, 8, kernel_size=1, padding=0),# 最后1x1卷积微调
            ]
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, C)
        # noise: (B, C_Out, L/8)
        # output: (B, C_Out, L/8)
        # NOTE: apply noise after encoding转置通道和序列维度
        x = x.transpose(1, 2)
        # x: (B, L, C) -> (B, C, L)
        # x = x.transpose(1, 2)
        # x: (B, C, L) -> (B, 8, L/8)逐个模块前向传播。
        for module in self.blocks:
            if getattr(module, "stride", None) == (2,):
                # Padding(left, right)
                x = F.pad(x, (0, 1))
            x = module(x)
            # print(x.shape)

        # (B, 8, L/8) -> 2 x (B, 4, L/8)输出潜空间的分布参数
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # 限制对数方差范围防止数值爆炸
        log_variance = torch.clamp(log_variance, -30, 20)
        # 计算方差和标准差，准备后面采样
        variance = log_variance.exp()
        stdev = variance.sqrt()

        # z ~ N(0, 1) -> x ~ N(mean, variance)
        # x = mean + stdev * z
        # (B, 4, L/8) -> (B, 4, L/8)
        if noise is None:
            noise = torch.randn(stdev.shape, device=stdev.device)
        x = mean + stdev * noise # 利用均值和标准差做重参数采样：z = mean + stdev * noise，这是VAE的核心采样步骤

        # Scale the output by a constant (magic number)
        x *= 0.18215

        return x, mean, log_variance


class VAE_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                # 第一段卷积层
                nn.Conv1d(4, 4, kernel_size=1, padding=0),
                nn.Conv1d(4, 512, kernel_size=3, padding=1),
                #深层残差与注意力模块
                VAE_ResidualBlock(512, 512),
                VAE_AttentionBlock(512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                # (B, 512, L/8) -> (B, 512, L/4)第一次上采样阶段
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                # (B, 512, L/4) -> (B, 512, L/2)第二次上采样阶段
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                VAE_ResidualBlock(512, 512),
                # (B, 256, L/2) -> (B, 256, L)第三次上采样阶段
                nn.Upsample(scale_factor=2),
                nn.Conv1d(512, 512, kernel_size=3, padding=1),
                VAE_ResidualBlock(512, 256),
                VAE_ResidualBlock(256, 256),
                VAE_ResidualBlock(256, 256),

                # nn.Upsample(scale_factor=5),
                # nn.Conv1d(256, 256, kernel_size=3, padding=1),
                # VAE_ResidualBlock(256, 128),
                # VAE_ResidualBlock(128, 128),
                # VAE_ResidualBlock(128, 128),末尾处理层

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


def loss_function(recons, x, mu, log_var, kld_weight=1e-4, perceptual_weight=1e-2, ecg_founder_model=None, device=None) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :para recons: reconstruction vector
    :para x: original vector
    :para mu: mean of latent gaussian distribution
    :log_var: log of latent gaussian distribution variance
    :kld_weight: weight of kl-divergence term
    """
    # recons, x: (B, L, 12) -> number, batch wise average
    # recons_loss = F.mse_loss(recons, x, reduction='sum').div(x.size(0))
    recons_loss = F.mse_loss(recons, x, reduction="mean")

    # (old) mu, log_var: (B, 4, L/8) -> number
    # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 2), dim=1).sum()

    # q(z|x): distribution learned by encoder
    q_z_x = Normal(mu, log_var.mul(0.5).exp())
    # p(z): prior of z, intended to be standard Gaussian
    p_z = Normal(torch.zeros_like(mu), torch.ones_like(log_var))
    # kld_loss: batch wise average
    kld_loss = kl_divergence(q_z_x, p_z).sum(1).mean()
    loss = recons_loss + kld_weight * kld_loss
    # loss = recons_loss

    return {"loss": loss, "recons_loss": recons_loss.detach(), "KLD_loss": kld_loss.detach(), "perceptual_loss": 0}


if __name__ == "__main__":
    encoder = VAE_Encoder()
    decoder = VAE_Decoder()

    data = torch.randn(64, 1000, 12)
    mask = torch.ones(64, 1000, 12)
    # mask[:, :, 0] = 0

    # 设置保留导联：II (1), V1 (6), V5 (10)
    mask[:, :, [1, 6, 10]] = 0

    # 设置保留导联：I (0), II (1), V3 (8)
    # mask[:, :, [0, 1, 8]] = 0
    # 掩码其余导联
    masked_data = data * (1 - mask)
    encoder_out = encoder(masked_data)[0]
    # print(encoder_out.shape)

    decoder_out = decoder(encoder_out)
    # print(decoder_out.shape)
