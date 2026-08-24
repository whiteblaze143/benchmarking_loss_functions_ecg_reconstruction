import math
import torch
import numpy as np
import torch.nn.functional as F


import sys
import os

project_root_dir = "/data1/1shared/jinjiarui/run/MLA-Diffusion"
sys.path.append(project_root_dir)
os.chdir(project_root_dir)

from torch import nn
from einops import rearrange, reduce, repeat
from models.interpretable_diffusion.model_utils import (
    LeadEmbedding,
    LearnablePositionalEncoding,
    Conv_MLP,
    AdaLayerNorm,
    Transpose,
    GELU2,
    series_decomp,
)


class FullAttention(nn.Module):
    def __init__(
        self,
        n_embd,  # the embed dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(
        self,
        n_embd,  # the embed dim
        condition_embd,  # condition dim
        n_head,  # the number of heads
        attn_pdrop=0.1,  # attention dropout prob
        resid_pdrop=0.1,  # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, condition_embed, mask=None):
        B, T, C = x.size()
        B, T_E, _ = condition_embed.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(condition_embed)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(condition_embed)
            .view(B, T_E, self.n_head, C // self.n_head)
            .transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
    ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))  # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        mlp_hidden_times=4,
        block_activate="GELU",
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            *[
                EncoderBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        activate="GELU",
        condition_dim=1024,
    ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.attn2 = CrossAttention(
            n_embd=n_embd,
            condition_embd=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ["GELU", "GELU2"]
        act = nn.GELU() if activate == "GELU" else GELU2()

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate="GELU",
        condition_dim=512,
    ):
        super().__init__()
        self.d_model = n_embd
        self.n_feat = n_feat
        self.blocks = nn.Sequential(
            *[
                DecoderBlock(
                    n_feat=n_feat,
                    n_channel=n_channel,
                    n_embd=n_embd,
                    n_head=n_head,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    mlp_hidden_times=mlp_hidden_times,
                    activate=block_activate,
                    condition_dim=condition_dim,
                )
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []

        for block_idx in range(len(self.blocks)):
            x = self.blocks[block_idx](
                x, enc, t, mask=padding_masks, label_emb=label_emb
            )

        return x


class PatchTransformer(nn.Module):

    def __init__(
        self,
        n_channel=12,
        n_feat=1000,
        n_layer_enc=6,
        n_layer_dec=3,
        n_embd=512,
        n_heads=4,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate="GELU",
        max_len=4,
        use_text=False,
        patch_len=100,
        patch_stride=100,
        **kwargs
    ):
        super().__init__()
        self.ecg_emb = Conv_MLP(n_channel, n_embd, resid_pdrop=resid_pdrop)
        self.inverse = Conv_MLP(n_embd, n_channel, resid_pdrop=resid_pdrop)
        self.encoder = Encoder(
            n_layer_enc,
            n_embd,
            n_heads,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
        )
        self.decoder = Decoder(
            n_channel,
            n_feat,
            n_embd,
            n_heads,
            n_layer_dec,
            attn_pdrop,
            resid_pdrop,
            mlp_hidden_times,
            block_activate,
            condition_dim=n_embd,
        )
        self.pos_enc = LearnablePositionalEncoding(
            d_model=n_embd, dropout=resid_pdrop, max_len=patch_len
        )
        self.pos_dec = LearnablePositionalEncoding(
            d_model=n_embd, dropout=resid_pdrop, max_len=patch_len
        )
        self.ecg_text_attn = CrossAttention(
            n_embd=n_embd,
            condition_embd=n_embd,
            n_head=n_heads,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )
        self.use_text = use_text

        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patches = (n_feat - patch_len) // patch_stride + 1
        # 将每个 patch 映射到 n_embd 维空间，即 token 化操作
        self.token_embedding = nn.Linear(self.n_patches, n_embd)
        # 逆映射：将 n_embd 映射回 patch_len（用于输出重构）
        self.token_inverse = nn.Linear(n_embd, self.n_patches)

    def forward(self, x, t, padding_masks=None, text_embed=None, return_res=False):
        x = x.transpose(1, 2)
        bs, n_channel, n_feat = x.size()
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        bs, n_channel, n_patches, patch_len = patches.size()

        t_expanded = t.unsqueeze(1).repeat(1, n_channel).reshape(-1)

        tokens = patches.reshape(bs * n_channel, n_patches, patch_len)  # [bs*channel, patch_num, patch_len]
        tokens = tokens.transpose(1, 2)  # [bs*channel, patch_len, patch_num]
        tokens = self.token_embedding(tokens)  # [bs*channel, patch_len, dim]
        tokens = self.pos_enc(tokens)  # [bs*channel, patch_len, dim]

        enc_emb = self.encoder(
            tokens, t_expanded, padding_masks=padding_masks
        )  # [bs*channel, patch_len, dim]

        inp_dec = self.pos_dec(enc_emb)  # [bs*channel, patch_len, dim]
        out = self.decoder(
            inp_dec, t_expanded, enc_emb, padding_masks=padding_masks
        )  # [bs*channel, patch_len, dim]

        out = out.reshape(
            bs, n_channel, self.patch_len, -1
        )  # [bs, channel, patch_len, dim]
        out = self.token_inverse(out)  # [bs, channel, patch_len, patch_num]
        out = out.reshape(
            bs, n_channel, n_patches * self.patch_len
        )  # [bs, channel, len]
        out = out.transpose(1, 2)
        return out


if __name__ == "__main__":
    model = PatchTransformer()
    t = torch.randint(0, 10, size=(32,))
    x = torch.randn((32, 1000, 12))
    out = model(x, t=t)
    print(out.shape)
