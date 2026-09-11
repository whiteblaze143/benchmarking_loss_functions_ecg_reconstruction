import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckBlock(nn.Module):

    def __init__(self, width, bottleneck_factor=4):
        super().__init__()
        bottleneck_width = int(width / bottleneck_factor)
        self.fc1 = nn.Linear(width, bottleneck_width)
        self.fc2 = nn.Linear(bottleneck_width, width)
        self.activation = nn.GELU()

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        out += identity
        out = self.activation(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(1)

        Q, K, V = self.query(x), self.key(x), self.value(x)

        scores = torch.bmm(Q, K.transpose(1, 2))
        attention_weights = self.softmax(scores / (K.size(-1) ** 0.5))
        
        context = torch.bmm(attention_weights, V)
        
        return context.squeeze(1)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        
        if x.dim() == 2:
            x = x.unsqueeze(1) 

        B, L, E = x.size() 
        H = self.num_heads
        D = self.head_dim

        Q = self.query(x)  # [B, L, E]
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(B, L, H, D).transpose(1, 2)
        K = K.view(B, L, H, D).transpose(1, 2)
        V = V.view(B, L, H, D).transpose(1, 2)

        # Attention scores: [B, H, L, L]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = self.softmax(scores)
        # Weighted sum: [B, H, L, D]
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, L, E)

        out = self.out_proj(context)

        return out.squeeze(1)

class ResMLP(nn.Module):
    def __init__(self, in_features=111, width=1032, num_blocks=3, num_classes=1, num_heads=1):
        super().__init__()
        self.initial_layer = nn.Linear(in_features, width)
        if num_heads<2:
            self.attention = SelfAttention(embed_dim=width)
        else:
            self.attention = MultiHeadSelfAttention(embed_dim=width, num_heads=num_heads)

        self.attention_norm = nn.LayerNorm(width) #####
        self.blocks = nn.ModuleList([BottleneckBlock(width) for _ in range(num_blocks)])
        
        self.final_layer = nn.Linear(width, num_classes)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.activation(x)

        identity = x #####
        x = self.attention_norm(x)  ####
        x = self.attention(x)  ####
        x = x + identity 
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_layer(x)
        return x