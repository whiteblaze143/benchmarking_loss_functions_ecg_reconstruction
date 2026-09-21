"""
GraphECG Model

Graph neural network for ECG classification using electrode-centric representation.
"""

import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from typing import Optional, Dict


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation = nn.Mish(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        return self.activation(out + residual)


class SphericalHarmonicEncoding(nn.Module):
    """Encode 3D electrode positions using spherical harmonic basis."""

    def __init__(self, max_degree: int = 4):
        super().__init__()
        self.max_degree = max_degree
        self.output_dim = sum(2 * l + 1 for l in range(max_degree + 1)) + 4

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2).clamp(min=1e-8)
        x_n, y_n, z_n = x / r, y / r, z / r

        features = [torch.ones_like(x_n) * 0.5 * math.sqrt(1 / math.pi)]

        if self.max_degree >= 1:
            c = 0.5 * math.sqrt(3 / math.pi)
            features.extend([c * y_n, c * z_n, c * x_n])

        if self.max_degree >= 2:
            c2 = 0.5 * math.sqrt(15 / math.pi)
            c2_0 = 0.25 * math.sqrt(5 / math.pi)
            features.extend([
                c2 * x_n * y_n, c2 * y_n * z_n, c2_0 * (3 * z_n**2 - 1),
                c2 * x_n * z_n, 0.5 * c2 * (x_n**2 - y_n**2)
            ])

        if self.max_degree >= 3:
            features.extend([
                y_n * (3 * x_n**2 - y_n**2), x_n * y_n * z_n,
                y_n * (5 * z_n**2 - 1), z_n * (5 * z_n**2 - 3),
                x_n * (5 * z_n**2 - 1), z_n * (x_n**2 - y_n**2),
                x_n * (x_n**2 - 3 * y_n**2)
            ])

        if self.max_degree >= 4:
            xy, xz, yz = x_n * y_n, x_n * z_n, y_n * z_n
            x2, y2, z2 = x_n**2, y_n**2, z_n**2
            features.extend([
                xy * (x2 - y2), yz * (3 * x2 - y2), xy * (7 * z2 - 1),
                yz * (7 * z2 - 3), 35 * z2**2 - 30 * z2 + 3,
                xz * (7 * z2 - 3), (x2 - y2) * (7 * z2 - 1),
                xz * (x2 - 3 * y2), x2**2 - 6 * x2 * y2 + y2**2
            ])

        sh = torch.stack(features, dim=-1)
        radial = torch.stack([r, torch.sin(math.pi * r), torch.cos(math.pi * r), torch.exp(-r)], dim=-1)
        return torch.cat([sh, radial], dim=-1)


class SignalEncoder(nn.Module):
    """Encode lead waveforms via 1D convolution."""

    def __init__(self, hidden_dim: int = 128, output_dim: int = 192):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.Mish(inplace=True),
            ResBlock1d(64),
            nn.Conv1d(64, hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.Mish(inplace=True),
            ResBlock1d(hidden_dim),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.Mish(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        return self.pool(self.encoder(x)).squeeze(-1)


class ECGMessagePassing(MessagePassing):
    """Message passing with edge (lead) features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.Mish(inplace=True),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return self.norm(x + out)

    def message(self, x_i, x_j, edge_attr):
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))


class GraphECG(nn.Module):
    """
    Graph Neural Network for ECG classification.

    Electrodes are nodes, leads are directed edges carrying waveform embeddings.
    Supports variable lead configurations through subgraph construction.
    """

    def __init__(
        self,
        node_dim: int = 128,
        edge_dim: int = 192,
        hidden_dim: int = 192,
        num_layers: int = 3,
        tabular_dim: int = 7,
        num_classes: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.tabular_dim = tabular_dim
        self.num_classes = num_classes
        self.pos_encoder = SphericalHarmonicEncoding(max_degree=4)
        self.node_proj = nn.Linear(self.pos_encoder.output_dim, node_dim)
        self.signal_encoder = SignalEncoder(hidden_dim=128, output_dim=edge_dim)

        self.gnn_layers = nn.ModuleList([
            ECGMessagePassing(node_dim, edge_dim, node_dim * 2)
            for _ in range(num_layers)
        ])

        self.graph_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Mish(inplace=True),
        )
        self.dropout = nn.Dropout(dropout)
        # num_classes defaults to 1 (binary, e.g. the pretrained EchoNext checkpoint);
        # set num_classes>1 for multiclass, tabular_dim=0 if you have no tabular features.
        self.classifier = nn.Linear(hidden_dim + tabular_dim, num_classes)

    def forward(self, data: Data, tabular: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch = getattr(data, 'batch', None)

        node_features = self.node_proj(self.pos_encoder(data.x))
        edge_embed = self.signal_encoder(data.edge_attr)

        x = node_features
        for gnn in self.gnn_layers:
            x = gnn(x, data.edge_index, edge_embed)

        if batch is None:
            graph_embed = x.mean(dim=0, keepdim=True)
        else:
            graph_embed = global_mean_pool(x, batch)

        graph_embed = self.dropout(self.graph_proj(graph_embed))

        if self.tabular_dim > 0:
            if tabular is None:
                tabular = graph_embed.new_zeros(graph_embed.shape[0], self.tabular_dim)
            fused = torch.cat([graph_embed, tabular], dim=-1)
        else:
            fused = graph_embed

        return {'logits': self.classifier(fused), 'embedding': graph_embed}
