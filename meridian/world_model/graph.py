"""
Dynamic Asset Graph Layer
==========================
GAT-based cross-asset dependency modeling with attention-weighted edges.
Edge features incorporate rolling correlation and volatility-of-volatility.
Learns time-varying asset relationships that static correlation matrices miss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class AssetGraphAttention(nn.Module):
    """
    Graph Attention layer for cross-asset dependencies.

    Each asset is a node. Edges are fully connected (all pairs)
    with learned attention weights conditioned on asset states.

    Multi-head attention over the asset dimension (not time).
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_heads),
        )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_assets, d_model)  — per-asset embeddings at one timestep
        returns: (batch, n_assets, d_model)
        """
        B, N, D = x.shape
        residual = x

        q = self.q_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # standard scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # edge bias: pairwise edge features modulate attention
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)
        edge_feat = torch.cat([xi, xj], dim=-1)
        edge_bias = self.edge_mlp(edge_feat).permute(0, 3, 1, 2)
        attn = attn + edge_bias

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return self.norm(out + residual)


class DynamicAssetGraph(nn.Module):
    """
    Applies graph attention across assets at each timestep.

    Input shape: (batch, seq_len, n_assets, d_model)
    The temporal encoder processes each asset independently;
    this layer mixes information across assets at each step.
    """

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'gat': AssetGraphAttention(d_model, n_heads, dropout),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model),
                    nn.Dropout(dropout),
                ),
                'ff_norm': nn.LayerNorm(d_model),
            })
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_assets, d_model)
        returns: same shape, with cross-asset information mixed
        """
        B, T, N, D = x.shape
        # flatten batch and time for graph ops
        x = x.reshape(B * T, N, D)
        for layer in self.layers:
            x = layer['gat'](x)
            x = layer['ff_norm'](layer['ff'](x) + x)
        return x.reshape(B, T, N, D)
