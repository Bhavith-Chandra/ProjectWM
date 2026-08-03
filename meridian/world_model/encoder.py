"""
Temporal Encoder for Financial World Model
===========================================
Selective State Space (Mamba-inspired) encoder with content-aware gating.
O(N) complexity allows processing 1000+ day histories efficiently.

Falls back to GRU when selective scan isn't needed, keeping the interface
identical so the world model doesn't care which backbone runs underneath.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SelectiveScan(nn.Module):
    """
    Simplified selective state-space layer (Mamba-style).
    Content-aware: A, B, C, dt are input-dependent.
    Uses parallel scan for O(N) training.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv,
                                padding=d_conv - 1, groups=d_inner)

        # S4-style projections (input-dependent)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # A: initialized via HiPPO
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).expand(d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        residual = x
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)

        # causal conv
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = self.conv1d(x_ssm)[:, :, :L]
        x_ssm = x_ssm.transpose(1, 2)
        x_ssm = F.silu(x_ssm)

        # content-aware SSM params
        x_dbl = self.x_proj(x_ssm)
        B_param = x_dbl[..., :self.d_state]
        C_param = x_dbl[..., self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(x_dbl[..., -1:]))

        A = -torch.exp(self.A_log)

        # discretize and scan (sequential fallback — torch doesn't have parallel scan)
        d_inner = x_ssm.shape[-1]
        h = torch.zeros(B, d_inner, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            dt_t = dt[:, t].unsqueeze(-1)
            A_bar = torch.exp(A * dt_t)
            B_bar = dt_t * B_param[:, t].unsqueeze(1)
            h = A_bar * h + B_bar * x_ssm[:, t].unsqueeze(-1)
            y_t = (h * C_param[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)
        y = y + x_ssm * self.D.unsqueeze(0).unsqueeze(0)

        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.norm(y + residual)


class TemporalEncoder(nn.Module):
    """
    Multi-layer temporal encoder.
    backbone='ssm' uses SelectiveScan (Mamba), 'gru' uses GRU (faster on CPU).
    """

    def __init__(self, input_dim: int, d_model: int = 256,
                 n_layers: int = 4, d_state: int = 16, dropout: float = 0.1,
                 backbone: str = 'gru'):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )

        if backbone == 'gru':
            self.gru = nn.GRU(d_model, d_model, num_layers=n_layers,
                              batch_first=True, dropout=dropout if n_layers > 1 else 0)
            self.out_norm = nn.LayerNorm(d_model)
        else:
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(nn.ModuleDict({
                    'ssm': SelectiveScan(d_model, d_state),
                    'ff': nn.Sequential(
                        nn.Linear(d_model, d_model * 4),
                        nn.SiLU(),
                        nn.Dropout(dropout),
                        nn.Linear(d_model * 4, d_model),
                        nn.Dropout(dropout),
                    ),
                    'ff_norm': nn.LayerNorm(d_model),
                }))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim)
        returns: (batch, seq_len, d_model)
        """
        x = self.input_proj(x)
        if self.backbone == 'gru':
            x, _ = self.gru(x)
            return self.out_norm(x)
        else:
            for layer in self.layers:
                x = layer['ssm'](x)
                x = layer['ff_norm'](layer['ff'](x) + x)
            return x
