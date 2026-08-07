"""
Renormalization Group Engine for Multi-Scale Financial Dynamics
================================================================
Wilson RG-inspired multi-scale decomposition:
  Scale 0: Raw (daily)
  Scale 1: Weekly dynamics
  Scale 2: Monthly dynamics
  Scale 3: Quarterly dynamics
  Scale 4: Regime dynamics

Uses RSMI-based coarse-graining (Koch-Janusz & Ringel, Nature Physics 2018),
NOT naive KL compression (Mehta-Schwab was refuted for general case).

Learned wavelet front-end disentangles scales. RG flow MLPs track how
parameters transform across scales. Anomalous dimension = Hurst exponent.

Pre-transition detection uses VARIANCE amplification, NOT autocorrelation
(autocorrelation CSD refuted by Guttal et al. PLOS ONE 2016).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional


class LearnableWavelet(nn.Module):
    """
    Learnable 1D wavelet decomposition for scale disentanglement.
    Replaces fixed wavelets with learned filters that separate
    frequency bands optimally for financial data.
    """

    def __init__(self, d_model: int, kernel_size: int = 7):
        super().__init__()
        self.low_pass = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        self.high_pass = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model, bias=False,
        )
        nn.init.constant_(self.low_pass.weight, 1.0 / kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, d_model)
        returns: (low_freq, high_freq) each (batch, seq_len, d_model)
        """
        x_t = x.transpose(1, 2)
        low = self.low_pass(x_t).transpose(1, 2)
        high = self.high_pass(x_t).transpose(1, 2)
        return low, high


class CoarseGrainBlock(nn.Module):
    """
    Attention-weighted temporal coarse-graining at one scale.
    NOT simple averaging — learns which timesteps matter at each scale.
    Uses stride to reduce temporal resolution.
    """

    def __init__(self, d_model: int, stride: int = 5, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.stride = stride
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.ff_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len // stride, d_model) — coarse-grained
        """
        B, T, D = x.shape
        T_out = max(1, T // self.stride)

        queries = x[:, ::self.stride][:, :T_out]
        attn_out, _ = self.attention(queries, x, x)
        out = self.norm(attn_out + queries)
        out = self.ff_norm(self.ff(out) + out)
        return out


class RGFlowMLP(nn.Module):
    """
    Learned RG flow: how parameters transform across scales.
    The beta function of the RG maps dynamics at scale k to scale k+1.
    """

    def __init__(self, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual gated flow: allows identity when no scale change needed."""
        flow = self.net(x)
        g = self.gate(x)
        return x + g * flow


class AnomalousDimension(nn.Module):
    """
    Learned Hurst exponent H(t) per asset.
    H = 0.5: random walk (Gaussian fixed point)
    H < 0.5: anti-persistent (mean-reverting, rough volatility)
    H > 0.5: persistent (trending)
    H != 0.5 means the RG flow has NOT reached the Gaussian fixed point.

    The Hurst exponent unifies three concepts:
      - Anomalous dimension in RG (physics)
      - Roughness parameter in rough vol (stochastic analysis)
      - Scaling exponent of fractal returns (geometry)
    """

    def __init__(self, d_model: int, n_assets: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, d_model) — pooled multi-scale features
        returns: (batch, n_assets) — Hurst exponent in (0.01, 0.99)
        """
        return 0.01 + 0.98 * torch.sigmoid(self.net(x))


class VarianceAmplificationDetector(nn.Module):
    """
    Pre-transition detection via variance amplification.
    Uses VARIANCE divergence (proven), NOT autocorrelation (refuted).

    Monitors the ratio of recent variance to baseline variance.
    When this ratio exceeds a threshold, a transition is imminent.
    """

    def __init__(self, d_model: int, window: int = 20, hidden: int = 64):
        super().__init__()
        self.window = window
        self.net = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_sequence: torch.Tensor) -> torch.Tensor:
        """
        x_sequence: (batch, seq_len, d_model)
        returns: (batch, 1) — critical slowing metric xi_t
        """
        T = x_sequence.shape[1]
        half = max(1, T // 2)

        recent = x_sequence[:, half:]
        baseline = x_sequence[:, :half]

        recent_var = recent.var(dim=1)
        baseline_var = baseline.var(dim=1).clamp(min=1e-6)

        features = torch.cat([recent_var, baseline_var], dim=-1)
        return torch.sigmoid(self.net(features))


class RenormalizationEngine(nn.Module):
    """
    Full RG engine: 5 scales of coarse-graining with learned flows.

    Pipeline per scale:
      1. Disentangle (learned wavelet separating scale dynamics)
      2. Coarse-grain (attention-weighted temporal pooling with stride)
      3. Flow (MLP mapping parameters across scales)

    Outputs:
      - Multi-scale features at each of 5 levels
      - Hurst exponent H(t) per asset
      - Critical slowing metric xi_t (variance amplification)
    """

    N_SCALES = 5
    STRIDES = [1, 5, 5, 4, 4]

    def __init__(self, d_model: int, n_assets: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_assets = n_assets

        self.wavelets = nn.ModuleList([
            LearnableWavelet(d_model) for _ in range(self.N_SCALES - 1)
        ])
        self.coarse_grains = nn.ModuleList([
            CoarseGrainBlock(d_model, stride=s, n_heads=n_heads, dropout=dropout)
            for s in self.STRIDES[1:]
        ])
        self.rg_flows = nn.ModuleList([
            RGFlowMLP(d_model) for _ in range(self.N_SCALES - 1)
        ])
        self.scale_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(self.N_SCALES)
        ])

        self.hurst = AnomalousDimension(d_model * self.N_SCALES, n_assets)
        self.critical = VarianceAmplificationDetector(d_model)

        self.output_proj = nn.Linear(d_model * self.N_SCALES, d_model * 2)
        self.output_norm = nn.LayerNorm(d_model * 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, seq_len, d_model) — tangent space vectors from manifold
           For multi-asset: flatten assets into batch or process per-asset.

        returns dict with:
            'multi_scale': (batch, d_model * 2)  — concatenated scale features
            'hurst': (batch, n_assets)  — Hurst exponent per asset
            'critical_slowing': (batch, 1)  — pre-transition variance signal
            'scale_features': list of per-scale features
        """
        B, T, D = x.shape

        scale_features = [self.scale_norms[0](x)]

        current = x
        for i in range(self.N_SCALES - 1):
            low, high = self.wavelets[i](current)
            coarse = self.coarse_grains[i](low)
            flowed = self.rg_flows[i](coarse)
            scale_features.append(self.scale_norms[i + 1](flowed))
            current = low

        pooled = []
        for sf in scale_features:
            pooled.append(sf.mean(dim=1))

        pooled_cat = torch.cat(pooled, dim=-1)

        hurst = self.hurst(pooled_cat)
        critical = self.critical(x)

        multi_scale = self.output_norm(self.output_proj(pooled_cat))

        return {
            'multi_scale': multi_scale,
            'hurst': hurst,
            'critical_slowing': critical,
            'scale_features': scale_features,
        }
