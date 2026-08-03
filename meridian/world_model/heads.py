"""
Emission Heads for the World Model
====================================
Decode latent states into observable quantities:
  - Return prediction (symlog-space)
  - Volatility (log-space, always positive)
  - Tail shape (Student-t degrees of freedom)
  - Regime classification

Each head reads from the combined (h, z) state of the RSSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rssm import symlog, symexp


class ReturnHead(nn.Module):
    """Predicts asset returns in symlog space."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

    def loss(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self.forward(state)
        return F.mse_loss(pred, symlog(target))


class VolatilityHead(nn.Module):
    """Predicts per-asset conditional volatility (log-space)."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns log-volatility."""
        return self.net(state)

    def loss(self, state: torch.Tensor, target_vol: torch.Tensor) -> torch.Tensor:
        pred_log_vol = self.forward(state)
        target_log_vol = torch.log(target_vol.clamp(min=1e-8))
        return F.mse_loss(pred_log_vol, target_log_vol)


class TailHead(nn.Module):
    """Predicts Student-t degrees of freedom (controls tail thickness)."""

    def __init__(self, state_dim: int, n_assets: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # df in [2.1, 50] — 2.1 is heavy tail, 50 is near-Gaussian
        return 2.1 + F.softplus(self.net(state)) * 10


class RegimeHead(nn.Module):
    """Classifies market regime from latent state."""

    def __init__(self, state_dim: int, n_regimes: int = 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_regimes),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class CovarianceHead(nn.Module):
    """
    Low-rank factor covariance: Sigma = L L^T + diag(d)
    Efficient for n_assets >> n_factors.
    """

    def __init__(self, state_dim: int, n_assets: int, n_factors: int = 8,
                 hidden: int = 256):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

        self.factor_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets * n_factors),
        )
        self.diag_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns full covariance matrix (batch, n_assets, n_assets)."""
        L = self.factor_net(state).reshape(-1, self.n_assets, self.n_factors)
        d = F.softplus(self.diag_net(state)) + 1e-6
        cov = L @ L.transpose(-1, -2) + torch.diag_embed(d)
        return cov
