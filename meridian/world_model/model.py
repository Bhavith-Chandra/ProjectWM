"""
Meridian World Model
=====================
Full world model combining:
  - TemporalEncoder (Mamba-style SSM)
  - DynamicAssetGraph (GAT cross-asset layer)
  - RSSM (DreamerV3-style latent dynamics)
  - Emission heads (returns, vol, tail, regime, covariance)

The model can:
  1. Encode historical observations into latent states
  2. Imagine forward trajectories (scenario generation)
  3. Predict returns, volatility, tail risk, and cross-asset covariance
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .encoder import TemporalEncoder
from .graph import DynamicAssetGraph
from .rssm import RSSM, symlog
from .heads import (ReturnHead, VolatilityHead, TailHead,
                    RegimeHead, CovarianceHead)


class MeridianWorldModel(nn.Module):
    """
    Full financial world model.

    Pipeline:
      raw obs → TemporalEncoder → DynamicAssetGraph → RSSM → heads

    For multi-asset inputs, the encoder runs per-asset, the graph mixes
    cross-asset info, then features are flattened for the RSSM.
    """

    def __init__(self, n_assets: int = 11, input_features: int = 5,
                 d_model: int = 128, hidden_dim: int = 512,
                 n_categoricals: int = 32, n_classes: int = 32,
                 n_encoder_layers: int = 3, n_graph_layers: int = 2,
                 n_factors: int = 8, n_regimes: int = 4):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model

        # per-asset temporal encoding
        self.encoder = TemporalEncoder(
            input_dim=input_features,
            d_model=d_model,
            n_layers=n_encoder_layers,
        )

        # cross-asset graph
        self.graph = DynamicAssetGraph(
            d_model=d_model,
            n_layers=n_graph_layers,
        )

        # RSSM operates on flattened asset embeddings
        rssm_obs_dim = n_assets * d_model
        self.rssm = RSSM(
            obs_dim=rssm_obs_dim,
            hidden_dim=hidden_dim,
            n_categoricals=n_categoricals,
            n_classes=n_classes,
            embed_dim=d_model * 2,
        )

        # emission heads read from combined (h, z) state
        state_dim = hidden_dim + n_categoricals * n_classes
        self.return_head = ReturnHead(state_dim, n_assets)
        self.vol_head = VolatilityHead(state_dim, n_assets)
        self.tail_head = TailHead(state_dim, n_assets)
        self.regime_head = RegimeHead(state_dim, n_regimes)
        self.cov_head = CovarianceHead(state_dim, n_assets, n_factors)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode raw observations.
        obs: (batch, seq_len, n_assets, input_features)
        returns: (batch, seq_len, n_assets * d_model)
        """
        B, T, N, F = obs.shape
        # encode each asset independently
        x = obs.reshape(B * N, T, F)
        x = self.encoder(x)  # (B*N, T, d_model)
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d_model)

        # cross-asset graph attention
        x = self.graph(x)  # (B, T, N, d_model)

        return x.reshape(B, T, N * self.d_model)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass (training).
        obs: (batch, seq_len, n_assets, input_features)
        """
        embedded = self.encode(obs)
        rssm_out = self.rssm.observe_sequence(embedded)

        h = rssm_out['h']  # (B, T, hidden_dim)
        z = rssm_out['z']  # (B, T, latent_dim)
        state = torch.cat([h, z], dim=-1)  # (B, T, state_dim)

        B, T, S = state.shape
        state_flat = state.reshape(B * T, S)

        return {
            'returns': self.return_head(state_flat).reshape(B, T, -1),
            'log_vol': self.vol_head(state_flat).reshape(B, T, -1),
            'tail_df': self.tail_head(state_flat).reshape(B, T, -1),
            'regime_logits': self.regime_head(state_flat).reshape(B, T, -1),
            'covariance': self.cov_head(state_flat).reshape(B, T, self.n_assets, self.n_assets),
            'prior_logits': rssm_out['prior_logits'],
            'post_logits': rssm_out['post_logits'],
            'h': h,
            'z': z,
        }

    def imagine(self, obs: torch.Tensor, horizon: int,
                n_scenarios: int = 1000) -> Dict[str, torch.Tensor]:
        """
        Generate future scenarios from current observations.

        1. Encode observations and run RSSM posterior
        2. Take final state
        3. Imagine forward `horizon` steps, sampling stochastic latents
        4. Decode each step through emission heads

        Returns scenario paths for returns, vol, covariance.
        """
        B = obs.shape[0]

        embedded = self.encode(obs)
        rssm_out = self.rssm.observe_sequence(embedded)

        # take final state and replicate for scenarios
        h_final = rssm_out['h'][:, -1]  # (B, hidden)
        z_final = rssm_out['z'][:, -1]  # (B, latent)

        # expand for multiple scenarios
        h = h_final.unsqueeze(1).expand(-1, n_scenarios, -1).reshape(B * n_scenarios, -1)
        z = z_final.unsqueeze(1).expand(-1, n_scenarios, -1).reshape(B * n_scenarios, -1)

        imagined = self.rssm.imagine_sequence(h, z, horizon)

        h_seq = imagined['h']  # (B*n_scenarios, horizon, hidden)
        z_seq = imagined['z']
        state = torch.cat([h_seq, z_seq], dim=-1)

        BN, H, S = state.shape
        state_flat = state.reshape(BN * H, S)

        returns = self.return_head(state_flat).reshape(B, n_scenarios, H, -1)
        log_vol = self.vol_head(state_flat).reshape(B, n_scenarios, H, -1)
        cov = self.cov_head(state_flat).reshape(B, n_scenarios, H, self.n_assets, self.n_assets)

        return {
            'returns': returns,
            'log_vol': log_vol,
            'covariance': cov,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def state_dim(self) -> int:
        return self.rssm.hidden_dim + self.rssm.prior_head.latent_dim
