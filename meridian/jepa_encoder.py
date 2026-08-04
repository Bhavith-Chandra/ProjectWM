"""Joint-Embedding Predictive Architecture (JEPA) for financial markets.

JEPA learns a shared representation space where:
  - Encoder maps observations (returns, vol, sentiment) → latent code z
  - Predictor maps past z's → future z (no reconstruction, just latent dynamics)
  - Loss: attract similar codes, repel dissimilar ones
  - No bottleneck: no need to reconstruct raw observations

Advantage for finance:
  - Handles sparse/missing data (order books, news)
  - Multi-scale temporal structure (1d, 1w, 1m)
  - Better representations than VAE when reconstruction is expensive
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque


class JEPAEncoder(nn.Module):
    """Maps observations → latent codes (no reconstruction loss)."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_dim) or (batch, input_dim)."""
        if x.ndim == 3:
            B, T, D = x.shape
            x_flat = x.reshape(B * T, D)
            z_flat = self.net(x_flat)
            return z_flat.reshape(B, T, self.latent_dim)
        else:
            return self.net(x)


class JEPAPredictor(nn.Module):
    """Predicts future latent codes from past latent codes (no reconstruction)."""

    def __init__(self, latent_dim: int, context_len: int = 120, hidden_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_len = context_len

        # GRU processes latent trajectory
        self.gru = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        # MLP projects hidden state to future latent mean
        self.head_mean = nn.Linear(hidden_dim, latent_dim)
        self.head_logstd = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        z_past: (batch, seq_len, latent_dim)
        Returns: (mean, logstd) of next latent code, each (batch, latent_dim)
        """
        _, h = self.gru(z_past)  # h: (1, batch, hidden_dim)
        h = h.squeeze(0)  # (batch, hidden_dim)

        mean = self.head_mean(h)
        logstd = self.head_logstd(h).clamp(-6, 2)  # Stability

        return mean, logstd

    def sample(self, z_past: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample future latent codes."""
        mean, logstd = self.forward(z_past)
        # Broadcast for multiple samples
        mean = mean.unsqueeze(1).expand(-1, n_samples, -1)  # (batch, n_samples, latent_dim)
        logstd = logstd.unsqueeze(1).expand(-1, n_samples, -1)

        eps = torch.randn_like(mean)
        z_future = mean + torch.exp(0.5 * logstd) * eps
        return z_future


class JEPADecoder(nn.Module):
    """Maps latent codes back to observation space (for validation/reconstruction)."""

    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (batch, seq_len, latent_dim) or (batch, latent_dim)."""
        if z.ndim == 3:
            B, T, D = z.shape
            z_flat = z.reshape(B * T, D)
            x_flat = self.net(z_flat)
            return x_flat.reshape(B, T, -1)
        else:
            return self.net(z)


class JEPAFinancialWorldModel(nn.Module):
    """Full JEPA model for financial markets."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        context_len: int = 120,
        use_decoder: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.context_len = context_len
        self.use_decoder = use_decoder

        self.encoder = JEPAEncoder(input_dim, latent_dim, hidden_dim)
        self.predictor = JEPAPredictor(latent_dim, context_len, hidden_dim)

        if use_decoder:
            self.decoder = JEPADecoder(latent_dim, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (batch, seq_len, input_dim)
        Returns: dict with losses and outputs
        """
        B, T, D = x.shape

        # Encode entire sequence
        z = self.encoder(x)  # (batch, seq_len, latent_dim)

        # Predict future latent from past
        z_past = z[:, :-1, :]  # (batch, seq_len-1, latent_dim)
        z_true_future = z[:, 1:, :]  # (batch, seq_len-1, latent_dim)

        mu, logstd = self.predictor(z_past)  # (batch, latent_dim)

        # JEPA loss: predict latent-space future (no reconstruction)
        # L = MSE(mu, z_true_future[:, -1, :]) + KL(N(mu, exp(logstd)), N(0,1))
        pred_loss = F.mse_loss(mu, z_true_future[:, -1, :])
        kl = -0.5 * (1 + logstd - mu**2 - torch.exp(logstd)).sum(dim=-1).mean()

        loss = pred_loss + 0.1 * kl

        output = {
            'loss': loss,
            'z': z,
            'z_pred_mean': mu,
            'z_pred_std': torch.exp(0.5 * logstd),
            'pred_loss': pred_loss,
            'kl_loss': kl
        }

        if self.use_decoder:
            x_recon = self.decoder(z)
            recon_loss = F.mse_loss(x_recon, x)
            output['recon_loss'] = recon_loss
            output['x_recon'] = x_recon

        return output

    @torch.no_grad()
    def get_latent_state(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent code for observations (inference)."""
        return self.encoder(x)

    @torch.no_grad()
    def generate_scenarios(self, x_history: torch.Tensor, horizon: int = 20, n_paths: int = 100) -> torch.Tensor:
        """
        Generate future scenarios from latent dynamics.

        Args:
            x_history: (batch, context_len, input_dim)
            horizon: number of steps to generate
            n_paths: number of parallel paths to generate

        Returns: (batch, n_paths, horizon, input_dim) - generated scenarios
        """
        B = x_history.shape[0]

        # Get current latent state
        z_current = self.encoder(x_history)[:, -1:, :]  # (batch, 1, latent_dim)

        # Generate latent trajectories
        scenarios_latent = []
        z_t = z_current.unsqueeze(1).expand(-1, n_paths, -1)  # (batch, n_paths, latent_dim)

        for _ in range(horizon):
            z_next = self.predictor.sample(z_t.reshape(B*n_paths, 1, -1), n_samples=1).squeeze(1)
            z_next = z_next.reshape(B, n_paths, -1)
            scenarios_latent.append(z_next)
            z_t = z_next.unsqueeze(2)  # Prepare for next step (keep as sequence)

        z_scenarios = torch.stack(scenarios_latent, dim=2)  # (batch, n_paths, horizon, latent_dim)

        # Decode to observation space if decoder exists
        if self.use_decoder:
            # Reshape for batch decoding
            z_flat = z_scenarios.reshape(B * n_paths * horizon, self.latent_dim)
            x_scenarios_flat = self.decoder(z_flat)
            x_scenarios = x_scenarios_flat.reshape(B, n_paths, horizon, self.input_dim)
            return x_scenarios
        else:
            return z_scenarios


class JEPATrainer:
    """Training harness for JEPA model."""

    def __init__(
        self,
        model: JEPAFinancialWorldModel,
        lr: float = 1e-3,
        weight_decay: float = 1e-5
    ):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = next(model.parameters()).device

    def train_epoch(self, train_loader, epochs: int = 1) -> dict:
        """Train for one or more epochs."""
        history = {'loss': [], 'pred_loss': [], 'kl_loss': []}

        for epoch in range(epochs):
            epoch_losses = []
            for x_batch in train_loader:
                x_batch = x_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = output['loss']

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                epoch_losses.append(loss.item())
                if 'recon_loss' in output:
                    history['recon_loss'].append(output['recon_loss'].item())

            history['loss'].append(np.mean(epoch_losses))
            history['pred_loss'].append(float(output.get('pred_loss', 0)))
            history['kl_loss'].append(float(output.get('kl_loss', 0)))

        return history
