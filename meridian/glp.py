"""Generative Latent Predictor (GLP) for financial markets.

GLPs learn:
  1. Latent factor structure: x_t ≈ A @ f_t + noise
  2. Factor dynamics: f_t = f(f_{t-1}, noise)
  3. Observation likelihood: p(x_t | f_t)

Advantage: Explicit interpretable latent factors (each = economic driver).
Each factor has its own half-life and evolution, making causal tracing easier.

Reference: Generative Latent Predictor (Gorti et al. 2024 ICLR)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from torch.optim import Adam


class FactorDynamics(nn.Module):
    """Temporal evolution of latent factors: f_t → f_t+1."""

    def __init__(self, n_factors: int, hidden_dim: int = 64):
        super().__init__()
        self.n_factors = n_factors

        # GRU for factor trajectories (preserves factor identity over time)
        self.gru = nn.GRU(n_factors, hidden_dim, batch_first=True)

        # Output: mean and covariance of next factors
        self.head_mean = nn.Linear(hidden_dim, n_factors)
        self.head_logstd = nn.Linear(hidden_dim, n_factors)

    def forward(self, f_past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        f_past: (batch, seq_len, n_factors)
        Returns: (mean, logstd) of f_t+1, each (batch, n_factors)
        """
        _, h = self.gru(f_past)  # h: (1, batch, hidden_dim)
        h = h.squeeze(0)

        mean = self.head_mean(h)
        logstd = self.head_logstd(h).clamp(-6, 2)

        return mean, logstd

    def sample(self, f_past: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample next factors."""
        mean, logstd = self.forward(f_past)

        mean = mean.unsqueeze(1).expand(-1, n_samples, -1)  # (batch, n_samples, n_factors)
        logstd = logstd.unsqueeze(1).expand(-1, n_samples, -1)

        eps = torch.randn_like(mean)
        f_next = mean + torch.exp(0.5 * logstd) * eps
        return f_next


class ObservationEmission(nn.Module):
    """Map factors to observations: f_t → p(x_t | f_t)."""

    def __init__(self, n_assets: int, n_factors: int, parametrization: str = 'low_rank'):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors
        self.parametrization = parametrization

        # Factor loadings (interpretable: each factor's impact on each asset)
        self.factor_loadings = nn.Parameter(torch.randn(n_assets, n_factors) * 0.1)

        # Per-asset noise variance
        self.logvar = nn.Parameter(torch.zeros(n_assets))

        if parametrization == 'low_rank':
            # Additional low-rank structure (cross-asset factors)
            self.rank = min(3, n_assets // 2)
            self.low_rank_mean = nn.Linear(n_factors, self.rank)
            self.low_rank_cov = nn.Linear(n_factors, n_assets * self.rank)

    def forward(self, f: torch.Tensor) -> dict:
        """
        f: (batch, seq_len, n_factors) or (batch, n_factors)
        Returns: dict with mean and covariance
        """
        if f.ndim == 3:
            B, T, K = f.shape
            f_flat = f.reshape(B * T, K)
            mean_flat = f_flat @ self.factor_loadings.t()  # (B*T, n_assets)
            mean = mean_flat.reshape(B, T, self.n_assets)
        else:
            mean = f @ self.factor_loadings.t()

        # Diagonal covariance (per-asset noise)
        diag_var = torch.exp(self.logvar) + 1e-6

        # Low-rank adjustment if applicable
        if self.parametrization == 'low_rank':
            # Additional structure from factors
            if f.ndim == 3:
                K_lr = self.low_rank_mean(f)  # (B, T, rank)
            else:
                K_lr = self.low_rank_mean(f).unsqueeze(0)  # (1, rank)

            cov_lr = torch.eye(self.n_assets).unsqueeze(0) * diag_var  # Start with diag
            # Add low-rank perturbation (would need more work for full implementation)
            cov = cov_lr
        else:
            cov = torch.diag_embed(diag_var)

        return {
            'mean': mean,
            'std': torch.sqrt(diag_var),
            'cov': cov if f.ndim == 2 else None,
            'logvar': self.logvar
        }

    def nll(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of x given f."""
        params = self.forward(f)
        mean = params['mean']
        std = params['std']

        # Gaussian NLL
        nll = 0.5 * ((x - mean) / std) ** 2 + 0.5 * self.logvar
        return nll.sum(dim=-1).mean()


class InferenceNetwork(nn.Module):
    """Encode observations → latent factor posterior: x_t → q(f_t | x_t)."""

    def __init__(self, n_assets: int, n_factors: int, hidden_dim: int = 128):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

        # GRU processes observation sequence
        self.gru = nn.GRU(n_assets, hidden_dim, batch_first=True)

        # Output: posterior mean and covariance
        self.head_mean = nn.Linear(hidden_dim, n_factors)
        self.head_logstd = nn.Linear(hidden_dim, n_factors)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, seq_len, n_assets)
        Returns: (mean, logstd) of q(f_t | x_1:t), each (batch, n_factors)
        """
        _, h = self.gru(x)  # h: (1, batch, hidden_dim)
        h = h.squeeze(0)

        mean = self.head_mean(h)
        logstd = self.head_logstd(h).clamp(-6, 2)

        return mean, logstd

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        """Sample factors from posterior."""
        mean, logstd = self.forward(x)
        eps = torch.randn_like(mean)
        f = mean + torch.exp(0.5 * logstd) * eps
        return f


class GLP(nn.Module):
    """Full Generative Latent Predictor."""

    def __init__(
        self,
        n_assets: int,
        n_factors: int = 8,
        hidden_dim: int = 128,
        parametrization: str = 'low_rank'
    ):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

        self.inference = InferenceNetwork(n_assets, n_factors, hidden_dim)
        self.dynamics = FactorDynamics(n_factors, hidden_dim)
        self.emission = ObservationEmission(n_assets, n_factors, parametrization)

        # Prior over factors (standard normal)
        self.register_buffer('prior_mean', torch.zeros(n_factors))
        self.register_buffer('prior_logstd', torch.zeros(n_factors))

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (batch, seq_len, n_assets)
        Returns: dict with ELBO, losses, and latent factors
        """
        B, T, N = x.shape

        # Inference: x_t → q(f_t | x_1:t)
        q_mean, q_logstd = self.inference(x)
        q_std = torch.exp(0.5 * q_logstd)
        f_posterior = q_mean + q_std * torch.randn_like(q_mean)  # Sample from posterior

        # Dynamics: predict f_t+1 from f_t
        f_past = f_posterior.unsqueeze(1)  # (batch, 1, n_factors) for dynamics
        dyn_mean, dyn_logstd = self.dynamics(f_past)

        # Emission: p(x_t | f_t)
        reconstruction_loss = self.emission.nll(x, f_posterior.unsqueeze(1).expand(-1, T, -1))

        # KL divergence: q(f | x) vs prior
        kl_prior = 0.5 * (
            -1 - q_logstd + (q_mean ** 2) + torch.exp(q_logstd)
        ).sum(dim=-1).mean()

        # Dynamics loss: predict next state
        dyn_loss = 0.5 * (f_posterior - dyn_mean) ** 2 / torch.exp(dyn_logstd)
        dyn_loss = dyn_loss.sum(dim=-1).mean()

        # Total ELBO
        elbo = reconstruction_loss + kl_prior + 0.1 * dyn_loss

        return {
            'elbo': elbo,
            'reconstruction_loss': reconstruction_loss,
            'kl_loss': kl_prior,
            'dynamics_loss': dyn_loss,
            'f_posterior': f_posterior,  # (batch, n_factors)
            'f_trajectory': f_posterior.unsqueeze(0).expand(T, -1, -1),  # (T, batch, n_factors)
        }

    @torch.no_grad()
    def get_factors(self, x: torch.Tensor) -> torch.Tensor:
        """Infer latent factors from observations."""
        f_mean, _ = self.inference(x)
        return f_mean

    @torch.no_grad()
    def generate_scenarios(
        self,
        x_history: torch.Tensor,
        horizon: int = 20,
        n_paths: int = 100
    ) -> torch.Tensor:
        """
        Generate future return scenarios from factor dynamics.

        Args:
            x_history: (batch, context_len, n_assets)
            horizon: number of steps
            n_paths: number of parallel paths

        Returns: (batch, n_paths, horizon, n_assets)
        """
        B = x_history.shape[0]

        # Get current factor state
        f_current = self.get_factors(x_history)  # (batch, n_factors)

        # Generate factor trajectories
        scenarios = []
        f_t = f_current.unsqueeze(1)  # (batch, 1, n_factors) for dynamics

        for _ in range(horizon):
            # Sample next factors
            f_next = self.dynamics.sample(f_t, n_samples=n_paths)  # (batch, n_paths, n_factors)

            # Emit observations
            f_next_flat = f_next.reshape(B * n_paths, self.n_factors)
            params = self.emission.forward(f_next_flat)
            mean = params['mean'].reshape(B, n_paths, self.n_assets)
            std = params['std'].reshape(1, self.n_assets).expand(B * n_paths, -1)

            # Sample returns
            noise = torch.randn(B * n_paths, self.n_assets) * std
            x_next = mean + noise.reshape(B, n_paths, self.n_assets)

            scenarios.append(x_next)
            f_t = f_next.reshape(B * n_paths, 1, self.n_factors)

        return torch.stack(scenarios, dim=2)  # (batch, n_paths, horizon, n_assets)


class GLPTrainer:
    """Training harness for GLP."""

    def __init__(self, model: GLP, lr: float = 1e-3, weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = next(model.parameters()).device

    def train_epoch(self, train_loader, epochs: int = 1) -> dict:
        """Train for epochs."""
        history = {'elbo': [], 'reconstruction': [], 'kl': [], 'dynamics': []}

        for epoch in range(epochs):
            epoch_elbos = []
            for x_batch in train_loader:
                x_batch = x_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(x_batch)
                loss = output['elbo']

                if not torch.isfinite(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

                epoch_elbos.append(loss.item())
                history['reconstruction'].append(float(output['reconstruction_loss']))
                history['kl'].append(float(output['kl_loss']))
                history['dynamics'].append(float(output['dynamics_loss']))

            history['elbo'].append(np.mean(epoch_elbos))

        return history
