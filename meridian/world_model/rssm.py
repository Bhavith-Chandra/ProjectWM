"""
Recurrent State-Space Model (RSSM) for Financial Markets
=========================================================
DreamerV3-inspired latent dynamics model adapted for multi-asset finance.

Architecture:
  h_t = GRU(h_{t-1}, z_{t-1})           # deterministic path
  prior:     p(z_t | h_t)                # dynamics predictor
  posterior: q(z_t | h_t, x_t)           # encoder (uses observations)

Latent z_t uses 32 categorical variables × 32 classes (1024-dim one-hot).
Discrete latents capture multimodal market regimes better than Gaussians.

Symlog transform handles fat-tailed return distributions:
  symlog(x) = sign(x) * ln(|x| + 1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class CategoricalLatent(nn.Module):
    """
    Discrete categorical latent variable (DreamerV3 style).
    32 categorical variables × 32 classes = 1024-dim one-hot.
    Straight-through gradients + 1% uniform mixture for stability.
    """

    def __init__(self, input_dim: int, n_categoricals: int = 32,
                 n_classes: int = 32, unimix: float = 0.01):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.unimix = unimix
        self.proj = nn.Linear(input_dim, n_categoricals * n_classes)

    @property
    def latent_dim(self) -> int:
        return self.n_categoricals * self.n_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (sample, logits). Sample uses straight-through."""
        logits = self.proj(x).reshape(-1, self.n_categoricals, self.n_classes)

        # uniform mixture for stability
        probs = F.softmax(logits, dim=-1)
        probs = (1 - self.unimix) * probs + self.unimix / self.n_classes

        # straight-through: sample in forward, use probs in backward
        if self.training:
            sample = F.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
            sample = F.one_hot(probs.argmax(-1), self.n_classes).float()

        sample_flat = sample.reshape(-1, self.latent_dim)
        logits_flat = logits.reshape(-1, self.n_categoricals * self.n_classes)
        return sample_flat, logits_flat

    def log_prob(self, logits: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """KL-friendly log probability."""
        logits = logits.reshape(-1, self.n_categoricals, self.n_classes)
        sample = sample.reshape(-1, self.n_categoricals, self.n_classes)
        log_probs = F.log_softmax(logits, dim=-1)
        return (sample * log_probs).sum(dim=(-2, -1))


class RSSM(nn.Module):
    """
    Financial Recurrent State-Space Model.

    State = (h, z) where:
      h: deterministic GRU hidden state (captures slow trends)
      z: stochastic discrete latent (captures regime/mode)

    Forward model (imagination):
      h_t = GRU(h_{t-1}, z_{t-1})
      z_t ~ p(z_t | h_t)

    Inference (with observations):
      h_t = GRU(h_{t-1}, z_{t-1})
      z_t ~ q(z_t | h_t, embed(x_t))
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 512,
                 n_categoricals: int = 32, n_classes: int = 32,
                 embed_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        latent_dim = n_categoricals * n_classes

        # observation embedding
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

        # recurrent backbone
        self.gru = nn.GRUCell(latent_dim, hidden_dim)

        # prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.prior_head = CategoricalLatent(embed_dim, n_categoricals, n_classes)

        # posterior: q(z_t | h_t, x_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )
        self.posterior_head = CategoricalLatent(embed_dim, n_categoricals, n_classes)

    def initial_state(self, batch_size: int,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.prior_head.latent_dim, device=device)
        return h, z

    def observe_step(self, obs: torch.Tensor, h: torch.Tensor,
                     z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single step with observation (training)."""
        # deterministic path
        h = self.gru(z, h)

        # embed observation
        obs_emb = self.obs_embed(symlog(obs))

        # prior
        prior_feat = self.prior_net(h)
        prior_z, prior_logits = self.prior_head(prior_feat)

        # posterior (uses observation)
        post_feat = self.posterior_net(torch.cat([h, obs_emb], dim=-1))
        post_z, post_logits = self.posterior_head(post_feat)

        return {
            'h': h,
            'z': post_z,
            'prior_logits': prior_logits,
            'post_logits': post_logits,
        }

    def imagine_step(self, h: torch.Tensor,
                     z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single step without observation (imagination/forecasting)."""
        h = self.gru(z, h)
        prior_feat = self.prior_net(h)
        z, logits = self.prior_head(prior_feat)
        return {'h': h, 'z': z, 'prior_logits': logits}

    def observe_sequence(self, obs_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process full observation sequence.
        obs_seq: (batch, seq_len, obs_dim)

        Returns dict of stacked states and logits.
        """
        B, T, _ = obs_seq.shape
        h, z = self.initial_state(B, obs_seq.device)

        hs, zs = [], []
        prior_logits_list, post_logits_list = [], []

        for t in range(T):
            out = self.observe_step(obs_seq[:, t], h, z)
            h, z = out['h'], out['z']
            hs.append(h)
            zs.append(z)
            prior_logits_list.append(out['prior_logits'])
            post_logits_list.append(out['post_logits'])

        return {
            'h': torch.stack(hs, dim=1),
            'z': torch.stack(zs, dim=1),
            'prior_logits': torch.stack(prior_logits_list, dim=1),
            'post_logits': torch.stack(post_logits_list, dim=1),
        }

    def imagine_sequence(self, h: torch.Tensor, z: torch.Tensor,
                         horizon: int) -> Dict[str, torch.Tensor]:
        """
        Imagine forward from a state (no observations).
        Used for scenario generation.
        """
        hs, zs = [], []
        for _ in range(horizon):
            out = self.imagine_step(h, z)
            h, z = out['h'], out['z']
            hs.append(h)
            zs.append(z)

        return {
            'h': torch.stack(hs, dim=1),
            'z': torch.stack(zs, dim=1),
        }

    @staticmethod
    def kl_loss(prior_logits: torch.Tensor, post_logits: torch.Tensor,
                alpha: float = 0.8, free_nats: float = 1.0) -> torch.Tensor:
        """
        KL balancing (DreamerV3): alpha weight on training prior toward posterior.
        free_nats prevents posterior collapse.
        """
        n_cat = 32
        n_cls = prior_logits.shape[-1] // n_cat

        prior = prior_logits.reshape(-1, n_cat, n_cls)
        post = post_logits.reshape(-1, n_cat, n_cls)

        prior_dist = torch.distributions.Categorical(logits=prior)
        post_dist = torch.distributions.Categorical(logits=post)

        # per-categorical KL, summed
        kl_per_cat = torch.distributions.kl_divergence(post_dist, prior_dist)
        kl = kl_per_cat.sum(-1)

        # KL balancing
        dyn_loss = torch.clamp(
            torch.distributions.kl_divergence(
                torch.distributions.Categorical(logits=post.detach()),
                prior_dist
            ).sum(-1),
            min=free_nats
        )
        rep_loss = torch.clamp(
            torch.distributions.kl_divergence(
                post_dist,
                torch.distributions.Categorical(logits=prior.detach())
            ).sum(-1),
            min=free_nats
        )

        return alpha * dyn_loss.mean() + (1 - alpha) * rep_loss.mean()
