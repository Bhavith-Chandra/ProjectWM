"""
World Model Trainer
====================
Training loop with proper loss composition:
  - Reconstruction losses (returns, volatility)
  - KL loss with DreamerV3 balancing
  - Symlog prediction loss
  - Covariance log-likelihood

Supports gradient clipping, LR scheduling, and mixed precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
from typing import Dict, Optional, Tuple

from .model import MeridianWorldModel
from .rssm import RSSM, symlog

logger = logging.getLogger(__name__)


class WorldModelTrainer:
    """
    Trains the MeridianWorldModel end-to-end.
    """

    def __init__(self, model: MeridianWorldModel, lr: float = 3e-4,
                 weight_decay: float = 1e-5, grad_clip: float = 100.0,
                 kl_weight: float = 1.0, return_weight: float = 1.0,
                 vol_weight: float = 0.5, cov_weight: float = 0.1,
                 device: str = 'cpu'):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.grad_clip = grad_clip

        # loss weights
        self.kl_weight = kl_weight
        self.return_weight = return_weight
        self.vol_weight = vol_weight
        self.cov_weight = cov_weight

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2
        )

    def compute_loss(self, obs: torch.Tensor,
                     returns: torch.Tensor,
                     realized_vol: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.

        Args:
            obs: (batch, seq_len, n_assets, n_features)
            returns: (batch, seq_len, n_assets) — raw returns
            realized_vol: optional (batch, seq_len, n_assets)
        """
        out = self.model(obs)

        # return prediction loss (symlog space)
        return_loss = F.mse_loss(out['returns'], symlog(returns))

        # KL divergence with DreamerV3 balancing
        kl_loss = RSSM.kl_loss(out['prior_logits'], out['post_logits'])

        total = self.return_weight * return_loss + self.kl_weight * kl_loss

        metrics = {
            'return_loss': return_loss.item(),
            'kl_loss': kl_loss.item(),
        }

        # volatility loss (if realized vol available)
        if realized_vol is not None:
            B, T, S = out['returns'].shape
            state = torch.cat([out['h'], out['z']], dim=-1)
            state_flat = state.reshape(B * T, -1)
            vol_loss = self.model.vol_head.loss(state_flat,
                                                 realized_vol.reshape(B * T, -1))
            total = total + self.vol_weight * vol_loss
            metrics['vol_loss'] = vol_loss.item()

        metrics['total_loss'] = total.item()
        return total, metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_metrics = {}
        n_batches = 0

        for batch in dataloader:
            obs = batch[0].to(self.device)
            returns = batch[1].to(self.device)
            realized_vol = batch[2].to(self.device) if len(batch) > 2 else None

            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(obs, returns, realized_vol)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1

        self.scheduler.step()

        return {k: v / n_batches for k, v in epoch_metrics.items()}

    def train(self, obs_data: np.ndarray, returns_data: np.ndarray,
              vol_data: Optional[np.ndarray] = None,
              n_epochs: int = 200, batch_size: int = 32,
              seq_len: int = 60, val_split: float = 0.1) -> list:
        """
        Full training run.

        Args:
            obs_data: (total_days, n_assets, n_features)
            returns_data: (total_days, n_assets)
            vol_data: optional (total_days, n_assets)
            n_epochs: training epochs
            batch_size: batch size
            seq_len: sequence length for each sample
        """
        # create sliding window sequences
        T = obs_data.shape[0]
        n_windows = T - seq_len
        if n_windows <= 0:
            raise ValueError(f"Need > {seq_len} days, got {T}")

        obs_windows = np.stack([obs_data[i:i+seq_len] for i in range(n_windows)])
        ret_windows = np.stack([returns_data[i:i+seq_len] for i in range(n_windows)])

        tensors = [
            torch.from_numpy(obs_windows).float(),
            torch.from_numpy(ret_windows).float(),
        ]
        if vol_data is not None:
            vol_windows = np.stack([vol_data[i:i+seq_len] for i in range(n_windows)])
            tensors.append(torch.from_numpy(vol_windows).float())

        # train/val split
        n_val = max(1, int(n_windows * val_split))
        n_train = n_windows - n_val

        train_ds = TensorDataset(*[t[:n_train] for t in tensors])
        val_ds = TensorDataset(*[t[n_train:] for t in tensors])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        history = []
        best_val = float('inf')

        for epoch in range(n_epochs):
            train_metrics = self.train_epoch(train_loader)

            # validation
            self.model.eval()
            val_metrics = {}
            n_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    obs = batch[0].to(self.device)
                    returns = batch[1].to(self.device)
                    rvol = batch[2].to(self.device) if len(batch) > 2 else None
                    _, metrics = self.compute_loss(obs, returns, rvol)
                    for k, v in metrics.items():
                        val_metrics[k] = val_metrics.get(k, 0) + v
                    n_val_batches += 1

            if n_val_batches > 0:
                val_metrics = {f'val_{k}': v / n_val_batches
                               for k, v in val_metrics.items()}

            record = {'epoch': epoch, **train_metrics, **val_metrics}
            history.append(record)

            val_loss = val_metrics.get('val_total_loss', float('inf'))
            if val_loss < best_val:
                best_val = val_loss

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"train={train_metrics['total_loss']:.4f} | "
                    f"val={val_loss:.4f} | "
                    f"KL={train_metrics['kl_loss']:.3f} | "
                    f"ret={train_metrics['return_loss']:.4f}"
                )

        return history
