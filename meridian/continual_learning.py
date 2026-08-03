"""Continual learning harness for online model updates with live market data.

Handles:
  - Live data ingestion (ticks, news, order book)
  - Experience replay with priority weighting (recent + large moves)
  - Catastrophic forgetting prevention (EWC, replay buffer strategy)
  - Regime detection (when model enters unfamiliar territory)
  - Adaptive learning rates based on data novelty
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import json


@dataclass
class MarketTick:
    """A single market observation."""
    timestamp: datetime
    returns: np.ndarray  # (n_assets,)
    volume: np.ndarray  # (n_assets,)
    volatility: np.ndarray  # Realized vol or IV
    sentiment: Optional[float] = None  # Sentiment score if available
    order_book_depth: Optional[Dict] = None
    news: Optional[List[str]] = None  # News stories
    shock_indicator: Optional[str] = None  # Label if this is a known shock


class ExperienceReplayBuffer:
    """Priority-weighted experience replay for continual learning."""

    def __init__(self, max_size: int = 10000, alpha_priority: float = 0.6):
        """
        max_size: maximum number of experiences to store
        alpha_priority: priority weighting exponent (0 = uniform, 1 = full priority)
        """
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.alpha_priority = alpha_priority
        self.max_priority = 1.0

    def add(self, experience: Dict, priority: Optional[float] = None):
        """Add experience to buffer with priority."""
        self.buffer.append(experience)

        if priority is None:
            # Default: new experiences get max priority
            priority = self.max_priority
        else:
            self.max_priority = max(self.max_priority, priority)

        self.priorities.append(priority)

    def sample(self, batch_size: int) -> Tuple[List[Dict], np.ndarray]:
        """Sample batch prioritized by experience importance."""
        if len(self.buffer) == 0:
            return [], np.array([])

        # Compute sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities ** self.alpha_priority
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probabilities, replace=False)

        # Extract experiences
        experiences = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probabilities[indices]) ** (-0.4)  # IS correction (beta=0.4)
        weights /= weights.max()  # Normalize

        return experiences, weights

    def decay_priorities(self, decay_rate: float = 0.99):
        """Decay priorities of old experiences (prefer recent data)."""
        self.priorities = deque([p * decay_rate for p in self.priorities], maxlen=self.buffer.maxlen)


class RegimeDetector:
    """Detect when model enters a new/unfamiliar regime."""

    def __init__(self, n_features: int = 11, window_size: int = 20):
        self.n_features = n_features
        self.window_size = window_size
        self.historical_stats = {
            'mean': np.zeros(n_features),
            'std': np.ones(n_features),
            'max_vol': 0.0,
            'min_vol': np.inf
        }
        self.anomaly_scores = deque(maxlen=window_size)

    def update_historical(self, x: np.ndarray):
        """Update baseline statistics (x: (n_features,) or (batch, n_features))."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        self.historical_stats['mean'] = 0.95 * self.historical_stats['mean'] + 0.05 * x.mean(axis=0)
        self.historical_stats['std'] = 0.95 * self.historical_stats['std'] + 0.05 * x.std(axis=0)
        self.historical_stats['max_vol'] = max(self.historical_stats['max_vol'], x.std())
        self.historical_stats['min_vol'] = min(self.historical_stats['min_vol'], x.std())

    def detect_anomaly(self, x: np.ndarray, threshold: float = 2.0) -> Tuple[bool, float]:
        """
        Detect if x is anomalous (many std devs from mean).

        Returns: (is_anomalous, anomaly_score)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Z-score
        z_scores = np.abs((x - self.historical_stats['mean']) / (self.historical_stats['std'] + 1e-8))
        anomaly_score = z_scores.mean()

        is_anomalous = anomaly_score > threshold

        self.anomaly_scores.append(anomaly_score)

        return is_anomalous, anomaly_score

    def is_regime_shift(self) -> bool:
        """Detect if recent anomaly scores indicate regime shift."""
        if len(self.anomaly_scores) < self.window_size:
            return False

        scores = np.array(self.anomaly_scores)
        # If recent anomalies are persistently high, regime shift
        return (scores[-10:] > 1.5).mean() > 0.7


class ContinualLearningHarness:
    """Full harness for online learning with live data."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer_size: int = 5000,
        batch_size: int = 64,
        update_frequency: int = 10,  # Update every 10 ticks
    ):
        """
        model: world model to train
        optimizer: torch optimizer
        replay_buffer_size: max replay buffer size
        batch_size: batch size for training
        update_frequency: how often to perform training step
        """
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.update_frequency = update_frequency

        self.replay_buffer = ExperienceReplayBuffer(max_size=replay_buffer_size)
        self.regime_detector = RegimeDetector()

        self.tick_counter = 0
        self.training_steps = 0
        self.regime_shifts = []

        # Fisher information matrix (for EWC)
        self.fisher_information = None
        self.prev_weights = None

    def ingest_tick(self, tick: MarketTick) -> Dict:
        """
        Ingest a market tick and potentially trigger training.

        Returns: dict with status, metrics
        """
        self.tick_counter += 1

        # Detect anomalies/regime shifts
        is_anomalous, anomaly_score = self.regime_detector.detect_anomaly(tick.returns)
        is_regime_shift = self.regime_detector.is_regime_shift()

        # Priority = magnitude of move + recency
        priority = 1.0 + np.abs(tick.returns).mean()
        if is_anomalous:
            priority *= 2.0  # Anomalies get higher priority

        # Store experience
        experience = {
            'timestamp': tick.timestamp,
            'returns': tick.returns,
            'volume': tick.volume,
            'volatility': tick.volatility,
            'sentiment': tick.sentiment,
            'anomaly': is_anomalous,
            'anomaly_score': anomaly_score
        }

        self.replay_buffer.add(experience, priority=priority)
        self.regime_detector.update_historical(tick.returns)

        # Training step if counter reached
        output = {
            'tick': self.tick_counter,
            'anomalous': is_anomalous,
            'anomaly_score': anomaly_score,
            'regime_shift': is_regime_shift,
            'trained': False
        }

        if self.tick_counter % self.update_frequency == 0:
            train_output = self.train_batch()
            output.update(train_output)

        if is_regime_shift:
            self.regime_shifts.append({
                'timestamp': tick.timestamp,
                'tick_counter': self.tick_counter,
                'anomaly_score': anomaly_score
            })
            output['regime_shift_detected'] = True

        return output

    def train_batch(self) -> Dict:
        """Perform one training batch."""
        experiences, weights = self.replay_buffer.sample(self.batch_size)

        if len(experiences) == 0:
            return {'trained': False, 'loss': None}

        # Convert to tensors
        returns_batch = torch.tensor([e['returns'] for e in experiences], dtype=torch.float32)
        volumes_batch = torch.tensor([e['volume'] for e in experiences], dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # Forward pass
        self.optimizer.zero_grad()

        # Compute loss (model-specific)
        if hasattr(self.model, 'elbo'):
            # VAE/RSSM style
            loss = self.model.elbo(returns_batch.unsqueeze(1))  # Add sequence dimension
        elif hasattr(self.model, 'forward'):
            output = self.model(returns_batch.unsqueeze(1))
            loss = output['loss']
        else:
            return {'trained': False, 'loss': None}

        # Apply importance weights
        loss = (loss * weights_tensor).mean()

        # EWC regularization (prevent catastrophic forgetting)
        if self.fisher_information is not None:
            ewc_loss = self._compute_ewc_loss()
            loss = loss + 0.4 * ewc_loss

        # Backward pass
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            self.training_steps += 1

        # Decay old experiences
        self.replay_buffer.decay_priorities(decay_rate=0.99)

        return {
            'trained': True,
            'loss': float(loss.detach().item()),
            'training_steps': self.training_steps
        }

    def _compute_ewc_loss(self) -> torch.Tensor:
        """Elastic Weight Consolidation regularization."""
        ewc_loss = 0
        for (name, param), (_, old_param) in zip(
            self.model.named_parameters(),
            zip(*[self.model.named_parameters()])  # Get old params
        ):
            if name in self.fisher_information:
                F_i = self.fisher_information[name]
                param_loss = F_i * (param - old_param) ** 2
                ewc_loss += param_loss.sum()

        return ewc_loss

    def compute_fisher_information(self, data_loader, n_batches: int = 10):
        """
        Compute Fisher information matrix on current data.
        Used to weight importance of parameters for EWC.
        """
        self.fisher_information = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        self.model.eval()
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            self.optimizer.zero_grad()

            if hasattr(self.model, 'elbo'):
                loss = self.model.elbo(batch)
            else:
                output = self.model(batch)
                loss = output['loss']

            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += (param.grad ** 2) / n_batches

        self.prev_weights = {name: param.clone().detach() for name, param in self.model.named_parameters()}

    def save_checkpoint(self, path: str):
        """Save model and training state."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'tick_counter': self.tick_counter,
            'training_steps': self.training_steps,
            'regime_shifts': self.regime_shifts
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model and training state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.tick_counter = checkpoint['tick_counter']
        self.training_steps = checkpoint['training_steps']
        self.regime_shifts = checkpoint.get('regime_shifts', [])

    def get_status(self) -> Dict:
        """Get current learning status."""
        return {
            'ticks_ingested': self.tick_counter,
            'training_steps': self.training_steps,
            'buffer_size': len(self.replay_buffer.buffer),
            'regime_shifts_detected': len(self.regime_shifts),
            'recent_anomalies': list(self.regime_detector.anomaly_scores)[-10:]
        }
