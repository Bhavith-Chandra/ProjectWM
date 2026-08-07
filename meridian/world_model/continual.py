"""
Continual Learning Module
===========================
Handles distribution shift in financial markets via three mechanisms:

1. ADWIN drift detection (Bifet & Gavalda 2007):
   Adaptive windowing that detects distributional change points
   without needing a fixed window size.

2. EWC weight consolidation (Kirkpatrick et al. 2017):
   Elastic Weight Consolidation — protects important weights from
   catastrophic forgetting when adapting to new regimes.

3. Regime-tagged replay buffer:
   Stores representative samples per regime so the model can
   rehearse past regimes while learning new ones.

References:
  Bifet & Gavalda "Learning from Time-Changing Data with Adaptive Windowing" SDM 2007
  Kirkpatrick et al. "Overcoming Catastrophic Forgetting" PNAS 2017
  Buzzega et al. "Dark Experience for General Continual Learning" NeurIPS 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from collections import deque


class ADWINDetector:
    """
    ADWIN — ADaptive WINdowing for drift detection.

    Maintains a variable-length window. At each step, tests whether
    the distribution of the recent sub-window differs significantly
    from the older sub-window. If so, drops the oldest data.

    For financial data: monitors prediction error or loss, detects
    when the model's performance has shifted (= market regime change).
    """

    def __init__(self, delta: float = 0.01, max_size: int = 2000):
        self.delta = delta
        self.max_size = max_size
        self.window: deque = deque(maxlen=max_size)
        self.drift_detected = False
        self.n_detections = 0

    def add(self, value: float) -> bool:
        """
        Add a new observation and check for drift.
        Returns True if drift is detected.
        """
        self.window.append(value)
        self.drift_detected = False

        if len(self.window) < 10:
            return False

        data = list(self.window)
        n = len(data)

        for split in range(max(5, n // 4), n - 5):
            w0 = data[:split]
            w1 = data[split:]

            n0, n1 = len(w0), len(w1)
            mu0 = sum(w0) / n0
            mu1 = sum(w1) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)

            epsilon = math.sqrt(0.5 * math.log(4.0 / self.delta) / m)

            if abs(mu0 - mu1) > epsilon:
                for _ in range(split):
                    if self.window:
                        self.window.popleft()
                self.drift_detected = True
                self.n_detections += 1
                return True

        return False

    @property
    def mean(self) -> float:
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    @property
    def size(self) -> int:
        return len(self.window)


class EWCRegularizer(nn.Module):
    """
    Elastic Weight Consolidation.

    After learning a task/regime, computes the Fisher information
    matrix (diagonal approximation) which measures how important
    each parameter is. When learning a new task, penalizes changes
    to important parameters.

    L_ewc = lambda/2 * sum_i F_i * (theta_i - theta_i*)^2
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 100.0):
        super().__init__()
        self.model = model
        self.ewc_lambda = ewc_lambda

        self._stored_params: Dict[str, torch.Tensor] = {}
        self._fisher: Dict[str, torch.Tensor] = {}
        self._consolidated = False

    @torch.enable_grad()
    def compute_fisher(self, dataloader, loss_fn,
                       n_samples: int = 200) -> None:
        """
        Estimate diagonal Fisher information from data.
        Call this after training on a regime, before switching to the next.
        """
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}

        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break

            self.model.zero_grad()
            loss = loss_fn(batch)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
                    count += 1

        for n in fisher:
            fisher[n] /= max(1, count)

        if self._consolidated:
            for n in fisher:
                if n in self._fisher:
                    self._fisher[n] = 0.5 * (self._fisher[n] + fisher[n])
                else:
                    self._fisher[n] = fisher[n]
        else:
            self._fisher = fisher

        self._stored_params = {n: p.data.clone()
                               for n, p in self.model.named_parameters()
                               if p.requires_grad}
        self._consolidated = True

    def penalty(self) -> torch.Tensor:
        """
        EWC penalty: sum_i F_i * (theta_i - theta_i*)^2
        Add this to the training loss.
        """
        if not self._consolidated:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self._fisher:
                loss += (self._fisher[n] * (p - self._stored_params[n]) ** 2).sum()

        return self.ewc_lambda * loss / 2.0


class RegimeReplayBuffer:
    """
    Regime-tagged experience replay buffer.

    Stores representative samples per regime so the model can
    rehearse past regimes while learning new ones. Each regime
    gets a fixed budget of slots.
    """

    def __init__(self, per_regime_size: int = 200, n_regimes: int = 4):
        self.per_regime_size = per_regime_size
        self.n_regimes = n_regimes
        self.buffers: Dict[int, deque] = {
            i: deque(maxlen=per_regime_size) for i in range(n_regimes)
        }
        self.regime_counts = {i: 0 for i in range(n_regimes)}

    def add(self, sample: Dict[str, torch.Tensor], regime: int) -> None:
        """Add a sample tagged with its regime."""
        detached = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                    for k, v in sample.items()}
        detached['regime'] = regime
        self.buffers[regime].append(detached)
        self.regime_counts[regime] += 1

    def sample(self, batch_size: int,
               regime_weights: Optional[Dict[int, float]] = None
               ) -> List[Dict[str, torch.Tensor]]:
        """
        Sample a mixed batch from all regimes.
        regime_weights: optional per-regime sampling probability.
        Default: uniform over regimes that have data.
        """
        available = {r: buf for r, buf in self.buffers.items() if len(buf) > 0}
        if not available:
            return []

        if regime_weights is None:
            regime_weights = {r: 1.0 / len(available) for r in available}

        samples = []
        for _ in range(batch_size):
            regimes = list(available.keys())
            weights = [regime_weights.get(r, 0) for r in regimes]
            total = sum(weights)
            if total == 0:
                break
            probs = [w / total for w in weights]

            r_idx = torch.multinomial(torch.tensor(probs), 1).item()
            regime = regimes[r_idx]

            buf = available[regime]
            idx = torch.randint(len(buf), (1,)).item()
            samples.append(buf[idx])

        return samples

    @property
    def total_size(self) -> int:
        return sum(len(buf) for buf in self.buffers.values())

    def regime_summary(self) -> Dict[str, int]:
        return {f'regime_{r}': len(buf) for r, buf in self.buffers.items()}


class ContinualLearner(nn.Module):
    """
    Orchestrates continual learning for the world model:
    1. Monitors for drift via ADWIN
    2. On drift: consolidate Fisher, tag replay buffer
    3. During training: EWC penalty + replay from past regimes

    Usage:
      learner = ContinualLearner(model)
      for batch in stream:
          loss = model_loss(batch)
          loss += learner.step(batch, loss.item(), current_regime)
          loss.backward()
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 100.0,
                 drift_delta: float = 0.01, replay_size: int = 200,
                 n_regimes: int = 4, replay_ratio: float = 0.25):
        super().__init__()
        self.model = model
        self.replay_ratio = replay_ratio

        self.drift_detector = ADWINDetector(delta=drift_delta)
        self.ewc = EWCRegularizer(model, ewc_lambda)
        self.replay = RegimeReplayBuffer(replay_size, n_regimes)

        self.current_regime = 0
        self.n_regime_changes = 0
        self.steps = 0

    def step(self, sample: Dict[str, torch.Tensor],
             loss_value: float,
             regime: int = 0) -> torch.Tensor:
        """
        One continual learning step.
        Returns additional loss to add (EWC penalty).
        """
        self.steps += 1

        drift = self.drift_detector.add(loss_value)

        if drift:
            self.n_regime_changes += 1

        self.replay.add(sample, regime)

        if regime != self.current_regime:
            self.current_regime = regime

        ewc_loss = self.ewc.penalty()

        return ewc_loss

    def get_replay_batch(self, batch_size: int
                         ) -> List[Dict[str, torch.Tensor]]:
        """Get a replay batch for rehearsal."""
        replay_size = max(1, int(batch_size * self.replay_ratio))
        return self.replay.sample(replay_size)

    def consolidate(self, dataloader, loss_fn,
                    n_samples: int = 200) -> None:
        """
        Consolidate current knowledge into Fisher information.
        Call this when transitioning between training phases.
        """
        self.ewc.compute_fisher(dataloader, loss_fn, n_samples)

    def diagnostics(self) -> Dict[str, float]:
        return {
            'steps': self.steps,
            'drift_detections': self.drift_detector.n_detections,
            'adwin_window_size': self.drift_detector.size,
            'adwin_mean_loss': self.drift_detector.mean,
            'replay_buffer_size': self.replay.total_size,
            'n_regime_changes': self.n_regime_changes,
            'current_regime': self.current_regime,
            'ewc_consolidated': self.ewc._consolidated,
        }
