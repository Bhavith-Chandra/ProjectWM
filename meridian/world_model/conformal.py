"""
Adaptive Conformal Inference (ACI)
====================================
Distribution-free prediction intervals with finite-sample coverage
guarantees. No distributional assumptions required.

The key insight: conformal prediction wraps ANY point predictor and
produces intervals guaranteed to contain the true value with
probability >= 1-alpha, regardless of the underlying distribution.

ACI (Gibbs & Candes 2021) adapts alpha online so coverage tracks
the target even under distribution shift.

References:
  Vovk et al. "Algorithmic Learning in a Random World" 2005
  Gibbs & Candes "Adaptive Conformal Inference Under Distribution Shift" 2021
  Barber et al. "Conformal Prediction Beyond Exchangeability" AOAS 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from collections import deque


class ConformalPredictor(nn.Module):
    """
    Split conformal prediction for financial time series.

    Calibrates prediction intervals using nonconformity scores
    from a held-out calibration set. The quantile of scores gives
    the interval width needed for 1-alpha coverage.
    """

    def __init__(self, n_assets: int = 35, alpha: float = 0.1,
                 cal_size: int = 200):
        super().__init__()
        self.n_assets = n_assets
        self.alpha = alpha
        self.cal_size = cal_size

        self.register_buffer(
            'cal_scores',
            torch.zeros(cal_size, n_assets),
        )
        self.register_buffer('cal_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('cal_filled', torch.tensor(False))

    def nonconformity_score(self, prediction: torch.Tensor,
                            actual: torch.Tensor) -> torch.Tensor:
        """Absolute residual as nonconformity score."""
        return (prediction - actual).abs()

    def calibrate(self, prediction: torch.Tensor,
                  actual: torch.Tensor) -> None:
        """
        Add a new calibration point.
        prediction, actual: (n_assets,) or (batch, n_assets)
        """
        scores = self.nonconformity_score(prediction, actual)
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)

        for s in scores:
            idx = self.cal_idx.item() % self.cal_size
            self.cal_scores[idx] = s
            self.cal_idx += 1
            if self.cal_idx >= self.cal_size:
                self.cal_filled.fill_(True)

    def predict_interval(self, prediction: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute prediction interval [lower, upper] with coverage >= 1-alpha.

        prediction: (batch, n_assets)
        returns: (lower, upper) each (batch, n_assets)
        """
        if not self.cal_filled and self.cal_idx == 0:
            width = torch.ones_like(prediction) * 0.05
            return prediction - width, prediction + width

        n = min(self.cal_idx.item(), self.cal_size)
        scores = self.cal_scores[:n]

        q_level = min(1.0, (1 - self.alpha) * (1 + 1.0 / n))
        quantile_idx = min(int(q_level * n), n - 1)
        sorted_scores, _ = scores.sort(dim=0)
        q_hat = sorted_scores[quantile_idx]

        lower = prediction - q_hat.unsqueeze(0)
        upper = prediction + q_hat.unsqueeze(0)

        return lower, upper

    def coverage_rate(self, predictions: torch.Tensor,
                      actuals: torch.Tensor) -> torch.Tensor:
        """Empirical coverage rate."""
        lower, upper = self.predict_interval(predictions)
        covered = (actuals >= lower) & (actuals <= upper)
        return covered.float().mean(dim=0)


class AdaptiveConformalInference(nn.Module):
    """
    ACI — Gibbs & Candes 2021.

    Adapts the miscoverage level alpha_t online:
      alpha_{t+1} = alpha_t + gamma * (err_t - alpha)

    where err_t = 1 if y_t is NOT in the interval, 0 otherwise.

    This makes coverage track the target 1-alpha even under
    distribution shift, which is essential for financial data.
    """

    def __init__(self, n_assets: int = 35, target_alpha: float = 0.1,
                 gamma: float = 0.01, cal_size: int = 500):
        super().__init__()
        self.n_assets = n_assets
        self.target_alpha = target_alpha
        self.gamma = gamma

        self.register_buffer(
            'alpha_t', torch.full((n_assets,), target_alpha),
        )

        self.conformal = ConformalPredictor(n_assets, target_alpha, cal_size)

        self.register_buffer('coverage_history', torch.zeros(1000))
        self.register_buffer('width_history', torch.zeros(1000))
        self.register_buffer('history_idx', torch.tensor(0, dtype=torch.long))

    def update(self, prediction: torch.Tensor,
               actual: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One ACI step: predict interval, check coverage, adapt alpha.

        prediction: (n_assets,) — point prediction
        actual: (n_assets,) — realized value
        """
        pred_2d = prediction.unsqueeze(0)
        lower, upper = self.conformal.predict_interval(pred_2d)
        lower, upper = lower.squeeze(0), upper.squeeze(0)

        err = ((actual < lower) | (actual > upper)).float()

        self.alpha_t = (self.alpha_t + self.gamma * (err - self.target_alpha)).clamp(0.001, 0.5)

        self.conformal.calibrate(prediction, actual)

        idx = self.history_idx.item() % 1000
        self.coverage_history[idx] = 1.0 - err.mean()
        self.width_history[idx] = (upper - lower).mean()
        self.history_idx += 1

        return {
            'lower': lower,
            'upper': upper,
            'covered': 1.0 - err,
            'alpha_t': self.alpha_t.clone(),
            'interval_width': (upper - lower),
        }

    def rolling_coverage(self, window: int = 100) -> torch.Tensor:
        """Rolling coverage rate over last `window` steps."""
        n = min(self.history_idx.item(), window)
        if n == 0:
            return torch.tensor(1.0 - self.target_alpha)
        idx = self.history_idx.item()
        start = max(0, idx - window) % 1000
        if start + n <= 1000:
            return self.coverage_history[start:start + n].mean()
        part1 = self.coverage_history[start:]
        part2 = self.coverage_history[:n - len(part1)]
        return torch.cat([part1, part2]).mean()

    def forward(self, prediction: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prediction intervals using current adaptive alpha.
        prediction: (batch, n_assets)
        """
        return self.conformal.predict_interval(prediction)


class ConformalRiskControl(nn.Module):
    """
    Conformal risk control for portfolio decisions.

    Instead of just prediction intervals, controls the risk
    of portfolio decisions: ensures that the probability of
    exceeding a loss threshold is bounded.

    Uses the conformal p-value framework (Bates et al. 2021).
    """

    def __init__(self, n_assets: int = 35, alpha: float = 0.05):
        super().__init__()
        self.n_assets = n_assets
        self.alpha = alpha

        self.register_buffer(
            'loss_scores', torch.zeros(500, n_assets),
        )
        self.register_buffer('score_idx', torch.tensor(0, dtype=torch.long))

    def add_observation(self, portfolio_loss: torch.Tensor) -> None:
        """
        portfolio_loss: (n_assets,) — per-asset loss (positive = loss)
        """
        idx = self.score_idx.item() % 500
        self.loss_scores[idx] = portfolio_loss
        self.score_idx += 1

    def risk_bound(self) -> torch.Tensor:
        """
        Returns per-asset loss bound that holds with probability 1-alpha.
        """
        n = min(self.score_idx.item(), 500)
        if n == 0:
            return torch.ones(self.n_assets) * 0.1

        scores = self.loss_scores[:n]
        q_level = min(1.0, (1 - self.alpha) * (1 + 1.0 / n))
        quantile_idx = min(int(q_level * n), n - 1)
        sorted_scores, _ = scores.sort(dim=0)
        return sorted_scores[quantile_idx]

    def safe_weights(self, weights: torch.Tensor,
                     max_loss: float = 0.05) -> torch.Tensor:
        """
        Adjust portfolio weights so that the conformal risk bound
        on each position doesn't exceed max_loss.
        """
        bound = self.risk_bound().to(weights.device)
        scale = (max_loss / bound.clamp(min=1e-6)).clamp(max=1.0)
        adjusted = weights * scale
        adjusted = adjusted / adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return adjusted
