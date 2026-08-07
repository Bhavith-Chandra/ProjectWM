"""
Interpretability Module
========================
Temporal attribution, factor decomposition, and explanation generation.

References:
  Sundararajan et al. "Axiomatic Attribution" ICML 2017 (integrated gradients)
  Shrikumar et al. "DeepLIFT" ICML 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class TemporalAttribution(nn.Module):
    """
    Integrated gradients over the time axis.
    Identifies which timesteps drove the prediction.
    """

    def __init__(self, n_steps: int = 20):
        super().__init__()
        self.n_steps = n_steps

    @torch.enable_grad()
    def attribute(self, model_fn, inputs: torch.Tensor,
                  target_idx: Optional[int] = None) -> torch.Tensor:
        """
        model_fn: callable, inputs -> scalar or (batch,) output
        inputs: (batch, seq_len, features)
        returns: (batch, seq_len, features) — attribution scores
        """
        baseline = torch.zeros_like(inputs)
        inputs_req = inputs.detach().requires_grad_(True)

        total_grads = torch.zeros_like(inputs)

        for k in range(self.n_steps):
            alpha = k / self.n_steps
            interpolated = baseline + alpha * (inputs_req - baseline)
            interpolated = interpolated.detach().requires_grad_(True)

            output = model_fn(interpolated)
            if output.dim() > 1 and target_idx is not None:
                output = output[:, target_idx]
            output = output.sum()

            grad = torch.autograd.grad(output, interpolated,
                                       create_graph=False)[0]
            total_grads += grad

        attribution = (inputs - baseline) * total_grads / self.n_steps
        return attribution

    def temporal_importance(self, attribution: torch.Tensor) -> torch.Tensor:
        """Aggregate feature attributions per timestep."""
        return attribution.abs().sum(dim=-1)


class FactorDecomposer(nn.Module):
    """
    Decompose predictions into interpretable factor contributions:
    trend, mean-reversion, momentum, carry, macro, residual.
    """

    FACTORS = ['trend', 'mean_reversion', 'momentum', 'carry', 'macro', 'residual']

    def __init__(self, state_dim: int, n_assets: int = 35, hidden: int = 128):
        super().__init__()
        self.n_factors = len(self.FACTORS)
        self.n_assets = n_assets

        self.factor_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets * self.n_factors),
        )

        self.factor_gate = nn.Sequential(
            nn.Linear(state_dim, n_assets * self.n_factors),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor,
                prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        state: (batch, state_dim)
        prediction: (batch, n_assets) — the return prediction to decompose
        returns: dict mapping factor name -> (batch, n_assets) contribution
        """
        B = state.shape[0]
        raw = self.factor_net(state).reshape(B, self.n_assets, self.n_factors)
        gates = self.factor_gate(state).reshape(B, self.n_assets, self.n_factors)

        weighted = raw * gates
        scale = prediction.unsqueeze(-1) / weighted.sum(-1, keepdim=True).clamp(min=1e-8)
        contributions = weighted * scale

        result = {}
        for i, name in enumerate(self.FACTORS):
            result[name] = contributions[:, :, i]

        return result


class RegimeExplainer(nn.Module):
    """
    Explains regime transitions: what features triggered the change,
    relative importance, and directionality.
    """

    def __init__(self, state_dim: int, n_regimes: int = 4, hidden: int = 64):
        super().__init__()
        self.n_regimes = n_regimes

        self.transition_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_regimes * n_regimes),
        )

        self.driver_net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, state_prev: torch.Tensor,
                state_curr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        state_prev, state_curr: (batch, state_dim)
        returns: transition probabilities and driver importance
        """
        B = state_prev.shape[0]
        combined = torch.cat([state_prev, state_curr], dim=-1)

        trans_logits = self.transition_net(combined).reshape(
            B, self.n_regimes, self.n_regimes
        )
        trans_probs = F.softmax(trans_logits, dim=-1)

        driver_importance = self.driver_net(combined)

        return {
            'transition_probs': trans_probs,
            'driver_importance': driver_importance,
        }


class ExplanationGenerator(nn.Module):
    """
    Generates natural-language explanations from model state.
    Template-based: maps internal signals to human-readable text.
    """

    SIGNAL_TEMPLATES = {
        'bullish_momentum': '{asset} showing bullish momentum (factor contribution: {value:.2%})',
        'bearish_momentum': '{asset} showing bearish momentum (factor contribution: {value:.2%})',
        'high_vol': 'elevated volatility in {asset} ({value:.1%} annualized)',
        'regime_shift': 'regime transition detected: {from_regime} → {to_regime} (confidence: {value:.0%})',
        'topological_warning': 'topological early warning: persistence norm elevated ({value:.2f})',
        'reflexivity_high': 'reflexivity gauge elevated (rho={value:.3f}), feedback loops intensifying',
        'causal_driver': '{asset} identified as causal driver of {target} (edge weight: {value:.3f})',
        'hurst_trending': '{asset} in trending regime (H={value:.3f} > 0.5)',
        'hurst_reverting': '{asset} in mean-reverting regime (H={value:.3f} < 0.5)',
        'systemic_risk': 'sheaf cohomology elevated: systemic risk signal ({value:.3f})',
    }

    def __init__(self, n_assets: int = 35):
        super().__init__()
        self.n_assets = n_assets

    def generate(self, signals: Dict[str, torch.Tensor],
                 asset_names: Optional[List[str]] = None,
                 top_k: int = 5) -> List[str]:
        """
        signals: dict of named tensors from various model components
        asset_names: list of asset ticker names
        top_k: max number of explanations to generate
        returns: list of explanation strings
        """
        if asset_names is None:
            asset_names = [f'Asset_{i}' for i in range(self.n_assets)]

        explanations = []

        if 'factor_contributions' in signals:
            fc = signals['factor_contributions']
            for factor_name, contribs in fc.items():
                if factor_name == 'momentum':
                    vals = contribs[0]
                    top_pos = vals.topk(min(2, len(vals)))
                    for idx, val in zip(top_pos.indices, top_pos.values):
                        if val > 0.001:
                            explanations.append(
                                self.SIGNAL_TEMPLATES['bullish_momentum'].format(
                                    asset=asset_names[idx], value=val.item()
                                )
                            )

        if 'rho' in signals:
            rho = signals['rho'][0, 0].item()
            if rho > 0.7:
                explanations.append(
                    self.SIGNAL_TEMPLATES['reflexivity_high'].format(value=rho)
                )

        if 'hurst' in signals:
            hurst = signals['hurst'][0]
            for i, h in enumerate(hurst):
                if h > 0.6:
                    explanations.append(
                        self.SIGNAL_TEMPLATES['hurst_trending'].format(
                            asset=asset_names[i] if i < len(asset_names) else f'Asset_{i}',
                            value=h.item()
                        )
                    )
                elif h < 0.4:
                    explanations.append(
                        self.SIGNAL_TEMPLATES['hurst_reverting'].format(
                            asset=asset_names[i] if i < len(asset_names) else f'Asset_{i}',
                            value=h.item()
                        )
                    )

        if 'systemic_risk' in signals:
            sr = signals['systemic_risk'][0, 0].item()
            if abs(sr) > 0.5:
                explanations.append(
                    self.SIGNAL_TEMPLATES['systemic_risk'].format(value=sr)
                )

        return explanations[:top_k]
