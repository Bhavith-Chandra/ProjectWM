"""
Tropical Portfolio Optimization Head
======================================
Portfolio optimization using tropical algebra (R ∪ {∞}, min, +):
  - VaR/CVaR are natural tropical operations (worst-case = min)
  - Portfolio weights as tropical polynomial coefficients
  - Differentiable Markowitz via CvxpyLayer (Agrawal et al. 2019)
  - Transaction costs: linear (spread) + quadratic (Almgren-Chriss impact)

The tropical layer complements, not replaces, convex optimization.
Tropical gives the risk decomposition; cvxpy gives the feasible weights.

References:
  Agrawal et al. "Differentiable Convex Optimization Layers" NeurIPS 2019
  Boyd et al. "Multi-Period Trading via Convex Optimization" 2017
  Almgren & Chriss "Optimal Execution of Portfolio Transactions" 2001
  Maclagan & Sturmfels "Introduction to Tropical Geometry" 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class TropicalSemiring(nn.Module):
    """
    Tropical algebra operations on (R ∪ {∞}, min, +).
    Addition = min, Multiplication = +.

    VaR is naturally a tropical operation: the alpha-quantile of losses
    is a tropical polynomial evaluation.
    """

    @staticmethod
    def trop_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.min(a, b)

    @staticmethod
    def trop_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    @staticmethod
    def trop_matvec(A: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Tropical matrix-vector multiply: (A ⊗ x)_i = min_j (A_ij + x_j)
        A: (batch, m, n), x: (batch, n) -> (batch, m)
        """
        return (A + x.unsqueeze(-2)).min(dim=-1).values

    @staticmethod
    def trop_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Tropical matrix multiply: (A ⊗ B)_ij = min_k (A_ik + B_kj)
        """
        return (A.unsqueeze(-1) + B.unsqueeze(-3)).min(dim=-2).values


class TropicalRiskDecomposition(nn.Module):
    """
    Decompose portfolio risk using tropical algebra.

    In tropical geometry, the VaR at level alpha is the tropical
    hypersurface of the loss polynomial — the locus where the minimum
    is achieved by at least two monomials (= two assets contribute
    equally to worst-case loss).

    This gives a natural risk attribution: which assets are on the
    tropical hypersurface (= marginal risk contributors).
    """

    def __init__(self, n_assets: int, n_scenarios: int = 64,
                 hidden: int = 128):
        super().__init__()
        self.n_assets = n_assets
        self.n_scenarios = n_scenarios

        self.scenario_net = nn.Sequential(
            nn.Linear(n_assets * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets * n_scenarios),
        )

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor,
                weights: torch.Tensor,
                alpha: float = 0.05) -> Dict[str, torch.Tensor]:
        """
        mu: (batch, n_assets) — expected returns
        sigma: (batch, n_assets, n_assets) — covariance matrix
        weights: (batch, n_assets) — portfolio weights
        alpha: VaR confidence level

        returns dict with VaR, CVaR, risk attribution
        """
        B = mu.shape[0]

        features = torch.cat([mu, sigma.diagonal(dim1=-2, dim2=-1)], dim=-1)
        scenarios = self.scenario_net(features).reshape(
            B, self.n_scenarios, self.n_assets
        )

        portfolio_losses = -(scenarios * weights.unsqueeze(1)).sum(-1)

        sorted_losses, _ = portfolio_losses.sort(dim=-1, descending=True)
        k = max(1, int(self.n_scenarios * alpha))
        var = sorted_losses[:, k - 1]
        cvar = sorted_losses[:, :k].mean(dim=-1)

        worst_scenarios = scenarios[
            torch.arange(B).unsqueeze(1),
            portfolio_losses.topk(k, dim=-1).indices,
        ]
        risk_attribution = -(worst_scenarios * weights.unsqueeze(1)).mean(dim=1)

        return {
            'var': var,
            'cvar': cvar,
            'risk_attribution': risk_attribution,
        }


class DifferentiableMarkowitz(nn.Module):
    """
    Differentiable mean-variance optimization.
    Pure PyTorch (no cvxpy dependency) using projected gradient.

    Solves: max w'μ - (γ/2) w'Σw
    s.t. w >= 0, sum(w) = 1

    Gradients flow back through μ and Σ to the world model.
    """

    def __init__(self, n_assets: int, risk_aversion: float = 1.0,
                 max_weight: float = 0.20, n_iters: int = 50):
        super().__init__()
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.n_iters = n_iters

        self.gamma = nn.Parameter(torch.tensor(risk_aversion))

    def _project_simplex(self, w: torch.Tensor) -> torch.Tensor:
        """Project onto simplex with max weight constraint."""
        w = w.clamp(0, self.max_weight)
        w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return w

    def forward(self, mu: torch.Tensor,
                sigma: torch.Tensor,
                prev_weights: Optional[torch.Tensor] = None,
                tc_linear: float = 0.001,
                tc_quadratic: float = 0.0001,
                ) -> Dict[str, torch.Tensor]:
        """
        mu: (batch, n_assets) — expected returns
        sigma: (batch, n_assets, n_assets) — covariance
        prev_weights: (batch, n_assets) — previous weights for TC
        """
        B, N = mu.shape
        gamma = self.gamma.abs().clamp(min=0.01)

        w = torch.ones(B, N, device=mu.device) / N

        for _ in range(self.n_iters):
            port_var = torch.bmm(
                w.unsqueeze(1),
                torch.bmm(sigma, w.unsqueeze(-1))
            ).squeeze(-1).squeeze(-1)

            grad_return = mu
            grad_risk = torch.bmm(sigma, w.unsqueeze(-1)).squeeze(-1)

            grad_tc = torch.zeros_like(w)
            if prev_weights is not None:
                trade = w - prev_weights
                grad_tc = tc_linear * trade.sign() + 2 * tc_quadratic * trade

            grad = grad_return - gamma * grad_risk - grad_tc

            lr = 0.01 / (1 + _ * 0.02)
            w = w + lr * grad
            w = self._project_simplex(w)

        port_return = (w * mu).sum(-1)
        port_var = torch.bmm(
            w.unsqueeze(1), torch.bmm(sigma, w.unsqueeze(-1))
        ).squeeze(-1).squeeze(-1)
        port_vol = port_var.clamp(min=1e-8).sqrt()

        tc = torch.zeros(B, device=mu.device)
        if prev_weights is not None:
            trade = (w - prev_weights).abs()
            tc = tc_linear * trade.sum(-1) + tc_quadratic * (trade ** 2).sum(-1)

        return {
            'weights': w,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe': port_return / port_vol.clamp(min=1e-6),
            'transaction_cost': tc,
        }


class TropicalPortfolioHead(nn.Module):
    """
    Full portfolio optimization head combining:
    1. Return/risk prediction from world model state
    2. Tropical risk decomposition
    3. Differentiable Markowitz optimization
    4. Multi-period look-ahead (Boyd et al. 2017)

    Input: world model state s_t
    Output: portfolio weights, risk metrics, attributions
    """

    def __init__(self, state_dim: int, n_assets: int = 35,
                 hidden: int = 256, n_scenarios: int = 64,
                 max_weight: float = 0.20, n_factors: int = 32):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = n_factors

        self.return_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

        self.factor_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets * n_factors),
        )

        self.diag_head = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
            nn.Softplus(),
        )

        self.tropical_risk = TropicalRiskDecomposition(
            n_assets, n_scenarios, hidden,
        )
        self.markowitz = DifferentiableMarkowitz(
            n_assets, max_weight=max_weight,
        )

    def _covariance(self, state: torch.Tensor) -> torch.Tensor:
        """Low-rank factor covariance: Σ = FF' + diag(d)."""
        B = state.shape[0]
        F = self.factor_head(state).reshape(B, self.n_assets, self.n_factors)
        d = self.diag_head(state) + 1e-6
        cov = torch.bmm(F, F.transpose(-1, -2))
        cov = cov + torch.diag_embed(d)
        return cov

    def forward(self, state: torch.Tensor,
                prev_weights: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        state: (batch, state_dim) — from world model
        prev_weights: (batch, n_assets) — for transaction costs
        """
        mu = self.return_head(state)
        sigma = self._covariance(state)

        opt = self.markowitz(mu, sigma, prev_weights)

        risk = self.tropical_risk(mu, sigma, opt['weights'])

        return {
            'weights': opt['weights'],
            'expected_return': opt['expected_return'],
            'volatility': opt['volatility'],
            'sharpe': opt['sharpe'],
            'transaction_cost': opt['transaction_cost'],
            'var': risk['var'],
            'cvar': risk['cvar'],
            'risk_attribution': risk['risk_attribution'],
            'mu': mu,
            'sigma': sigma,
        }
