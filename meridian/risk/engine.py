"""
Risk Engine
============
Computes risk metrics from world model scenarios:
  - Value at Risk (VaR)
  - Expected Shortfall (ES / CVaR)
  - Marginal / Component risk decomposition
  - Stress testing
  - Tail risk indicators
"""

import numpy as np
from typing import Dict, Optional, List, Tuple


class RiskEngine:
    """
    Scenario-based risk analytics.
    Takes Monte Carlo paths from the world model's ScenarioGenerator.
    """

    def __init__(self, asset_names: Optional[List[str]] = None):
        self.asset_names = asset_names or []

    def portfolio_pnl(self, scenario_returns: np.ndarray,
                      weights: np.ndarray) -> np.ndarray:
        """
        Compute portfolio P&L paths.
        scenario_returns: (n_scenarios, horizon, n_assets)
        weights: (n_assets,)
        returns: (n_scenarios, horizon)
        """
        return (scenario_returns * weights[None, None, :]).sum(axis=-1)

    def var(self, pnl: np.ndarray, confidence: float = 0.99,
            horizon_days: Optional[int] = None) -> float:
        """
        Value at Risk.
        pnl: (n_scenarios,) or (n_scenarios, horizon)
        If horizon given, uses cumulative returns over that horizon.
        """
        if pnl.ndim == 2:
            h = horizon_days or pnl.shape[1]
            cum_pnl = pnl[:, :h].sum(axis=1)
        else:
            cum_pnl = pnl
        return -np.percentile(cum_pnl, (1 - confidence) * 100)

    def expected_shortfall(self, pnl: np.ndarray,
                           confidence: float = 0.99,
                           horizon_days: Optional[int] = None) -> float:
        """
        Expected Shortfall (CVaR) — average loss beyond VaR.
        """
        if pnl.ndim == 2:
            h = horizon_days or pnl.shape[1]
            cum_pnl = pnl[:, :h].sum(axis=1)
        else:
            cum_pnl = pnl
        cutoff = np.percentile(cum_pnl, (1 - confidence) * 100)
        tail = cum_pnl[cum_pnl <= cutoff]
        if len(tail) == 0:
            return -cutoff
        return -tail.mean()

    def marginal_var(self, scenario_returns: np.ndarray,
                     weights: np.ndarray,
                     confidence: float = 0.99,
                     delta: float = 0.01) -> np.ndarray:
        """
        Marginal VaR: sensitivity of portfolio VaR to each asset weight.
        """
        base_pnl = self.portfolio_pnl(scenario_returns, weights)
        base_var = self.var(base_pnl, confidence)

        n_assets = weights.shape[0]
        m_var = np.zeros(n_assets)
        for i in range(n_assets):
            w_up = weights.copy()
            w_up[i] += delta
            w_up /= w_up.sum()
            pnl_up = self.portfolio_pnl(scenario_returns, w_up)
            m_var[i] = (self.var(pnl_up, confidence) - base_var) / delta

        return m_var

    def component_var(self, scenario_returns: np.ndarray,
                      weights: np.ndarray,
                      confidence: float = 0.99) -> np.ndarray:
        """Component VaR: how much each asset contributes to total VaR."""
        m_var = self.marginal_var(scenario_returns, weights, confidence)
        return m_var * weights

    def risk_report(self, scenario_returns: np.ndarray,
                    weights: np.ndarray,
                    horizons: List[int] = [1, 5, 10, 20],
                    confidence: float = 0.99) -> Dict:
        """
        Full risk report across multiple horizons.
        """
        pnl = self.portfolio_pnl(scenario_returns, weights)
        report = {'confidence': confidence, 'horizons': {}}

        for h in horizons:
            if h > pnl.shape[1]:
                continue
            report['horizons'][h] = {
                'var': self.var(pnl, confidence, h),
                'es': self.expected_shortfall(pnl, confidence, h),
                'mean_return': pnl[:, :h].sum(axis=1).mean(),
                'vol': pnl[:, :h].sum(axis=1).std(),
                'worst_case': -pnl[:, :h].sum(axis=1).min(),
                'best_case': pnl[:, :h].sum(axis=1).max(),
            }

        # component decomposition at 1-day
        if 1 in report['horizons']:
            report['component_var'] = dict(zip(
                self.asset_names or [f'asset_{i}' for i in range(weights.shape[0])],
                self.component_var(scenario_returns, weights, confidence)
            ))

        return report

    def stress_scenarios(self, scenario_returns: np.ndarray,
                         weights: np.ndarray,
                         shocks: Dict[str, Dict[int, float]]) -> Dict:
        """
        Named stress scenarios.
        shocks: {'2020 crash': {0: -0.12, 1: -0.15, ...}, ...}
        """
        results = {}
        for name, shock in shocks.items():
            stressed = scenario_returns.copy()
            for asset_idx, ret in shock.items():
                stressed[:, 0, asset_idx] = ret
            pnl = self.portfolio_pnl(stressed, weights)
            results[name] = {
                'day1_loss': -pnl[:, 0].mean(),
                'var_99': self.var(pnl, 0.99),
                'es_99': self.expected_shortfall(pnl, 0.99),
                'max_loss': -pnl.sum(axis=1).min(),
            }
        return results
