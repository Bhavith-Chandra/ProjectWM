"""Portfolio optimization using world-model estimated covariance and risk metrics.

Strategies:
  - Mean-variance: maximize Sharpe ratio subject to vol constraint
  - Risk-parity: allocate so each asset contributes equally to portfolio risk
  - Maximum Sharpe: maximize (μ - rf) / σ
  - Minimum variance: minimize portfolio volatility
  - Robust: use scenarios instead of point estimates for robustness
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Tuple, List
import scipy.optimize as opt
from scipy.linalg import cholesky


class PortfolioOptimizer:
    """Optimize portfolio allocations using world-model beliefs."""

    def __init__(self, n_assets: int, constraints: Optional[Dict] = None):
        """
        n_assets: number of assets in portfolio
        constraints: dict with 'max_weight', 'min_weight', 'max_leverage', etc.
        """
        self.n_assets = n_assets

        self.constraints = constraints or {
            'max_weight': 0.3,  # No single asset > 30%
            'min_weight': 0.0,  # Long-only
            'max_leverage': 1.0  # Full investment (no borrowing)
        }

    def minimum_variance(
        self,
        cov_matrix: np.ndarray,
        constraint_type: str = 'long_only'
    ) -> np.ndarray:
        """
        Minimize portfolio volatility.

        Args:
            cov_matrix: (n_assets, n_assets) covariance matrix
            constraint_type: 'long_only', 'long_short', 'risk_parity'

        Returns: (n_assets,) weight vector
        """
        def objective(w):
            return w @ cov_matrix @ w

        if constraint_type == 'long_only':
            bounds = [(self.constraints['min_weight'], self.constraints['max_weight']) for _ in range(self.n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        else:  # long_short
            bounds = [(-0.5, 0.5) for _ in range(self.n_assets)]
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 0}]

        x0 = np.ones(self.n_assets) / self.n_assets
        result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def maximum_sharpe(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.01
    ) -> np.ndarray:
        """
        Maximize Sharpe ratio: (μ - rf) / σ.

        Args:
            mean_returns: (n_assets,) expected returns
            cov_matrix: (n_assets, n_assets) covariance
            risk_free_rate: annual risk-free rate

        Returns: (n_assets,) weight vector
        """
        def neg_sharpe(w):
            portfolio_return = w @ mean_returns
            portfolio_vol = np.sqrt(w @ cov_matrix @ w)
            if portfolio_vol < 1e-6:
                return 1e6
            return -(portfolio_return - risk_free_rate) / portfolio_vol

        x0 = np.ones(self.n_assets) / self.n_assets
        bounds = [(self.constraints['min_weight'], self.constraints['max_weight']) for _ in range(self.n_assets)]
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = opt.minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else x0

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Risk-parity allocation: each asset contributes equally to portfolio risk.

        w_i ∝ 1 / σ_i (inverse volatility weighting)
        """
        vol = np.diag(cov_matrix) ** 0.5
        weights = 1.0 / vol
        weights /= weights.sum()

        # Enforce max position size
        weights = np.minimum(weights, self.constraints['max_weight'])
        weights /= weights.sum()

        return weights

    def equal_weight(self) -> np.ndarray:
        """Simple equal-weight baseline."""
        return np.ones(self.n_assets) / self.n_assets

    def from_scenarios(
        self,
        scenarios: np.ndarray,  # (n_paths, horizon, n_assets)
        objective: str = 'min_var',
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """
        Optimize using scenario-based statistics.

        Args:
            scenarios: generated return paths
            objective: 'min_var', 'max_sharpe', 'risk_parity'
            horizon: if specified, use only last horizon days

        Returns: (n_assets,) weight vector
        """
        if horizon:
            scenarios = scenarios[:, -horizon:, :]

        # Compute covariance and expected returns from scenarios
        scenarios_flat = scenarios.reshape(-1, self.n_assets)  # (n_paths*horizon, n_assets)
        cov_matrix = np.cov(scenarios_flat.T)
        mean_returns = scenarios_flat.mean(axis=0)

        if objective == 'min_var':
            return self.minimum_variance(cov_matrix)
        elif objective == 'max_sharpe':
            return self.maximum_sharpe(mean_returns, cov_matrix)
        elif objective == 'risk_parity':
            return self.risk_parity(cov_matrix)
        else:
            return self.equal_weight()

    def rebalance_optimization(
        self,
        scenarios: np.ndarray,
        current_holdings: np.ndarray,
        turnover_limit: float = 0.1  # Max 10% turnover
    ) -> np.ndarray:
        """
        Optimize rebalance with turnover constraint.

        Args:
            scenarios: return paths
            current_holdings: current weight vector
            turnover_limit: max absolute change in positions

        Returns: new weight vector respecting turnover
        """
        # Unconstrained optimal weights
        optimal_w = self.from_scenarios(scenarios, objective='min_var')

        # Turnover constraint: |w_new - w_old| ≤ turnover_limit per asset
        min_w = np.maximum(current_holdings - turnover_limit, self.constraints['min_weight'])
        max_w = np.minimum(current_holdings + turnover_limit, self.constraints['max_weight'])

        def objective(w):
            return np.sum((w - optimal_w) ** 2)  # Distance from optimal

        x0 = current_holdings
        bounds = list(zip(min_w, max_w))
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

        return result.x if result.success else current_holdings

    def utility_maximization(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_aversion: float = 2.0,  # Higher = more risk-averse
        risk_free_rate: float = 0.01
    ) -> np.ndarray:
        """
        Maximize utility: U(w) = w^T μ - (γ/2) w^T Σ w

        Args:
            mean_returns: expected returns
            cov_matrix: covariance matrix
            risk_aversion: gamma parameter
            risk_free_rate: rate for cash

        Returns: optimal weights
        """
        # Analytical solution for unconstrained problem
        # w* = (1/γ) Σ^{-1} (μ - rf)

        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_matrix)

        excess_returns = mean_returns - risk_free_rate
        weights = (1.0 / risk_aversion) * (cov_inv @ excess_returns)

        # Project to constraints
        weights = np.maximum(weights, self.constraints['min_weight'])
        weights = np.minimum(weights, self.constraints['max_weight'])
        weights /= weights.sum()  # Renormalize

        return weights


class LivePortfolioManager:
    """Manage portfolio allocation in live trading using world-model predictions."""

    def __init__(self, optimizer: PortfolioOptimizer, update_frequency: int = 5):
        """
        optimizer: PortfolioOptimizer instance
        update_frequency: rebalance every N days
        """
        self.optimizer = optimizer
        self.update_frequency = update_frequency

        self.current_weights = np.ones(optimizer.n_assets) / optimizer.n_assets
        self.rebalance_counter = 0

    def get_next_allocation(
        self,
        world_model,
        recent_returns: np.ndarray,
        scenarios: np.ndarray,
        force_rebalance: bool = False
    ) -> Dict:
        """
        Get next portfolio allocation based on world-model predictions.

        Args:
            world_model: fitted JEPA/GLP/RSSM
            recent_returns: recent return history
            scenarios: generated return scenarios (n_paths, horizon, n_assets)
            force_rebalance: if True, always rebalance (ignore schedule)

        Returns: dict with 'weights', 'turnover', 'expected_return', 'expected_vol'
        """
        self.rebalance_counter += 1

        should_rebalance = (self.rebalance_counter >= self.update_frequency) or force_rebalance

        if should_rebalance:
            # Compute new optimal weights
            new_weights = self.optimizer.from_scenarios(
                scenarios,
                objective='min_var',
                horizon=20  # 20-day horizon
            )

            # Turnover
            turnover = np.abs(new_weights - self.current_weights).sum() / 2

            # Expected portfolio statistics
            scenarios_flat = scenarios.reshape(-1, len(self.current_weights))
            expected_return = (new_weights @ scenarios_flat.mean(axis=0))
            expected_vol = np.sqrt(new_weights @ np.cov(scenarios_flat.T) @ new_weights)

            self.current_weights = new_weights
            self.rebalance_counter = 0

            return {
                'weights': new_weights,
                'turnover': turnover,
                'expected_return': expected_return,
                'expected_vol': expected_vol,
                'rebalanced': True
            }
        else:
            return {
                'weights': self.current_weights,
                'turnover': 0.0,
                'rebalanced': False
            }

    def simulate_strategy(
        self,
        returns_history: np.ndarray,  # (n_days, n_assets)
        scenarios_generator,  # Function that generates scenarios
        start_date: int = 500,  # Start after 500 days of data
        rebalance_freq: int = 5,
        objective: str = 'min_var'
    ) -> Dict:
        """
        Backtest strategy.

        Returns:
            dict with 'portfolio_returns', 'weights_history', 'turnover_history'
        """
        n_days, n_assets = returns_history.shape
        portfolio_returns = []
        weights_history = []
        turnover_history = []

        current_w = np.ones(n_assets) / n_assets

        for t in range(start_date, n_days):
            # Every rebalance_freq days, generate scenarios and reoptimize
            if (t - start_date) % rebalance_freq == 0:
                # Get scenarios for next period
                scenarios = scenarios_generator(t, horizon=20)

                # Reoptimize
                new_w = self.optimizer.from_scenarios(scenarios, objective=objective)

                # Turnover
                turnover = np.abs(new_w - current_w).sum() / 2
                turnover_history.append(turnover)

                current_w = new_w

            weights_history.append(current_w.copy())

            # Realize returns with current weights
            pnl = current_w @ returns_history[t]
            portfolio_returns.append(pnl)

        return {
            'portfolio_returns': np.array(portfolio_returns),
            'weights_history': np.array(weights_history),
            'turnover_history': np.array(turnover_history),
            'cumulative_return': np.cumprod(1 + np.array(portfolio_returns)) - 1,
            'sharpe_ratio': np.mean(portfolio_returns) / (np.std(portfolio_returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._compute_max_drawdown(np.array(portfolio_returns))
        }

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        """Compute maximum drawdown."""
        cumul = np.cumprod(1 + returns) - 1
        running_max = np.maximum.accumulate(cumul)
        drawdown = cumul - running_max
        return drawdown.min()
