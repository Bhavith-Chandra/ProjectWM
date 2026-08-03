"""
Portfolio Optimizer
====================
Scenario-aware portfolio optimization:
  - Mean-Variance (Markowitz)
  - Hierarchical Risk Parity (HRP)
  - Risk Parity (equal risk contribution)
  - Black-Litterman with world model views
  - CVaR optimization
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list


class PortfolioOptimizer:

    def __init__(self, asset_names: Optional[List[str]] = None,
                 risk_free_rate: float = 0.04 / 252):
        self.asset_names = asset_names or []
        self.rf = risk_free_rate

    def mean_variance(self, expected_returns: np.ndarray,
                      cov_matrix: np.ndarray,
                      target_vol: Optional[float] = None) -> np.ndarray:
        """
        Mean-variance optimization.
        Returns optimal weights.
        """
        n = len(expected_returns)

        def neg_sharpe(w):
            ret = w @ expected_returns - self.rf
            vol = np.sqrt(w @ cov_matrix @ w)
            return -ret / (vol + 1e-10)

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n

        if target_vol is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda w: target_vol - np.sqrt(w @ cov_matrix @ w)
            })

        result = minimize(neg_sharpe, np.ones(n) / n,
                          bounds=bounds, constraints=constraints,
                          method='SLSQP')
        return result.x

    def risk_parity(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Equal risk contribution portfolio."""
        n = cov_matrix.shape[0]

        def risk_budget_obj(w):
            vol = np.sqrt(w @ cov_matrix @ w)
            mrc = cov_matrix @ w / vol
            rc = w * mrc
            target_rc = vol / n
            return np.sum((rc - target_rc) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0.01, 1)] * n
        result = minimize(risk_budget_obj, np.ones(n) / n,
                          bounds=bounds, constraints=constraints,
                          method='SLSQP')
        return result.x

    def hrp(self, cov_matrix: np.ndarray,
            expected_returns: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Hierarchical Risk Parity (Lopez de Prado).
        Cluster assets by correlation, then allocate inversely to cluster vol.
        """
        n = cov_matrix.shape[0]
        corr = self._cov_to_corr(cov_matrix)
        dist = np.sqrt(0.5 * (1 - np.clip(corr, -1, 1)))
        np.fill_diagonal(dist, 0)

        condensed = dist[np.triu_indices(n, k=1)]
        link = linkage(condensed, method='single')
        order = leaves_list(link).tolist()

        weights = np.ones(n)
        cluster_items = [order]

        while any(len(c) > 1 for c in cluster_items):
            new_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    new_clusters.append(cluster)
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]

                left_var = self._cluster_var(cov_matrix, left)
                right_var = self._cluster_var(cov_matrix, right)
                alpha = 1 - left_var / (left_var + right_var)

                for i in left:
                    weights[i] *= alpha
                for i in right:
                    weights[i] *= (1 - alpha)

                new_clusters.extend([left, right])
            cluster_items = new_clusters

        return weights / weights.sum()

    def black_litterman(self, cov_matrix: np.ndarray,
                        market_weights: np.ndarray,
                        views: np.ndarray,
                        view_confidences: np.ndarray,
                        tau: float = 0.05) -> np.ndarray:
        """
        Black-Litterman with world model views.

        views: (n_views, n_assets) — pick matrix P
        view_confidences: (n_views,) — expected excess returns from views
        """
        n = cov_matrix.shape[0]
        # equilibrium returns
        delta = 2.5
        pi = delta * cov_matrix @ market_weights

        P = views
        Q = view_confidences
        omega = np.diag(np.diag(P @ (tau * cov_matrix) @ P.T))

        # posterior
        tau_sigma = tau * cov_matrix
        M = np.linalg.inv(
            np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
        )
        mu_bl = M @ (np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ Q)

        return self.mean_variance(mu_bl, cov_matrix)

    def cvar_optimize(self, scenario_returns: np.ndarray,
                      confidence: float = 0.95) -> np.ndarray:
        """
        CVaR (Expected Shortfall) minimization.
        scenario_returns: (n_scenarios, n_assets)
        """
        n_scenarios, n_assets = scenario_returns.shape

        def cvar_obj(w):
            pnl = scenario_returns @ w
            cutoff = np.percentile(pnl, (1 - confidence) * 100)
            tail = pnl[pnl <= cutoff]
            return -tail.mean() if len(tail) > 0 else -cutoff

        constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1}]
        bounds = [(0, 1)] * n_assets
        result = minimize(cvar_obj, np.ones(n_assets) / n_assets,
                          bounds=bounds, constraints=constraints,
                          method='SLSQP')
        return result.x

    def optimize_from_scenarios(self, scenario_returns: np.ndarray,
                                method: str = 'hrp') -> Dict:
        """
        Convenience: extract stats from scenarios and optimize.
        scenario_returns: (n_scenarios, horizon, n_assets)
        """
        # use 1-day returns for covariance
        daily = scenario_returns[:, 0, :]
        mu = daily.mean(axis=0)
        cov = np.cov(daily.T)

        if method == 'mean_variance':
            w = self.mean_variance(mu, cov)
        elif method == 'risk_parity':
            w = self.risk_parity(cov)
        elif method == 'hrp':
            w = self.hrp(cov)
        elif method == 'cvar':
            w = self.cvar_optimize(daily)
        else:
            raise ValueError(f"Unknown method: {method}")

        port_ret = mu @ w
        port_vol = np.sqrt(w @ cov @ w)
        sharpe = (port_ret - self.rf) / (port_vol + 1e-10)

        return {
            'weights': dict(zip(
                self.asset_names or [f'asset_{i}' for i in range(len(w))],
                w
            )),
            'expected_return': float(port_ret * 252),
            'expected_vol': float(port_vol * np.sqrt(252)),
            'sharpe': float(sharpe * np.sqrt(252)),
            'method': method,
        }

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        d = np.sqrt(np.diag(cov))
        d[d == 0] = 1
        return cov / np.outer(d, d)

    @staticmethod
    def _cluster_var(cov: np.ndarray, indices: List[int]) -> float:
        sub_cov = cov[np.ix_(indices, indices)]
        w = np.ones(len(indices)) / len(indices)
        return w @ sub_cov @ w
