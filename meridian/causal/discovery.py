"""
Causal Discovery and Propagation
==================================
Learns causal DAGs from multi-asset data and propagates shocks
through nth-order effects.

Methods:
  - PC algorithm (constraint-based) for skeleton
  - NOTEARS (continuous DAG constraint) for weighted adjacency
  - Transfer entropy for directed information flow
  - Nth-order propagation via matrix powers of the causal graph
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.stats import pearsonr


class CausalGraph:
    """
    Causal DAG over assets with nth-order propagation.
    """

    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n = len(asset_names)
        self.adjacency = np.zeros((self.n, self.n))

    def fit_notears(self, data: np.ndarray, lambda1: float = 0.1,
                    max_iter: int = 100, h_tol: float = 1e-8) -> np.ndarray:
        """
        NOTEARS: continuous optimization for DAG learning.
        data: (T, n_assets) — returns or features
        Solves: min ||X - XW||^2 + lambda1*|W|
                s.t. tr(e^{W*W}) - n = 0  (DAG constraint)
        """
        X = data - data.mean(axis=0)
        n = X.shape[1]
        W = np.zeros((n, n))

        rho, alpha = 1.0, 0.0

        for _ in range(max_iter):
            # gradient of least squares
            M = X @ W - X
            grad = 2 * X.T @ M / X.shape[0]

            # DAG constraint gradient
            E = np.linalg.matrix_power(np.eye(n) + W * W / n, n)
            h = np.trace(E) - n
            grad_h = E.T * 2 * W / n

            # augmented Lagrangian gradient
            total_grad = grad + (alpha + rho * h) * grad_h + lambda1 * np.sign(W)

            # simple gradient step with projection
            W -= 0.001 * total_grad
            np.fill_diagonal(W, 0)

            # threshold small values
            W[np.abs(W) < 0.05] = 0

            # update multipliers
            h_new = np.trace(np.linalg.matrix_power(
                np.eye(n) + W * W / n, n)) - n
            alpha += rho * h_new
            if h_new > 0.25 * h:
                rho *= 10
            if abs(h_new) < h_tol:
                break

        self.adjacency = W
        return W

    def fit_transfer_entropy(self, data: np.ndarray,
                             lags: int = 5) -> np.ndarray:
        """
        Pairwise transfer entropy (linear approximation via Granger causality).
        data: (T, n_assets)
        """
        from numpy.linalg import lstsq

        T, n = data.shape
        te_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # restricted model: y_t ~ y_{t-1:t-k}
                Y = data[lags:, i]
                T_eff = T - lags
                X_r = np.column_stack([data[lags-k:lags-k+T_eff, i]
                                       for k in range(1, lags+1)])
                X_u = np.column_stack([X_r] + [
                    data[lags-k:lags-k+T_eff, j]
                    for k in range(1, lags+1)
                ])

                res_r = Y - X_r @ lstsq(X_r, Y, rcond=None)[0]
                res_u = Y - X_u @ lstsq(X_u, Y, rcond=None)[0]

                var_r = np.var(res_r)
                var_u = np.var(res_u)

                if var_u > 0 and var_r > var_u:
                    te_matrix[j, i] = 0.5 * np.log(var_r / var_u)

        self.adjacency = te_matrix
        return te_matrix

    def nth_order_impact(self, shock_asset: int, shock_magnitude: float,
                         max_order: int = 5) -> Dict[str, List[Dict]]:
        """
        Propagate a shock through the causal graph up to nth order.

        Returns impacts at each order:
          order 1: direct effects from shock_asset
          order 2: effects through intermediaries
          ...
          order n: nth-order cascading effects
        """
        A = self.adjacency.copy()
        # normalize rows for propagation
        row_sums = np.abs(A).sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        A_norm = A / row_sums

        impacts = {}
        current_shock = np.zeros(self.n)
        current_shock[shock_asset] = shock_magnitude

        cumulative = current_shock.copy()

        for order in range(1, max_order + 1):
            propagated = A_norm.T @ current_shock
            propagated[shock_asset] = 0  # don't re-shock source

            order_impacts = []
            for i in range(self.n):
                if abs(propagated[i]) > 1e-6:
                    order_impacts.append({
                        'asset': self.asset_names[i],
                        'impact': float(propagated[i]),
                        'cumulative': float(cumulative[i] + propagated[i]),
                    })

            impacts[f'order_{order}'] = sorted(
                order_impacts, key=lambda x: abs(x['impact']), reverse=True
            )

            cumulative += propagated
            current_shock = propagated

        impacts['total'] = {
            self.asset_names[i]: float(cumulative[i])
            for i in range(self.n) if abs(cumulative[i]) > 1e-6
        }

        return impacts

    def get_dag_edges(self, threshold: float = 0.05) -> List[Dict]:
        """Return edges above threshold for visualization."""
        edges = []
        for i in range(self.n):
            for j in range(self.n):
                if abs(self.adjacency[i, j]) > threshold:
                    edges.append({
                        'from': self.asset_names[i],
                        'to': self.asset_names[j],
                        'weight': float(self.adjacency[i, j]),
                    })
        return sorted(edges, key=lambda e: abs(e['weight']), reverse=True)
