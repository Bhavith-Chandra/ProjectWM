"""
Proper Scoring Rules for World Model Evaluation
=================================================
Energy score and variogram score for multivariate probabilistic forecasts.
These are strictly proper: they are uniquely minimized by the true distribution.
"""

import numpy as np
from typing import Dict, Optional, Tuple


def energy_score(scenarios: np.ndarray, observed: np.ndarray,
                 n_pairs: int = 20) -> float:
    """
    Energy score: E||X - obs|| - 0.5 * E||X - X'||

    scenarios: (n_scenarios, n_assets) — simulated paths at one horizon
    observed: (n_assets,) — realized observation
    Lower is better.
    """
    n_sc = scenarios.shape[0]

    # E||X - obs||
    diffs = scenarios - observed[None, :]
    term1 = np.sqrt((diffs ** 2).sum(axis=1)).mean()

    # E||X - X'|| via random pairs
    rng = np.random.default_rng(42)
    idx1 = rng.integers(0, n_sc, size=n_pairs * n_sc)
    idx2 = rng.integers(0, n_sc, size=n_pairs * n_sc)
    pair_diffs = scenarios[idx1] - scenarios[idx2]
    term2 = np.sqrt((pair_diffs ** 2).sum(axis=1)).mean()

    return term1 - 0.5 * term2


def variogram_score(scenarios: np.ndarray, observed: np.ndarray,
                    p: float = 0.5) -> float:
    """
    Variogram score: measures cross-asset dependence calibration.
    Uses pairwise distances between assets.

    p: order of the variogram (0.5 recommended for robustness)
    Lower is better.
    """
    n_assets = observed.shape[0]
    score = 0.0

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            obs_diff = abs(observed[i] - observed[j]) ** p
            sc_diff = np.abs(scenarios[:, i] - scenarios[:, j]) ** p
            score += (obs_diff - sc_diff.mean()) ** 2

    return score


def evaluate_scenarios(scenarios: np.ndarray,
                       observations: np.ndarray,
                       horizons: list = [1, 5, 10, 20]) -> Dict:
    """
    Evaluate scenario quality across multiple horizons.

    scenarios: (n_scenarios, max_horizon, n_assets)
    observations: (max_horizon, n_assets) — realized values
    """
    results = {}
    for h in horizons:
        if h > scenarios.shape[1] or h > observations.shape[0]:
            continue
        # cumulative returns over horizon
        sc_cum = scenarios[:, :h, :].sum(axis=1)
        obs_cum = observations[:h, :].sum(axis=0)

        es = energy_score(sc_cum, obs_cum)
        vs = variogram_score(sc_cum, obs_cum)

        results[f'{h}d'] = {
            'energy_score': float(es),
            'variogram_score': float(vs),
            'mean_bias': float((sc_cum.mean(axis=0) - obs_cum).mean()),
            'coverage_90': float(
                np.mean((obs_cum >= np.percentile(sc_cum, 5, axis=0)) &
                        (obs_cum <= np.percentile(sc_cum, 95, axis=0)))
            ),
        }

    return results
