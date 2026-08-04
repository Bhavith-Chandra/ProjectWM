"""
Scenario Generation Engine
============================
Generates Monte Carlo scenario paths from the world model's latent space.
Supports:
  - Unconditional simulation (sample from prior)
  - Conditional simulation (condition on regime, stress events)
  - Antithetic sampling (variance reduction)
"""

import torch
import numpy as np
from typing import Dict, Optional, List


class ScenarioGenerator:
    """
    Wraps the world model for practical scenario generation.
    Handles batching, device management, and output formatting.
    """

    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, history: np.ndarray, horizon: int = 20,
                 n_scenarios: int = 1000,
                 antithetic: bool = True) -> Dict[str, np.ndarray]:
        """
        Generate scenario paths from historical data.

        Args:
            history: (seq_len, n_assets, n_features) — recent observations
            horizon: number of future steps to simulate
            n_scenarios: number of Monte Carlo paths
            antithetic: if True, generate n/2 paths and mirror them

        Returns:
            dict with 'returns', 'volatility', 'covariance' arrays
        """
        obs = torch.from_numpy(history).float().unsqueeze(0).to(self.device)

        if antithetic:
            n_half = n_scenarios // 2
            result = self.model.imagine(obs, horizon, n_half)

            # mirror returns for antithetic paths
            returns = result['returns'].cpu().numpy()
            returns_anti = np.concatenate([returns, -returns], axis=1)
            returns_anti = returns_anti[:, :n_scenarios]

            log_vol = result['log_vol'].cpu().numpy()
            vol = np.exp(log_vol)
            vol_full = np.concatenate([vol, vol], axis=1)[:, :n_scenarios]

            cov = result['covariance'].cpu().numpy()
            cov_full = np.concatenate([cov, cov], axis=1)[:, :n_scenarios]
        else:
            result = self.model.imagine(obs, horizon, n_scenarios)
            returns_anti = result['returns'].cpu().numpy()
            vol_full = np.exp(result['log_vol'].cpu().numpy())
            cov_full = result['covariance'].cpu().numpy()

        return {
            'returns': returns_anti[0],      # (n_scenarios, horizon, n_assets)
            'volatility': vol_full[0],       # (n_scenarios, horizon, n_assets)
            'covariance': cov_full[0],       # (n_scenarios, horizon, n_assets, n_assets)
        }

    @torch.no_grad()
    def stress_test(self, history: np.ndarray, shocks: Dict[int, float],
                    horizon: int = 20,
                    n_scenarios: int = 500) -> Dict[str, np.ndarray]:
        """
        Conditional scenario generation under stress.

        Args:
            shocks: {asset_index: return_shock} e.g. {0: -0.10} for SPY -10%
        """
        scenarios = self.generate(history, horizon, n_scenarios, antithetic=False)

        # condition first-step returns on the shock
        for asset_idx, shock_return in shocks.items():
            scenarios['returns'][:, 0, asset_idx] = shock_return

        return scenarios

    @torch.no_grad()
    def encode_state(self, history: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract the current latent state (useful for regime analysis)."""
        obs = torch.from_numpy(history).float().unsqueeze(0).to(self.device)
        embedded = self.model.encode(obs)
        rssm_out = self.model.rssm.observe_sequence(embedded)

        h = rssm_out['h'][:, -1].cpu().numpy()
        z = rssm_out['z'][:, -1].cpu().numpy()
        state = np.concatenate([h, z], axis=-1)

        regime_logits = self.model.regime_head(
            torch.from_numpy(state).float().to(self.device)
        ).cpu().numpy()

        return {
            'h': h[0],
            'z': z[0],
            'regime_probs': np.exp(regime_logits[0]) / np.exp(regime_logits[0]).sum(),
        }
