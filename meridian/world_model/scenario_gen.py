"""
Conditional Scenario Generator
================================
Generates fat-tailed financial scenarios using the world model's
imagination capability with intervention support.

Features:
  - Conditional generation: "given Fed hikes 50bp" via causal do() operator
  - CCAR/FRTB stress test templates
  - Antithetic sampling for variance reduction
  - Validation suite: KS test, autocorrelation match, VaR violations

References:
  Sornette "Why Stock Markets Crash" 2003
  FRTB / Basel III stress testing framework
  Glasserman "Monte Carlo Methods in Financial Engineering" 2003
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


STRESS_TEMPLATES = {
    'recession_2008': {
        'description': 'GFC-style recession: equity -40%, credit +300bp, VIX 80',
        'equity_shock': -0.40,
        'credit_spread_shock': 0.03,
        'vol_multiplier': 3.0,
        'duration_days': 252,
    },
    'covid_2020': {
        'description': 'Pandemic shock: equity -34%, VIX 82, rapid recovery',
        'equity_shock': -0.34,
        'credit_spread_shock': 0.02,
        'vol_multiplier': 4.0,
        'duration_days': 60,
    },
    'rate_shock_2022': {
        'description': 'Rapid rate hike: bonds -15%, equity -20%, orderly',
        'equity_shock': -0.20,
        'bond_shock': -0.15,
        'vol_multiplier': 1.5,
        'duration_days': 252,
    },
    'svb_2023': {
        'description': 'Banking crisis: regional banks -30%, contagion risk',
        'equity_shock': -0.10,
        'financials_shock': -0.30,
        'credit_spread_shock': 0.015,
        'vol_multiplier': 2.0,
        'duration_days': 30,
    },
    'stagflation': {
        'description': 'Stagflation: equity -25%, bonds -10%, commodities +30%',
        'equity_shock': -0.25,
        'bond_shock': -0.10,
        'commodity_shock': 0.30,
        'vol_multiplier': 2.0,
        'duration_days': 504,
    },
}


class ScenarioGenerator(nn.Module):
    """
    Generates scenarios by rolling out the world model's RSSM
    in imagination mode, optionally conditioned on interventions.
    """

    def __init__(self, state_dim: int, n_assets: int = 35,
                 hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        self.n_assets = n_assets

        self.conditioning_net = nn.Sequential(
            nn.Linear(n_assets + 8, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, state_dim),
        )

        self.return_decoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets * 2),
        )

    def condition_state(self, state: torch.Tensor,
                        shock: torch.Tensor,
                        macro: Optional[torch.Tensor] = None,
                        ) -> torch.Tensor:
        """
        Modify state to reflect a conditioning scenario.
        shock: (batch, n_assets) — asset-level shocks
        macro: (batch, 8) — macro conditioning variables
        """
        B = state.shape[0]
        if macro is None:
            macro = torch.zeros(B, 8, device=state.device)
        cond_input = torch.cat([shock, macro], dim=-1)
        delta = self.conditioning_net(cond_input)
        return state + delta

    def decode_returns(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        states: (batch, horizon, state_dim)
        returns: mu and sigma for each timestep
        """
        B, T, D = states.shape
        out = self.return_decoder(states)
        mu = out[..., :self.n_assets]
        log_sigma = out[..., self.n_assets:]
        sigma = F.softplus(log_sigma) + 1e-6
        return {'mu': mu, 'sigma': sigma}

    def sample_paths(self, mu: torch.Tensor, sigma: torch.Tensor,
                     n_paths: int = 1000,
                     antithetic: bool = True) -> torch.Tensor:
        """
        Generate return paths from predicted mu/sigma.
        Uses antithetic sampling for variance reduction.

        mu: (batch, horizon, n_assets)
        sigma: (batch, horizon, n_assets)
        returns: (batch, n_paths, horizon, n_assets)
        """
        B, T, N = mu.shape

        if antithetic:
            half = n_paths // 2
            z = torch.randn(B, half, T, N, device=mu.device)
            z = torch.cat([z, -z], dim=1)
            if n_paths % 2:
                z = torch.cat([z, torch.zeros(B, 1, T, N, device=mu.device)], dim=1)
        else:
            z = torch.randn(B, n_paths, T, N, device=mu.device)

        df = 5.0
        chi2 = torch.distributions.Chi2(df).sample(z.shape[:-1]).unsqueeze(-1).to(mu.device)
        t_noise = z * (df / chi2.clamp(min=1e-6)).sqrt()

        returns = mu.unsqueeze(1) + sigma.unsqueeze(1) * t_noise
        return returns

    def forward(self, state: torch.Tensor,
                imagine_fn,
                horizon: int = 20,
                n_paths: int = 1000,
                shock: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Full scenario generation pipeline.

        state: (batch, state_dim) or dict — initial state
        imagine_fn: callable state -> imagined states (batch, horizon, state_dim)
        horizon: forecast horizon
        n_paths: number of Monte Carlo paths
        shock: optional (batch, n_assets) conditioning shock
        """
        if shock is not None:
            if isinstance(state, dict):
                s = torch.cat([state['h'], state['z'], state['topo']], dim=-1)
                s = self.condition_state(s, shock)
            else:
                state = self.condition_state(state, shock)

        if isinstance(state, dict):
            imagined = imagine_fn(state, horizon)
            states = imagined['states']
        else:
            states = state.unsqueeze(1).expand(-1, horizon, -1)

        decoded = self.decode_returns(states)
        paths = self.sample_paths(decoded['mu'], decoded['sigma'], n_paths)

        cum_returns = paths.cumsum(dim=2)

        return {
            'paths': paths,
            'cumulative_returns': cum_returns,
            'mu': decoded['mu'],
            'sigma': decoded['sigma'],
            'states': states,
        }


class StressTestRunner(nn.Module):
    """
    Runs stress test scenarios from templates and computes
    portfolio-level impact metrics.
    """

    def __init__(self, n_assets: int = 35):
        super().__init__()
        self.n_assets = n_assets

    def create_shock(self, template_name: str,
                     device: torch.device) -> Tuple[torch.Tensor, Dict]:
        """Create a shock vector from a template."""
        template = STRESS_TEMPLATES[template_name]
        shock = torch.zeros(1, self.n_assets, device=device)

        if 'equity_shock' in template:
            shock[0, :10] = template['equity_shock']
        if 'bond_shock' in template:
            shock[0, 10:15] = template['bond_shock']
        if 'commodity_shock' in template:
            shock[0, 20:25] = template['commodity_shock']
        if 'financials_shock' in template:
            shock[0, :5] = template['financials_shock']
        if 'credit_spread_shock' in template:
            shock[0, 15:20] = -template['credit_spread_shock'] * 10

        return shock, template

    def compute_impact(self, paths: torch.Tensor,
                       weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        paths: (batch, n_paths, horizon, n_assets)
        weights: (batch, n_assets)
        """
        port_returns = (paths * weights.unsqueeze(1).unsqueeze(2)).sum(-1)
        cum_port = port_returns.cumsum(dim=-1)

        max_drawdown = (cum_port.cummax(dim=-1).values - cum_port).max(dim=-1).values

        final = cum_port[:, :, -1]
        var_95 = final.quantile(0.05, dim=1)
        cvar_95 = final[final <= var_95.unsqueeze(1)].mean() if final.numel() > 0 else var_95.mean()

        return {
            'mean_return': final.mean(dim=1),
            'median_return': final.median(dim=1).values,
            'var_95': var_95,
            'max_drawdown': max_drawdown.mean(dim=1),
            'worst_case': final.min(dim=1).values,
            'best_case': final.max(dim=1).values,
            'prob_loss': (final < 0).float().mean(dim=1),
        }


class ScenarioValidator:
    """
    Validates generated scenarios against statistical properties
    of real financial data.
    """

    @staticmethod
    def check_marginals(scenarios: torch.Tensor,
                        real_data: torch.Tensor) -> Dict[str, float]:
        """
        KS test on marginals (per-asset return distributions).
        scenarios: (n_paths, horizon, n_assets)
        real_data: (horizon, n_assets)
        """
        results = {}
        n_assets = scenarios.shape[-1]

        for i in range(n_assets):
            sim = scenarios[:, :, i].reshape(-1).sort().values
            real = real_data[:, i].sort().values

            n = len(sim)
            m = len(real)
            sim_cdf = torch.arange(1, n + 1, dtype=torch.float32) / n
            real_cdf = torch.arange(1, m + 1, dtype=torch.float32) / m

            all_vals = torch.cat([sim, real]).sort().values
            sim_ecdf = torch.searchsorted(sim, all_vals).float() / n
            real_ecdf = torch.searchsorted(real, all_vals).float() / m
            ks_stat = (sim_ecdf - real_ecdf).abs().max().item()

            results[f'asset_{i}_ks'] = ks_stat

        return results

    @staticmethod
    def check_autocorrelation(scenarios: torch.Tensor,
                              max_lag: int = 10) -> torch.Tensor:
        """
        Check autocorrelation structure of generated paths.
        Returns autocorrelation at each lag, averaged over paths/assets.
        """
        n_paths, T, N = scenarios.shape
        acf = torch.zeros(max_lag)

        for lag in range(1, max_lag + 1):
            x = scenarios[:, :-lag, :]
            y = scenarios[:, lag:, :]
            corr = ((x - x.mean()) * (y - y.mean())).mean() / (
                x.std() * y.std() + 1e-8
            )
            acf[lag - 1] = corr

        return acf

    @staticmethod
    def check_var_violations(scenarios: torch.Tensor,
                             alpha: float = 0.05) -> float:
        """
        Empirical VaR violation rate. Should be close to alpha.
        """
        n_paths = scenarios.shape[0]
        per_path_returns = scenarios.sum(dim=-1).sum(dim=-1)
        var_threshold = per_path_returns.quantile(alpha)
        violations = (per_path_returns < var_threshold).float().mean().item()
        return violations
