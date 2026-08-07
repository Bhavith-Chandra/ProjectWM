"""
Meridian Genesis — Composed World Model
=========================================
Integrates all 6 computational primitives into a single forward pass:

  Input → Sheaf → Manifold → RG → Topo-RSSM → Reflexive → Emission Heads
                                                          → Tropical Portfolio
                                                          → Scenario Generator

~150M parameters. The most architecturally advanced financial world model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class GenesisConfig:
    # Assets
    n_assets: int = 35
    asset_class_map: Dict[str, List[int]] = field(default_factory=lambda: {
        'equity': list(range(10)),
        'fixed_income': list(range(10, 15)),
        'commodity': list(range(15, 20)),
        'fx': list(range(20, 25)),
        'alternatives': list(range(25, 30)),
        'crypto': list(range(30, 35)),
    })
    input_dims: Dict[str, int] = field(default_factory=lambda: {
        'equity': 18, 'fixed_income': 14, 'commodity': 12,
        'fx': 10, 'alternatives': 10, 'crypto': 12,
    })

    # Sheaf
    common_dim: int = 576
    n_diffusion_layers: int = 4
    frac_diff_d: float = 0.4

    # Manifold
    h_dim: int = 192
    s_dim: int = 96
    e_dim: int = 96

    # RG
    rg_n_heads: int = 8

    # RSSM
    hidden_dim: int = 2048
    topo_dim: int = 256
    n_categoricals: int = 32
    n_classes: int = 32

    # Reflexive
    belief_dim: int = 2048
    max_deq_iter: int = 10
    jac_reg: float = 0.01

    # Tropical
    n_scenarios: int = 128
    max_weight: float = 0.20
    n_factors: int = 96

    # Conformal
    conformal_alpha: float = 0.1
    conformal_lr: float = 0.01

    # General
    dropout: float = 0.1
    return_window: int = 60

    @property
    def tangent_dim(self) -> int:
        return self.h_dim + self.s_dim + self.e_dim

    @property
    def rssm_state_dim(self) -> int:
        return self.hidden_dim + self.n_categoricals * self.n_classes + self.topo_dim

    @property
    def adjusted_state_dim(self) -> int:
        return self.rssm_state_dim + self.belief_dim + 1


class MeridianGenesis(nn.Module):
    """
    The full Meridian Genesis world model.

    Forward pass:
      1. Sheaf encoder: heterogeneous stalks + Laplacian diffusion
      2. Product manifold: embed on H^n × S^k × R^m
      3. Renormalization: multi-scale decomposition + Hurst
      4. Topological RSSM: state-space dynamics with persistence
      5. Reflexive equilibrium: Soros fixed-point + causal DAG
      6. Emission heads: returns, vol, tail, regime, covariance
      7. Tropical portfolio: optimized weights with risk decomposition
    """

    def __init__(self, config: Optional[GenesisConfig] = None):
        super().__init__()
        if config is None:
            config = GenesisConfig()
        self.config = config

        from .sheaf import SheafEncoder
        from .manifold import ProductManifold
        from .renorm import RenormalizationEngine
        from .topo_rssm import TopologicalRSSM
        from .reflexive import ReflexiveEquilibrium, CausalDAG, CausalAttention
        from .tropical import TropicalPortfolioHead
        from .heads import ReturnHead, VolatilityHead, TailHead, RegimeHead, CovarianceHead

        self.sheaf = SheafEncoder(
            asset_class_map=config.asset_class_map,
            input_dims=config.input_dims,
            common_dim=config.common_dim,
            n_diffusion_layers=config.n_diffusion_layers,
            dropout=config.dropout,
            frac_diff_d=config.frac_diff_d,
        )

        self.manifold = ProductManifold(
            input_dim=config.common_dim,
            h_dim=config.h_dim,
            s_dim=config.s_dim,
            e_dim=config.e_dim,
        )

        self.rg = RenormalizationEngine(
            d_model=config.tangent_dim,
            n_assets=config.n_assets,
            n_heads=config.rg_n_heads,
            dropout=config.dropout,
        )

        rg_out_dim = config.tangent_dim * 2
        self.rssm = TopologicalRSSM(
            obs_dim=rg_out_dim,
            n_assets=config.n_assets,
            hidden_dim=config.hidden_dim,
            topo_dim=config.topo_dim,
            n_categoricals=config.n_categoricals,
            n_classes=config.n_classes,
            use_ttt=True,
            return_window=config.return_window,
            dropout=config.dropout,
        )

        state_dim = config.rssm_state_dim
        self.reflexive = ReflexiveEquilibrium(
            state_dim=state_dim,
            belief_dim=config.belief_dim,
            max_iter=config.max_deq_iter,
            jac_reg=config.jac_reg,
            dropout=config.dropout,
        )

        self.causal_dag = CausalDAG(
            n_assets=config.n_assets,
            state_dim=state_dim,
        )

        adj_state_dim = config.adjusted_state_dim

        self.return_head = ReturnHead(adj_state_dim, config.n_assets)
        self.vol_head = VolatilityHead(adj_state_dim, config.n_assets)
        self.tail_head = TailHead(adj_state_dim, config.n_assets)
        self.regime_head = RegimeHead(adj_state_dim)
        self.cov_head = CovarianceHead(adj_state_dim, config.n_assets)

        self.portfolio = TropicalPortfolioHead(
            state_dim=adj_state_dim,
            n_assets=config.n_assets,
            n_scenarios=config.n_scenarios,
            max_weight=config.max_weight,
            n_factors=config.n_factors,
        )

    def encode(self, features: Dict[str, torch.Tensor]
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run sheaf → manifold → RG pipeline.
        features: dict mapping class name -> (B, T, N_class, D_class)
        returns: (rg_output, systemic_risk)
          rg_output: (B, T', d_model*2) — multi-scale features
          systemic_risk: (B, T, 1)
        """
        encoded, systemic_risk = self.sheaf(features)

        B, N, T, D = encoded.shape
        tangent_list = []
        for t in range(T):
            frame = encoded[:, :, t, :]
            tangent = self.manifold(frame)
            tangent_list.append(tangent)
        tangent_seq = torch.stack(tangent_list, dim=1)

        tangent_flat = tangent_seq.reshape(B * N, T, -1)
        rg_out = self.rg(tangent_flat)

        rg_features = rg_out['multi_scale'].reshape(B, N, -1)
        rg_pooled = rg_features.mean(dim=1)

        return rg_pooled.unsqueeze(1).expand(-1, T, -1), systemic_risk, rg_out

    def forward(self, features: Dict[str, torch.Tensor],
                returns_seq: Optional[torch.Tensor] = None,
                prev_weights: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all modules.

        features: dict mapping class name -> (B, T, N_class, D_class)
        returns_seq: (B, T, n_assets) — for topology computation
        prev_weights: (B, n_assets) — for transaction costs

        returns: dict with all predictions and diagnostics
        """
        rg_seq, systemic_risk, rg_diagnostics = self.encode(features)

        rssm_out = self.rssm.observe_sequence(
            rg_seq, returns_seq, window=self.config.return_window,
        )

        states = rssm_out['states']
        B, T, SD = states.shape

        last_state = states[:, -1]

        ref_out = self.reflexive(last_state)
        dag_out = self.causal_dag(last_state)

        adj_state = ref_out['adjusted_state']

        returns = self.return_head(adj_state)
        vol = self.vol_head(adj_state)
        tail_df = self.tail_head(adj_state)
        regime = self.regime_head(adj_state)
        cov = self.cov_head(adj_state)

        port_out = self.portfolio(adj_state, prev_weights)

        kl_loss = self.rssm.kl_loss(
            rssm_out['prior_logits'][:, -1],
            rssm_out['post_logits'][:, -1],
        )

        if self.training:
            jac_loss = self.reflexive.jacobian_regularization(
                last_state, ref_out['beliefs'],
            )
        else:
            jac_loss = torch.tensor(0.0, device=last_state.device)

        return {
            'returns': returns,
            'volatility': vol,
            'tail_df': tail_df,
            'regime': regime,
            'covariance': cov,
            'portfolio': port_out,
            'systemic_risk': systemic_risk,
            'rho': ref_out['rho'],
            'adjacency': dag_out['adjacency'],
            'hurst': rg_diagnostics['hurst'],
            'critical_slowing': rg_diagnostics['critical_slowing'],
            'kl_loss': kl_loss,
            'acyclicity_loss': dag_out['acyclicity_loss'],
            'jacobian_loss': jac_loss,
            'rssm_states': states,
            'beliefs': ref_out['beliefs'],
            'adjusted_state': adj_state,
            'persistence_norms': rssm_out.get('persistence_norms'),
        }

    def imagine(self, state: Dict[str, torch.Tensor],
                horizon: int = 20) -> Dict[str, torch.Tensor]:
        """
        Imagination rollout from a given RSSM state.
        No observations — uses prior only (for scenario generation).
        """
        imagined = self.rssm.imagine_sequence(state, horizon)
        states = imagined['states']
        B, T, SD = states.shape

        all_returns, all_vol, all_regime = [], [], []
        for t in range(T):
            ref_out = self.reflexive(states[:, t])
            adj = ref_out['adjusted_state']
            all_returns.append(self.return_head(adj))
            all_vol.append(self.vol_head(adj))
            all_regime.append(self.regime_head(adj))

        return {
            'states': states,
            'returns': torch.stack(all_returns, dim=1),
            'volatility': torch.stack(all_vol, dim=1),
            'regime': torch.stack(all_regime, dim=1),
        }

    def count_parameters(self) -> Dict[str, int]:
        """Parameter count by module."""
        counts = {}
        for name, module in [
            ('sheaf', self.sheaf),
            ('manifold', self.manifold),
            ('renorm', self.rg),
            ('topo_rssm', self.rssm),
            ('reflexive', self.reflexive),
            ('causal_dag', self.causal_dag),
            ('return_head', self.return_head),
            ('vol_head', self.vol_head),
            ('tail_head', self.tail_head),
            ('regime_head', self.regime_head),
            ('cov_head', self.cov_head),
            ('portfolio', self.portfolio),
        ]:
            counts[name] = sum(p.numel() for p in module.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        return counts

    def loss(self, output: Dict[str, torch.Tensor],
             targets: Dict[str, torch.Tensor],
             alpha_kl: float = 1.0,
             alpha_dag: float = 0.1,
             alpha_jac: float = 1.0,
             ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss from model outputs and targets.

        targets should contain:
          'returns': (B, n_assets) — actual returns
          'volatility': (B, n_assets) — realized vol (optional)
        """
        from .rssm import symlog

        return_loss = F.mse_loss(
            output['returns'],
            symlog(targets['returns']),
        )

        losses = {
            'return_loss': return_loss,
            'kl_loss': alpha_kl * output['kl_loss'],
            'acyclicity_loss': alpha_dag * output['acyclicity_loss'],
            'jacobian_loss': alpha_jac * output['jacobian_loss'],
        }

        if 'volatility' in targets:
            vol_loss = F.mse_loss(
                output['volatility'],
                targets['volatility'].log().clamp(min=-10),
            )
            losses['vol_loss'] = vol_loss

        losses['total'] = sum(losses.values())
        return losses
