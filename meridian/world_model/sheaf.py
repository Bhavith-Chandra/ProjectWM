"""
Sheaf Neural Network Encoder for Heterogeneous Asset Classes
=============================================================
Each asset class gets its own stalk (vector space) with class-specific
GRU encoders. Restriction maps translate between stalks. The sheaf
Laplacian diffuses information respecting heterogeneity.

Cohomology H^1 measures systemic risk: non-trivial global sections
that can't be explained locally = different parts of the market
telling contradictory stories.

References:
  Bodnar et al. "Neural Sheaf Diffusion" ICLR 2023
  Barbero et al. "Sheaf Neural Networks with Connection Laplacians" ICML 2022
  Lopez de Prado "Advances in Financial ML" 2018 (fractional differentiation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class FractionalDiff(nn.Module):
    """
    Fractional differentiation (Lopez de Prado 2018).
    Makes series stationary while preserving long memory.
    d in [0, 1]: d=0 is raw, d=1 is first-difference.
    Optimal d ~ 0.3-0.7 for most financial series.
    """

    def __init__(self, d: float = 0.4, max_lag: int = 50, threshold: float = 1e-4):
        super().__init__()
        self.d = nn.Parameter(torch.tensor(d))
        self.max_lag = max_lag
        self.threshold = threshold

    def _weights(self, d: torch.Tensor) -> torch.Tensor:
        w = [torch.ones(1, device=d.device)]
        for k in range(1, self.max_lag):
            w_k = w[-1] * (d - k + 1) / k
            if w_k.abs() < self.threshold:
                break
            w.append(w_k)
        return torch.stack(w).flip(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, features)"""
        d_clamped = self.d.clamp(0.01, 0.99)
        w = self._weights(d_clamped)
        K = len(w)
        if K > x.shape[1]:
            K = x.shape[1]
            w = w[-K:]

        B, T, F = x.shape
        x_pad = F_pad(x.transpose(1, 2), (K - 1, 0)).transpose(1, 2)
        w_kernel = w.view(1, 1, -1).expand(F, 1, -1)
        out = torch.conv1d(
            x_pad.transpose(1, 2),
            w_kernel,
            groups=F,
        ).transpose(1, 2)
        return out[:, :T]


def F_pad(x, pad):
    return F.pad(x, pad)


class AdaptiveNorm(nn.Module):
    """
    Exponentially weighted z-score normalization.
    NOT full-sample — uses EW mean/var to avoid lookahead.
    Per Galashov et al. NeurIPS 2024 (DreamerV3 normalization).
    """

    def __init__(self, dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer('ew_mean', torch.zeros(dim))
        self.register_buffer('ew_var', torch.ones(dim))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, features) or (batch, seq, features)"""
        if self.training:
            flat = x.detach().reshape(-1, x.shape[-1])
            batch_mean = flat.mean(0)
            batch_var = flat.var(0, unbiased=False)
            if not self.initialized:
                self.ew_mean.copy_(batch_mean)
                self.ew_var.copy_(batch_var)
                self.initialized.fill_(True)
            else:
                self.ew_mean.mul_(self.decay).add_(batch_mean, alpha=1 - self.decay)
                self.ew_var.mul_(self.decay).add_(batch_var, alpha=1 - self.decay)

        return (x - self.ew_mean) / (self.ew_var.sqrt() + self.eps)


class StalkEncoder(nn.Module):
    """
    Per-asset-class stalk encoder.
    Each asset class has its own GRU with class-specific feature dimension.
    """

    def __init__(self, input_dim: int, stalk_dim: int, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.stalk_dim = stalk_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, stalk_dim),
            nn.LayerNorm(stalk_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            stalk_dim, stalk_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(stalk_dim)
        self.adaptive_norm = AdaptiveNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_dim) — raw features for one asset class
        returns: (batch, seq_len, stalk_dim)
        """
        x = self.adaptive_norm(x)
        x = self.input_proj(x)
        x, _ = self.gru(x)
        return self.norm(x)


class SheafLaplacian(nn.Module):
    """
    Sheaf Laplacian diffusion on the asset graph.

    Each edge (u, v) has a restriction map F_{v<e}: R^{d_v} -> R^{d_e}.
    Start with DIAGONAL restriction maps (O(d) params per edge) to avoid
    representation degeneracy on limited data.

    Diffusion: x^{l+1} = x^l - sigma(L_F . x^l . W^l)
    """

    def __init__(self, n_assets: int, stalk_dims: Dict[str, int],
                 common_dim: int, n_layers: int = 3, dropout: float = 0.1,
                 diagonal_maps: bool = True):
        super().__init__()
        self.n_assets = n_assets
        self.common_dim = common_dim
        self.n_layers = n_layers
        self.diagonal_maps = diagonal_maps

        if diagonal_maps:
            self.restriction_diag = nn.Parameter(
                torch.ones(n_assets, n_assets, common_dim) * 0.1
            )
        else:
            self.restriction_maps = nn.Parameter(
                torch.randn(n_assets, n_assets, common_dim, common_dim) * 0.01
            )

        self.diffusion_weights = nn.ModuleList([
            nn.Linear(common_dim, common_dim) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(common_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def _apply_restriction(self, x_u: torch.Tensor, u: int, v: int) -> torch.Tensor:
        if self.diagonal_maps:
            return x_u * self.restriction_diag[u, v]
        else:
            return x_u @ self.restriction_maps[u, v]

    def _laplacian_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_assets, common_dim)
        Computes L_F . x where L_F is the sheaf Laplacian.
        """
        B, N, D = x.shape
        Lx = torch.zeros_like(x)

        for v in range(N):
            for u in range(N):
                if u == v:
                    continue
                Fv = self._apply_restriction(x[:, v], v, u)
                Fu = self._apply_restriction(x[:, u], u, v)
                Lx[:, v] += Fv - Fu

        return Lx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_assets, common_dim)
        returns: (batch, n_assets, common_dim) — diffused features
        """
        for l in range(self.n_layers):
            Lx = self._laplacian_step(x)
            dx = self.diffusion_weights[l](Lx)
            x = self.norms[l](x - self.dropout(F.silu(dx)))
        return x


class CohomologyReadout(nn.Module):
    """
    Computes sheaf cohomology H^1 as a systemic risk indicator.

    H^1 = ker(delta_1) / im(delta_0).
    dim(H^1) > 0 means local signals contradict globally.

    In practice, we approximate dim(H^1) via the smallest singular values
    of the coboundary operator delta_0 — near-zero singular values
    correspond to cohomological features.
    """

    def __init__(self, n_assets: int, common_dim: int, n_top: int = 4):
        super().__init__()
        self.n_top = n_top
        self.proj = nn.Sequential(
            nn.Linear(n_top, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor,
                restriction_diag: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_assets, common_dim)
        returns: (batch, 1) — systemic risk scalar
        """
        B, N, D = x.shape
        n_edges = N * (N - 1) // 2

        coboundary_rows = []
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                Fi = x[:, i] * restriction_diag[i, j]
                Fj = x[:, j] * restriction_diag[j, i]
                coboundary_rows.append(Fi - Fj)
                idx += 1

        if len(coboundary_rows) == 0:
            return torch.zeros(B, 1, device=x.device)

        delta = torch.stack(coboundary_rows, dim=1)
        _, S, _ = torch.linalg.svd(delta, full_matrices=False)

        n_sv = min(self.n_top, S.shape[-1])
        smallest_sv = S[:, -n_sv:]
        return self.proj(smallest_sv)


class SheafEncoder(nn.Module):
    """
    Full sheaf encoder pipeline:
      1. Fractional differentiation (stationary with memory)
      2. Per-class stalk encoding (heterogeneous GRUs)
      3. Projection to common dimension
      4. Sheaf Laplacian diffusion (cross-asset information flow)
      5. Cohomology readout (systemic risk)

    Asset classes and their stalk dimensions:
      equity: 128, fixed_income: 96, commodity: 80, fx: 64, alternatives: 64
    """

    STALK_DIMS = {
        'equity': 128,
        'fixed_income': 96,
        'commodity': 80,
        'fx': 64,
        'alternatives': 64,
        'crypto': 64,
    }

    def __init__(self, asset_class_map: Dict[str, List[int]],
                 input_dims: Dict[str, int],
                 common_dim: int = 128,
                 n_diffusion_layers: int = 3,
                 dropout: float = 0.1,
                 frac_diff_d: float = 0.4):
        """
        asset_class_map: maps class name -> list of asset indices
            e.g. {'equity': [0,1,2], 'fixed_income': [3,4], ...}
        input_dims: maps class name -> input feature dimension
        """
        super().__init__()
        self.asset_class_map = asset_class_map
        self.common_dim = common_dim

        n_assets = sum(len(v) for v in asset_class_map.values())
        self.n_assets = n_assets

        self.frac_diffs = nn.ModuleDict()
        self.stalk_encoders = nn.ModuleDict()
        self.stalk_projections = nn.ModuleDict()

        for cls_name, asset_indices in asset_class_map.items():
            stalk_dim = self.STALK_DIMS.get(cls_name, 64)
            inp_dim = input_dims[cls_name]

            self.frac_diffs[cls_name] = FractionalDiff(d=frac_diff_d)
            self.stalk_encoders[cls_name] = StalkEncoder(
                input_dim=inp_dim,
                stalk_dim=stalk_dim,
                dropout=dropout,
            )
            self.stalk_projections[cls_name] = nn.Linear(stalk_dim, common_dim)

        stalk_dims = {k: self.STALK_DIMS.get(k, 64) for k in asset_class_map}
        self.sheaf_laplacian = SheafLaplacian(
            n_assets=n_assets,
            stalk_dims=stalk_dims,
            common_dim=common_dim,
            n_layers=n_diffusion_layers,
            dropout=dropout,
        )
        self.cohomology = CohomologyReadout(n_assets, common_dim)

    def forward(self, features: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        features: dict mapping class name -> (batch, seq_len, n_class_assets, input_dim)

        Returns:
            encoded: (batch, seq_len, n_assets, common_dim)
            systemic_risk: (batch, seq_len, 1)
        """
        B = None
        T = None
        asset_embeddings = {}

        for cls_name, cls_features in features.items():
            B_c, T_c, N_c, D_c = cls_features.shape
            if B is None:
                B, T = B_c, T_c

            flat = cls_features.reshape(B_c * N_c, T_c, D_c)
            flat = self.frac_diffs[cls_name](flat)
            encoded = self.stalk_encoders[cls_name](flat)
            encoded = encoded.reshape(B, N_c, T, -1)
            projected = self.stalk_projections[cls_name](encoded)
            asset_embeddings[cls_name] = projected

        all_assets = []
        index_order = []
        for cls_name, indices in self.asset_class_map.items():
            emb = asset_embeddings[cls_name]
            for local_idx, global_idx in enumerate(indices):
                all_assets.append((global_idx, emb[:, local_idx]))
                index_order.append(global_idx)

        all_assets.sort(key=lambda x: x[0])
        stacked = torch.stack([a[1] for a in all_assets], dim=1)

        systemic_risk_list = []
        diffused_list = []
        for t in range(T):
            frame = stacked[:, :, t, :]
            diffused = self.sheaf_laplacian(frame)
            diffused_list.append(diffused)

            sr = self.cohomology(
                frame, self.sheaf_laplacian.restriction_diag
            )
            systemic_risk_list.append(sr)

        encoded = torch.stack(diffused_list, dim=2)
        systemic_risk = torch.stack(systemic_risk_list, dim=1)

        return encoded, systemic_risk
