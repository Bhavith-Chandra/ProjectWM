"""
Reflexive Equilibrium + Causal Discovery
==========================================
Soros reflexivity as learned dynamics via Deep Equilibrium Models (DEQ):
  f(reality -> beliefs) and g(beliefs -> reality) iterated to fixed point.

Convergence speed rho_t is a regime signal:
  rho -> 0: stable (calm)
  rho -> 1: critical (transition)
  rho > 1: unstable (bubble/crash)

Causal structure via NOTEARS (Zheng et al. 2018):
  Differentiable DAG discovery with trace-exponential acyclicity penalty.
  Enables do(X) interventions: "what if Fed hikes 50bp?"

References:
  Bai, Kolter, Koltun "Deep Equilibrium Models" NeurIPS 2019
  Sornette "Why Stock Markets Crash" 2003
  Zheng et al. "DAGs with NO TEARS" NeurIPS 2018
  Filimonov & Sornette 2012: branching ratio n=0.7-0.8
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class CognitiveFunction(nn.Module):
    """
    f: market_state -> participant_beliefs
    How reality shapes beliefs. A transformer block that reads the
    current market state and produces a belief embedding.
    """

    def __init__(self, state_dim: int, belief_dim: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(state_dim + belief_dim, belief_dim)
        self.attn = nn.MultiheadAttention(
            belief_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(belief_dim)
        self.ff = nn.Sequential(
            nn.Linear(belief_dim, belief_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(belief_dim * 2, belief_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(belief_dim)

    def forward(self, state: torch.Tensor,
                beliefs: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(torch.cat([state, beliefs], dim=-1))
        x_3d = x.unsqueeze(1)
        attn_out, _ = self.attn(x_3d, x_3d, x_3d)
        x = self.norm1(attn_out.squeeze(1) + x)
        x = self.norm2(self.ff(x) + x)
        return x


class ManipulativeFunction(nn.Module):
    """
    g: beliefs -> market_impact
    How beliefs change reality. Models the feedback from participant
    positioning/behavior back into market dynamics.
    """

    def __init__(self, belief_dim: int, state_dim: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.proj_in = nn.Linear(belief_dim + state_dim, belief_dim)
        self.attn = nn.MultiheadAttention(
            belief_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(belief_dim)
        self.ff = nn.Sequential(
            nn.Linear(belief_dim, belief_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(belief_dim * 2, belief_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(belief_dim)

    def forward(self, beliefs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(torch.cat([beliefs, state], dim=-1))
        x_3d = x.unsqueeze(1)
        attn_out, _ = self.attn(x_3d, x_3d, x_3d)
        x = self.norm1(attn_out.squeeze(1) + x)
        x = self.norm2(self.ff(x) + x)
        return x


class ReflexiveEquilibrium(nn.Module):
    """
    DEQ-based reflexive equilibrium finder.

    Forward: iterate b_{k+1} = f(g(b_k, s_t), s_t) until convergence.
    Backward: implicit differentiation (O(1) memory, exact gradients).

    Outputs:
      b*: equilibrium beliefs
      rho_t: convergence speed (regime signal)
    """

    def __init__(self, state_dim: int, belief_dim: int = 512,
                 max_iter: int = 10, tol: float = 1e-4,
                 jac_reg: float = 0.01, dropout: float = 0.1):
        super().__init__()
        self.belief_dim = belief_dim
        self.max_iter = max_iter
        self.tol = tol
        self.jac_reg = jac_reg

        self.cognitive = CognitiveFunction(state_dim, belief_dim, dropout=dropout)
        self.manipulative = ManipulativeFunction(belief_dim, state_dim, dropout=dropout)

        self.init_beliefs = nn.Linear(state_dim, belief_dim)

    def _fixed_point_iteration(self, state: torch.Tensor,
                               beliefs: torch.Tensor) -> torch.Tensor:
        """One step of the reflexive iteration: b -> f(g(b, s), s)."""
        impact = self.manipulative(beliefs, state)
        new_beliefs = self.cognitive(state, impact)
        return new_beliefs

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        state: (batch, state_dim) — RSSM state
        returns dict with:
          'beliefs': (batch, belief_dim) — equilibrium beliefs
          'rho': (batch, 1) — convergence speed (regime signal)
          'adjusted_state': (batch, state_dim + belief_dim + 1)
          'n_iters': int — actual iterations used
        """
        B = state.shape[0]
        b = self.init_beliefs(state)

        initial_diff = None
        final_diff = None
        converged = False

        for k in range(self.max_iter):
            b_new = self._fixed_point_iteration(state, b)
            diff = (b_new - b).norm(dim=-1, keepdim=True)

            if k == 0:
                initial_diff = diff.detach()
            final_diff = diff.detach()

            if diff.max() < self.tol:
                converged = True
                b = b_new
                break
            b = b_new

        rho = final_diff / initial_diff.clamp(min=1e-8)
        rho = rho.clamp(max=5.0)

        adjusted_state = torch.cat([state, b, rho], dim=-1)

        return {
            'beliefs': b,
            'rho': rho,
            'adjusted_state': adjusted_state,
            'n_iters': k + 1,
            'converged': converged,
        }

    def jacobian_regularization(self, state: torch.Tensor,
                                beliefs: torch.Tensor) -> torch.Tensor:
        """
        Regularize the Jacobian df/db to keep spectral radius < 1.
        Prevents gradient explosion in backward pass.
        """
        beliefs_rg = beliefs.detach().requires_grad_(True)
        b_new = self._fixed_point_iteration(state, beliefs_rg)

        jvp = torch.autograd.grad(
            b_new.sum(), beliefs_rg, create_graph=True,
        )[0]

        return self.jac_reg * (jvp ** 2).mean()


class CausalDAG(nn.Module):
    """
    Differentiable causal graph discovery via NOTEARS.

    Learns an N x N weighted adjacency matrix A where A[i,j] > 0
    means asset i causally influences asset j. Acyclicity enforced
    via trace-exponential penalty: h(A) = tr(e^{A*A}) - N = 0.

    Enables do(X) interventions: clamp a node and propagate.
    """

    def __init__(self, n_assets: int, state_dim: int, hidden: int = 128):
        super().__init__()
        self.n_assets = n_assets

        self.adj_net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets * n_assets),
        )

        self.structural_eq = nn.Sequential(
            nn.Linear(n_assets + state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def adjacency(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (batch, state_dim)
        returns: (batch, n_assets, n_assets) — weighted adjacency matrix
        """
        raw = self.adj_net(state).reshape(-1, self.n_assets, self.n_assets)
        A = torch.sigmoid(raw)
        mask = 1.0 - torch.eye(self.n_assets, device=A.device).unsqueeze(0)
        return A * mask

    def acyclicity_penalty(self, A: torch.Tensor) -> torch.Tensor:
        """
        NOTEARS acyclicity constraint: h(A) = tr(e^{A*A}) - N = 0
        This equals zero iff A is a DAG.
        """
        N = self.n_assets
        A_sq = A * A
        M = torch.matrix_exp(A_sq)
        h = torch.diagonal(M, dim1=-2, dim2=-1).sum(-1) - N
        return h.mean()

    def intervene(self, state: torch.Tensor, node: int,
                  value: torch.Tensor) -> torch.Tensor:
        """
        do(node = value): clamp a node and propagate through the DAG.
        This is the causal intervention interface.

        state: (batch, state_dim)
        node: which asset to intervene on
        value: (batch,) — the intervention value
        returns: (batch, n_assets) — predicted effect on all assets
        """
        A = self.adjacency(state)
        x = torch.zeros(state.shape[0], self.n_assets, device=state.device)
        x[:, node] = value

        for _ in range(3):
            effect = torch.bmm(A.transpose(-1, -2), x.unsqueeze(-1)).squeeze(-1)
            x = x + effect
            x[:, node] = value

        return x

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        state: (batch, state_dim)
        returns: adjacency matrix + acyclicity penalty
        """
        A = self.adjacency(state)
        h = self.acyclicity_penalty(A)

        return {
            'adjacency': A,
            'acyclicity_loss': h,
        }


class CausalAttention(nn.Module):
    """
    Attention weights constrained by the discovered DAG.
    Only attend to causal parents — prevents spurious correlations
    from leaking through attention.
    """

    def __init__(self, d_model: int, n_assets: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                adjacency: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, n_assets, d_model)
        adjacency: (batch, n_assets, n_assets) — DAG mask
        """
        B, N, D = x.shape
        residual = x

        q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        dag_mask = adjacency.unsqueeze(1)
        dag_mask = dag_mask + torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0)
        attn = attn.masked_fill(dag_mask < 0.1, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = attn.nan_to_num(0.0)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.norm(self.out(out) + residual)
