"""Meridian World Model — a genuine deep state-space model of the market (so the label is honest).

This is NOT a point-forecaster. It is a learned latent-dynamics model that satisfies the technical
definition of a world model (Ha-Schmidhuber; Dreamer; Deep Markov Models, Krishnan-Shalit-Sontag):

  • LATENT STATE   z_t ∈ R^K — a learned, compact representation of the market's condition.
  • LEARNED DYNAMICS  p(z_t | z_{t-1}) — a neural stochastic transition (the "forward model").
  • EMISSION        p(r_t | z_t) — a DYNAMIC-COVARIANCE Gaussian: latent state sets each asset's
                    volatility and a low-rank cross-asset factor structure (latent stochastic vol).
  • INFERENCE       q(z_t | r_{≤t}) — a GRU filter (structured inference network).
  • ROLLOUT         sample z_t → z_{t+1} → … and DECODE coherent multi-asset return PATHS (imagination).
  • INTERVENTION    clamp / shock a latent (or an asset's vol) and roll forward → genuine "what-if".

Trained by maximizing the ELBO. It legitimately models the world's dynamics and imagines futures —
that is what earns the name "world model", independent of point-forecast QLIKE (which the specialist
modules own). Value = COHERENT SCENARIO SIMULATION the specialists cannot provide.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def mlp(sizes, act=nn.GELU):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [act()]
    return nn.Sequential(*layers)


class MeridianWorldModel(nn.Module):
    """Deep state-space market model with latent stochastic volatility + low-rank factor covariance."""

    def __init__(self, n_assets: int, K: int = 12, n_factors: int = 3, hid: int = 96):
        super().__init__()
        self.N, self.K, self.F = n_assets, K, n_factors
        # inference (filtering) network: GRU over returns -> posterior q(z_t | r_<=t)
        self.enc = nn.GRU(n_assets, hid, batch_first=True)
        self.q_mu = nn.Linear(hid, K)
        self.q_ls = nn.Linear(hid, K)
        # transition (prior) p(z_t | z_{t-1}) — gated, so it can persist (vol clustering)
        self.tr = mlp([K, hid, hid])
        self.tr_mu = nn.Linear(hid, K)
        self.tr_ls = nn.Linear(hid, K)
        self.tr_gate = nn.Linear(K, K)
        # emission p(r_t | z_t): per-asset log-variance + low-rank factor loadings (dynamic covariance)
        self.em = mlp([K, hid, hid])
        self.em_logvar = nn.Linear(hid, n_assets)          # idiosyncratic log-variance per asset
        self.em_load = nn.Linear(hid, n_assets * n_factors)  # factor loadings (common risk)
        self.z0 = nn.Parameter(torch.zeros(K))

    # ---- distributions ----
    def transition(self, z):
        h = self.tr(z); g = torch.sigmoid(self.tr_gate(z))
        mu = g * self.tr_mu(h) + (1 - g) * z               # gated ~ persistent latent
        ls = self.tr_ls(h).clamp(-6, 2)
        return mu, ls

    def emit_params(self, z):
        h = self.em(z)
        logvar = self.em_logvar(h).clamp(-16, 4)           # idiosyncratic variance
        load = self.em_load(h).reshape(*z.shape[:-1], self.N, self.F) * 0.3
        return logvar, load

    def nll_returns(self, r, z):
        """-log N(r; 0, D + L L^T) with D=diag(exp(logvar)), via Woodbury (F<<N)."""
        logvar, L = self.emit_params(z)                    # logvar [B,N], L [B,N,F]
        d = torch.exp(logvar) + EPS                        # [B,N]
        Dinv = 1.0 / d
        # Woodbury: (D+LL^T)^-1 = Dinv - Dinv L (I + L^T Dinv L)^-1 L^T Dinv
        Lt_Dinv = L.transpose(-1, -2) * Dinv.unsqueeze(-2)  # [B,F,N]
        M = (1.0 + 1e-4) * torch.eye(self.F, device=r.device) + Lt_Dinv @ L  # [B,F,F] (jittered, PD)
        rDinv = r * Dinv                                    # [B,N]
        quad_d = (r * rDinv).sum(-1)                        # r^T Dinv r
        w = Lt_Dinv @ r.unsqueeze(-1)                       # [B,F,1]
        sol = torch.linalg.solve(M, w)                     # M^-1 w
        quad = (quad_d - (w.squeeze(-1) * sol.squeeze(-1)).sum(-1)).clamp(min=0)
        logdet = logvar.sum(-1) + torch.linalg.slogdet(M)[1]   # slogdet: numerically stable
        return 0.5 * (quad + logdet + self.N * np.log(2 * np.pi))

    # ---- ELBO ----
    def elbo(self, R, beta=1.0):
        """R: [B, T, N] returns. Returns (loss, recon, kl)."""
        B, T, N = R.shape
        h, _ = self.enc(R)                                 # [B,T,hid]
        qm, qls = self.q_mu(h), self.q_ls(h).clamp(-6, 2)
        z = qm + torch.randn_like(qm) * torch.exp(0.5 * qls)   # reparameterized posterior samples
        # transition prior from previous z
        zprev = torch.cat([self.z0.expand(B, 1, self.K), z[:, :-1]], 1)
        pm, pls = self.transition(zprev)
        # KL(q(z_t) || p(z_t|z_{t-1})) per step, Gaussian closed form
        kl = 0.5 * (pls - qls + (torch.exp(qls) + (qm - pm) ** 2) / torch.exp(pls) - 1).sum(-1)
        recon = self.nll_returns(R.reshape(B * T, N), z.reshape(B * T, self.K)).reshape(B, T)
        loss = (recon + beta * kl).mean()
        return loss, recon.mean(), kl.mean()

    # ---- filtering (get current latent state from history) ----
    @torch.no_grad()
    def filter_state(self, R):
        h, _ = self.enc(R)
        return self.q_mu(h[:, -1])                          # posterior mean latent at last step

    @torch.no_grad()
    def emit_sample(self, z, n_paths=500):
        """Sample next-step returns directly from a latent state (1-step predictive; no prior drift)."""
        zr = z.unsqueeze(0).repeat(n_paths, 1)
        logvar, L = self.emit_params(zr); d = torch.exp(logvar)
        eps_i = torch.randn(n_paths, self.N) * torch.sqrt(d)
        eps_f = (L @ torch.randn(n_paths, self.F, 1)).squeeze(-1)
        return eps_i + eps_f                               # [n_paths, N]

    # ---- ROLLOUT: imagine coherent multi-step return paths ----
    @torch.no_grad()
    def rollout(self, z0, steps, n_paths=500, intervene=None, temperature=1.0):
        """From latent z0 [K], sample n_paths of length `steps` of multi-asset returns.
        intervene: optional fn(z, t)->z to inject a shock (what-if).
        temperature<1 tempers the transition noise to stabilize long free-running rollouts
        (DreamerV3 stabilization) — report at 1.0, use <1 only for conditional readouts."""
        z = z0.unsqueeze(0).repeat(n_paths, 1)             # [P,K]
        paths = torch.zeros(n_paths, steps, self.N)
        for t in range(steps):
            mu, ls = self.transition(z)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * ls) * temperature
            if intervene is not None:
                z = intervene(z, t)
            logvar, L = self.emit_params(z)
            d = torch.exp(logvar)
            eps_i = torch.randn(n_paths, self.N) * torch.sqrt(d)
            eps_f = (L @ torch.randn(n_paths, self.F, 1)).squeeze(-1)   # common factor shock
            paths[:, t] = eps_i + eps_f
        return paths                                       # [P, steps, N]
