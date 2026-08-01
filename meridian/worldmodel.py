"""Meridian World Model — a genuine deep state-space model of the market (so the label is honest).

A learned latent-dynamics model that satisfies the technical definition of a world model
(Ha-Schmidhuber; Dreamer; Deep Markov Models, Krishnan-Shalit-Sontag):

  • LATENT STATE   z_t ∈ R^K — a learned representation of the market's condition.
  • LEARNED DYNAMICS  p(z_t | z_{t-1}, c_{t-1}) — a SIGN-CONDITIONED neural stochastic transition
                    (the "forward model"); conditioning on sign(r)·|r| gives the LEVERAGE effect,
                    and an exogenous slot u_t is the structural "do-hook" for what-ifs.
  • EMISSION        p(r_t | z_t) — a heavy-tailed STUDENT-t with dynamic low-rank factor covariance
                    (latent stochastic vol); Student-t emission + Gaussian transition = fat daily
                    tails without destabilizing long rollouts (DreamerV3 stabilization principle).
  • INFERENCE       q(z_t | r_{≤t}) — a GRU filter (structured inference network).
  • ROLLOUT         sample z_t → z_{t+1} … and DECODE coherent multi-asset return PATHS (imagination).
  • INTERVENTION    set u_t / clamp a latent and roll forward → genuine "what-if".

Trained by maximizing the ELBO. Passive (observational) world model — V + M + rollout + intervention,
but NOT agentic (no controller trained inside). Value = coherent SCENARIO SIMULATION the specialist
forecasters cannot provide. See WORLDMODEL_CORE.md for exactly what we can and cannot claim.
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


def sfeat(r):
    """signed magnitude feature sign(r)·log(1+|r|) — carries the leverage sign into the transition."""
    return torch.sign(r) * torch.log1p(r.abs())


class MeridianWorldModel(nn.Module):
    def __init__(self, n_assets: int, K: int = 12, n_factors: int = 3, n_u: int = 4, hid: int = 96):
        super().__init__()
        self.N, self.K, self.F, self.n_u = n_assets, K, n_factors, n_u
        self.cond = n_assets + n_u                          # sign-conditioning + exogenous shocks
        # inference (filtering) network
        self.enc = nn.GRU(n_assets, hid, batch_first=True)
        self.q_mu = nn.Linear(hid, K)
        self.q_ls = nn.Linear(hid, K)
        # transition p(z_t | z_{t-1}, cond_{t-1}) — gated; conditioned on prev signed-return + u
        self.tr = mlp([K + self.cond, hid, hid])
        self.tr_mu = nn.Linear(hid, K)
        self.tr_ls = nn.Linear(hid, K)
        self.tr_gate = nn.Linear(K + self.cond, K)
        # emission p(r_t | z_t): per-asset log-var + low-rank factor loadings; Student-t dof
        self.em = mlp([K, hid, hid])
        self.em_logvar = nn.Linear(hid, n_assets)
        self.em_load = nn.Linear(hid, n_assets * n_factors)
        self.nu_raw = nn.Parameter(torch.tensor(2.0))       # emission degrees of freedom (learned)
        self.z0 = nn.Parameter(torch.zeros(K))

    def nu(self):
        return 2.5 + F.softplus(self.nu_raw)                # ν > 2.5 → finite variance

    # ---- distributions ----
    def transition(self, z, cond):
        zc = torch.cat([z, cond], -1)
        h = self.tr(zc); g = torch.sigmoid(self.tr_gate(zc))
        mu = g * self.tr_mu(h) + (1 - g) * z               # gated ⇒ persistent latent (vol clustering)
        ls = self.tr_ls(h).clamp(-6, 2)
        return mu, ls

    def emit_params(self, z):
        h = self.em(z)
        logvar = self.em_logvar(h).clamp(-16, 4)
        load = self.em_load(h).reshape(*z.shape[:-1], self.N, self.F) * 0.3
        return logvar, load

    def _quad_logdet(self, r, logvar, L):
        """r^T Σ^-1 r and logdet Σ for Σ = diag(exp(logvar)) + L L^T (Woodbury, F<<N)."""
        d = torch.exp(logvar) + EPS; Dinv = 1.0 / d
        Lt_Dinv = L.transpose(-1, -2) * Dinv.unsqueeze(-2)
        M = (1.0 + 1e-4) * torch.eye(self.F, device=r.device) + Lt_Dinv @ L
        quad_d = (r * (r * Dinv)).sum(-1)
        w = Lt_Dinv @ r.unsqueeze(-1)
        sol = torch.linalg.solve(M, w)
        quad = (quad_d - (w.squeeze(-1) * sol.squeeze(-1)).sum(-1)).clamp(min=0)
        logdet = logvar.sum(-1) + torch.linalg.slogdet(M)[1]
        return quad, logdet

    def nll_returns(self, r, z):
        """-log StudentT_ν(r; 0, Σ(z)) with dynamic factor covariance Σ."""
        logvar, L = self.emit_params(z)
        quad, logdet = self._quad_logdet(r, logvar, L)
        nu = self.nu(); N = self.N
        logp = (torch.lgamma((nu + N) / 2) - torch.lgamma(nu / 2) - 0.5 * N * torch.log(nu * np.pi)
                - 0.5 * logdet - 0.5 * (nu + N) * torch.log1p(quad / nu))
        return -logp

    # ---- ELBO ----
    def elbo(self, R, beta=1.0, free_bits=0.0):    # free_bits>0 over-activates the free-run on this data
        B, T, N = R.shape
        h, _ = self.enc(R)
        qm, qls = self.q_mu(h), self.q_ls(h).clamp(-6, 2)
        z = qm + torch.randn_like(qm) * torch.exp(0.5 * qls)
        zprev = torch.cat([self.z0.expand(B, 1, self.K), z[:, :-1]], 1)
        # conditioning at t = (signed return_{t-1}, u=0); pad first step with zeros
        sf = torch.cat([sfeat(R), torch.zeros(B, T, self.n_u, device=R.device)], -1)   # [B,T,cond]
        cond_prev = torch.cat([torch.zeros(B, 1, self.cond, device=R.device), sf[:, :-1]], 1)
        pm, pls = self.transition(zprev, cond_prev)
        kl_dim = 0.5 * (pls - qls + (torch.exp(qls) + (qm - pm) ** 2) / torch.exp(pls) - 1)  # [B,T,K]
        kl = torch.clamp(kl_dim, min=free_bits).sum(-1)    # FREE-BITS: keep the latent informative
        recon = self.nll_returns(R.reshape(B * T, N), z.reshape(B * T, self.K)).reshape(B, T)
        loss = (recon + beta * kl).mean()
        return loss, recon.mean(), kl.mean()

    @torch.no_grad()
    def filter_state(self, R):
        h, _ = self.enc(R)
        return self.q_mu(h[:, -1])

    # ---- Student-t emission sampling ----
    @torch.no_grad()
    def emit_sample(self, z, prev_r=None, n_paths=500):
        zr = z.unsqueeze(0).repeat(n_paths, 1)
        logvar, L = self.emit_params(zr); d = torch.exp(logvar)
        g = torch.randn(n_paths, self.N) * torch.sqrt(d) + (L @ torch.randn(n_paths, self.F, 1)).squeeze(-1)
        nu = self.nu()
        w = torch.distributions.Chi2(nu).sample((n_paths, 1)).clamp(min=1e-3) / nu
        return g / torch.sqrt(w)                            # multivariate Student-t

    # ---- ROLLOUT: imagine coherent multi-step return paths ----
    @torch.no_grad()
    def rollout(self, z0, steps, n_paths=500, u_shock=None, intervene=None, temperature=1.0):
        """From latent z0, sample n_paths of multi-asset returns; sign-condition on each emitted
        return and thread an optional exogenous shock u_shock(t)->[n_u] (the structural do-hook)."""
        z = z0.unsqueeze(0).repeat(n_paths, 1)
        prev_r = torch.zeros(n_paths, self.N)
        paths = torch.zeros(n_paths, steps, self.N)
        for t in range(steps):
            u = torch.zeros(n_paths, self.n_u)
            if u_shock is not None:
                u = u + torch.as_tensor(u_shock(t), dtype=torch.float32)
            cond = torch.cat([sfeat(prev_r), u], -1)
            mu, ls = self.transition(z, cond)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * ls) * temperature
            if intervene is not None:
                z = intervene(z, t)
            r = self.emit_sample_batch(z)
            paths[:, t] = r; prev_r = r
        return paths

    @torch.no_grad()
    def emit_sample_batch(self, z):
        """emit one Student-t return per row of a batched latent z [P,K]."""
        logvar, L = self.emit_params(z); d = torch.exp(logvar); P = z.shape[0]
        g = torch.randn(P, self.N) * torch.sqrt(d) + (L @ torch.randn(P, self.F, 1)).squeeze(-1)
        nu = self.nu()
        w = torch.distributions.Chi2(nu).sample((P, 1)).clamp(min=1e-3) / nu
        return g / torch.sqrt(w)
