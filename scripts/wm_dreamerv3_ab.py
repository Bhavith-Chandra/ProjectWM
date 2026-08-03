"""DreamerV3-stabilized world model — built and tested against the CURRENT architecture and against
block-bootstrap, on the SAME proper scoring rules (energy score, variogram score), split calm/stress.

Diagnoses WHY the current model loses to block-bootstrap with two independent fixes, tested separately
so we know which mechanism (if either) is the actual problem:

  FIX A — DreamerV3 stabilizations applied to the SAME parametric (Student-t + low-rank) emission:
    1. symlog encoder input: symlog(x) = sign(x)*log(1+|x|) — compresses large-magnitude/fat-tail
       observations before the GRU sees them (Hafner et al. 2023 §"Robust Predictions").
    2. KL balancing (not just free-bits): separate DYNAMICS loss (train transition to match a
       stop-grad posterior) and REPRESENTATION loss (train posterior toward a stop-grad prior),
       weighted 0.8/0.2 — stops the encoder from being dragged toward an undertrained prior while
       still training the prior to be sample-able (DreamerV2/V3 "KL balancing").
    3. free-bits floor per-dimension (already present; kept, tuned jointly).

  FIX B — diagnostic control: if the LEARNED LATENT STATE carries real information but the *parametric*
    Student-t+low-rank emission family is simply a worse fit to the true joint return distribution than
    resampling real historical days, then swapping to a NONPARAMETRIC emission — z-conditioned filtered
    historical simulation (scale historical residual DAYS by the model's emitted per-asset vol, keep the
    REAL empirical joint dependence) — should beat block-bootstrap even with the exact same latent state.
    If FIX B wins and the raw parametric emission (with or without FIX A) does not, the mechanism is
    EMISSION MISSPECIFICATION, not representation collapse — and the right fix is nonparametric emission,
    not more stabilization.

Honest verdict on each, independently, vs block-bootstrap, calm and stress.
"""
from __future__ import annotations

import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.worldmodel import TRAINED_UNIVERSE as UNI, WM_SCALE, sfeat
from scripts.wm_eval import energy_score, variogram_score

DEV = "cpu"           # exact linalg (solve/slogdet) — MPS is flaky for this, as documented elsewhere
K, NFAC, HID, NU_MIN = 12, 3, 96, 2.5
WIN, STRIDE, EPOCHS, BATCH = 120, 20, 220, 64
TEST_DAYS = 300


def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (torch.expm1(x.abs()))


class WM(nn.Module):
    """Same architecture as meridian.worldmodel.MeridianWorldModel, with two optional stabilizations."""

    def __init__(self, n, use_symlog=False, kl_balance=False):
        super().__init__()
        self.N, self.K, self.F, self.n_u = n, K, NFAC, 4
        self.cond = n + self.n_u
        self.use_symlog, self.kl_balance = use_symlog, kl_balance
        self.enc = nn.GRU(n, HID, batch_first=True)
        self.q_mu = nn.Linear(HID, K); self.q_ls = nn.Linear(HID, K)
        self.tr = nn.Sequential(nn.Linear(K + self.cond, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.tr_mu = nn.Linear(HID, K); self.tr_ls = nn.Linear(HID, K); self.tr_gate = nn.Linear(K + self.cond, K)
        self.em = nn.Sequential(nn.Linear(K, HID), nn.GELU(), nn.Linear(HID, HID), nn.GELU())
        self.em_logvar = nn.Linear(HID, n); self.em_load = nn.Linear(HID, n * NFAC)
        self.nu_raw = nn.Parameter(torch.tensor(2.0)); self.z0 = nn.Parameter(torch.zeros(K))

    def nu(self):
        return NU_MIN + F.softplus(self.nu_raw)

    def transition(self, z, cond):
        zc = torch.cat([z, cond], -1); h = self.tr(zc); g = torch.sigmoid(self.tr_gate(zc))
        mu = g * self.tr_mu(h) + (1 - g) * z
        return mu, self.tr_ls(h).clamp(-6, 2)

    def emit_params(self, z):
        h = self.em(z)
        return self.em_logvar(h).clamp(-16, 4), self.em_load(h).reshape(*z.shape[:-1], self.N, self.F) * 0.3

    def _quad_logdet(self, r, logvar, L):
        d = torch.exp(logvar) + 1e-6; Dinv = 1.0 / d
        LtD = L.transpose(-1, -2) * Dinv.unsqueeze(-2)
        M = (1 + 1e-4) * torch.eye(self.F) + LtD @ L
        quad_d = (r * (r * Dinv)).sum(-1); w = LtD @ r.unsqueeze(-1)
        sol = torch.linalg.solve(M, w)
        quad = (quad_d - (w.squeeze(-1) * sol.squeeze(-1)).sum(-1)).clamp(min=0)
        return quad, logvar.sum(-1) + torch.linalg.slogdet(M)[1]

    def nll(self, r, z):
        logvar, L = self.emit_params(z); quad, logdet = self._quad_logdet(r, logvar, L)
        nu = self.nu(); N = self.N
        logp = (torch.lgamma((nu + N) / 2) - torch.lgamma(nu / 2) - 0.5 * N * torch.log(nu * np.pi)
                - 0.5 * logdet - 0.5 * (nu + N) * torch.log1p(quad / nu))
        return -logp

    def elbo(self, R, free_bits=0.02):
        B, T, N = R.shape
        Renc = symlog(R) if self.use_symlog else R
        h, _ = self.enc(Renc)
        qm, qls = self.q_mu(h), self.q_ls(h).clamp(-6, 2)
        z = qm + torch.randn_like(qm) * torch.exp(0.5 * qls)
        zprev = torch.cat([self.z0.expand(B, 1, K), z[:, :-1]], 1)
        sf = torch.cat([sfeat(R), torch.zeros(B, T, self.n_u)], -1)
        cprev = torch.cat([torch.zeros(B, 1, self.cond), sf[:, :-1]], 1)
        pm, pls = self.transition(zprev, cprev)
        recon = self.nll(R.reshape(B * T, N), z.reshape(B * T, K)).reshape(B, T)
        if self.kl_balance:
            # DYNAMICS loss: train the prior toward a STOP-GRAD posterior (so it becomes sample-able)
            kl_dyn = 0.5 * (pls - qls.detach() + (torch.exp(qls.detach()) + (qm.detach() - pm) ** 2) / torch.exp(pls) - 1)
            # REPRESENTATION loss: train the posterior toward a STOP-GRAD prior (regularize, don't over-drag)
            kl_rep = 0.5 * (pls.detach() - qls + (torch.exp(qls) + (qm - pm.detach()) ** 2) / torch.exp(pls.detach()) - 1)
            kl = 0.8 * torch.clamp(kl_dyn, min=free_bits).sum(-1) + 0.2 * torch.clamp(kl_rep, min=free_bits).sum(-1)
        else:
            kl_dim = 0.5 * (pls - qls + (torch.exp(qls) + (qm - pm) ** 2) / torch.exp(pls) - 1)
            kl = torch.clamp(kl_dim, min=free_bits).sum(-1)
        return (recon + kl).mean()

    @torch.no_grad()
    def filter_state(self, R):
        Renc = symlog(R) if self.use_symlog else R
        h, _ = self.enc(Renc)
        return self.q_mu(h[:, -1])

    @torch.no_grad()
    def emit_sample(self, z, n_paths=300):
        zr = z.unsqueeze(0).repeat(n_paths, 1)
        logvar, L = self.emit_params(zr); d = torch.exp(logvar)
        g = torch.randn(n_paths, self.N) * torch.sqrt(d) + (L @ torch.randn(n_paths, self.F, 1)).squeeze(-1)
        nu = self.nu()
        w = torch.distributions.Chi2(nu).sample((n_paths, 1)).clamp(min=1e-3) / nu
        return g / torch.sqrt(w)

    @torch.no_grad()
    def emit_vol(self, z):
        logvar, _ = self.emit_params(z)
        return torch.exp(0.5 * logvar)                      # per-asset conditional sigma [N]


def load_panel():
    df = pd.DataFrame({a: np.log(fetch_yahoo(a)["adjclose"]).diff() for a in UNI}).dropna()
    return df[df.index >= "2008-01-01"]


def train(model, Rtr, epochs=EPOCHS):
    opt = torch.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    chunks = [Rtr[i:i + WIN] for i in range(0, len(Rtr) - WIN, STRIDE)]
    X = torch.tensor(np.stack(chunks), dtype=torch.float32)
    torch.manual_seed(0)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), BATCH):
            b = X[perm[i:i + BATCH]]
            opt.zero_grad(); loss = model.elbo(b)
            if not torch.isfinite(loss):
                continue
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def evaluate(model, R, split, nonparam_fhs=False):
    """Returns dict[regime] -> (energy, variogram) for this model's emitted scenarios, vs realized R[t+1].
    If nonparam_fhs: emission = z-conditioned scaled historical residual days (FIX B diagnostic)."""
    T, N = R.shape
    mkt_vol = pd.Series(R.mean(1)).rolling(10).std().to_numpy()
    thr = np.nanquantile(mkt_vol[split:T - 1], 0.70)
    out = {"all": {"es": [], "vs": []}, "calm": {"es": [], "vs": []}, "stress": {"es": [], "vs": []}}
    rng = np.random.RandomState(0)
    hist_std = R[:split].std(0) + 1e-9                       # long-run per-asset vol for FHS standardization
    hist_z = R[:split] / hist_std                            # standardized historical residual DAYS (real joint dep.)
    for t in range(split, T - 1):
        hist = R[t - WIN:t]; y = R[t + 1]
        z = model.filter_state(torch.tensor(hist[None], dtype=torch.float32))[0]
        if nonparam_fhs:
            sig = model.emit_vol(z).numpy()                  # [N] model's conditional vol per asset
            idx = rng.randint(0, len(hist_z), size=300)
            X = hist_z[idx] * sig[None, :]                    # real joint shape, model-conditioned scale
        else:
            X = model.emit_sample(z, n_paths=300).numpy()
        es, vs = energy_score(X, y), variogram_score(X, y)
        reg = "stress" if mkt_vol[t] >= thr else "calm"
        for g in ("all", reg):
            out[g]["es"].append(es); out[g]["vs"].append(vs)
    return {g: (np.mean(v["es"]), np.mean(v["vs"])) for g, v in out.items()}


def block_bootstrap_baseline(R, split):
    T, N = R.shape
    mkt_vol = pd.Series(R.mean(1)).rolling(10).std().to_numpy()
    thr = np.nanquantile(mkt_vol[split:T - 1], 0.70)
    rng = np.random.RandomState(0)
    out = {"all": {"es": [], "vs": []}, "calm": {"es": [], "vs": []}, "stress": {"es": [], "vs": []}}
    for t in range(split, T - 1):
        hist = R[t - WIN:t]; y = R[t + 1]
        X = hist[rng.randint(0, len(hist), size=300)]
        es, vs = energy_score(X, y), variogram_score(X, y)
        reg = "stress" if mkt_vol[t] >= thr else "calm"
        for g in ("all", reg):
            out[g]["es"].append(es); out[g]["vs"].append(vs)
    return {g: (np.mean(v["es"]), np.mean(v["vs"])) for g, v in out.items()}


def main():
    t0 = time.time()
    df = load_panel(); R = df.to_numpy() * WM_SCALE; N = R.shape[1]
    split = len(R) - TEST_DAYS
    Rtr = R[:split]
    print(f"DreamerV3-stabilization A/B — {len(R)} days x {N} assets, train {split} / test {TEST_DAYS}\n")

    variants = [
        ("baseline (current arch)", dict(use_symlog=False, kl_balance=False)),
        ("+ symlog input", dict(use_symlog=True, kl_balance=False)),
        ("+ KL-balance", dict(use_symlog=False, kl_balance=True)),
        ("+ symlog + KL-balance", dict(use_symlog=True, kl_balance=True)),
    ]
    results = {}
    for name, kw in variants:
        m = WM(N, **kw)
        train(m, Rtr)
        results[name] = evaluate(m, R, split, nonparam_fhs=False)
        print(f"  trained: {name} [{time.time()-t0:.0f}s]")

    # FIX B diagnostic: nonparametric z-conditioned FHS, on the BEST parametric variant's latent state
    best_name = min(results, key=lambda k: results[k]["all"][0])
    m_best = WM(N, **dict(variants)[best_name]); train(m_best, Rtr)
    fhs_result = evaluate(m_best, R, split, nonparam_fhs=True)
    print(f"  trained: nonparametric FHS on '{best_name}' latent [{time.time()-t0:.0f}s]")

    bb = block_bootstrap_baseline(R, split)
    print(f"\n  {'variant':>30} {'all E':>9} {'all V':>9} {'calm E':>9} {'stress E':>10}")
    print(f"  {'block-bootstrap (baseline)':>30} {bb['all'][0]:>9.5f} {bb['all'][1]:>9.5f} "
          f"{bb['calm'][0]:>9.5f} {bb['stress'][0]:>10.5f}")
    for name, _ in variants:
        r = results[name]
        print(f"  {name:>30} {r['all'][0]:>9.5f} {r['all'][1]:>9.5f} {r['calm'][0]:>9.5f} {r['stress'][0]:>10.5f}")
    print(f"  {'FIX B: nonparam FHS (z-cond)':>30} {fhs_result['all'][0]:>9.5f} {fhs_result['all'][1]:>9.5f} "
          f"{fhs_result['calm'][0]:>9.5f} {fhs_result['stress'][0]:>10.5f}")

    best_param = min(results.values(), key=lambda r: r["all"][0])["all"][0]
    fhs_beats = fhs_result["all"][0] < bb["all"][0]
    param_beats = best_param < bb["all"][0]
    print(f"\n  DIAGNOSIS [{time.time()-t0:.0f}s]:")
    print(f"    best parametric-emission variant beats block-boot: {param_beats} ({best_name})")
    print(f"    nonparametric z-conditioned FHS beats block-boot:  {fhs_beats}")
    if fhs_beats and not param_beats:
        print("    -> MECHANISM = EMISSION MISSPECIFICATION. The latent state carries real info (FHS wins),")
        print("       but the parametric Student-t+low-rank family fits worse than real historical shape.")
        print("       FIX: adopt z-conditioned nonparametric emission, not more stabilization.")
    elif param_beats:
        print("    -> DreamerV3 stabilization (best variant) genuinely fixes it. Adopt that variant.")
    else:
        print("    -> Neither fixes it. The latent state itself carries too little information (both")
        print("       parametric AND nonparametric-with-z-scale lose) — a deeper representation problem.")


if __name__ == "__main__":
    main()
