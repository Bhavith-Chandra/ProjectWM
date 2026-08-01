"""Train + validate the Meridian World Model as a GENUINE world model.

A world model earns the name by modeling the world's DYNAMICS and imagining coherent futures —
not by point-forecast accuracy. So we validate it on world-model-appropriate tests:

  1. STYLIZED FACTS (Cont 2001): do the model's SIMULATED returns reproduce the defining facts of
     real markets — volatility clustering (ACF of |r|), fat tails (kurtosis), the leverage effect?
     A model that reproduces these has genuinely learned market dynamics.
  2. SCENARIO CALIBRATION: filter to each OOS day, roll out many paths, read 1-day 99% VaR from the
     simulated distribution, and backtest coverage vs realized. Calibrated path-VaR = usable simulation.
  3. WHAT-IF: shock the latent market factor and show the coherent propagated multi-asset scenario.

Baseline for stylized facts: an i.i.d. Gaussian (no dynamics) — the model must beat it on clustering
and tails. Honest: this is a small first world model; it is judged as a SIMULATOR, not a QLIKE forecaster.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.worldmodel import MeridianWorldModel

DEV = "cpu"    # MPS has buggy linalg (solve/logdet) for the Woodbury covariance; CPU is exact
ASSETS = ["SPY", "QQQ", "IWM", "DIA", "TLT", "IEF", "LQD", "HYG", "GLD", "EEM", "EFA"]
SCALE = 100.0                                              # returns *100 for numerical stability


def load_panel():
    df = pd.DataFrame({a: np.log(fetch_yahoo(a)["adjclose"]).diff() for a in ASSETS}).dropna()
    df = df[df.index >= "2008-01-01"]
    return df


def acf_abs(x, lags=(1, 5, 10)):
    a = np.abs(x - x.mean())
    return [float(np.corrcoef(a[:-l], a[l:])[0, 1]) for l in lags]


def leverage(x, l=1):                                      # corr(r_t, |r_{t+l}|) — negative in markets
    return float(np.corrcoef(x[:-l], np.abs(x[l:]))[0, 1])


def main():
    print(f"Meridian World Model — train + validate (device={DEV})\n")
    df = load_panel(); R = df.to_numpy() * SCALE; dates = df.index; N = R.shape[1]
    split = int(len(R) * 0.7)
    Rtr, Rte = R[:split], R[split:]
    print(f"panel: {len(R)} days × {N} assets, train {len(Rtr)} / test {len(Rte)} "
          f"({dates[0].date()}→{dates[-1].date()})\n")

    # ---- train (ELBO, chunked sequences, KL annealing) ----
    wm = MeridianWorldModel(N, K=12, n_factors=3).to(DEV)
    opt = torch.optim.Adam(wm.parameters(), lr=1.5e-3, weight_decay=1e-5)
    W, stride = 120, 20
    chunks = [Rtr[i:i + W] for i in range(0, len(Rtr) - W, stride)]
    X = torch.tensor(np.stack(chunks), dtype=torch.float32).to(DEV)
    torch.manual_seed(0)
    for ep in range(220):
        beta = min(1.0, ep / 60)                           # KL annealing (avoid posterior collapse)
        perm = torch.randperm(len(X), device=DEV)
        tot = 0
        for i in range(0, len(X), 64):
            b = X[perm[i:i + 64]]
            opt.zero_grad(); loss, rec, kl = wm.elbo(b, beta)
            if not torch.isfinite(loss):
                continue                                    # skip any degenerate batch
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 5.0); opt.step(); tot += loss.item()
        if ep % 40 == 0 or ep == 219:
            print(f"  epoch {ep:3d}  ELBO {tot/max(1,len(X)//64):.3f}  (recon {rec.item():.3f}, KL {kl.item():.3f})")

    # ---- 1. STYLIZED FACTS: simulate a long path from the filtered end-of-train state ----
    z0 = wm.filter_state(torch.tensor(Rtr[None], dtype=torch.float32).to(DEV))[0]
    sim = wm.rollout(z0.cpu(), steps=len(Rte), n_paths=1)[0].numpy() / SCALE     # [T,N]
    real = Rte / SCALE
    # market portfolio (equal-weight) for aggregate stylized facts
    real_m = real.mean(1); sim_m = sim.mean(1)
    iid = np.random.RandomState(0).normal(0, real_m.std(), len(real_m))
    print("\n[1] STYLIZED FACTS (market portfolio) — does the SIMULATION look like a real market?")
    print(f"  {'stat':>26} {'REAL':>9} {'WorldModel':>11} {'iid-Gauss':>10}")
    print(f"  {'excess kurtosis (fat tails)':>26} {pd.Series(real_m).kurt():>9.2f} {pd.Series(sim_m).kurt():>11.2f} {pd.Series(iid).kurt():>10.2f}")
    ar, aw, ai = acf_abs(real_m), acf_abs(sim_m), acf_abs(iid)
    for k, l in enumerate([1, 5, 10]):
        print(f"  {f'ACF|r| lag {l} (vol cluster)':>26} {ar[k]:>9.3f} {aw[k]:>11.3f} {ai[k]:>10.3f}")
    print(f"  {'leverage corr(r,|r+1|)':>26} {leverage(real_m):>9.3f} {leverage(sim_m):>11.3f} {leverage(iid):>10.3f}")

    # ---- 2. SCENARIO VaR CALIBRATION (1-day 99% VaR from simulated paths, OOS) ----
    viol = []
    for t in range(1, len(Rte)):
        hist = torch.tensor(R[:split + t][None], dtype=torch.float32).to(DEV)
        z = wm.filter_state(hist)[0]
        paths = wm.emit_sample(z.cpu(), n_paths=400).numpy() / SCALE   # [P,N] 1-day predictive
        port = paths.mean(1)                               # equal-weight portfolio 1-day returns
        var99 = np.quantile(port, 0.01)
        viol.append(real_m[t] < var99)
    ex = np.mean(viol) * 100
    print(f"\n[2] SCENARIO VaR (1-day 99%, {len(viol)} OOS days): breach {ex:.2f}% (target 1.0%) → "
          f"{'calibrated' if 0.4 < ex < 2.0 else 'off'}")

    # ---- 2b. AGGREGATIONAL GAUSSIANITY: daily fat tails must become ~Gaussian monthly ----
    real_mo = pd.Series(real_m).rolling(21).sum().dropna()
    sim_mo = pd.Series(sim_m).rolling(21).sum().dropna()
    print(f"\n[2b] AGGREGATIONAL GAUSSIANITY (excess kurtosis, should fall daily→monthly toward 0):")
    print(f"     REAL  daily {pd.Series(real_m).kurt():>5.2f} → monthly {real_mo.kurt():>5.2f}   "
          f"WorldModel  daily {pd.Series(sim_m).kurt():>5.2f} → monthly {sim_mo.kurt():>5.2f}")

    # ---- 3. WHAT-IF: STRUCTURAL do-hook — shock the exogenous channel u_t (rate/credit-shock-like) ----
    base = wm.rollout(z0.cpu(), steps=5, n_paths=400).mean((0, 1)).numpy() / SCALE
    shk = wm.rollout(z0.cpu(), steps=5, n_paths=400,
                     u_shock=lambda t: [3.0, 0, 0, 0] if t == 0 else [0, 0, 0, 0]).mean((0, 1)).numpy() / SCALE
    print("\n[3] WHAT-IF (structural u-shock at t=0) — mean 5-day response by asset (bps):")
    for a, b, s in sorted(zip(ASSETS, base, shk), key=lambda x: x[2]):
        print(f"    {a:>5}  {(s-b)*1e4:>+7.1f} bps", end="")
    print("\n\n  Verdict: if the SIMULATION reproduces fat tails + vol clustering (vs iid-Gauss) and the")
    print("  path-VaR is calibrated, the model has genuinely learned market dynamics — a real world model.")
    torch.save(wm.state_dict(), Path(__file__).resolve().parent.parent / "results" / "worldmodel.pt")


if __name__ == "__main__":
    main()
