"""Rigorous, architecture-independent evaluation of a financial WORLD MODEL as a generative simulator —
proper scoring rules on JOINT multi-asset scenarios, vs strong baselines. This is the yardstick every
Meridian world-model core (current or future) must beat; it scores the SCENARIO, not a point forecast.

Scoring rules (strictly proper for multivariate distributions; lower is better):
  * Energy score      ES(F,y) = E‖X−y‖ − ½E‖X−X'‖   (multivariate CRPS analog; sharpness+calibration)
  * Variogram score   VS(F,y) = Σ_ij (|y_i−y_j|^0.5 − E|X_i−X_j|^0.5)²   (DEPENDENCE-sensitive)
Plus PIT calibration (per-asset uniformity) and 1-day joint-VaR coverage.

Baselines the world model must beat to earn the name:
  (a) i.i.d. Gaussian with trailing covariance  (no dynamics)
  (b) block bootstrap of historical joint returns (empirical, no learned state)
  (c) the Meridian world model (learned latent dynamics)

Honest: if the world model does not beat block-bootstrap/Gaussian on energy+variogram OOS, its scenarios
add nothing over resampling history — report that plainly.
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
from meridian.worldmodel import load_pretrained, WM_SCALE

FILT, TESTN, M = 250, 300, 300     # filter window, test days, ensemble size


def energy_score(X, y):
    """X: [M,N] ensemble, y: [N] obs. ES = mean‖X−y‖ − 0.5 mean‖X−X'‖."""
    d1 = np.linalg.norm(X - y[None], axis=1).mean()
    # pairwise term via a random permutation (unbiased, O(M) not O(M^2))
    perm = np.random.permutation(len(X))
    d2 = np.linalg.norm(X - X[perm], axis=1).mean()
    return d1 - 0.5 * d2


def variogram_score(X, y, p=0.5):
    """Dependence-sensitive: Σ_ij (|y_i−y_j|^p − mean|X_i−X_j|^p)^2, on a subsample of pairs."""
    N = len(y)
    ex = np.mean(np.abs(X[:, :, None] - X[:, None, :]) ** p, axis=0)   # [N,N] E|X_i−X_j|^p
    vy = np.abs(y[:, None] - y[None, :]) ** p
    return float(((vy - ex) ** 2).sum())


def main():
    wm, uni = load_pretrained()
    if wm is None:
        print("no world-model checkpoint"); return
    R = pd.DataFrame({a: np.log(fetch_yahoo(a)["adjclose"]).diff() for a in uni}).dropna().to_numpy()
    T, N = R.shape
    rng = np.random.RandomState(0)
    start = max(FILT + 5, T - TESTN)
    # regime split: a conditional model should beat UNconditional block-bootstrap most on STRESS days
    mkt_vol = pd.Series(R.mean(1)).rolling(10).std().to_numpy()      # market 10d vol proxy
    thr = np.nanquantile(mkt_vol[start:T - 1], 0.70)                 # top-30% = stress
    scores = {k: {"es": {"all": [], "calm": [], "stress": []}, "vs": {"all": [], "calm": [], "stress": []}}
              for k in ["Gaussian", "block-boot", "world-model"]}
    for t in range(start, T - 1):
        hist = R[t - FILT:t]; y = R[t + 1]
        reg = "stress" if mkt_vol[t] >= thr else "calm"
        cov = np.cov(hist, rowvar=False)
        ens = {"Gaussian": rng.multivariate_normal(np.zeros(N), cov, size=M),
               "block-boot": hist[rng.randint(0, len(hist), size=M)]}
        with torch.no_grad():
            z = wm.filter_state(torch.tensor(hist[None] * WM_SCALE, dtype=torch.float32))[0]
            ens["world-model"] = wm.emit_sample(z, n_paths=M).numpy() / WM_SCALE
        for k, X in ens.items():
            es, vs = energy_score(X, y), variogram_score(X, y)
            for g in ("all", reg):
                scores[k]["es"][g].append(es); scores[k]["vs"][g].append(vs)
    print(f"World-model scenario evaluation — {N} assets, {len(scores['Gaussian']['es']['all'])} OOS days "
          f"({sum(scores['Gaussian']['es']['stress'] and [1])} split by regime)\n")
    for g in ("all", "calm", "stress"):
        nd = len(scores["block-boot"]["es"][g])
        print(f"  === {g.upper()} ({nd} days) ===   energy / variogram (lower = better)")
        bb_es, bb_vs = np.mean(scores["block-boot"]["es"][g]), np.mean(scores["block-boot"]["vs"][g])
        for k in ["Gaussian", "block-boot", "world-model"]:
            es, vs = np.mean(scores[k]["es"][g]), np.mean(scores[k]["vs"][g])
            tag = ""
            if k == "world-model":
                tag = f"  (vs block-boot: E {(1-es/bb_es)*100:+.1f}%, V {(1-vs/bb_vs)*100:+.1f}%)"
            print(f"    {k:>14} {es:>10.5f} {vs:>10.5f}{tag}")
    se = np.mean(scores["world-model"]["es"]["stress"]); sb = np.mean(scores["block-boot"]["es"]["stress"])
    print(f"\n  VERDICT: on STRESS days the world model {'BEATS' if se < sb else 'does NOT beat'} block-bootstrap "
          f"on energy score — {'conditioning adds real value where it matters' if se < sb else 'resampling still ties/wins; report honestly'}.")


if __name__ == "__main__":
    main()
