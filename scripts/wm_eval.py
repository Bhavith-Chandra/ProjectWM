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
    scores = {k: {"es": [], "vs": []} for k in ["Gaussian", "block-boot", "world-model"]}
    for t in range(start, T - 1):
        hist = R[t - FILT:t]; y = R[t + 1]
        cov = np.cov(hist, rowvar=False)
        ens = {}
        ens["Gaussian"] = rng.multivariate_normal(np.zeros(N), cov, size=M)
        ens["block-boot"] = hist[rng.randint(0, len(hist), size=M)]        # resample whole-day joint returns
        with torch.no_grad():
            z = wm.filter_state(torch.tensor(hist[None] * WM_SCALE, dtype=torch.float32))[0]
            ens["world-model"] = wm.emit_sample(z, n_paths=M).numpy() / WM_SCALE
        for k, X in ens.items():
            scores[k]["es"].append(energy_score(X, y))
            scores[k]["vs"].append(variogram_score(X, y))
    print(f"World-model scenario evaluation — {N} assets, {start}→{T-1} ({len(scores['Gaussian']['es'])} OOS days)\n")
    print(f"  {'method':>14} {'energy score':>13} {'variogram score':>16}  (lower = better)")
    base = None
    for k in ["Gaussian", "block-boot", "world-model"]:
        es, vs = np.mean(scores[k]["es"]), np.mean(scores[k]["vs"])
        print(f"  {k:>14} {es:>13.5f} {vs:>16.5f}")
        if k == "block-boot":
            base = (es, vs)
    wm_es, wm_vs = np.mean(scores["world-model"]["es"]), np.mean(scores["world-model"]["vs"])
    print(f"\n  world model vs block-bootstrap: energy {(1-wm_es/base[0])*100:+.1f}%, "
          f"variogram {(1-wm_vs/base[1])*100:+.1f}%")
    win = wm_es < base[0] and wm_vs < base[1]
    print(f"  VERDICT: the learned world model {'BEATS' if win else 'does NOT beat'} block-bootstrap on joint "
          f"scenario scoring → {'its learned dynamics add real value' if win else 'resampling history is as good; report honestly'}.")


if __name__ == "__main__":
    main()
