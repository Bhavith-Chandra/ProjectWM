"""Stabilize the world model's FREE-RUN (the kurtosis-26 overshoot). The long rollout compounds
transition noise into unrealistically fat tails. `rollout(..., temperature)` scales the transition
noise; lower temperature = less latent wandering. Sweep it and find the setting whose long free-run
best matches REAL stylized facts (kurtosis ~4.5, ACF|r| ~0.13), WITHOUT killing vol clustering.

Honest target: bring free-run kurtosis toward real while keeping ACF|r| positive (clustering). If a
temperature does both, that becomes the default rollout temperature. Reports the full sweep.
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

STEPS, NPATH = 900, 8


def acf_abs(x, l=1):
    a = np.abs(x - x.mean())
    return float(np.corrcoef(a[:-l], a[l:])[0, 1])


def stats_of(series):
    return pd.Series(series).kurt(), acf_abs(series, 1), acf_abs(series, 5)


def main():
    wm, uni = load_pretrained()
    if wm is None:
        print("no checkpoint"); return
    R = pd.DataFrame({a: np.log(fetch_yahoo(a)["adjclose"]).diff() for a in uni}).dropna().to_numpy()
    real_m = R.mean(1)
    rk, ra1, ra5 = stats_of(real_m)
    print(f"world-model free-run stabilization — {STEPS}-step rollout, {NPATH} paths, market portfolio\n")
    print(f"  {'REAL market':>20}: kurtosis {rk:6.2f} | ACF|r| lag1 {ra1:.3f} lag5 {ra5:.3f}\n")
    z0 = wm.filter_state(torch.tensor(R[None] * WM_SCALE, dtype=torch.float32))[0]
    print(f"  {'temperature':>11} {'kurtosis':>9} {'ACF|r|1':>8} {'ACF|r|5':>8}   (target: kurt~{rk:.1f}, ACF>0)")
    best = None
    for temp in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        torch.manual_seed(0)
        paths = wm.rollout(z0, steps=STEPS, n_paths=NPATH, temperature=temp).numpy() / WM_SCALE  # [P,T,N]
        ks, a1s, a5s = [], [], []
        for p in range(NPATH):
            m = paths[p].mean(1)
            k, a1, a5 = stats_of(m)
            if np.isfinite(k):
                ks.append(k); a1s.append(a1); a5s.append(a5)
        kurt, a1, a5 = np.median(ks), np.median(a1s), np.median(a5s)
        # score: close to real kurtosis AND still clustering (ACF>0.05)
        score = abs(kurt - rk) + (0.5 if a1 < 0.05 else 0.0)
        flag = ""
        if a1 > 0.05 and (best is None or score < best[1]):
            best = (temp, score); flag = ""
        print(f"  {temp:>11.1f} {kurt:>9.2f} {a1:>8.3f} {a5:>8.3f}")
    if best:
        print(f"\n  BEST stabilizing temperature: {best[0]:.1f} — free-run kurtosis closest to real while")
        print(f"  keeping vol clustering (ACF|r|>0.05). Set as the default rollout temperature.")
    else:
        print("\n  No temperature both tames kurtosis and keeps clustering — report honestly.")


if __name__ == "__main__":
    main()
