"""Does HYBRID calibration fix the world model's joint-scenario VaR? (world-model stabilization.)

The raw world-model 1-day scenario VaR breaches ~3% (target 1%) — under-calibrated. The hybrid keeps
the WM's JOINT DEPENDENCE but rescales each marginal to the specialist EWMA vol. This backtests, OOS,
the equal-weight in-universe portfolio's 1-day 99% VaR coverage: RAW world-model vs HYBRID. Target 1.0%.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.worldmodel import load_pretrained, WM_SCALE


def _ewma_daily_vol(R, lam=0.94):
    s2 = np.var(R[:20], axis=0)
    for t in range(len(R)):
        s2 = lam * s2 + (1 - lam) * R[t] ** 2
    return np.sqrt(s2)

FILT = 250     # trailing history fed to the filter
TESTN = 500    # OOS test days (most recent)
NP = 1200


def kupiec(nb, n, p=0.01):
    if nb == 0 or nb == n:
        return float("nan")
    ph = nb / n
    lr = -2 * (np.log((1 - p) ** (n - nb) * p ** nb) - np.log((1 - ph) ** (n - nb) * ph ** nb))
    return float(1 - stats.chi2.cdf(lr, 1))


def main():
    wm, uni = load_pretrained()
    if wm is None:
        print("no checkpoint"); return
    R = pd.DataFrame({a: np.log(fetch_yahoo(a)["adjclose"]).diff() for a in uni}).dropna()
    Rn = R.to_numpy(); T, N = Rn.shape
    w = np.ones(N) / N
    start = max(FILT + 5, T - TESTN)
    br_raw = br_cal = 0; m = 0; raw_hits = []; cal_hits = []
    for t in range(start, T - 1):
        hist = Rn[t - FILT:t]
        z = wm.filter_state(torch.tensor(hist[None] * WM_SCALE, dtype=torch.float32))[0]
        paths = wm.emit_sample(z, n_paths=NP).numpy() / WM_SCALE           # [P,N]
        port_raw = paths @ w
        Zc = (paths - paths.mean(0)) / (paths.std(0) + 1e-12)
        sig = _ewma_daily_vol(hist)                                        # specialist marginal vol at t
        port_cal = (Zc * sig) @ w
        realized = Rn[t + 1] @ w
        vr = np.quantile(port_raw, 0.01); vc = np.quantile(port_cal, 0.01)
        hr = realized < vr; hc = realized < vc
        br_raw += hr; br_cal += hc; raw_hits.append(hr); cal_hits.append(hc); m += 1
    print(f"world-model joint VaR calibration — equal-weight {N}-asset in-universe book, {m} OOS days\n")
    print(f"  {'engine':>38} {'breach%':>8} {'Kupiec p':>9}")
    print(f"  {'RAW world-model scenario':>38} {br_raw/m*100:>7.2f}% {kupiec(br_raw, m):>9.3f}")
    print(f"  {'HYBRID (WM dependence + EWMA marginals)':>38} {br_cal/m*100:>7.2f}% {kupiec(br_cal, m):>9.3f}")
    print(f"\n  target 1.00%. Kupiec p>0.05 = coverage not rejected. The hybrid should move the breach")
    print(f"  rate toward 1% while KEEPING the world model's joint cross-asset dependence.")


if __name__ == "__main__":
    main()
