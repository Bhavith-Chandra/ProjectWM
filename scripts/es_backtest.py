"""Regulatory-grade tail backtest of the EVT-GPD VaR/ES engine (Roadmap #2).

The tail estimator (engine._tail, McNeil-Frey conditional EVT-GPD) already showed the best 99% VaR
CALIBRATION (scripts/cf_vs_evt.py). This adds the three tests a risk regulator actually runs:
  * Kupiec POF        — unconditional coverage: is the breach rate = 1%?
  * Christoffersen    — conditional coverage + independence: are breaches also NON-CLUSTERED?
  * Acerbi-Szekely Z2 — does realized Expected Shortfall match the predicted ES (VaR alone isn't enough)?

Rolling OOS: at each t, VaR/ES(t) from data < t; test against realized return(t+1). Honest verdict.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.engine import _tail
from meridian.features import realized_variance

ASSETS = ["SPY", "QQQ", "IWM", "EEM", "TLT", "GLD", "USO", "AAPL"]
WIN, Q = 750, 0.99
TRADING = 252


def kupiec(nb, n, p=1 - Q):
    if nb == 0:
        return float("nan")
    pihat = nb / n
    lr = -2 * (np.log(((1 - p) ** (n - nb)) * p ** nb) - np.log(((1 - pihat) ** (n - nb)) * pihat ** nb))
    return float(1 - stats.chi2.cdf(lr, 1))


def christoffersen(hits):
    h = np.asarray(hits).astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(h)):
        a, b = h[i - 1], h[i]
        n00 += (a == 0 and b == 0); n01 += (a == 0 and b == 1)
        n10 += (a == 1 and b == 0); n11 += (a == 1 and b == 1)
    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return float("nan")
    p01 = n01 / (n00 + n01); p11 = n11 / (n10 + n11); p = (n01 + n11) / (n00 + n01 + n10 + n11)
    num = (1 - p) ** (n00 + n10) * p ** (n01 + n11)
    den = (1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11
    lr_ind = -2 * np.log(num / (den + 1e-300))
    return float(1 - stats.chi2.cdf(lr_ind, 1))            # independence p-value


def acerbi_z2(ret, var, es, p=1 - Q):
    """Acerbi-Szekely Test 2: Z = mean( ret*1[ret<-var] / (n*p*(-es)) ) + 1; ~0 if ES correct."""
    r, v, e = map(np.asarray, (ret, var, es))
    hit = r < -v
    if hit.sum() == 0:
        return float("nan")
    z = np.sum(r[hit] / (len(r) * p * (-e[hit]))) + 1.0
    return float(z)                                        # >0 ⇒ ES too optimistic (under-predicts loss)


def main():
    print(f"EVT-GPD tail backtest — 1-day 99% VaR/ES, {WIN}d rolling OOS\n")
    print(f"  {'asset':>6} {'n':>5} {'breach%':>8} {'Kupiec':>7} {'Chris.ind':>9} {'Acerbi Z2':>10}  verdict")
    agg_hits = []; agg_z = []
    for a in ASSETS:
        try:
            ret = realized_variance(fetch_yahoo(a))["ret"].dropna()
        except Exception:
            continue
        r = ret.to_numpy(); n = len(r)
        vars_, ess_, rl = [], [], []
        for t in range(WIN, n - 1):
            w = pd.Series(r[t - WIN:t])
            sig = np.std(r[t - 252:t]) * np.sqrt(TRADING)   # trailing realized vol as sigma_now
            var, es = _tail(w, sig, q=Q)                    # signed (negative)
            vars_.append(-var); ess_.append(-es); rl.append(r[t + 1])   # store positive VaR/ES
        vars_, ess_, rl = map(np.array, (vars_, ess_, rl))
        hits = rl < -vars_; nb = int(hits.sum()); m = len(rl)
        kp = kupiec(nb, m); ci = christoffersen(hits); z2 = acerbi_z2(rl, vars_, ess_)
        agg_hits.append(hits.mean()); agg_z.append(z2)
        ok = (nb / m < 0.02) and (not np.isnan(kp) and kp > 0.05)
        print(f"  {a:>6} {m:>5} {nb/m*100:>7.2f}% {kp:>7.3f} {ci:>9.3f} {z2:>10.3f}  "
              f"{'PASS' if ok else 'review'}")
    print(f"\n  MEAN breach {np.mean(agg_hits)*100:.2f}% (target 1.0%) | mean Acerbi Z2 {np.nanmean(agg_z):.3f} "
          f"(0 = ES exact; >0 = ES slightly optimistic)")
    print("  Kupiec/Christoffersen p>0.05 ⇒ coverage & independence not rejected. |Z2| small ⇒ ES well-sized.")


if __name__ == "__main__":
    main()
