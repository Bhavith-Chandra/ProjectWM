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
    """Acerbi-Szekely (2014) Test 2: Z2 = (1/(N·p))·Σ X_t·I_t/ES_t + 1, with ES_t the POSITIVE expected
    shortfall and X_t the (signed) return. ≈0 if ES correct; <0 ⇒ ES too conservative (predicts worse
    than realized); >0 ⇒ ES too optimistic (realized losses exceed predicted ES). `es`/`var` are passed
    as positive loss magnitudes here."""
    r, v, e = map(np.asarray, (ret, var, es))
    hit = r < -v
    if hit.sum() == 0:
        return float("nan")
    z = np.sum(r[hit] / (len(r) * p * e[hit])) + 1.0       # X_t negative / ES positive → sum ≈ -1 if exact
    return float(z)


def ewma_vol(r, lam=0.94):
    """RiskMetrics EWMA daily-vol series (fast-reacting), annualized. Causal: v[t] uses r[<t]."""
    v = np.empty(len(r)); s2 = np.var(r[:20]) if len(r) > 20 else np.var(r)
    for i in range(len(r)):
        v[i] = np.sqrt(s2 * TRADING)
        s2 = lam * s2 + (1 - lam) * r[i] ** 2               # update AFTER emitting (no lookahead)
    return v


def main():
    print(f"EVT-GPD tail backtest — 1-day 99% VaR/ES, {WIN}d rolling OOS")
    print("Compares sigma_now = TRAILING realized vol (slow) vs EWMA dynamic vol (fast — what the live")
    print("engine effectively uses via the HAR-lev+IV forecast). Fix hypothesis: dynamic sigma stops")
    print("breach CLUSTERING (Christoffersen independence).\n")
    for tag, use_ewma in [("TRAILING sigma (slow)", False), ("EWMA dynamic sigma (fast)", True)]:
        print(f"  === {tag} ===")
        print(f"  {'asset':>6} {'n':>5} {'breach%':>8} {'Kupiec':>7} {'Chris.ind':>9} {'Acerbi Z2':>10}")
        agg_hits = []; agg_z = []; agg_ci = []
        for a in ASSETS:
            try:
                ret = realized_variance(fetch_yahoo(a))["ret"].dropna()
            except Exception:
                continue
            r = ret.to_numpy(); n = len(r)
            ev = ewma_vol(r) if use_ewma else None
            vars_, ess_, rl = [], [], []
            for t in range(WIN, n - 1):
                w = pd.Series(r[t - WIN:t])
                sig = ev[t] if use_ewma else np.std(r[t - 252:t]) * np.sqrt(TRADING)
                var, es = _tail(w, sig, q=Q)
                vars_.append(-var); ess_.append(-es); rl.append(r[t + 1])
            vars_, ess_, rl = map(np.array, (vars_, ess_, rl))
            hits = rl < -vars_; nb = int(hits.sum()); m = len(rl)
            kp = kupiec(nb, m); ci = christoffersen(hits); z2 = acerbi_z2(rl, vars_, ess_)
            agg_hits.append(hits.mean()); agg_z.append(z2); agg_ci.append(ci)
            print(f"  {a:>6} {m:>5} {nb/m*100:>7.2f}% {kp:>7.3f} {ci:>9.3f} {z2:>10.3f}")
        passes = int(np.sum([c > 0.05 for c in agg_ci if not np.isnan(c)]))
        print(f"  MEAN breach {np.mean(agg_hits)*100:.2f}% | mean Acerbi Z2 {np.nanmean(agg_z):.3f} | "
              f"Christoffersen independence PASSES {passes}/{len(agg_ci)} assets\n")


if __name__ == "__main__":
    main()
