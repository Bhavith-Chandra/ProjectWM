"""Does REGIME-CONDITIONAL (stress-covariance) shock propagation beat the single LINEAR network at
predicting REALIZED crisis co-moves — out-of-sample? (external review #3, the reflexivity claim.)

Honest test. At impact the generalized-IRF response of asset i to a shock in the source j is
beta_i = Sigma_ij / Sigma_jj (the regression beta). The reflexivity claim is that in crises the
cross-asset betas SURGE (correlations gap toward 1), so a stress-estimated covariance should predict
crash-day co-moves better than a full-sample one. We test exactly that, walk-forward, purged:

  For each LARGE market-down day t in the test period (SPY return <= SHOCK_THR):
    - estimate betas on a trailing window ending EMB days before t (no peeking):
        beta_linear  from the full-window covariance   (what network.py's girf uses)
        beta_stress  from the covariance of the window's STRESS days only (regime_sigmas)
    - predict each other asset's return: pred_i = beta_i * realized_SPY_return(t)
    - error vs REALIZED return_i(t).
  Aggregate |error| over all crisis days, linear vs stress. Lower wins. Paired sign-test for significance.

Also reports calm-day behavior (the switching model must NOT hurt normal days).
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from scipy.stats import binomtest

# diversified liquid universe (clean daily co-moves across asset classes)
UNIV = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLU", "XLV", "XLP",
        "EEM", "EFA", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO"]
SOURCE = "SPY"
WIN = 500            # trailing estimation window (days)
EMB = 5             # embargo before the test day
SHOCK_THR = -0.02   # "crisis / large-down" day: SPY <= -2%
STRESS_Q = 0.90     # top-decile broad-shock days define the stress covariance
SHRINK = 0.25       # shrink stress cov toward pooled (few obs)


def load():
    cols = {}
    for s in UNIV:
        try:
            cols[s] = np.log(fetch_yahoo(s)["adjclose"]).diff().rename(s)
        except Exception:
            pass
    R = pd.DataFrame(cols).dropna()
    return R


def betas(cov, j):
    return cov[:, j] / (cov[j, j] + 1e-18)


def main():
    R = load()
    names = list(R.columns); j = names.index(SOURCE)
    X = R.to_numpy(); dates = R.index
    n = len(X)
    lin_err, str_err, sw_err = [], [], []      # per (crisis-day, asset) abs errors
    calm_lin, calm_sw = [], []
    n_crisis = 0
    for t in range(WIN + EMB, n):
        shock = X[t, j]
        tr = X[t - EMB - WIN:t - EMB]           # trailing window, embargoed
        Sig_full = np.cov(tr, rowvar=False)
        b_lin = betas(Sig_full, j)
        # stress covariance, DOWNSIDE-conditioned: days where the source (SPY) fell in its worst decile
        # within the window — the proper asymmetric/exceedance form (Longin-Solnik 2001; Ang-Chen 2002).
        src = tr[:, j]
        stress = src <= np.quantile(src, 1 - STRESS_Q)     # bottom-decile SPY days
        Sig_str = np.cov(tr[stress], rowvar=False) if stress.sum() > len(names) + 2 else Sig_full
        Sig_str = (1 - SHRINK) * Sig_str + SHRINK * Sig_full
        b_str = betas(Sig_str, j)
        realized = X[t]
        is_crisis = shock <= SHOCK_THR
        # switching model: stress betas on crisis days, linear betas otherwise
        b_sw = b_str if is_crisis else b_lin
        for i in range(len(names)):
            if i == j:
                continue
            pl = abs(b_lin[i] * shock - realized[i])
            ps = abs(b_str[i] * shock - realized[i])
            pw = abs(b_sw[i] * shock - realized[i])
            if is_crisis:
                lin_err.append(pl); str_err.append(ps); sw_err.append(pw)
            else:
                calm_lin.append(pl); calm_sw.append(pw)
        n_crisis += int(is_crisis)

    lin_err, str_err = np.array(lin_err), np.array(str_err)
    ml, ms = lin_err.mean(), str_err.mean()
    wins = int((str_err < lin_err).sum()); tot = len(str_err)
    p = binomtest(wins, tot, 0.5, alternative="greater").pvalue
    print(f"reflexive-propagation validation — {len(names)} assets, {n} days "
          f"({dates[0].date()}→{dates[-1].date()})")
    print(f"crisis days (SPY <= {SHOCK_THR:.0%}): {n_crisis}   |   crisis (day,asset) predictions: {tot}\n")
    print("  CRISIS-DAY co-move prediction error (mean |pred - realized|, lower is better):")
    print(f"    linear (full-sample betas):  {ml:.5f}")
    print(f"    stress (regime betas):       {ms:.5f}   ({(1-ms/ml)*100:+.1f}% vs linear)")
    print(f"    stress beats linear on {wins}/{tot} predictions  (sign-test p={p:.2e})")
    cl, cw = np.array(calm_lin).mean(), np.array(calm_sw).mean()
    print(f"\n  CALM-DAY error — switching model must not hurt: linear {cl:.5f} vs switching {cw:.5f} "
          f"({'OK, identical (uses calm betas)' if abs(cl-cw)<1e-9 else 'differs'})")
    verdict = ("REFLEXIVITY CONFIRMED — stress betas predict crash co-moves better OOS"
               if ms < ml and p < 0.05 else
               "NOT CONFIRMED on this universe — keep linear GIRF as default, report honestly")
    print(f"\n  VERDICT: {verdict}.")


if __name__ == "__main__":
    main()
