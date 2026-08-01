"""OUT-OF-SAMPLE gauntlet for the ThresholdShockNetwork (external review #3, historical-grounding form).

Does a crisis covariance grounded in ACTUAL stress blocks (2008 GFC, 2020 COVID) predict realized
crisis co-moves better than the linear GIRF baseline — on a HELD-OUT later crisis?

Leave-one-crisis-out, CHRONOLOGICAL (no test crisis informs its own covariance — the integrity fix):
  * Predict COVID-2020 using a crisis covariance built from GFC-2008 only.
  * Predict 2022     using a crisis covariance built from GFC-2008 + COVID-2020.
Linear baseline = full return covariance estimated on all data strictly BEFORE the test crisis.

For each day t in the test-crisis block, given the source's realized return that day, predict every
other asset's return as beta_i * r_source(t); score squared error vs the REALIZED return. Report
per-crisis RMSE (threshold vs linear) and a one-sided Diebold-Mariano (threshold better if p<0.05).
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.network import crisis_calm_cov, ThresholdShockNetwork

UNIV = ["SPY", "QQQ", "IWM", "DIA", "XLF", "XLK", "XLE", "XLU", "XLV", "XLP",
        "EEM", "EFA", "TLT", "IEF", "LQD", "HYG", "GLD", "SLV", "USO"]
SOURCE = "SPY"
CRISES = {
    "GFC-2008":  ("2008-09-01", "2009-03-31"),
    "COVID-2020": ("2020-02-20", "2020-04-15"),
    "Rate-2022":  ("2022-08-16", "2022-10-14"),
}
ORDER = ["GFC-2008", "COVID-2020", "Rate-2022"]


def load():
    cols = {}
    for s in UNIV:
        try:
            cols[s] = np.log(fetch_yahoo(s)["adjclose"]).diff().rename(s)
        except Exception:
            pass
    return pd.DataFrame(cols).dropna()


def dm_one_sided(loss_lin, loss_thr):
    """H1: threshold has LOWER loss. d = loss_lin - loss_thr; test mean(d) > 0 (Newey-West lag-5)."""
    d = np.asarray(loss_lin) - np.asarray(loss_thr)
    n = len(d); dbar = d.mean()
    g0 = np.var(d)
    lrv = g0
    for L in range(1, 6):
        cov = np.cov(d[:-L], d[L:])[0, 1]
        lrv += 2 * (1 - L / 6) * cov
    se = np.sqrt(max(lrv, 1e-18) / n)
    from scipy.stats import norm
    stat = dbar / se
    return float(stat), float(1 - norm.cdf(stat))   # one-sided p (small = threshold better)


def betas(cov, j):
    return cov[:, j] / (cov[j, j] + 1e-18)


def main():
    R = load(); names = list(R.columns); j = names.index(SOURCE)
    print(f"network gauntlet — {len(names)} assets, {len(R)} days "
          f"({R.index[0].date()}→{R.index[-1].date()})\n")
    print(f"  {'test crisis':>12} {'n_days':>6} {'RMSE_linear':>12} {'RMSE_thresh':>12} {'Δ%':>7} {'DM p(1-sided)':>14}")
    for ti in range(1, len(ORDER)):                    # can't test the first crisis (no prior)
        test = ORDER[ti]; a, b = CRISES[test]
        priors = [CRISES[ORDER[k]] for k in range(ti)]  # crises strictly before the test crisis
        start = pd.Timestamp(a)
        pre = R[R.index < start]                        # all data before the test crisis
        if len(pre) < 300:
            continue
        calm_cov, crisis_cov = crisis_calm_cov(pre, priors)
        full_cov = np.cov(pre.to_numpy(), rowvar=False)  # linear GIRF baseline covariance
        net = ThresholdShockNetwork(names, calm_cov, crisis_cov)
        block = R[(R.index >= start) & (R.index <= pd.Timestamp(b))]
        b_lin = betas(full_cov, j)
        loss_lin, loss_thr = [], []
        for _, row in block.iterrows():
            shock = row[SOURCE]
            pred_thr = net.propagate_shock(SOURCE, shock)     # crisis branch on these down days
            for i, nm in enumerate(names):
                if i == j:
                    continue
                realized = row[nm]
                loss_lin.append((b_lin[i] * shock - realized) ** 2)
                loss_thr.append((pred_thr[nm] - realized) ** 2)
        rl = np.sqrt(np.mean(loss_lin)); rt = np.sqrt(np.mean(loss_thr))
        stat, p = dm_one_sided(loss_lin, loss_thr)
        print(f"  {test:>12} {len(block):>6} {rl:>12.5f} {rt:>12.5f} {(1-rt/rl)*100:>+6.1f}% {p:>14.3f}")
    print("\n  Δ% > 0 and DM p < 0.05 ⇒ the crisis-grounded threshold network beats linear GIRF OOS on")
    print("  that held-out crisis. Otherwise linear GIRF stays the default and we report it honestly.")

    # ---- the crisis covariance's STRONGEST case: portfolio tail-risk coverage DURING a crisis ----
    # "correlations gap to 1" should make full-sample cov UNDER-estimate portfolio vol in a crisis
    # (too-many breaches); a crisis cov should cover better. Equal-weight risk-asset book, OOS.
    RISK = [s for s in ["SPY", "QQQ", "IWM", "XLF", "XLK", "XLE", "EEM", "EFA", "HYG"] if s in names]
    ix = [names.index(s) for s in RISK]; w = np.ones(len(ix)) / len(ix); Z99 = 2.326
    print(f"\n  PORTFOLIO stress-VaR (equal-weight {len(RISK)}-asset risk book; target 1.0% breaches in-crisis):")
    print(f"  {'test crisis':>12} {'breach_fullcov':>14} {'breach_crisiscov':>17} {'width_full%':>12} {'width_crisis%':>13}")
    for ti in range(1, len(ORDER)):
        test = ORDER[ti]; a, b = CRISES[test]
        priors = [CRISES[ORDER[k]] for k in range(ti)]
        pre = R[R.index < pd.Timestamp(a)]
        if len(pre) < 300:
            continue
        calm_cov, crisis_cov = crisis_calm_cov(pre, priors)
        full_cov = np.cov(pre.to_numpy(), rowvar=False)
        sub_full = full_cov[np.ix_(ix, ix)]; sub_cris = crisis_cov[np.ix_(ix, ix)]
        vol_full = np.sqrt(w @ sub_full @ w); vol_cris = np.sqrt(w @ sub_cris @ w)
        block = R[(R.index >= pd.Timestamp(a)) & (R.index <= pd.Timestamp(b))]
        port_ret = block[RISK].to_numpy() @ w
        # decompose: is coverage from crisis CORRELATIONS or crisis VOLS? build hybrid covariances.
        df_ = np.sqrt(np.diag(sub_full)); dc_ = np.sqrt(np.diag(sub_cris))
        Cf = sub_full / np.outer(df_, df_); Cc = sub_cris / np.outer(dc_, dc_)
        vol_corrOnly = np.sqrt(w @ (np.outer(df_, df_) * Cc) @ w)   # crisis corr + calm vol
        br_full = float((port_ret < -Z99 * vol_full).mean() * 100)
        br_cris = float((port_ret < -Z99 * vol_cris).mean() * 100)
        br_corr = float((port_ret < -Z99 * vol_corrOnly).mean() * 100)
        print(f"  {test:>12} {br_full:>13.2f}% {br_cris:>16.2f}% {Z99*vol_full*100:>11.2f}% {Z99*vol_cris*100:>12.2f}%"
              f"   [crisis-corr-only: breach {br_corr:.1f}%, width {Z99*vol_corrOnly*100:.2f}%]")
    print("  If crisis-cov breaches are closer to 1% (fewer) than full-cov, the crisis covariance earns")
    print("  a role in PORTFOLIO stress-VaR even though it did not improve point co-move propagation.")


if __name__ == "__main__":
    main()
