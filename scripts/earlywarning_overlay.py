"""Does ACTING on the IV early-warning improve realized risk-adjusted outcomes — not just VaR coverage?
(Roadmap #1, the actionable-value test.)

The term-structure inversion (VIX9D/VIX3M > 1) leads stress onset by ~6 days. Strategy test: hold SPY,
but cut exposure to (1 - haircut) for the next day whenever the structure is inverted at today's close
(causal). Compare buy-and-hold vs the de-risking overlay OUT-OF-SAMPLE on realized return, volatility,
Sortino, max drawdown, and 99% VaR breaches — net of a transaction cost on position changes.

Honest either way: a timing overlay must cut risk by MORE than it gives up return, after costs, or it
is not worth wiring. Reports the full ledger.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian import exog

TRADING = 252
COST = 0.0002          # 2 bps per unit turnover (round-trip ~4 bps)


def metrics(r):
    r = r[np.isfinite(r)]
    ann = r.mean() * TRADING
    vol = r.std() * np.sqrt(TRADING)
    downside = r[r < 0].std() * np.sqrt(TRADING)
    sortino = ann / (downside + 1e-12)
    eq = np.cumprod(1 + r); dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    dsig = pd.Series(r).rolling(252).std().shift(1) * 2.326
    breach = float((pd.Series(r).values < -dsig.values)[252:].mean() * 100)
    return {"ret_ann_pct": ann * 100, "vol_ann_pct": vol * 100, "sortino": sortino,
            "max_dd_pct": dd * 100, "var99_breach_pct": breach}


def main():
    spy = fetch_yahoo("SPY")
    ret = np.log(spy["adjclose"]).diff().rename("ret")
    iv = exog.load_iv(); v9d, v3m = iv.get("VIX9D"), iv.get("VIX3M")
    if v9d is None or v3m is None:
        print("term-structure data unavailable"); return
    ts = (v9d.reindex(ret.index).ffill() / v3m.reindex(ret.index).ffill())
    df = pd.DataFrame({"ret": ret, "ts": ts}).dropna()
    inverted = (df["ts"] > 1.0).astype(int)                 # signal at close t

    print(f"early-warning overlay — SPY, {len(df)} days ({df.index[0].date()}→{df.index[-1].date()})")
    print(f"inverted {inverted.mean()*100:.0f}% of days\n")
    bh = df["ret"].to_numpy()
    print(f"  {'strategy':>22} {'ret%':>7} {'vol%':>7} {'Sortino':>8} {'maxDD%':>8} {'VaR99 br%':>10}")
    m = metrics(bh)
    print(f"  {'buy & hold':>22} {m['ret_ann_pct']:>+6.1f} {m['vol_ann_pct']:>7.1f} {m['sortino']:>8.2f} "
          f"{m['max_dd_pct']:>8.1f} {m['var99_breach_pct']:>9.2f}%")
    for hc in (0.5, 1.0):                                    # de-risk to 50% or to 0% when inverted
        pos = (1 - hc * inverted.shift(1).fillna(0)).to_numpy()   # position for day t (causal)
        turn = np.abs(np.diff(np.r_[1.0, pos]))
        r_ov = pos * bh - COST * turn
        mo = metrics(r_ov)
        print(f"  {f'overlay (cut {int(hc*100)}% on inv)':>22} {mo['ret_ann_pct']:>+6.1f} {mo['vol_ann_pct']:>7.1f} "
              f"{mo['sortino']:>8.2f} {mo['max_dd_pct']:>8.1f} {mo['var99_breach_pct']:>9.2f}%")
    # verdict: risk-adjusted improvement = higher Sortino AND shallower drawdown
    pos = (1 - 0.5 * inverted.shift(1).fillna(0)).to_numpy()
    r_ov = pos * bh - COST * np.abs(np.diff(np.r_[1.0, pos]))
    mo = metrics(r_ov)
    better = mo["sortino"] > m["sortino"] and mo["max_dd_pct"] > m["max_dd_pct"]
    print(f"\n  VERDICT (cut-50%): Sortino {m['sortino']:.2f}→{mo['sortino']:.2f}, "
          f"maxDD {m['max_dd_pct']:.1f}%→{mo['max_dd_pct']:.1f}% — "
          f"{'the overlay improves risk-adjusted outcomes; wire it as an optional de-risk rule' if better else 'no clear risk-adjusted gain after costs; keep the signal as a FLAG only, do not auto-trade it'}.")


if __name__ == "__main__":
    main()
