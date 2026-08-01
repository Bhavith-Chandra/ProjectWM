"""Does an IMPLIED-VOL signal give EARLIER stress warning than the backward-looking realized-vol
labeler — and does an IV-gated VaR improve coverage at stress onset? (external review #5.)

The IV *level* is already inside the champion forecaster (HAR-lev+IV, +10.3%). The open question is
whether the FORWARD-LOOKING structure — VIX term-structure inversion (VIX9D/VIX3M > 1, near-term fear
exceeding medium-term) and the IV/RV divergence — LEADS the realized-vol regime, per the review's
'instant anomaly gate'. Two honest tests on SPY:

  (A) LEAD TIME. Define RV-stress onset = the day SPY's 5-day realized vol first crosses its trailing
      1-year 85th percentile. For each onset, how many days EARLIER did the IV signal fire?
  (B) VaR COVERAGE AT ONSET. Compare next-day 99% VaR breaches in the 10 days from each onset:
      reactive VaR (trailing realized vol only) vs IV-gated VaR (widen instantly to the VIX-implied
      daily vol when the term structure inverts). Fewer onset breaches at similar average width = better.

Everything causal: signals at close t, evaluated against return t+1. Honest verdict either way.
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
Z99 = 2.326


def main():
    spy = fetch_yahoo("SPY")
    ret = np.log(spy["adjclose"]).diff().rename("ret")
    rv_daily = ret ** 2
    rvol = np.sqrt(rv_daily.rolling(5).mean() * TRADING)          # 5-day realized vol, annualized
    iv = exog.load_iv()
    vix, v9d, v3m = iv.get("VIX"), iv.get("VIX9D"), iv.get("VIX3M")
    if vix is None or v9d is None or v3m is None:
        print("VIX term-structure data unavailable; cannot run."); return
    df = pd.DataFrame({"ret": ret, "rvol": rvol,
                       "vix": vix.reindex(ret.index).ffill(),
                       "v9d": v9d.reindex(ret.index).ffill(),
                       "v3m": v3m.reindex(ret.index).ffill()}).dropna()
    # backward RV-stress state: 5d realized vol > trailing 1y 85th percentile (the labeler's logic)
    thr = df["rvol"].rolling(252, min_periods=60).quantile(0.85)
    df["rv_stress"] = (df["rvol"] > thr).astype(int)
    # IV early-warning signal: term-structure inversion (near-term > medium-term = backwardation)
    df["ts_inv"] = (df["v9d"] / df["v3m"] > 1.0).astype(int)
    df = df.dropna()

    # (A) lead time at each RV-stress onset
    onset = (df["rv_stress"].diff() == 1)
    onsets = df.index[onset]
    leads = []
    for o in onsets:
        w = df.loc[:o].tail(11)                                   # up to 10 days before onset + onset
        fired = w.index[w["ts_inv"] == 1]
        if len(fired):
            leads.append((o - fired[0]).days)                    # calendar days the signal led by
    leads = np.array(leads)
    print(f"IV early-warning — SPY, {len(df)} days ({df.index[0].date()}→{df.index[-1].date()})")
    print(f"RV-stress onsets: {len(onsets)}   |   preceded by a term-structure inversion: {len(leads)} "
          f"({len(leads)/max(len(onsets),1)*100:.0f}%)")
    if len(leads):
        print(f"  median lead time: {np.median(leads):.0f} calendar days  (mean {leads.mean():.1f}, "
              f"fired at-or-before onset in {(leads>=0).mean()*100:.0f}% of cases)\n")

    # (A2) PRECISION / false-positive rate: of all inversion days, how many are followed by an
    # RV-stress onset within 15 days? A leading signal that fires constantly is useless.
    inv_days = df.index[df["ts_inv"] == 1]
    onset_set = list(onsets)
    hit = 0
    for d in inv_days:
        future = [o for o in onset_set if 0 <= (o - d).days <= 15]
        hit += int(len(future) > 0)
    base = float(df["ts_inv"].mean())
    prec = hit / max(len(inv_days), 1)
    print(f"  false-positive check: term structure is inverted {base*100:.0f}% of all days; "
          f"of those, {prec*100:.0f}% precede an onset within 15d (precision).\n")

    # (B) VaR coverage in the 10 days following each onset
    daily_rvol = df["rvol"] / np.sqrt(TRADING)
    var_react = Z99 * daily_rvol                                  # reactive: realized vol only
    iv_daily = (df["vix"] / 100.0) / np.sqrt(TRADING)
    gated = df["ts_inv"] == 1
    var_gated = var_react.copy()
    var_gated[gated] = np.maximum(var_react[gated], Z99 * iv_daily[gated])   # widen on inversion
    nxt = df["ret"].shift(-1)
    onset_win = pd.Series(False, index=df.index)
    for o in onsets:
        onset_win.loc[o:] = False
        loc = df.index.get_loc(o)
        onset_win.iloc[loc:loc + 10] = True
    m = onset_win & nxt.notna()
    br_r = float((nxt[m] < -var_react[m]).mean() * 100)
    br_g = float((nxt[m] < -var_gated[m]).mean() * 100)
    wr = float(var_react[m].mean() * 100); wg = float(var_gated[m].mean() * 100)
    print("  VaR breaches in the 10 days from each onset (target 1.0%):")
    print(f"    reactive (realized vol):  {br_r:.2f}%  breach   | avg width {wr:.2f}%")
    print(f"    IV-gated (widen on inv):  {br_g:.2f}%  breach   | avg width {wg:.2f}%")
    improved = br_g < br_r
    print(f"\n  VERDICT: term-structure inversion "
          f"{'LEADS onset and the IV-gate reduces onset breaches' if (len(leads) and improved) else 'does not clearly improve onset coverage'} "
          f"— {'wire it as an early-warning flag' if improved else 'keep as context only, report honestly'}.")


if __name__ == "__main__":
    main()
