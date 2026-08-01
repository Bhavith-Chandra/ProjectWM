"""Adversarial test of the 'market-RV factor = lookahead leakage' claim (external review #4).

The market factor is mktrv_t = cross-sectional mean of log-RV at date t (contemporaneous, known at
close t; NOT t+1). The reviewer's steelman fix: use a STRICTLY 1-DAY-LAGGED factor (mktrv_{t-1}),
which is unambiguously available in any real-time feed. If the OOS edge over HAR SURVIVES with the
lagged factor, the 'the beat is leakage' claim is empirically refuted.

Purged/embargoed walk-forward, QLIKE (proxy-robust), Diebold-Mariano vs HAR, on the 24 HELD-OUT
never-trained assets. Linear (the recommended model). Compares:
  HAR                          — cascade only
  HAR + mktrv (contemporaneous, t)   — as shipped
  HAR + mktrv (STRICT lag, t-1)      — the reviewer's fix
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.heldout import load_heldout
from meridian.features import realized_variance
from meridian.evalproto import qlike, diebold_mariano

EPS = 1e-12
MIN_TRAIN_D, TEST_D, EMB = 1000, 378, 22


def build():
    rows = []
    for a, ohlc in load_heldout().items():
        rvf = realized_variance(ohlc); rv = rvf["rv"]; ret = rvf["ret"]
        lrv = np.log(rv + EPS)
        rows.append(pd.DataFrame({
            "asset": a, "date": rv.index, "har_d": lrv,
            "har_w": np.log(rv.rolling(5).mean() + EPS), "har_m": np.log(rv.rolling(22).mean() + EPS),
            "lev": np.log((ret.clip(upper=0) ** 2) + EPS), "pos": np.log((ret.clip(lower=0) ** 2) + EPS),
            "y": lrv.shift(-1), "rv_next": rv.shift(-1)}))
    R = pd.concat(rows, ignore_index=True)
    # contemporaneous market factor (as shipped): cross-sectional mean of log-RV at date t
    R["mktrv"] = R.groupby("date")["har_d"].transform("mean")
    # STRICT 1-day lag PER ASSET (reviewer's fix): only info available at close t-1
    R = R.sort_values(["asset", "date"])
    R["mktrv_lag1"] = R.groupby("asset")["mktrv"].shift(1)
    return R


LADDER = {
    "HAR":                 ["har_d", "har_w", "har_m", "lev", "pos"],
    "HAR+mktrv(t)":        ["har_d", "har_w", "har_m", "lev", "pos", "mktrv"],
    "HAR+mktrv(t-1) LAG":  ["har_d", "har_w", "har_m", "lev", "pos", "mktrv_lag1"],
}


def ols(tr, te, cols):
    A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
    jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return np.exp(B @ beta + jb)


def main():
    R = build()
    feats = sorted({c for cs in LADDER.values() for c in cs})
    R = R.dropna(subset=feats + ["y", "rv_next"]).reset_index(drop=True)
    print(f"leakage test — {R['asset'].nunique()} held-out assets, {len(R)} rows "
          f"({R['date'].min().date()}→{R['date'].max().date()})\n")
    dates = np.array(sorted(R["date"].unique()))
    preds, i = [], MIN_TRAIN_D
    while i < len(dates):
        td = set(dates[i:i + TEST_D]); cut = dates[i - EMB]
        tr = R[R["date"] < cut]; te = R[R["date"].isin(td)]
        if len(tr) < 1500 or len(te) < 60:
            i += TEST_D; continue
        o = te[["y", "rv_next"]].copy()
        for m, cs in LADDER.items():
            o[m] = ols(tr, te, cs)
        preds.append(o); i += TEST_D
    P = pd.concat(preds, ignore_index=True); rv = P["rv_next"].to_numpy()
    loss = {m: qlike(rv, P[m].to_numpy()) for m in LADDER}
    qh = float(np.nanmean(loss["HAR"]))
    print(f"  {'model':>22} {'QLIKE':>8} {'vsHAR%':>8} {'DM vs HAR':>10}")
    for m in LADDER:
        q = float(np.nanmean(loss[m]))
        dm = "—" if m == "HAR" else f"{diebold_mariano(loss[m], loss['HAR'])[1]:.4f}"
        print(f"  {m:>22} {q:>8.4f} {(1-q/qh)*100:>+7.2f}% {dm:>10}")
    e_c = (1 - np.nanmean(loss["HAR+mktrv(t)"]) / qh) * 100
    e_l = (1 - np.nanmean(loss["HAR+mktrv(t-1) LAG"]) / qh) * 100
    print(f"\n  VERDICT: contemporaneous edge {e_c:+.2f}% vs strict-lag edge {e_l:+.2f}% "
          f"→ {'edge SURVIVES strict lagging → not leakage' if e_l > 0.5 * e_c and e_l > 0 else 'edge collapses under lag → leakage-driven'}.")


if __name__ == "__main__":
    main()
