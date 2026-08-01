"""Does the implied-vol edge GROW at longer horizons? (roadmap test — research flagged weekly/monthly
as the untapped lever). Same matched-IV ladder as benchmark_exog, but the target is forward H-day mean
RV for H in {1, 5, 22}. If +IV / +exog beat HAR by MORE at H=5/22 than at H=1, multi-horizon is the
next expansion; if not, it isn't. Purged walk-forward, QLIKE, Diebold-Mariano vs HAR."""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.features import realized_variance
from meridian.exog import load_iv, exog_features
from meridian.evalproto import qlike, diebold_mariano

EPS = 1e-12
ASSETS = ["SPY", "QQQ", "IWM", "DIA", "USO", "GLD", "EEM"]
MIN_TRAIN_D, TEST_D, EMB = 1000, 252, 22
LADDER = {"HAR": ["har_d", "har_w", "har_m"],
          "Meridian": ["har_d", "har_w", "har_m", "lev", "pos", "mktrv"],
          "Meridian+IV": ["har_d", "har_w", "har_m", "lev", "pos", "mktrv", "iv", "iv_chg"],
          "Meridian+exog": ["har_d", "har_w", "har_m", "lev", "pos", "mktrv", "iv", "iv_chg", "ts_short", "ts_long", "vrp"]}


def build(H):
    iv = load_iv(); rows = []
    for a in ASSETS:
        try:
            rvf = realized_variance(fetch_yahoo(a))
        except Exception:
            continue
        rv = rvf["rv"]; ret = rvf["ret"]; lrv = np.log(rv + EPS)
        fwd = rv.rolling(H).mean().shift(-H)                       # forward H-day mean RV
        base = pd.DataFrame({"asset": a, "date": rv.index,
                             "har_d": lrv, "har_w": np.log(rv.rolling(5).mean() + EPS),
                             "har_m": np.log(rv.rolling(22).mean() + EPS),
                             "lev": np.log((ret.clip(upper=0) ** 2) + EPS),
                             "pos": np.log((ret.clip(lower=0) ** 2) + EPS),
                             "y": np.log(fwd + EPS), "rv_next": fwd})
        ex = exog_features(a, rv.index, rv=rv, iv=iv, macro={}).reset_index(drop=True)
        rows.append(pd.concat([base.reset_index(drop=True), ex], axis=1))
    R = pd.concat(rows, ignore_index=True)
    R["mktrv"] = R.groupby("date")["har_d"].transform("mean")
    for k in LADDER:
        LADDER[k] = [c for c in LADDER[k] if c in R.columns]
    return R


def ols(tr, te, cols):
    A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
    jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return np.exp(B @ beta + jb)


def run(H):
    R = build(H)
    feats = sorted({c for cs in LADDER.values() for c in cs})
    R = R.dropna(subset=feats + ["y", "rv_next"]).reset_index(drop=True)
    dates = np.array(sorted(R["date"].unique())); preds, i = [], MIN_TRAIN_D
    while i < len(dates):
        td = set(dates[i:i + TEST_D]); cut = dates[i - EMB]
        tr = R[R["date"] < cut]; te = R[R["date"].isin(td)]
        if len(tr) < 1500 or len(te) < 60:
            i += TEST_D; continue
        o = te[["rv_next"]].copy()
        for m, cs in LADDER.items():
            o[m] = ols(tr, te, cs)
        preds.append(o); i += TEST_D
    P = pd.concat(preds, ignore_index=True); rv = P["rv_next"].to_numpy()
    loss = {m: qlike(rv, P[m].to_numpy()) for m in LADDER}
    qh = float(np.nanmean(loss["HAR"]))
    edges = {m: (1 - float(np.nanmean(loss[m])) / qh) * 100 for m in LADDER}
    dmiv = diebold_mariano(loss["Meridian+IV"], loss["HAR"])[1]
    return edges, dmiv, len(P)


def main():
    print("IV edge vs HAR by forecast horizon (QLIKE % improvement over HAR, matched-IV assets)\n")
    print(f"  {'horizon':>9} {'Meridian':>9} {'+IV':>8} {'+exog':>8} {'DM(+IV)':>9} {'n_oos':>8}")
    grow = {}
    for H, lab in [(1, "1-day"), (5, "1-week"), (22, "1-month")]:
        e, dm, n = run(H)
        grow[lab] = e["Meridian+exog"]
        print(f"  {lab:>9} {e['Meridian']:>+8.2f}% {e['Meridian+IV']:>+7.2f}% {e['Meridian+exog']:>+7.2f}% {dm:>9.3f} {n:>8}")
    print(f"\n  VERDICT: +exog edge  1d {grow['1-day']:+.1f}%  →  1w {grow['1-week']:+.1f}%  →  1m {grow['1-month']:+.1f}%")
    print("  If it rises with horizon, multi-horizon targeting is the next expansion.")


if __name__ == "__main__":
    main()
