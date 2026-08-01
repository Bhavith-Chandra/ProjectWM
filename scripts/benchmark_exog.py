"""Does RICHER FREE DATA materially improve the volatility edge? Adds the matched implied-vol
family + VIX term structure + variance-risk-premium + macro/credit to the model, and benchmarks
the OOS gain over HAR and over the current Meridian — on assets that HAVE a matched implied-vol
index (SPY→VIX, QQQ→VXN, IWM→RVX, DIA→VXD, USO→OVX, GLD→GVZ, EEM→VXEEM).

Purged walk-forward, QLIKE (proxy-robust), Diebold-Mariano vs HAR and vs Meridian. Honest: whichever
feature set wins, wins; a feature is kept only if it beats the current model DM-significantly OOS.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import fetch_yahoo
from meridian.features import realized_variance
from meridian.exog import load_iv, load_macro_exog, exog_features
from meridian.evalproto import qlike, diebold_mariano
from scipy.stats import spearmanr

EPS = 1e-12
ASSETS = ["SPY", "QQQ", "IWM", "DIA", "USO", "GLD", "EEM"]     # all have a matched implied-vol index
MIN_TRAIN_D, TEST_D, EMB = 1000, 252, 22


def build():
    iv = load_iv(); macro = load_macro_exog()
    rows = []
    for a in ASSETS:
        try:
            oh = fetch_yahoo(a)
        except Exception:
            continue
        rvf = realized_variance(oh); rv = rvf["rv"]; ret = rvf["ret"]
        lrv = np.log(rv + EPS)
        base = pd.DataFrame({"asset": a, "date": rv.index,
                             "har_d": lrv, "har_w": np.log(rv.rolling(5).mean() + EPS),
                             "har_m": np.log(rv.rolling(22).mean() + EPS),
                             "lev": np.log((ret.clip(upper=0) ** 2) + EPS),
                             "pos": np.log((ret.clip(lower=0) ** 2) + EPS),
                             "y": lrv.shift(-1), "rv_next": rv.shift(-1)})
        ex = exog_features(a, rv.index, rv=rv, iv=iv, macro=macro).reset_index(drop=True)
        base = pd.concat([base.reset_index(drop=True), ex], axis=1)
        rows.append(base)
    R = pd.concat(rows, ignore_index=True)
    R["mktrv"] = R.groupby("date")["har_d"].transform("mean")
    return R


LADDER = {
    "HAR":          ["har_d", "har_w", "har_m"],
    "Meridian":     ["har_d", "har_w", "har_m", "lev", "pos", "mktrv"],
    "Meridian+IV":  ["har_d", "har_w", "har_m", "lev", "pos", "mktrv", "iv", "iv_chg"],
    "Meridian+exog": ["har_d", "har_w", "har_m", "lev", "pos", "mktrv", "iv", "iv_chg",
                      "ts_short", "ts_long", "vrp", "hy_oas", "nfci"],
}


def ols(tr, te, cols):
    A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
    jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return B @ beta + jb


def main():
    R = build()
    # keep only features that actually loaded (e.g. FRED macro may be unavailable)
    for k in list(LADDER):
        LADDER[k] = [c for c in LADDER[k] if c in R.columns]
    feats = sorted({c for cs in LADDER.values() for c in cs})
    R = R.dropna(subset=feats + ["y", "rv_next"]).reset_index(drop=True)
    print(f"exog benchmark — {R['asset'].nunique()} assets w/ matched implied vol, "
          f"{len(R)} rows ({R['date'].min().date()}→{R['date'].max().date()})")
    print(f"exog features present: {[c for c in ['iv','ts_short','ts_long','vrp','hy_oas','nfci'] if c in R.columns]}\n")
    dates = np.array(sorted(R["date"].unique()))
    preds = []
    i = MIN_TRAIN_D
    while i < len(dates):
        td = set(dates[i:i + TEST_D]); cut = dates[i - EMB]
        tr = R[R["date"] < cut]; te = R[R["date"].isin(td)]
        if len(tr) < 1500 or len(te) < 60:
            i += TEST_D; continue
        o = te[["y", "rv_next"]].copy()
        for m, cs in LADDER.items():
            o[m] = np.exp(ols(tr, te, cs))
        preds.append(o); i += TEST_D
    P = pd.concat(preds, ignore_index=True); rv = P["rv_next"].to_numpy(); y = P["y"].to_numpy()
    loss = {m: qlike(rv, P[m].to_numpy()) for m in LADDER}
    qh = float(np.nanmean(loss["HAR"])); qm = loss["Meridian"]
    print(f"  {'model':>16} {'QLIKE':>8} {'vsHAR%':>8} {'IC':>6} {'DMvsHAR':>9} {'DMvsMeridian':>13}")
    for m in LADDER:
        q = float(np.nanmean(loss[m]))
        ic = float(spearmanr(np.sqrt(P[m].to_numpy()), np.sqrt(rv))[0])
        dh = "—" if m == "HAR" else f"{diebold_mariano(loss[m], loss['HAR'])[1]:.3f}"
        dm = "—" if m in ("HAR", "Meridian") else f"{diebold_mariano(loss[m], qm)[1]:.3f}"
        star = "  ★" if (m not in ("HAR", "Meridian") and diebold_mariano(loss[m], qm)[1] < 0.05 and q < np.nanmean(qm)) else ""
        print(f"  {m:>16} {q:>8.4f} {(1-q/qh)*100:>+7.2f}% {ic:>6.3f} {dh:>9} {dm:>13}{star}")
    print("\n  ★ = significantly beats the current Meridian OOS (the free-data lever genuinely helps).")


if __name__ == "__main__":
    main()
