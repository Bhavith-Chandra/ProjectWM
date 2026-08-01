"""EXPERIMENT — can we push past regime-Meridian? Two evidence-backed levers on OMI:
  1. FORECAST COMBINATION (Bates-Granger / Timmermann): a simple average of good models often
     beats any single one ("forecast combination puzzle"). Combine regime / intra+ / CJ / lin.
  2. FINER 3-STATE REGIME (calm / normal / stress) instead of 2-state.
Champion to beat = Meridian-regime. Keep a variant ONLY if it beats it DM-significantly OOS.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from scripts.frontier_intraday import build, LADDER, MIN_TRAIN_D, TEST_D, EMBARGO_D
from meridian.evalproto import qlike, diebold_mariano
from scipy.stats import spearmanr

EPS = 1e-12
COMPONENTS = ["Meridian-regime", "Meridian-intra+", "Meridian-CJ", "Meridian-lin"]


def ols_var(tr, te, cols):
    A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
    jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return np.exp(B @ beta + jb)


def main():
    R = build().reset_index(drop=True)
    # 3-state regime features (thresholds fit per fold on train; here add interaction columns)
    dates = np.array(sorted(R["date"].unique()))
    preds = []
    i = MIN_TRAIN_D
    while i < len(dates):
        td = set(dates[i:i + TEST_D]); cut = dates[i - EMBARGO_D]
        tr = R[R["date"] < cut]; te = R[R["date"].isin(td)]
        if len(tr) < 2000 or len(te) < 100:
            i += TEST_D; continue
        o = te[["asset", "date", "y", "rv_next"]].copy()
        # component models + HAR
        o["HAR"] = ols_var(tr, te, LADDER["HAR"])
        for m in COMPONENTS:
            o[m] = ols_var(tr, te, LADDER[m])
        # 1. combinations (simple average in variance space)
        o["Combo-all"] = np.mean([o[m].to_numpy() for m in COMPONENTS], 0)
        o["Combo-reg+intra"] = 0.5 * (o["Meridian-regime"].to_numpy() + o["Meridian-intra+"].to_numpy())
        # 2. 3-state regime: low/mid/high by train-quantiles of har_d
        q1, q2 = np.quantile(tr["har_d"], [0.4, 0.8])
        for frame, src in [(tr, tr), (te, te)]:
            pass
        def reg3_cols(fr):
            hd = fr["har_d"].to_numpy()
            r_mid = ((hd >= q1) & (hd < q2)).astype(float); r_hi = (hd >= q2).astype(float)
            base = ["har_d", "har_w", "har_m", "lev", "pos", "mktrv"]
            X = {c: fr[c].to_numpy() for c in base}
            for c in ["har_d", "har_w", "lev", "mktrv"]:
                X[c + "_mid"] = fr[c].to_numpy() * r_mid; X[c + "_hi"] = fr[c].to_numpy() * r_hi
            X["r_mid"] = r_mid; X["r_hi"] = r_hi
            return pd.DataFrame(X)
        trX, teX = reg3_cols(tr), reg3_cols(te)
        cols3 = list(trX.columns)
        A = np.column_stack([np.ones(len(trX))] + [trX[c].to_numpy() for c in cols3])
        beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
        jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
        B = np.column_stack([np.ones(len(teX))] + [teX[c].to_numpy() for c in cols3])
        o["Meridian-regime3"] = np.exp(B @ beta + jb)
        preds.append(o); i += TEST_D
    P = pd.concat(preds, ignore_index=True)
    rv = P["rv_next"].to_numpy()
    champ = "Meridian-regime"
    models = ["HAR", "Meridian-lin", "Meridian-regime", "Meridian-regime3",
              "Combo-reg+intra", "Combo-all"]
    loss = {m: qlike(rv, P[m].to_numpy()) for m in models}
    qh = float(np.nanmean(loss["HAR"])); qc = loss[champ]
    print(f"BOOST experiment — OMI ({P['asset'].nunique()} indices, {len(P)} OOS forecasts)")
    print(f"champion = {champ}\n")
    print(f"  {'model':>18} {'QLIKE':>8} {'vsHAR%':>8} {'IC':>6} {'DMvsHAR':>9} {'DMvsRegime':>11}")
    for m in models:
        q = float(np.nanmean(loss[m]))
        ic = float(spearmanr(np.sqrt(P[m].to_numpy()), np.sqrt(rv))[0])
        dh = "—" if m == "HAR" else f"{diebold_mariano(loss[m], loss['HAR'])[1]:.3f}"
        dc = "—" if m == champ else f"{diebold_mariano(loss[m], qc)[1]:.3f}"
        star = "  ← beats champ" if (m != champ and diebold_mariano(loss[m], qc)[1] < 0.05 and q < np.nanmean(qc)) else ""
        print(f"  {m:>18} {q:>8.4f} {(1-q/qh)*100:>+7.2f}% {ic:>6.3f} {dh:>9} {dc:>11}{star}")
    print("\n  A variant is kept ONLY if DMvsRegime<0.05 AND lower QLIKE. Else regime-Meridian stands.")


if __name__ == "__main__":
    main()
