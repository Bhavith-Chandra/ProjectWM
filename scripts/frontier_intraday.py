"""FRONTIER experiment — does the intraday realized-measure richness push Meridian's edge
BEYOND the daily-style model? Tested on the independent Oxford-Man indices (17), which ship
the full measure set: 5-min RV, realized KERNEL (noise-robust), BIPOWER variation (continuous
/ jump-free), subsampled RV, median RV, realized semivariance.

Model ladder (all LINEAR — established that neural nets don't help daily vol), forecasting
next-day log RV(5-min), purged walk-forward, QLIKE + Diebold-Mariano vs HAR + Model Confidence Set:
  HAR              : HAR cascade on 5-min RV (baseline)
  Meridian-lin     : + realized semivariance (leverage) + common market-RV factor  (current model)
  Meridian-RK      : HAR cascade on the noise-robust REALIZED KERNEL + semivar + market
  Meridian-CJ      : + continuous (bipower) / JUMP (RV-BV) decomposition (HAR-CJ) + semivar + market
  Meridian-intra+  : everything — HAR + semivar + jump/continuous + subsampled RV + kernel + medRV + market

If the richer measures beat Meridian-lin (DM-significant), the frontier genuinely moved.
If they tie, that's the honest ceiling of daily-frequency forecasting from intraday measures.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data_omi import load_omi
from meridian.evalproto import qlike, diebold_mariano, mz_r2
from scipy.stats import spearmanr

EPS = 1e-12
MIN_TRAIN_D, TEST_D, EMBARGO_D = 1260, 378, 22


def cascade(x):
    lx = np.log(np.clip(x, EPS, None))
    return lx, pd.Series(lx).rolling(5).mean().to_numpy(), pd.Series(lx).rolling(22).mean().to_numpy()


def build():
    d = load_omi(refresh=True)
    rows = []
    for a, df in d.items():
        rv = df["rv"].to_numpy()
        rk = np.where(np.isfinite(df["rk"].to_numpy()) & (df["rk"].to_numpy() > 0), df["rk"].to_numpy(), rv)
        bv = np.where(np.isfinite(df["bv"].to_numpy()) & (df["bv"].to_numpy() > 0), df["bv"].to_numpy(), rv)
        rvss = np.where(np.isfinite(df["rvss"].to_numpy()) & (df["rvss"].to_numpy() > 0), df["rvss"].to_numpy(), rv)
        medrv = np.where(np.isfinite(df["medrv"].to_numpy()) & (df["medrv"].to_numpy() > 0), df["medrv"].to_numpy(), rv)
        rsv = np.clip(df["rsv"].to_numpy(), EPS, None)
        good = np.clip(rv - rsv, EPS, None)
        jump = np.clip(rv - bv, 0.0, None)                    # jump variation
        hd, hw, hm = cascade(rv)
        rkd, rkw, rkm = cascade(rk)
        fwd = pd.Series(rv).rolling(1).mean().shift(-1).to_numpy()   # next-day RV
        dts = df.index
        for t in range(22, len(rv) - 1):
            y = np.log(fwd[t] + EPS)
            feat = dict(har_d=hd[t], har_w=hw[t], har_m=hm[t],
                        lev=np.log(rsv[t]), pos=np.log(good[t]),
                        rk_d=rkd[t], rk_w=rkw[t], rk_m=rkm[t],
                        cont_d=np.log(bv[t] + EPS), jump_d=np.log(jump[t] + EPS),
                        rvss_d=np.log(rvss[t]), medrv_d=np.log(medrv[t]))
            if not np.isfinite([y] + list(feat.values())).all():
                continue
            rows.append(dict(asset=a, date=dts[t], y=y, rv_next=fwd[t], **feat))
    R = pd.DataFrame(rows)
    R["mktrv"] = R.groupby("date")["har_d"].transform("mean")
    # regime-conditional: stress indicator (current vol above its monthly trend) + interactions
    hi = (R["har_d"] > R["har_m"]).astype(float)
    R["hi"] = hi
    for c in ["har_d", "har_w", "har_m", "lev", "mktrv"]:
        R[c + "_hi"] = R[c] * hi
    return R


LADDER = {
    "HAR":            ["har_d", "har_w", "har_m"],
    "Meridian-lin":   ["har_d", "har_w", "har_m", "lev", "pos", "mktrv"],
    "Meridian-RK":    ["rk_d", "rk_w", "rk_m", "lev", "pos", "mktrv"],
    "Meridian-CJ":    ["har_d", "har_w", "har_m", "cont_d", "jump_d", "lev", "mktrv"],
    "Meridian-intra+": ["har_d", "har_w", "har_m", "lev", "pos", "cont_d", "jump_d",
                        "rk_d", "rvss_d", "medrv_d", "mktrv"],
    "Meridian-regime": ["har_d", "har_w", "har_m", "lev", "pos", "mktrv",
                        "hi", "har_d_hi", "har_w_hi", "har_m_hi", "lev_hi", "mktrv_hi"],
}


def main():
    R = build().reset_index(drop=True)
    dates = np.array(sorted(R["date"].unique()))
    preds = []
    i = MIN_TRAIN_D
    while i < len(dates):
        test_dates = set(dates[i:i + TEST_D]); cutoff = dates[i - EMBARGO_D]
        tr = R[R["date"] < cutoff]; te = R[R["date"].isin(test_dates)]
        if len(tr) < 2000 or len(te) < 100:
            i += TEST_D; continue
        out = te[["asset", "date", "y", "rv_next"]].copy()
        for name, cols in LADDER.items():
            A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
            beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
            jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
            B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
            pm = B @ beta
            out[name] = pm; out[name + "__v"] = np.exp(pm + jb)
        preds.append(out); i += TEST_D
    P = pd.concat(preds, ignore_index=True)
    rv_true = P["rv_next"].to_numpy(); y = P["y"].to_numpy()
    loss = {m: qlike(rv_true, P[m + "__v"].to_numpy()) for m in LADDER}
    qh = float(np.nanmean(loss["HAR"]))
    print(f"FRONTIER — intraday realized measures on OMI ({P['asset'].nunique()} indices, {len(P)} OOS forecasts)\n")
    print(f"  {'model':>16} {'QLIKE':>8} {'vsHAR%':>8} {'RMSE':>7} {'MZ-R2':>7} {'IC':>6} {'DMvsHAR':>9} {'DMvsMer-lin':>12}")
    ql = loss["Meridian-lin"]
    for m in LADDER:
        q = float(np.nanmean(loss[m]))
        rmse = float(np.sqrt(np.nanmean((y - P[m].to_numpy()) ** 2)))
        mz = float(mz_r2(y, P[m].to_numpy()))
        ic = float(spearmanr(np.sqrt(P[m + "__v"].to_numpy()), np.sqrt(rv_true))[0])
        dh = "—" if m == "HAR" else f"{diebold_mariano(loss[m], loss['HAR'])[1]:.3f}"
        dl = "—" if m in ("HAR", "Meridian-lin") else f"{diebold_mariano(loss[m], ql)[1]:.3f}"
        print(f"  {m:>16} {q:>8.4f} {(1-q/qh)*100:>+7.2f}% {rmse:>7.4f} {mz:>7.3f} {ic:>6.3f} {dh:>9} {dl:>12}")
    # MCS
    from arch.bootstrap import MCS
    LM = pd.DataFrame({m: loss[m] for m in LADDER}).replace([np.inf, -np.inf], np.nan).dropna()
    mcs = MCS(LM, size=0.10, reps=1000, block_size=22, method="R", seed=0); mcs.compute()
    print(f"\n  MCS 90%: {sorted(mcs.included)}")
    print("  DMvsMer-lin<0.05 ⇒ the intraday-rich model significantly beats the current Meridian.")
    # save for the master comparison
    import json
    out = {"n_forecasts": int(len(P)), "n_indices": int(P["asset"].nunique()), "models": {}}
    for m in LADDER:
        q = float(np.nanmean(loss[m]))
        out["models"][m] = {"QLIKE": q, "RMSE_log": float(np.sqrt(np.nanmean((y - P[m].to_numpy()) ** 2))),
                            "MZ_R2": float(mz_r2(y, P[m].to_numpy())),
                            "IC": float(spearmanr(np.sqrt(P[m + "__v"].to_numpy()), np.sqrt(rv_true))[0]),
                            "R2_vs_HAR_pct": float((1 - q / qh) * 100),
                            "DM_vs_HAR_p": None if m == "HAR" else float(diebold_mariano(loss[m], loss["HAR"])[1]),
                            "in_mcs": m in mcs.included}
    (Path(__file__).resolve().parent.parent / "results" / "frontier_omi.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
