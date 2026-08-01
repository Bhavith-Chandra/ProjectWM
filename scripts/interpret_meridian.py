"""LAYER-BY-LAYER interpretation of the Meridian volatility model (on the independent OMI
indices). Because the winning model is LINEAR, it is interpretable from foundations — the
coefficients ARE the mechanism. We report:

  1. COEFFICIENTS   — the fitted weight on each feature (sign + magnitude), averaged over folds.
  2. ABLATION       — drop each feature, measure the OOS QLIKE change. A feature "earns its place"
                      only if removing it HURTS out-of-sample (causal-necessity test).
  3. PER-ASSET      — QLIKE ratio vs HAR per index: where Meridian wins and where it doesn't.
  4. PER-REGIME     — does the edge come from calm or stressed days (vol-percentile split)?
  5. CALIBRATION    — Mincer-Zarnowitz (a, b, R2): is the forecast unbiased?
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from scripts.frontier_intraday import build, MIN_TRAIN_D, TEST_D, EMBARGO_D
from meridian.evalproto import qlike, mz_r2

EPS = 1e-12
CHAMP = ["har_d", "har_w", "har_m", "lev", "pos", "cont_d", "jump_d", "rk_d", "rvss_d", "medrv_d", "mktrv"]
NICE = {"har_d": "RV daily", "har_w": "RV weekly", "har_m": "RV monthly", "lev": "bad-vol (semivar-)",
        "pos": "good-vol (semivar+)", "cont_d": "continuous (bipower)", "jump_d": "jump",
        "rk_d": "realized kernel", "rvss_d": "subsampled RV", "medrv_d": "median RV",
        "mktrv": "market-RV factor"}


def wf_predict(R, dates, cols):
    """walk-forward OLS forecasts for a feature set → aligned (loss-ready) frame."""
    preds = []
    i = MIN_TRAIN_D
    coefs = []
    while i < len(dates):
        tds = set(dates[i:i + TEST_D]); cut = dates[i - EMBARGO_D]
        tr = R[R["date"] < cut]; te = R[R["date"].isin(tds)]
        if len(tr) < 2000 or len(te) < 100:
            i += TEST_D; continue
        A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
        beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
        jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
        B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
        o = te[["asset", "date", "y", "rv_next", "har_d"]].copy()
        o["v"] = np.exp(B @ beta + jb)
        preds.append(o); coefs.append(beta); i += TEST_D
    P = pd.concat(preds, ignore_index=True)
    return P, np.mean(coefs, 0)


def main():
    R = build().reset_index(drop=True)
    dates = np.array(sorted(R["date"].unique()))
    Pc, coef = wf_predict(R, dates, CHAMP)
    Ph, _ = wf_predict(R, dates, ["har_d", "har_w", "har_m"])
    rv = Pc["rv_next"].to_numpy()
    qc = qlike(rv, Pc["v"].to_numpy()); qh = qlike(rv, Ph["v"].to_numpy())
    base = float(np.nanmean(qc))
    print(f"Meridian interpretation — OMI ({Pc['asset'].nunique()} indices, {len(Pc)} OOS forecasts)")
    print(f"champion OOS QLIKE {base:.4f}  vs HAR {np.nanmean(qh):.4f}  ({(1-base/np.nanmean(qh))*100:+.2f}%)\n")

    print("[1] COEFFICIENTS (avg over folds, on standardized-scale log features):")
    for c, b in sorted(zip(CHAMP, coef[1:]), key=lambda kv: -abs(kv[1])):
        print(f"    {NICE[c]:>22}  {b:+.3f}")

    print("\n[2] ABLATION — drop one feature, ΔQLIKE vs champion (positive = feature HELPS):")
    rows = []
    for c in CHAMP:
        sub = [x for x in CHAMP if x != c]
        Ps, _ = wf_predict(R, dates, sub)
        qa = float(np.nanmean(qlike(Ps["rv_next"].to_numpy(), Ps["v"].to_numpy())))
        rows.append((c, (qa - base) / base * 100))
    for c, d in sorted(rows, key=lambda kv: -kv[1]):
        flag = "  ← earns its place" if d > 0.05 else ("  (noise)" if d < -0.02 else "")
        print(f"    {NICE[c]:>22}  {d:+.2f}%{flag}")

    print("\n[3] PER-ASSET — QLIKE ratio Meridian/HAR (<1 = Meridian wins):")
    for a, idx in Pc.groupby("asset").groups.items():
        m = Pc.index.isin(idx)
        r = float(np.nanmean(qc[m]) / np.nanmean(qh[m]))
        bar = "✓" if r < 0.99 else ("≈" if r < 1.01 else "✗")
        print(f"    {a:>8} {r:6.3f} {bar}", end="   ")
    print()
    wins = sum(1 for a, idx in Pc.groupby("asset").groups.items()
               if np.nanmean(qc[Pc.index.isin(idx)]) / np.nanmean(qh[Pc.index.isin(idx)]) < 0.99)
    print(f"    → Meridian beats HAR on {wins}/{Pc['asset'].nunique()} indices")

    print("\n[4] PER-REGIME — edge in calm vs stressed days (by same-day RV percentile):")
    pct = pd.Series(Pc["har_d"].to_numpy()).rank(pct=True).to_numpy()
    for lo, hi, name in [(0.0, 0.5, "calm (low RV)"), (0.5, 0.85, "normal"), (0.85, 1.01, "stress (high RV)")]:
        m = (pct >= lo) & (pct < hi)
        edge = (1 - np.nanmean(qc[m]) / np.nanmean(qh[m])) * 100
        print(f"    {name:>18}: QLIKE edge vs HAR {edge:+.2f}%   (n={m.sum()})")

    print("\n[5] CALIBRATION (Mincer-Zarnowitz, log-space):")
    y = Pc["y"].to_numpy(); f = np.log(Pc["v"].to_numpy())
    X = np.column_stack([np.ones(len(f)), f]); ab, *_ = np.linalg.lstsq(X, y, rcond=None)
    print(f"    a={ab[0]:+.3f} (0=unbiased)  b={ab[1]:.3f} (1=correct scale)  MZ-R2={mz_r2(y, f):.3f}")


if __name__ == "__main__":
    main()
