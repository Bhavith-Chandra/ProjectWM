"""A/B: does the EB-JEPA LEARNED energy beat the fixed latent-MSE energy as a surprise/regime signal?

Loads the two walk-forward outputs (identical config except energy_mode) and compares the energy purely
as a SURPRISE detector — the metric that matters, since neither beats HAR on point forecasting:

  * vol lift        : realized vol on top-decile-energy days ÷ the rest (higher = sharper surprise)
  * Spearman(E, vol): rank-correlation of energy with realized log-RV
  * Spearman(E, err): does energy flag the model's own forecast misses
  * onset lead      : of realized-vol stress onsets, fraction preceded by an energy spike within 5 days,
                      and the median lead (trading days) — the leading-indicator value

Honest verdict: EB-JEPA wins only if it raises the vol lift AND/OR the onset lead over the L2 baseline.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent.parent / "results"


def onset_lead(df):
    """Per-asset: RV-stress onset = 5-day mean log-RV crossing its trailing-1y 85th pct. Lead = how many
    days earlier the energy (top-decile per asset) fired before the onset."""
    leads, hit, tot = [], 0, 0
    for a, g in df.groupby("asset"):
        g = g.sort_values("date")
        lrv = g["y_true_log"].rolling(5).mean()
        thr = lrv.rolling(252, min_periods=60).quantile(0.85)
        stress = (lrv > thr).astype(int)
        onsets = g.index[stress.diff() == 1]
        ethr = g["energy"].quantile(0.90)
        fired = g["energy"] >= ethr
        for o in onsets:
            tot += 1
            window = g.loc[:o].tail(6)
            f = window.index[fired.loc[window.index]]
            if len(f):
                hit += 1
                leads.append((g.loc[o, "date"] - g.loc[f[0], "date"]).days)
    return (hit / max(tot, 1) * 100, float(np.median(leads)) if leads else float("nan"))


def metrics(path):
    P = pd.read_parquet(path).dropna(subset=["y_true_log", "energy"]).reset_index(drop=True)
    rv = np.exp(P["y_true_log"].to_numpy()); e = P["energy"].to_numpy()
    err = np.abs(P["y_true_log"].to_numpy() - P["y_pred_log"].to_numpy())
    hi = e >= np.quantile(e, 0.9)
    lift = np.mean(rv[hi]) / np.mean(rv[~hi])
    sens, lead = onset_lead(P)
    return {"n": len(P), "lift": lift, "rho_vol": spearmanr(e, P["y_true_log"])[0],
            "rho_err": spearmanr(e, err)[0], "onset_sens": sens, "onset_lead": lead}


def main():
    pairs = [("L2 (latent-MSE)", "meridian_pred_l2.parquet"),
             ("EB-JEPA (learned)", "meridian_pred_ebjepa.parquet")]
    rows = []
    for name, f in pairs:
        p = RES / f
        if not p.exists():
            print(f"  (missing {f} — run its training first)"); continue
        m = metrics(p); m["name"] = name; rows.append(m)
    if not rows:
        return
    print("Energy-as-surprise A/B — EB-JEPA learned energy vs fixed latent-MSE\n")
    print(f"  {'energy':>18} {'vol lift':>9} {'ρ(E,vol)':>9} {'ρ(E,err)':>9} {'onset%':>8} {'lead(d)':>8}")
    for m in rows:
        print(f"  {m['name']:>18} {m['lift']:>8.2f}x {m['rho_vol']:>+9.3f} {m['rho_err']:>+9.3f} "
              f"{m['onset_sens']:>7.0f}% {m['onset_lead']:>8.1f}")
    if len(rows) == 2:
        l2, eb = rows[0], rows[1]
        better = (eb["lift"] > l2["lift"] * 1.02) or (eb["onset_sens"] > l2["onset_sens"] + 3)
        print(f"\n  VERDICT: EB-JEPA learned energy "
              f"{'BEATS' if better else 'does NOT beat'} the latent-MSE energy as a surprise/regime "
              f"signal → {'adopt the learned energy' if better else 'keep the simpler L2 energy (report honestly)'}.")


if __name__ == "__main__":
    main()
