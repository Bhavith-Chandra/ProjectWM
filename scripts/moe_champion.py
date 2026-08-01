"""HONEST check before wiring any module bridge: is the +2.94% MoE gain robust, or does it hide
asset-class heterogeneity? Split the HAR->HAR+neural bridge by IV-availability.

Finding: the neural expert HELPS where it is well-behaved (index ETFs) and POISONS the combination
where its QLIKE blows up (noisy single stocks / FX). So the pooled +2.94% (6-way equal weight) only
survives because the neural is diluted to 1/6 weight. A 2-way HAR+neural bridge is +6% on ETFs but
-26% on single stocks — NOT safe to wire blindly. Conclusion: keep the modular, asset-conditional
routing (HAR-lev+IV for index ETFs; HAR-lev for single stocks/FX). Bridging is real but must be gated.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
RES = Path(__file__).resolve().parent.parent / "results"
MATCHED = {"SPY", "QQQ", "IWM", "DIA", "GLD", "EEM"}     # assets with a matched free implied-vol index


def ql(f, y):
    r = np.exp(y - f)
    return float((r - (y - f) - 1).mean())


def main():
    b = pd.read_parquet(RES / "baseline_predictions.parquet"); b["f"] = b["y_pred_log"] + b["logvar_bias"]
    har = b[b["model"] == "HAR-RV"].set_index(["asset", "date"])["f"]
    yt = b.groupby(["asset", "date"])["y_true_log"].first()
    m = pd.read_parquet(RES / "meridian_predictions.parquet"); m["f"] = m["y_pred_log"] + m.get("logvar_bias", 0.0)
    nn = m.groupby(["asset", "date"])["f"].first()
    D = pd.concat([har.rename("har"), nn.rename("nn"), yt.rename("y")], axis=1).dropna()
    print(f"MoE bridge robustness — {len(D)} aligned rows; is the +2.94% pooled gain uniform?\n")
    print(f"  {'asset class':>26} {'n':>7} {'HAR QLIKE':>10} {'HAR+NN bridge':>14}")
    for mask, name in [(D.index.get_level_values("asset").isin(MATCHED), "index-ETF (IV available)"),
                       (~D.index.get_level_values("asset").isin(MATCHED), "single-stock/FX (no IV)")]:
        d = D[mask]; y = d["y"].to_numpy()
        qh = ql(d["har"].to_numpy(), y); qc = ql(0.5 * d["har"].to_numpy() + 0.5 * d["nn"].to_numpy(), y)
        print(f"  {name:>26} {len(d):>7} {qh:>10.4f} {(1-qc/qh)*100:>+13.2f}%")
    print("\n  CONCLUSION: the module bridge is REAL but ASSET-CONDITIONAL — the neural helps on smooth")
    print("  index ETFs and poisons noisy single stocks (its QLIKE blow-up). Do NOT wire a blanket bridge;")
    print("  keep the modular routing (IV champion for ETFs, HAR-lev for single stocks). Gain is single-digit,")
    print("  regime/asset-dependent, and fragile — nowhere near a blanket 3x. Honesty over a flattering number.")


if __name__ == "__main__":
    main()
