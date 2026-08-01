"""Compile every benchmark_vol*.json into ONE master results table (all metrics, all
universes, all horizons). Writes results/benchmark_master_table.md and prints it."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# (universe label, horizon) -> json filename
CELLS = [
    ("Training", 1, "benchmark_vol.json"),
    ("Held-out", 1, "benchmark_vol_heldout.json"),
    ("Held-out", 5, "benchmark_vol_heldout_h5.json"),
    ("Held-out", 22, "benchmark_vol_heldout_h22.json"),
    ("OMI-indep", 1, "benchmark_vol_omi.json"),
    ("OMI-indep", 5, "benchmark_vol_omi_h5.json"),
    ("OMI-indep", 22, "benchmark_vol_omi_h22.json"),
]
# display order of models (best-first-ish); only those present are shown
MORDER = ["Meridian-WM", "HAR-full", "HAR-IV", "Meridian", "HAR", "TimeMixer", "EWMA", "GARCH"]
NICE = {"Meridian-WM": "Meridian-net", "HAR-full": "Meridian-lin"}


def fmt(x, d=4, pct=False, sign=False):
    if x is None:
        return "—"
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return s + ("%" if pct else "")


def main():
    lines = ["# Meridian — Master Benchmark Table (all metrics · all universes · all horizons)\n",
             "*QLIKE/MSE/RMSE/MAE lower = better; MZ-R², R²-vs-HAR%, IC higher = better; bias→0 best. "
             "DM-p = one-sided prob. of lower QLIKE than HAR (bold <0.05 = significantly beats HAR). "
             "MCS = in the 90% Model Confidence Set. Meridian-lin=linear, Meridian-net=MLP ensemble.*\n",
             "| Universe | h | Model | QLIKE | MSE | RMSE | MAE | MZ-R² | R²vHAR% | IC | bias | DM-p | MCS |",
             "|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|"]
    n_cells = 0
    for uni, h, fn in CELLS:
        f = RESULTS / fn
        if not f.exists():
            continue
        n_cells += 1
        d = json.loads(f.read_text())
        mcs = set(d.get("mcs_90_included", [])); mp = d.get("mcs_pvalues", {})
        present = [m for m in MORDER if m in d["metrics"]]
        present.sort(key=lambda m: d["metrics"][m]["QLIKE"])          # best QLIKE first
        for m in present:
            v = d["metrics"][m]
            dmp = v.get("DM_vs_HAR_p")
            dm_s = "—" if dmp is None else (f"**{dmp:.3f}**" if dmp < 0.05 else f"{dmp:.3f}")
            nm = NICE.get(m, m)
            if m == present[0]:
                nm = "**" + nm + "**"                                  # bold the cell's best model
            lines.append(
                f"| {uni} | {h} | {nm} | {fmt(v['QLIKE'])} | {fmt(v.get('MSE_log'))} | "
                f"{fmt(v['RMSE_log'])} | {fmt(v.get('MAE_log'))} | {fmt(v.get('MZ_R2'),3)} | "
                f"{fmt(v.get('R2_vs_HAR_pct'),2,pct=True,sign=True)} | {fmt(v['IC'],3)} | "
                f"{fmt(v.get('bias'),3,sign=True)} | {dm_s} | {'✅' if m in mcs else '·'} |")
        na = len(d.get("assets", [])); nf = d.get("n_forecasts", "?")
        lines.append(f"| *{uni} h={h}* | | *{na} assets, {nf} OOS forecasts* | | | | | | | | | | |")
    out = RESULTS / "benchmark_master_table.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[{n_cells}/{len(CELLS)} cells available] → saved {out}")


if __name__ == "__main__":
    main()
