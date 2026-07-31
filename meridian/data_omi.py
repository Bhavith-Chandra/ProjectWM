"""Oxford-Man Realized Library parser — the INDEPENDENT gold-standard RV dataset.

Cross-source validation: RV here is computed by OMI (Heber, Lunde, Shephard & Sheppard) from
5-minute intraday returns with peer-reviewed cleaning — a completely different construction and
vendor than our Yahoo/Garman-Klass pipeline. Scoring every model on the SAME OMI-computed proxy
isolates model skill from data-pipeline differences (the biggest confound in RV benchmarking).

We use per index: Realized Variance (5-minute) [the proxy/target], Realized Semivariance
(5-minute) [good/bad vol], and Return. Implied vol is NOT available for most OMI indices, so the
OMI arm of Meridian-WM runs WITHOUT the VIX feature — a cleaner test of the realized-measure
architecture on truly independent data.

Snapshot: OxfordManRealizedVolatilityIndices(20160928).xlsx, 2000-01-03 → 2016-09, ~20 indices.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
OMI_XLSX = Path(__file__).resolve().parent.parent / "data" / "omi" / "omi_2016.xlsx"
CACHE = Path(__file__).resolve().parent.parent / "data" / "omi" / "omi_parsed.parquet"

# OMI index name -> clean symbol.  S&P 500 EXCLUDED (SPY is in the training universe).
INDEX_MAP = {
    "FTSE 100": "FTSE", "DAX": "DAX", "CAC 40": "CAC", "Nikkei 225": "N225",
    "Hang Seng": "HSI", "KOSPI Composite Index": "KOSPI", "Russell 2000": "RUT",
    "EURO STOXX 50": "STOXX50", "IPC Mexico": "IPC", "Bovespa": "BVSP",
    "S&P/TSX Composite": "TSX", "All Ordinaries": "AORD", "Swiss Market Index": "SMI",
    "IBEX 35": "IBEX", "FTSE MIB": "MIB", "AEX": "AEX", "S&P CNX Nifty": "NIFTY",
    "Shanghai Composite": "SSEC", "Straits Times": "STI", "BSE Sensex": "SENSEX",
}


def _match(name: str) -> str | None:
    n = name.replace(" (Live)", "").strip()
    for k, v in INDEX_MAP.items():
        if k.lower() in n.lower():
            return v
    return None


def load_omi(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """{symbol: DataFrame[rv, rsv, ret]} indexed by date, from the OMI 5-min measures."""
    if CACHE.exists() and not refresh:
        df = pd.read_parquet(CACHE)
        return {s: g.drop(columns="sym").set_index("date") for s, g in df.groupby("sym")}
    raw = pd.read_excel(OMI_XLSX, engine="openpyxl", header=[0, 1])
    dates = pd.to_datetime(raw.iloc[:, 0].astype(str), format="%Y%m%d", errors="coerce")
    out = {}
    top = raw.columns.get_level_values(0)
    for idx_name in pd.unique(top):
        sym = _match(str(idx_name))
        if sym is None:
            continue
        block = raw.loc[:, idx_name]
        def col(sub):
            hits = [c for c in block.columns if sub.lower() in str(c).lower()]
            return pd.to_numeric(block[hits[0]], errors="coerce") if hits else None
        rv = col("Realized Variance (5-minute)")
        rsv = col("Realized Semivariance (5-minute)")
        ret = col("Return")
        if rv is None or ret is None:
            continue
        d = pd.DataFrame({"date": dates, "rv": rv.values,
                          "rsv": (rsv.values if rsv is not None else np.nan), "ret": ret.values})
        d = d.dropna(subset=["date", "rv"]).set_index("date").sort_index()
        d = d[d["rv"] > 0]
        if len(d) > 800:
            out[sym] = d
    # cache
    alld = pd.concat([g.assign(sym=s).reset_index() for s, g in out.items()], ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    alld.to_parquet(CACHE)
    return out


if __name__ == "__main__":
    d = load_omi(refresh=True)
    print(f"OMI indices parsed: {len(d)}  (S&P 500 excluded — trained via SPY)\n")
    for s, df in sorted(d.items()):
        print(f"  {s:8} {len(df):5d} days  {df.index.min().date()} → {df.index.max().date()}  "
              f"rsv={'yes' if df['rsv'].notna().any() else 'no'}")
