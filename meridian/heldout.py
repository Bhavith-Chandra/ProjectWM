"""Held-out universe for GENUINE out-of-sample validation.

None of these tickers is in the training universe (SPY, QQQ, IWM, AAPL, MSFT, JPM, XOM, JNJ,
EURUSD, USDJPY, GBPUSD). They span new US sectors, index ETFs, commodity/bond ETFs, new FX
pairs, and — importantly — four INTERNATIONAL indices (different markets/time zones), so the
benchmark tests cross-asset AND cross-market generalization.

Data-source note (honest): the ideal is a *different provider* than the training source (Yahoo).
Stooq's free CSV endpoint is now gated behind a JavaScript proof-of-work bot-detection challenge,
which we do not bypass; keyed providers (Tiingo/AlphaVantage/Nasdaq Data Link) need credentials
we don't have. So held-out data is fetched from Yahoo on NEVER-TRAINED-ON assets (the primary
generalization test), and the Oxford-Man Realized Library (independent, intraday realized
measures) is used as the cross-source check where an accessible trusted mirror exists.
"""
from __future__ import annotations

from meridian.data import fetch_yahoo

HELDOUT = {
    # US single stocks — new names & sectors (training had only AAPL/MSFT/JPM/XOM/JNJ)
    "KO": "KO", "WMT": "WMT", "DIS": "DIS", "BA": "BA", "CAT": "CAT",
    "GS": "GS", "NKE": "NKE", "CVX": "CVX", "PFE": "PFE", "VZ": "VZ",
    # index / sector ETFs (training had SPY/QQQ/IWM)
    "DIA": "DIA", "EEM": "EEM", "EFA": "EFA",
    # commodity & bond ETFs
    "GLD": "GLD", "USO": "USO", "TLT": "TLT", "HYG": "HYG",
    # FX not in training
    "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X",
    # INTERNATIONAL indices — different markets
    "DAX": "^GDAXI", "FTSE": "^FTSE", "NIKKEI": "^N225", "HSI": "^HSI",
}

CLASS = {**{k: "us_stock" for k in ["KO", "WMT", "DIS", "BA", "CAT", "GS", "NKE", "CVX", "PFE", "VZ"]},
         **{k: "etf_index" for k in ["DIA", "EEM", "EFA"]},
         **{k: "etf_cmdty_bond" for k in ["GLD", "USO", "TLT", "HYG"]},
         **{k: "fx" for k in ["AUDUSD", "USDCAD", "USDCHF"]},
         **{k: "intl_index" for k in ["DAX", "FTSE", "NIKKEI", "HSI"]}}


def load_heldout(min_rows: int = 800) -> dict:
    """{name: OHLC df} for held-out tickers with enough history."""
    out = {}
    for name, sym in HELDOUT.items():
        try:
            df = fetch_yahoo(sym)
            if len(df) >= min_rows:
                out[name] = df
        except Exception:
            pass
    return out


if __name__ == "__main__":
    d = load_heldout()
    print(f"held-out universe: {len(d)}/{len(HELDOUT)} loaded")
    for n, df in d.items():
        print(f"  {n:8} {CLASS[n]:16} {len(df):5d} rows  {df.index.min().date()} → {df.index.max().date()}")
