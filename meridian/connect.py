"""Pluggable data connectors — connect ANY data source to Meridian with one small adapter.

A source is just a callable that returns a tidy DataFrame/Series. Register it once and every
Meridian module can use it. Built-in adapters: Yahoo (global prices), FRED (macro), the free
implied-vol family, user CSV upload, and an honest news hook. Users add their own (a broker API,
a data vendor, a database) by writing a ~5-line adapter and calling `register(name, fn)`.

Honesty notes wired in:
  • CSV/vendor data you supply is trusted as-is — Meridian will say so and never invent values.
  • The news hook returns HEADLINES for context; it does NOT claim news predicts prices (the
    OOS evidence for news→return/vol prediction is weak — see BENCHMARK/RESEARCH). It surfaces
    relevant items and, at most, an uncertainty flag — never a fabricated market call.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from meridian.data import fetch_yahoo, fetch_fred, _get

# ---- registry ----
_SOURCES: dict[str, Callable] = {}


def register(name: str, fn: Callable):
    """Plug in a data source: fn(*args, **kwargs) -> DataFrame/Series. Overrides same-name."""
    _SOURCES[name] = fn
    return fn


def get(name: str):
    if name not in _SOURCES:
        raise KeyError(f"no data source '{name}'. registered: {sorted(_SOURCES)}")
    return _SOURCES[name]


def sources() -> list[str]:
    return sorted(_SOURCES)


# ---- built-in adapters ----
def yahoo_prices(symbol: str, start: str = "2007-01-01"):
    """Daily OHLC(V) for any global symbol (equities/ETF/FX/crypto/futures/indices)."""
    return fetch_yahoo(symbol, start=start)


def fred_series(series_id: str, start: str = "2000-01-01"):
    """Any FRED macro series (rates, credit spreads, conditions indices)."""
    return fetch_fred(series_id, start=start)


def csv_upload(path_or_buffer, date_col: str = "date", value_cols: list | None = None):
    """User-supplied CSV/dataframe. Trusted as-is; Meridian never alters or invents values."""
    df = pd.read_csv(path_or_buffer)
    dc = date_col if date_col in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc])
    df = df.set_index(dc).sort_index()
    if value_cols:
        df = df[value_cols]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(how="all")


def news_headlines(query: str, limit: int = 8):
    """HONEST news hook: fetches recent headlines for CONTEXT only (no fabricated market call).
    Uses a free RSS/JSON feed; returns [{title, source, published}]. Users can register a richer
    provider (Bloomberg/Reuters/NewsAPI) via register('news', their_fn). If unavailable, returns []."""
    import json
    import urllib.parse
    try:
        url = ("https://news.google.com/rss/search?q=" + urllib.parse.quote(query) +
               "&hl=en-US&gl=US&ceid=US:en")
        raw = _get(url, tries=2, timeout=15).decode("utf-8", "ignore")
        items = []
        for block in raw.split("<item>")[1:limit + 1]:
            def between(tag):
                a = block.find(f"<{tag}>"); b = block.find(f"</{tag}>")
                return block[a + len(tag) + 2:b].strip() if a >= 0 and b > a else ""
            title = between("title").replace("<![CDATA[", "").replace("]]>", "")
            items.append({"title": title, "source": between("source"), "published": between("pubDate")})
        return items
    except Exception:
        return []


# register built-ins
register("yahoo", yahoo_prices)
register("fred", fred_series)
register("csv", csv_upload)
register("news", news_headlines)


def freshness(series: pd.Series, max_age_days: int = 7) -> dict:
    """Provenance/freshness audit for any exogenous series (the review's data-alignment concern).

    Mixing daily equity data (NYSE calendar) with FRED macro (weekly/monthly, revised, different
    holidays) means a naive ffill can pair a stale macro value with today's tape. This returns the age
    of the latest real print and a `stale` flag so the caller can flag informational decay in the thesis
    rather than silently trusting a decayed value. Use per exogenous input, not on the price tape."""
    s = series.dropna()
    if not len(s):
        return {"available": False}
    last = s.index[-1]
    age = int((pd.Timestamp.utcnow().tz_localize(None) - pd.Timestamp(last)).days)
    return {"available": True, "as_of": str(pd.Timestamp(last).date()),
            "age_days": age, "stale": age > max_age_days}


@dataclass
class Connection:
    """A named handle over registered sources, so a user 'connects' once and reuses."""
    name: str = "default"
    def prices(self, symbol, **kw):
        return get("yahoo")(symbol, **kw)
    def macro(self, series_id, **kw):
        return get("fred")(series_id, **kw)
    def upload(self, path, **kw):
        return get("csv")(path, **kw)
    def news(self, query, **kw):
        return get("news")(query, **kw)


if __name__ == "__main__":
    print("registered sources:", sources())
    c = Connection()
    px = c.prices("AAPL")
    print(f"yahoo AAPL: {len(px)} rows, last {px['close'].iloc[-1]:.2f}")
    h = c.news("Apple stock")
    print(f"news 'Apple stock': {len(h)} headlines" + (f" — e.g. \"{h[0]['title'][:70]}…\"" if h else ""))
