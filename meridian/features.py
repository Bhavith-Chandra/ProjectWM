"""Feature engineering: realized volatility estimators, HAR features, targets.

All features at row t use only information available at the close of day t.
The target is next-day realized variance (t+1), so nothing here peeks ahead
except the explicit target column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
EPS = 1e-12


def realized_variance(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Daily realized-variance proxies from OHLC (no intraday needed).

    Returns a frame with:
      ret       : close-to-close log return
      rv_cc     : squared log return (noisy 1-obs estimator)
      rv_park   : Parkinson (high-low range) variance estimator
      rv_gk     : Garman-Klass estimator (uses OHLC)
      rv        : primary target basis = Garman-Klass, floored
    """
    o, h, l, c = (np.log(ohlc[x]) for x in ("open", "high", "low", "close"))
    ret = c.diff()

    park = (h - l) ** 2 / (4.0 * np.log(2.0))
    gk = 0.5 * (h - l) ** 2 - (2.0 * np.log(2.0) - 1.0) * (c - o) ** 2

    out = pd.DataFrame(index=ohlc.index)
    out["ret"] = ret
    out["rv_cc"] = ret ** 2
    out["rv_park"] = park.clip(lower=EPS)
    out["rv_gk"] = gk.clip(lower=EPS)
    out["rv"] = out["rv_gk"].clip(lower=EPS)
    return out


def har_features(rv: pd.Series) -> pd.DataFrame:
    """Corsi (2009) HAR components on realized variance.

    rv_d = RV_t, rv_w = mean RV over last 5d, rv_m = mean RV over last 22d.
    All are known at close of day t.
    """
    f = pd.DataFrame(index=rv.index)
    f["rv_d"] = rv
    f["rv_w"] = rv.rolling(5).mean()
    f["rv_m"] = rv.rolling(22).mean()
    return f


def build_asset_frame(ohlc: pd.DataFrame, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full per-asset modeling frame with features (t) and target (t+1).

    Columns:
      log-space HAR features, realized measures, optional macro context,
      and `y = log RV_{t+1}` as the prediction target.
    """
    rvf = realized_variance(ohlc)
    rv = rvf["rv"]

    df = pd.DataFrame(index=ohlc.index)
    df["ret"] = rvf["ret"]
    df["rv"] = rv
    df["log_rv"] = np.log(rv + EPS)

    har = har_features(rv)
    df["har_d"] = np.log(har["rv_d"] + EPS)
    df["har_w"] = np.log(har["rv_w"] + EPS)
    df["har_m"] = np.log(har["rv_m"] + EPS)

    # extra descriptive context features (available at t)
    df["ret_abs"] = rvf["ret"].abs()
    df["ret_5"] = rvf["ret"].rolling(5).sum()
    df["rv_cc"] = np.log(rvf["rv_cc"].clip(lower=EPS) + EPS)

    if macro is not None:
        m = macro.reindex(df.index).ffill()
        df["vix"] = np.log(m["VIXCLS"].clip(lower=EPS))
        df["term"] = (m["DGS10"] - m["DGS2"])  # yield-curve slope
        df["vix_chg"] = df["vix"].diff()

    # TARGET: next-day log realized variance
    df["y"] = df["log_rv"].shift(-1)
    # secondary target: next-day close-to-close return (for the distributional/VaR head)
    df["r_next"] = rvf["ret"].shift(-1)

    return df


if __name__ == "__main__":
    from meridian.data import load_all

    d = load_all()
    spy = build_asset_frame(d["prices"]["SPY"], d["macro"])
    print(spy.tail(6).round(4).to_string())
    print("\nshape:", spy.shape, "| NaNs per col:")
    print(spy.isna().sum().to_string())
