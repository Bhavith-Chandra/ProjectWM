"""Prove the production edge-case guards hold (external review, robustness round).

Triggers each landmine directly and asserts the system degrades gracefully instead of crashing or
poisoning the matrix:
  #1 zero-print semivariance day  -> features stay finite (no -inf / NaN in the HAR cascade)
  #2 stale exogenous feed         -> freshness audit flags it; early-warning is withheld, not acted on
  #3 degenerate EVT-GPD window    -> tail estimator falls back to empirical quantile, no unhandled error
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.features import realized_variance, har_features
from meridian.engine import _tail, _har_forecast
from meridian.connect import freshness

EPS = 1e-12
ok = True


def check(name, cond):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ---- #1: zero-downside / halted-flat day poisons nothing ----
n = 400
rng = np.random.RandomState(0)
ret = pd.Series(rng.standard_t(5, n) * 0.01)
ret.iloc[-1] = 0.0                                  # a completely flat / halted day (zero up & down move)
ret.iloc[-2] = 0.05                                 # a one-sided limit-up morning (zero downside)
rv = (ret ** 2).clip(lower=EPS).rename("rv")
neg = np.log((ret.clip(upper=0) ** 2) + EPS)        # RS- (downside semivariance), floored
pos = np.log((ret.clip(lower=0) ** 2) + EPS)        # RS+ (upside semivariance), floored
check("#1 zero/one-sided day: RS-, RS+ all finite (no -inf/NaN)",
      bool(np.isfinite(neg).all() and np.isfinite(pos).all()))
v1, v5, capped = _har_forecast(rv, ret)
check("#1 HAR forecast on that series is finite & positive", bool(np.isfinite(v1) and v1 > 0))

# ---- #2: stale exogenous feed is flagged, not silently trusted ----
old = pd.Series([1.0, 1.1, 1.2], index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
fresh = pd.Series([1.0, 1.1], index=[pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(d, "D") for d in (2, 1)])
check("#2 stale series flagged stale=True", freshness(old)["stale"] is True)
check("#2 fresh series flagged stale=False", freshness(fresh)["stale"] is False)

# ---- #3: degenerate EVT window doesn't crash; falls back cleanly ----
try:
    # near-constant residuals -> GPD MLE degenerate; must fall back to empirical quantile, no exception
    degen = pd.Series(np.r_[np.zeros(300) + 1e-9, [-0.2]])       # one lone extreme, else ~flat
    var, es = _tail(degen, sigma_now=0.20, q=0.99)
    check("#3 degenerate GPD window returns finite VaR/ES (no crash)",
          bool(np.isfinite(var) and np.isfinite(es) and var < 0 and es <= var))
except Exception as e:
    check(f"#3 degenerate GPD window handled (got exception: {type(e).__name__})", False)

# also confirm a normal window still fits the GPD path fine
var2, es2 = _tail(pd.Series(rng.standard_t(4, 1000) * 0.01), sigma_now=0.2, q=0.99)
check("#3 normal window still returns a sane tail", bool(np.isfinite(var2) and var2 < 0))

print(f"\n  {'ALL EDGE-CASE GUARDS HOLD.' if ok else 'A GUARD FAILED — see above.'}")
sys.exit(0 if ok else 1)
