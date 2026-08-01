# Meridian roadmap — what's validated worth building, and what the data says to skip

Every item below was **tested**, not asserted. The headline: the free-data *forecast* edge is at its
ceiling (~10-11%), so future value is in turning validated signals into action, hardening the tail, and
honest documentation of dead ends — not more forecast features.

## Tested this round

| Direction | Result | Verdict |
|---|---|---|
| **Act on IV early-warning** (`earlywarning_overlay.py`) | Cut-50%-on-inversion overlay: max drawdown −35.7%→**−26.2%**, vol 17.1→13.8, Sortino 0.93→**0.98**, for ~2.5pp/yr return. Cut-100% over-de-risks (Sortino 0.78). | ✅ **Validated** — a *moderate* de-risk on the signal improves risk-adjusted outcomes. Offer as an optional rule; do NOT full-exit. |
| **ES / regulatory backtest** (`es_backtest.py`) | VaR coverage PASSES (Kupiec, mean breach 0.94%) AND ES severity **PASSES** (Acerbi Z2≈0.02 after fixing a sign bug in the stat — the tail is well-sized). Only Christoffersen **independence** partially fails (breaches cluster). Dynamic EWMA σ helps (independence 1/8→3/8 assets) but doesn't fully fix — residual clustering is intrinsic to daily tail risk. | ✅ **Mostly regulatory-grade** — level + ES both pass; the live engine already uses dynamic σ (HAR-lev+IV), the better side of the clustering result. Clustering is a documented, partly-intrinsic limit. |
| **Horizon extension** (`horizon_iv_edge.py`) | +exog edge +10.3% (1d) → +10.9% (1w) → +4.6% (1m). | ◑ **Minor** — 1-week is a small first-class win; monthly decays. Not the doubling the literature hinted. |
| **Breadth: more IV markets** (probed) | Yahoo free vol-index coverage is poor (VSTOXX absent; VXFXI/VXSLV/VXTLT no history; EVZ discontinued 2025). Only `^MOVE` has history — and it does NOT help bond vol OOS (TLT −0.06% p=0.53, IEF +0.55% p=0.42, LQD −0.88% p=0.72). | ❌ **Dead end** — the matched-IV family can't be meaningfully extended for free. |
| **Credit / fixed-income spreads** (research + fetch) | Marginal, stress-only (Christiansen-Schmeling-Schrimpf: OOS R² 0.00–0.08 monthly; Paye: recession-onset-only). FRED currently unreachable. | ❌ **Not a forecast lever** — at most a stress-confirmation flag, unvalidated pending data. |

## Recommended next builds (ranked)

1. **Wire the validated de-risk overlay** as an optional, clearly-labeled rule (moderate haircut on
   term-structure inversion) — the one change with proven realized-outcome value.
2. **Fix the ES independence gap** — feed the dynamic vol forecast as σ into `_tail` so breaches stop
   clustering; re-run `es_backtest.py` to confirm Christoffersen passes.
3. **Promote the 1-week forecast** to a first-class output (peak edge, small but real).

## Explicitly NOT recommended (tested, low/no value)

- More matched-IV markets (no free data).
- Credit spreads as a forecast input (marginal; and unfetchable right now).
- Longer than 1-week horizons (edge decays).
- A crisis-correlation network (earlier round: vol, not correlation, is the driver).
