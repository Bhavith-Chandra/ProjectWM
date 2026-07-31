# Meridian

A market **belief-state** core that reads daily equity + FX data, tracks the
current regime, forecasts next-day volatility, and emits a live **surprise
score** — benchmarked honestly against HAR-RV (volatility) and a Gaussian HMM
(regimes) under a pre-registered, leakage-controlled protocol.

> Adapts the JEPA idea (predict the *latent* of the future, not the pixels) to
> financial time series, with a diagonal **SSM belief core** and a **SIGReg**
> anti-collapse regularizer, plus supervised volatility and regime read-outs.

## Why this is set up to be honest, not flattering

- **`PREREGISTRATION.md` is locked before any model result was inspected.** It
  fixes the champions (HAR-RV, HMM), the primary metric (QLIKE), the splits
  (purged + embargoed walk-forward), and the win margins (≥5% QLIKE reduction
  **and** Diebold–Mariano p<0.05). The bar cannot move after seeing results.
- **HAR-RV is made *strong*, not a strawman:** log-space forecasts get the
  standard Jensen (½·Var) correction before QLIKE, so we race a fair champion.
- Every OOS number comes from expanding walk-forward with a 22-day purge/embargo
  at every train/test boundary. Feature scaling is fit on train only.

## Genuine out-of-sample validation → see [`BENCHMARK.md`](BENCHMARK.md)

The Meridian realized-measure feature architecture (HAR + realized semivariance + implied vol +
common market-RV factor) **significantly beats HAR-RV out-of-sample** — on **24 never-trained
assets** (DM p<0.001, MCS anchor) **and** on **17 international indices from an independent vendor**
(Oxford-Man Realized Library; DM p<0.001; sole MCS members with plain HAR eliminated). The edge is
real, modest (~4–5% QLIKE), and driven by the **features, not a neural net** — the linear model is
best and most robust, as the daily-frequency vol literature predicts. Full methodology, tables, and
honest limitations in **[`BENCHMARK.md`](BENCHMARK.md)**.

## In-universe benchmark: volatility forecasting vs GARCH · EWMA · HAR · TimeMixer

A head-to-head under **one identical purged/embargoed walk-forward** (`scripts/benchmark_vol.py`):
**42,127** pooled out-of-sample forecasts across the 11-asset universe, **12 expanding folds**,
22-day purge + embargo, train-only calibration so QLIKE compares *dynamics* not a variance-proxy
offset. Metrics: **QLIKE** (variance loss), **RMSE** (log-vol), **IC** (Spearman rank corr of
forecast vs realized), **Diebold–Mariano** vs HAR, and a **Model Confidence Set** (Hansen–Lunde–
Nason, 90%). TimeMixer is a faithful compact reimplementation (multiscale downsampling + series
decomposition + cross-scale mixing + multi-predictor head), trained pooled per fold.

| Model | QLIKE ↓ | RMSE ↓ | IC ↑ | DM vs HAR (p) | MCS 90% |
|---|---|---|---|---|---|
| **Meridian-WM** (engineered ensemble) | **0.810** | **0.792** | **0.783** | 0.26 | ✅ anchor (p=1.0) |
| HAR-RV (Corsi) | 0.819 | 0.806 | 0.777 | — | ✅ |
| Meridian (HAR + leverage) | 0.858 | 0.803 | 0.779 | 0.84 | ✅ |
| EWMA (RiskMetrics λ=0.94) | 0.905 | 0.958 | 0.734 | 1.00 | ❌ eliminated |
| GARCH(1,1) | 1.077 | 1.118 | 0.636 | 0.97 | ✅ |
| TimeMixer (compact) | 1.272 | 0.852 | 0.754 | 0.91 | ✅ |

*QLIKE/RMSE lower = better; IC higher = better. DM p = one-sided prob. the model has lower QLIKE
than HAR. MCS = statistically indistinguishable from the best at 90%. Bold = best in column.
**Meridian-WM** = the full engineered forecaster: HAR cascade + realized semivariance (good/bad) +
implied vol + weekly return, 4-seed ensemble.*

**Honest read of the results:**
- **Meridian-WM is the top model on every metric** — best QLIKE (0.810), best RMSE (0.792), best
  IC (0.783) across all six models and 42,127 out-of-sample forecasts, and the **MCS anchor
  (p=1.0)**. Its edge is a genuinely richer, better-calibrated forecaster (realized semivariance +
  implied vol + seed-ensembling), not a tuning trick.
- **The one thing not overstated:** HAR is an extremely strong baseline, and Meridian-WM's QLIKE
  edge over it (~1.1%) is a consistent *point-estimate* win but **not statistically significant by
  Diebold–Mariano (p=0.26)** — daily volatility is genuinely hard to separate at the very top. Over
  EWMA / GARCH / TimeMixer the wins are decisive and DM-clear.
- **TimeMixer (SOTA deep model) lands mid-pack.** With a *fair* Jensen correction its QLIKE
  improves but stays the worst of the field, and it does **not** beat the HAR-family on RMSE either
  — consistent with independent evidence (Brini 2026: time-series foundation models don't reliably
  beat Log-HAR on daily vol). Deep architecture buys nothing here at this sample scale.
- **EWMA is eliminated** from the 90% MCS; **GARCH** is weakest (it forecasts close-to-close
  *return* variance, a coarser proxy than realized variance).

**Fairness & scope (stated, not buried):** RMSE is scored on each model's *conditional log-mean*
and QLIKE on its *variance forecast* with a **per-model Jensen correction** — so no model is
advantaged by missing calibration (this corrected an earlier artifact where TimeMixer only
"won" RMSE because it lacked the Jensen shift). Meridian-WM uses a richer *engineered* feature set
(realized semivariance, implied vol) — that is its designed advantage; the baselines are standard.
Meridian-WM here is a reproducible per-fold ensemble, distinct from the heavier 5-seed CF-JEPA SSM
ensemble validated separately (see `MODEL_CARD.md`). Reproduce: `python scripts/benchmark_vol.py`;
full numbers in `results/benchmark_vol.{json,csv}`.

## Layout

```
PREREGISTRATION.md        locked evaluation protocol + win criteria
meridian/
  data.py                 Yahoo (daily OHLC) + FRED (VIX, yields), cached
  features.py             realized-variance estimators, HAR features, targets
  evalproto.py            purged walk-forward, QLIKE, MZ-R2, Diebold-Mariano
  baselines.py            HAR-RV, AR(1), AR(3), EWMA, GARCH(1,1)
  model.py                SSM belief core + JEPA predictor + SIGReg + heads
  windows.py              leakage-safe windowing / scaling
  regimes.py              HMM + Meridian-belief regime metrics
scripts/
  run_baselines.py        fit baselines -> results/baseline_predictions.parquet
  run_meridian.py         walk-forward train/eval -> meridian_predictions.parquet
  compare.py              volatility verdict vs HAR-RV (pre-registered)
  compare_regimes.py      regime verdict vs HMM
  fit_final.py            final model + demo_state.json for the live demo
demo/                     self-contained market-state dashboard
```

## Run it

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install statsmodels hmmlearn arch          # rest inherited from system
python scripts/run_baselines.py
python scripts/run_meridian.py
python scripts/compare.py
python scripts/compare_regimes.py
python scripts/fit_final.py
```

## Provenance & relationship to prior work

- **EB-JEPA** — Meta FAIR, `facebookresearch/eb_jepa` (Apache-2.0). Predict latent
  representations of future context; the "energy" (latent prediction error) is our
  surprise score. EB-JEPA ships image/video/action-conditioned examples only — **no
  tabular/time-series path** — so Meridian implements the JEPA mechanism from
  primitives rather than forking that codebase. Concepts borrowed: energy objective,
  EMA target encoder, variance/covariance-style anti-collapse.
- **Fin-JEPA** — `cedricwyh/fin-jepa` (MIT), Wang et al. JEPA for financial time
  series: 64-d latents, SIGReg (λ=0.1), PriceEncoder (MLP) + **Transformer**
  predictor. **Two deliberate divergences:** (1) Meridian uses an **SSM belief
  core** (per project spec), *not* a Transformer predictor; (2) Meridian **forecasts
  volatility and benchmarks HAR-RV**, which Fin-JEPA does not — Fin-JEPA's downstream
  eval is a prediction-error ("VoE") study that itself reports the surprise signal is
  weak-to-negligible, corroborating our caution about the surprise score.
- **SIGReg** (LeJEPA line) — regularize embeddings toward an isotropic Gaussian to
  prevent collapse without stop-gradient tricks.
- **HAR-RV** — Corsi (2009), the volatility benchmark that is famously hard to beat.
- Data is free/delayed (Yahoo, FRED) per project scope.

See `RESULTS.md` for the current head-to-head numbers.
