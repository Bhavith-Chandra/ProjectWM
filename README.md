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
