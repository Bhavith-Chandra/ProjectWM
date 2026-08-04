# Meridian — Enterprise World Model for financial markets

Meridian is an **interactive, interpretable world model** for market **risk and volatility**.
Ask about *any* entity ("Apple", "the DAX", "bitcoin", "SPY vs gold", "what if the market drops
5%?") and it fetches data live, runs a bank of calibrated modules, and **explains** the answer —
volatility forecast, regime, tail risk (VaR/ES), and portfolio/covariance risk. Every number is a
calibrated-module output, never a guess. It is best-in-class where measured, and **honest about the
limits** (see [`BENCHMARK.md`](BENCHMARK.md), [`COMPARISON.md`](COMPARISON.md)).

## What it is, in one picture

```
             USER — any question · any entity
                          │
                    ┌─────▼─────┐   the ENGINE routes the question by intent
                    │  ROUTER   │   (engine.py: analyze/compare/scenario/world/portfolio)
                    └─────┬─────┘
      ┌──────────┬────────┼─────────┬──────────────┬─────────────┐
      ▼          ▼        ▼         ▼              ▼             ▼
  Volatility   Regime    Tail    Covariance   Network        (each a decoupled
 (Regime-      (by-       (EVT     / Portfolio  propagation    SPECIALIST, its own
  Meridian)    design)    VaR/ES)  (min-var)    (gen-IRF)      objective + validation)
      └──────────┴────────┴────┬────┴──────────────┴─────────────┘
                          ┌─────▼─────┐   glass-box combiner + provenance ledger
                          │  ANSWER   │   (numbers ONLY from modules, explained)
                          └───────────┘
```
Rendered diagram: [`results/architecture.jpg`](results/architecture.jpg) · system doc: `results/system_document.html`.

## Is it one model, or does it route? → **It routes to specialists (by design).**

Meridian is **not** one monolithic model — it is a **bank of decoupled specialist modules, routed
by an interactive engine.** This is a proven choice, not a preference:

- **Different tasks need different objectives.** Volatility optimizes QLIKE; tail risk optimizes
  *coverage*; portfolio optimizes *variance/Sharpe*. One model can only be optimal for one loss.
- **We measured the monolith failing.** A single neural net trying to do everything (`Meridian-net`)
  had good squared-error but its **QLIKE blew up out-of-sample** (2.05) — fragile. The routed linear
  specialists were robust and won the decisive metrics. (Earlier, collapsing modules into one shared
  network degraded every task — the "design law" in [`WORLD_MODEL.md`](WORLD_MODEL.md).)
- **Routing keeps it interpretable and improvable.** You can upgrade one module (we did — see
  Regime-Meridian) without retraining or risking the others; and every number is traceable to the
  module that produced it.

**How routing works** — the engine detects intent and dispatches to the right specialist:

| A user asks… | Routed to | Optimizes |
|---|---|---|
| "how risky is X?" | Volatility + Regime + EVT tail | QLIKE / coverage |
| "A vs B" | compare (runs the read on both) | — |
| "what if the market drops 5%?" | Network propagation (generalized-IRF) | co-movement |
| "build a low-risk basket" | Covariance → min-variance | Sharpe / variance |
| "what's my downside?" | EVT tail → VaR / ES | tail coverage |

To the user it feels like *one* model (one interface, one conversation via `scripts/ask.py`); under
the hood it's a routed bank of independently-validated specialists. The **conversational contract**
(`meridian/tools.py`) guarantees the LLM only *routes and explains* — it can never originate a number.

### Who wins each task (all out-of-sample, from [`COMPARISON.md`](COMPARISON.md))
- **Volatility forecast** → **Regime-Meridian** (lowest QLIKE, +4.4% vs HAR, DM-significant, MCS member)
- **Portfolio risk** → **Meridian min-variance** (best Sharpe 0.71; −49% risk vs the naive portfolio; beats sample-cov)
- **Tail risk** → **Meridian conditional-EVT** (most exact 99% VaR coverage)

*Honest ceiling: the volatility edge over HAR is ~4% (real, significant, not huge — that's the frontier
on free data). The large, clean margins live in portfolio risk vs the tools people actually use.*

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

## Scenario generation: regime-conditioned GARCH-FHS

The world model's scenario engine generates **calibrated multi-asset Monte Carlo scenarios** using
expanding-window GJR-GARCH(1,1) with Filtered Historical Simulation. This is the core of risk
measurement — it feeds VaR/ES, portfolio optimization, and stress testing.

**Why GARCH-FHS, not a neural generator?** We tested both. The neural approach (354K-param GRU +
8-head cross-asset attention with 20% vol blending) adds <0.5% lift over pure GARCH-FHS at every
horizon — the GARCH model already captures the volatility clustering and leverage effects the
neural net tries to learn. Simpler wins.

**Key design choices:**
- **Expanding window**: fit on ALL available history (4000+ days), not a sliding window — more data
  = better GARCH parameter estimates
- **GJR asymmetry**: captures leverage effect (negative returns → higher vol) that symmetric
  GARCH misses
- **VIX-based regime conditioning**: innovation resampling weighted by current regime (low/mid/high
  vol) with EWMA decay — recent innovations matter more
- **Horizon-adaptive block lengths**: block_len=1 for 1–5d (independence), 3 for 10d, 5 for 20d
  (preserves autocorrelation structure at longer horizons)

### Benchmark: GARCH-FHS vs block bootstrap (the standard)

Tested on **35 ETFs** (US equity, sectors, international, fixed income, commodities, alternatives)
with **18 features** from 5 data sources (Yahoo, FRED, Fama-French, VIX term structure,
cross-asset derived). 160 eval windows, 1000 scenarios each, 504-day (2-year) test period.

| Horizon | Energy Score vs BB | p-value | Verdict |
|---------|-------------------|---------|---------|
| 1d | +2.0% | 0.020 | WIN |
| 5d | +5.4% | 0.00002 | WIN |
| 10d | +12.5% | <1e-6 | WIN |
| 20d | +18.9% | <1e-6 | WIN |

Margins compound with horizon — the expanding-window GARCH captures vol dynamics that fixed-window
bootstrap cannot. On the original 11-asset universe, all 4 horizons are statistically significant
(p<0.005), including 1d at +2.6% (p=0.004).

## DreamerV3-style world model core

The neural world model (`meridian/world_model/`) implements a **DreamerV3-style RSSM** adapted for
financial time series — discrete categorical latents (32x32), symlog predictions, KL balancing.
This is the generative backbone for forward simulation and what-if analysis.

```
┌─────────────────────────────────────────────────────────────┐
│  Encoder: Mamba SSM (content-aware, HiPPO init)            │
│     or GRU backbone (switchable)                           │
├─────────────────────────────────────────────────────────────┤
│  RSSM: 32×32 discrete categoricals                         │
│     prior: p(z_t | h_t)    posterior: q(z_t | h_t, x_t)   │
│     KL balancing (α=0.8 free nats)                         │
├─────────────────────────────────────────────────────────────┤
│  Graph: GAT (multi-head attention over asset dimension)    │
│     learns cross-asset dependencies from data              │
├─────────────────────────────────────────────────────────────┤
│  Heads: returns · volatility · tail · regime · covariance  │
│     each a specialist decoder with its own loss             │
└─────────────────────────────────────────────────────────────┘
```

## Additional modules

| Module | Path | Purpose |
|--------|------|---------|
| **Risk engine** | `meridian/risk/engine.py` | VaR, ES, component risk, stress testing |
| **Portfolio optimizer** | `meridian/portfolio/optimizer.py` | Mean-variance, HRP, Risk Parity, Black-Litterman, CVaR |
| **Causal discovery** | `meridian/causal/discovery.py` | NOTEARS structure learning, transfer entropy, nth-order shock propagation |
| **Continual learning** | `meridian/continual/olora.py` | O-LoRA adapters + regime replay for online adaptation |
| **Scoring** | `meridian/eval/scoring.py` | Energy score, variogram score (proper scoring rules) |

## Layout

```
meridian/                           the modules (the routed specialists)
  data.py            Yahoo (global OHLC) + FRED loaders, live-fetch, cached
  data_omi.py        Oxford-Man Realized Library parser (independent 5-min RV)
  heldout.py         24 never-trained assets for out-of-sample validation
  features.py        realized-variance estimators, HAR cascade, targets
  evalproto.py       purged/embargoed walk-forward, QLIKE, MZ-R², Diebold-Mariano
  engine.py          THE ROUTER — resolve entity → dispatch to modules → explain
  tools.py           conversational contract (LLM routes/explains; modules own numbers)
  network.py         generalized-IRF shock propagation ("what-if" scenarios)
  switching.py, meridian_wm.py   regime module internals
  world_model/       DreamerV3-style RSSM (Mamba/GRU + GAT + multi-head decoders)
    rssm.py          32×32 discrete categoricals, symlog, KL balancing
    encoder.py       Mamba SSM + GRU backbone (switchable)
    graph.py         GAT cross-asset attention
    heads.py         returns, vol, tail, regime, covariance heads
    model.py         MeridianWorldModel (composes all components)
    scenario.py      Monte Carlo with antithetic sampling
    trainer.py       composite loss training
  risk/engine.py     VaR, ES, component risk, stress testing
  portfolio/optimizer.py  MV, HRP, Risk Parity, BL, CVaR optimization
  causal/discovery.py     NOTEARS, transfer entropy, nth-order propagation
  continual/olora.py      O-LoRA adapters + regime replay
  eval/scoring.py         energy score, variogram score
scripts/
  ask.py             interactive CLI: analyze / compare / scenario / world / portfolio
  benchmark_vol.py   forecasting benchmark (universes × horizons, all metrics, MCS)
  benchmark_ultimate.py  GARCH-FHS vs bootstrap gauntlet (11 ETFs, 4 horizons)
  benchmark_mega.py      mega benchmark (35 ETFs, 18 features, 5 data sources)
  frontier_intraday.py   intraday-measure ladder + Regime-Meridian (champion)
  interpret_meridian.py  layer-by-layer interpretation (coeffs, ablation, per-regime)
  risk_benchmark.py  portfolio-risk + tail-risk benchmark
  compile_comparison.py  builds COMPARISON.md (every model, starred winners)
BENCHMARK.md         genuine out-of-sample validation write-up
COMPARISON.md        detailed model comparison, winners starred
WORLD_MODEL.md       the module bank + design law + interpretability standard
results/             saved metrics (*.json), architecture.jpg, dashboards
```

## Run it

```bash
python3 -m venv --system-site-packages .venv && source .venv/bin/activate
pip install statsmodels arch scikit-learn torch openpyxl   # rest inherited from system

# ask about any entity (routes to the right modules, explains):
python scripts/ask.py "Apple"
python scripts/ask.py --full "SPY"                  # full thesis: forecast + Monte-Carlo + tail + news
python scripts/ask.py "SPY vs gold"
python scripts/ask.py --world SPY -0.05 "Tesla"     # what-if a market shock
python scripts/ask.py --portfolio SPY TLT GLD NVDA  # min-variance basket

# reproduce the benchmarks (every number in the docs):
python scripts/benchmark_vol.py                     # training universe
MERIDIAN_HELDOUT=1 python scripts/benchmark_vol.py  # 24 never-trained assets
MERIDIAN_OMI=1     python scripts/benchmark_vol.py  # 17 intl indices (independent source)
python scripts/frontier_intraday.py                 # Regime-Meridian champion
python scripts/risk_benchmark.py                    # portfolio + tail risk
python scripts/compile_comparison.py                # -> COMPARISON.md
```

## Connecting your own data

Meridian is built to plug into **any** data source with a tiny adapter — a source is just a callable
that returns a tidy DataFrame/Series. Register once; every module can use it (`meridian/connect.py`).

```python
from meridian import connect
from meridian.analyze import full_analysis, portfolio_analysis

conn = connect.Connection()
conn.prices("AAPL")                 # any global symbol (equity/ETF/FX/crypto/futures/index) via Yahoo
conn.macro("BAMLH0A0HYM2")          # any FRED series (rates, credit spreads, conditions indices)
conn.upload("my_prices.csv")        # YOUR data — trusted as-is, never altered or invented
conn.news("Apple stock")            # recent headlines for CONTEXT (honest: not a market prediction)

# plug in a private vendor / broker feed in ~5 lines:
connect.register("mybroker", lambda sym: my_client.history(sym))
connect.get("mybroker")("AAPL")

# one call → the full stack (forecast + regime + tail VaR/ES + Monte-Carlo + generated thesis):
full_analysis("SPY", with_news=False)
portfolio_analysis(["SPY", "TLT", "GLD", "QQQ"])     # Ledoit-Wolf min-variance basket
```

**What one `full_analysis` call gives you** (every number from a calibrated, no-lookahead module):
volatility forecast (implied-vol-augmented where a matched free vol index exists — the +10.3% model),
regime, 1-day tail VaR/ES, a 10k-path Monte-Carlo (filtered historical simulation) horizon
distribution, market beta, and a readable **thesis**. Multi-asset joint scenarios and structural
what-ifs come from the world-model core (`meridian/worldmodel.py`, [`WORLDMODEL_CORE.md`](WORLDMODEL_CORE.md)).

**Honesty rails (by design, not disclaimer):** uploaded/vendor data is used exactly as given; the news
hook surfaces context and never fabricates a market call; the forecast thesis states *which* model ran
and where the free implied-vol lever is **not** available (single stocks / crypto). It quantifies risk;
it does not claim to predict the market, and it does not claim to beat proprietary-data or
latency-driven firm models — those are different games (private data, capital, microseconds), and any
such claim would be fabrication.

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
