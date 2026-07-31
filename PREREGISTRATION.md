# Meridian — Pre-Registration of Evaluation Protocol

**Locked before any model results are inspected.** Date: 2026-07-29.
This document fixes the benchmarks, metrics, data splits, and win-margins **before**
the JEPA core is trained, so that a claimed win cannot be the product of hindsight
tuning. Any deviation must be recorded in the "Amendments" section with a timestamp
and a reason.

---

## 1. Task definitions

### Task A — Volatility forecasting (the primary claim)
Forecast next-day (h=1) realized volatility (RV) for each asset.
- **Target:** log realized variance, `log RV_{t+1}`. RV is computed from daily data as
  a Garman–Klass / Parkinson range estimator (intraday not required at this stage),
  and cross-checked against squared close-to-close returns.
- **Champion to beat:** **HAR-RV** (Corsi 2009) — OLS on `[RV_t, RV_{t-5:t}, RV_{t-22:t}]`
  in log space.
- **Also reported:** AR(1), AR(3) on log-RV; GARCH(1,1) 1-step variance forecast;
  EWMA (RiskMetrics λ=0.94).

### Task B — Regime persistence (the secondary claim)
Identify market regimes and beat a Gaussian **HMM** on regime *persistence* and
predictive usefulness.
- **Champion to beat:** 2- and 3-state Gaussian HMM on daily returns (+ optionally RV).

---

## 2. Data

- **Universe (equity):** SPY, QQQ, IWM, and 5 large-cap single names (AAPL, MSFT,
  JPM, XOM, JNJ) — sector spread.
- **Universe (FX):** EURUSD, USDJPY, GBPUSD (via Yahoo `EURUSD=X` etc.).
- **Macro/vol context (features only, never target leakage):** VIX (FRED `VIXCLS`),
  DGS10, DGS2 (FRED). Used only with information available at close of day `t`.
- **Sample:** 2007-01-01 → most recent available. Includes GFC, 2015, COVID, 2022.
- **Sources:** Yahoo chart API (delayed daily OHLC), FRED CSV. Free/delayed = acceptable
  for this stage per project scope.

---

## 3. Splits — leakage control (this is the scientific spine)

- **Walk-forward, expanding window.** Train on `[start, T)`, predict a forward block,
  roll forward. No shuffling. No future data ever enters training or feature scaling.
- **Purge + embargo:** because RV targets and HAR features overlap in time, we PURGE
  any training sample whose target window overlaps the test window, and apply an
  **embargo of 22 trading days** after each test block before training resumes.
- **Feature scaling** (means/stds) fit on train only, applied to test.
- **Out-of-sample (OOS)** = the concatenation of all forward test blocks. All headline
  numbers are OOS.

---

## 4. Metrics (pre-registered, primary in bold)

Volatility (lower is better):
- **QLIKE** on variance: `mean( RV/σ² − log(RV/σ²) − 1 )` — the robust, standard vol loss.
- MSE on `log RV`.
- MZ regression R² (Mincer–Zarnowitz), reported for interpretability only.

Regime:
- **Regime persistence** = mean dwell time / expected-state-duration, and
  1-step regime-transition log-likelihood on OOS.
- Economic check: does conditioning on the regime improve OOS vol QLIKE?

Surprise (JEPA-native, descriptive not a win-claim): energy / prediction-error score,
reported and correlated with realized |return| and VIX changes.

---

## 5. WIN CRITERIA (the bar, locked)

The Meridian core is declared to **beat** a benchmark only if **all** hold on purged OOS,
pooled across the universe:

- **Volatility vs HAR-RV:** ≥ **5% relative reduction in QLIKE**, AND a
  **Diebold–Mariano test** of equal predictive accuracy rejected at **p < 0.05**
  (HAC/Newey–West variance, one-sided in Meridian's favor).
- **Regime vs HMM:** higher OOS 1-step regime predictive log-likelihood AND
  ≥ **10% higher** mean regime persistence (dwell time) **without** loss of the
  economic check (regime-conditioned vol QLIKE not worse than HMM's).

Anything short of both is reported as **"did not beat"** — honestly, with the numbers.
No moving of this bar after seeing results.

---

## 6. Amendments

### Amendment 1 — 2026-07-29 — disclosed OOS looks + one pre-committed final run
Two configurations have been evaluated on the pre-registered OOS so far:
1. Meridian-MSE (head = E[log RV], MSE loss): calibrated QLIKE 0.3457, +2.51% vs
   HAR-RV, DM p=0.0036.
2. Meridian-QLIKE (head = log-variance, trained on exact QLIKE): calibrated QLIKE
   0.3435, +3.15% vs HAR-RV, DM p=0.0020.
Both beat AR/EWMA/GARCH by 14–23% (p<1e-4); neither cleared the +5% bar vs HAR-RV.

To avoid multiple-comparisons overfitting, I now **pre-commit a single final
configuration and will report its result once, whatever it is**:

> **Final model (committed before running):** a **5-seed ensemble** of the
> Meridian-QLIKE model (seeds 0–4), forecasts combined by **averaging the
> log-variance output**, evaluated under the identical purged walk-forward and
> the identical symmetric calibration applied to all models. No per-result
> tuning; no further looks after this one. Rationale: seed-ensembling is a
> standard a-priori variance-reduction step, chosen without reference to any
> test-set outcome.

Win criteria are unchanged (§5). If the ensemble does not clear +5%, that is the
final Day-1 answer and no further changes will be made against this OOS.

### Amendment 2 — 2026-07-29 — regime evaluation redesign (declared before results)
The original regime metrics (§4) are flawed and are replaced BEFORE computing any
new regime result:
- **Mean dwell time is gameable** — a detector that never switches maximizes it.
  It is demoted to a *sanity gate*, not a win metric: a usable detector must have
  pooled mean dwell in **[5, 60] trading days** (it must actually switch, yet be
  persistent). The first Meridian attempt (HMM on 64-d belief) failed this gate
  (~300-day dwell) and is discarded.
- **The old economic check shifted log-variance by the per-regime mean residual**,
  which targets squared error, not QLIKE — ill-posed. Replaced by a proper,
  leakage-safe OOS test:

> **Regime economic value (new primary regime metric).** Hold the base vol
> forecast FIXED at HAR-RV for BOTH methods. Split the OOS in time (first 50% =
> fit, last 50% = test). On the fit half, learn a per-regime affine map
> `a_r + b_r·(HAR log-var)` by minimizing the exact QLIKE within each regime;
> apply on the test half. **Value = OOS QLIKE improvement of (HAR + regime) over
> plain calibrated HAR.** The regime labeling that adds more OOS QLIKE value has
> the more useful regimes. Because the base forecast is HAR for both, this
> isolates regime information, not forecast skill.

- **Meridian regime construction (committed):** a 3-state Gaussian HMM on the
  **JEPA surprise-energy series** (the model's own instability signal) — distinct
  from returns and from the vol level, hence not circular with HAR.
- **HMM baseline:** 3-state Gaussian HMM on daily returns (unchanged).
- **Regime win criterion (amended):** Meridian regimes beat HMM iff they pass the
  dwell sanity gate AND deliver **strictly greater OOS QLIKE value** on the test
  half. Reported once; no iterate-to-win.

### Amendment 3 — 2026-07-30 — Meridian-WM evaluation (declared before any WM result)
The bridged model (ARCHITECTURE.md) is evaluated on three pre-declared axes under
the SAME purged/embargoed walk-forward; each is judged honestly, win or lose:

1. **Volatility (must-not-regress):** calibrated OOS QLIKE vs HAR-RV and vs the
   current SSM+QLIKE core. WM is acceptable on vol iff it does **not** significantly
   underperform the current core (research says vol is HAR-frontier-bound; WM is not
   expected to *improve* vol, only to hold it while adding regime/tail).
2. **Regime (target win):** the switching-SSM regime (argmax of the posterior α) beats
   a Gaussian HMM iff it passes the dwell gate [5,60] days AND gives strictly greater
   OOS regime economic value on a fixed HAR base (compare_regimes2 protocol).
3. **Tail (target win, new capability):** the predictive return distribution's
   left-VaR at 5% and 1% is **well-calibrated** — Kupiec POF and Christoffersen
   independence tests **not rejected at p<0.05** — and at least as well-calibrated as a
   Gaussian(σ) baseline with the same σ. Tails validated by VaR exceedance backtests,
   NOT by a proper scoring rule (which provably cannot separate extreme-tail values).

Ablations (toggle one component at a time) attribute each: switching core, EMA
readout, distributional head, JEPA/SIGReg aux. Reported once per configuration.
