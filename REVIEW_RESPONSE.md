# Response to the external technical review — verdicts by evidence

Each claim was audited against the actual source and, where testable, settled with a committed
experiment. Two claims produced genuine upgrades; three were refuted by measurement. Honest outcomes,
not agreement.

| # | Claim | Verdict | Decisive evidence | Action taken |
|---|---|---|---|---|
| 1 | Market-RV factor → cross-sectional lookahead leakage | **Refuted** | Factor is contemporaneous (time-t), not t+1. OOS edge **identical** contemporaneous +0.54% vs strict 1-day-lag +0.56%, both DM-sig (`scripts/leakage_mktrv_test.py`, 24 held-out assets, 117,928 rows). Leakage would collapse it to 0. | Adopted the **strict 1-day lag** as default anyway — free hygiene, provably equivalent (`features.market_rv_factor`, `MERIDIAN_MKT_LAG`). |
| 2 | FHS/EVT tail too slow → replace with Cornish-Fisher | **Refuted** | FHS = **3.45 ms** (10k×21d); 100k paths = 41 ms — no bottleneck. And CF is **less accurate**: 99% VaR breach 1.30% vs EVT-GPD **1.18%** (target 1.0%), and CF was **non-monotone (invalid) on 9% of windows** (`scripts/cf_vs_evt.py`, 10 assets). | **Kept EVT-GPD.** CF would be a downgrade solving a non-problem. |
| 3 | Linear GIRF severely understates crisis contagion → Threshold-VAR | **Not confirmed (mechanism overturned)** | Two independent OOS tests. (a) Rolling stress betas don't beat full-sample at realized crash co-moves: −2%/−5.8%, p=1.00 (`reflexive_validate.py`). (b) Crisis-grounded `ThresholdShockNetwork`, leave-one-crisis-out: −0.7%/−1.1%, DM p=0.90/0.99 (`validate_network.py`). Portfolio VaR *does* under-cover in crises (COVID 25.6% breach), but the decomposition shows that is a crisis-**volatility** effect — crisis **correlations** alone barely move it (25.6%→23.1%). | Linear GIRF stays default; `regime_propagate` + `ThresholdShockNetwork` retained **diagnostic-only**. The real lever (dynamic vol) is already shipped via HAR-lev+IV + the #5 gate. |
| 4 | LLM router on the hot serving path adds latency | **Refuted (false premise)** | `grep`: **zero** LLM/API calls anywhere. `ask.py` routes via `resolve()` (keyword/ticker) straight to numpy modules. `tools.py` is an optional provenance registry, not wired in. The proposed "Clock-0 triage gate" **is already the architecture.** | No change needed; documented. |
| 5 | Daily regime detection lags → IV/RV early-warning gate | **Confirmed** ✅ | VIX term-structure inversion (VIX9D/VIX3M>1) **leads** realized-vol stress onset by a median **~6 trading days**, catches **70%** of onsets at **52% precision** vs a 13% base rate, never lags (`scripts/iv_earlywarning.py`). IV-gated VaR also cut onset breaches 3.43%→3.10%. | **Wired live** as an early-warning flag (`exog.term_structure_warning`, surfaced in every thesis when inverted). |

## What genuinely improved the model

- **#5 — IV term-structure early-warning (new capability).** A validated leading stress indicator now
  surfaces in the thesis: *"VIX term structure is inverted … leads realized-vol stress onset by ~6
  trading days."* Honest scope: market-wide US-equity-vol signal; single stocks have no free per-asset
  IV. Validation: `scripts/iv_earlywarning.py`.
- **#1 — strict-lag factor (hardening).** No accuracy change (proven equivalent), but the factor is now
  leakage-proof by construction, removing the entire argument surface at zero cost.

## What the evidence said not to do

- **Don't** swap EVT-GPD for Cornish-Fisher (#2): measurably worse and mathematically breaks in fat tails.
- **Don't** replace the linear GIRF with a stress-switched one (#3): it does not improve OOS crisis
  prediction on this universe; the correlation-surge stylized fact is real but already priced by the
  full-sample beta.
- **Don't** build a "Clock-0 LLM bypass" (#4): there is no LLM in the path to bypass.

The reviewer's strongest intuition — that a forward-looking implied-vol signal beats a purely backward
realized-vol labeler at the *onset* of stress — was **correct and is now shipped**. The intuitions that
rested on assumed components (an LLM gate, contemporaneous-as-future leakage, an FHS bottleneck) did not
survive contact with the code or the measurements.

## Postscript — the Threshold-VAR deep dive (claim #3, fully worked)

The reviewer asked specifically to build the `ThresholdShockNetwork` with historically-grounded crisis
covariance and validate it. Done (`meridian/network.py`, `scripts/validate_network.py`), with the one
integrity fix that the crisis covariance is built **only from prior crises** (predict COVID from GFC;
predict 2022 from GFC+COVID) — otherwise scoring on the same windows used to build the matrix is
in-sample and wins trivially.

Result, worked to the mechanism:
1. **Propagation** (predict each asset's crisis-day move from the market's move): crisis covariance
   loses to linear GIRF OOS (DM p=0.90 / 0.99). The full-sample beta already prices the average crash
   co-move; the residual is idiosyncratic noise no beta can capture.
2. **Portfolio tail risk**: here the full-sample covariance genuinely fails — a 9-asset risk book breached
   its 99% VaR **25.6%** of days in COVID. So the reviewer's "systemic risk under-estimated" is real —
   but at the joint/portfolio level.
3. **Which mechanism?** Decomposition (crisis correlations + calm vols): coverage barely improves
   (25.6% → 23.1%). Crisis **volatility** carries the entire fix (→ 7.7%). So it is **not** the
   correlation gap. The actionable lever is dynamic volatility — which Meridian already applies via the
   HAR-lev+IV forecast and the IV term-structure early-warning gate.

Net: the threshold network is a useful **diagnostic / scenario** object (it shows the calm-vs-crisis
split), but it is **not** a validated forecasting upgrade, and shipping it as the default would dress up
a volatility effect as a contagion-network effect. The honest fix for crisis under-coverage — scale risk
to current volatility, and lead it with the implied-vol gate — is already in the model.
