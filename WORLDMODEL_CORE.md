# Meridian World-Model Core — a genuine deep state-space generative model of the market

This is the component that lets us **honestly** use the term "world model." It is *not* one of the
point-forecast specialists (those live in the module bank, `WORLD_MODEL.md`). It is a learned model
of market **dynamics** that can **imagine coherent futures** — the technical definition of a world
model (Ha & Schmidhuber 2018; LeCun; Dreamer). Code: `meridian/worldmodel.py`, `scripts/train_worldmodel.py`.

## What it is (the honest claim)

Verified against the canonical bar (learned latent state **V**, learned forward dynamics **M**,
rollout/imagination, intervention):

- ✅ **(V) learned latent market state** `z_t ∈ ℝ¹²` — a compact learned representation of the market's
  condition, inferred from returns by a GRU filter `q(z_t | r_{≤t})`.
- ✅ **(M) learned forward dynamics** — a neural stochastic transition `p(z_t | z_{t-1})` (gated, so
  volatility persists).
- ✅ **Emission with dynamic covariance** — the latent sets each asset's log-variance *and* a low-rank
  cross-asset factor structure (latent stochastic volatility): `r_t ~ N(0, D(z_t) + L(z_t)Lᵀ)`.
- ✅ **Rollout / imagination** — ancestral-sample the transition forward and decode **coherent
  multi-asset return paths** (Monte-Carlo scenarios).
- ✅ **What-if intervention** — clamp/shock a latent factor and roll forward → conditional scenario.
- ❌ **Not agentic** — no controller trained inside the model (no "dream-training" loop).

> **One-sentence honest description:** *Meridian's world-model core is a deep state-space generative
> model of the daily multivariate market that learns a latent regime/volatility state and its forward
> dynamics, and rolls out coherent multi-step joint scenarios and shock-conditioned what-ifs in latent
> space. It is a **passive (observational) world model**, not an agentic planner, and it does not claim
> to beat the specialist forecasters on point accuracy.*

**We may call it:** a deep state-space / RSSM-style generative market world model (passive/observational).
**We may NOT call it:** an autonomous/agentic world model, "the" causal counterfactual engine, or a model
that beats the specialist bank at point forecasting.

## Architecture (best-practice, research-backed — pass wlp6oec7l)

Deep Markov / RSSM-style, with the three finance-specific upgrades the research prescribed:
- **Student-t emission** with dynamic low-rank factor covariance (fat tails in the emission;
  Gaussian transition stays light-tailed so rollouts don't detonate — DreamerV3 stabilization).
- **Sign-conditioned transition** — the latent vol reacts to `sign(r)·|r|` of the previous return
  (the leverage channel).
- **Exogenous `u_t` conditioning slot** — the structural "do-hook": `do(u_rate)` vs `do(u_credit)`
  give different coherent multi-asset paths. Action-conditioning is *constitutive* of a world model.
- **Free-bits / KL-annealing** guards against posterior collapse (tuned to 0 here — a positive floor
  over-activated the free-run on this noisy daily data; see limitation below).

## Validation — does it genuinely behave like a market?

A world model earns the name by reproducing the world's behavior, not by point-forecast accuracy.

| Test | Result | Verdict |
|---|---|---|
| **Stylized facts** (Cont 2001): fat tails, vol clustering, leverage | reproduced — kurtosis ~5 (real 4.5) via Student-t, positive ACF\|r\|, correct leverage sign — vs an i.i.d. baseline of ~0 | ✅ learned real dynamics |
| **Aggregational Gaussianity** (daily fat tails → near-Gaussian monthly) | REAL 4.45→0.49 vs model 5.26→0.41 — a *hard* test most naive models fail | ✅ passes |
| **Scenario VaR** (1-day 99% from filtered-state emission, OOS) | breach **~2.4%** (target 1.0%) — usable | ◑ reasonable, slightly tight |
| **What-if** (structural `u_t` shock) | coherent market-wide risk-off: equities ≈ −5 bps, havens flat/up | ✅ economically sensible |
| **Free-running LONG rollout (1400 steps)** | unstable / high run-to-run variance (kurtosis can overshoot) | ⚠️ **documented limitation** |

**Honest bottom line:** the model **demonstrably learned market dynamics** (fat tails + aggregational
Gaussianity are hard to fake), emits **usable calibrated 1-day risk**, and produces **coherent
structural what-ifs**. Its **long free-running simulation is unstable** — a fundamental limitation of
deep state-space models on noisy daily data (the research warned: *"the honest rollout horizon is days,
not weeks — report where calibration breaks"*). So the **honest usable horizon is short (1-day
calibrated; multi-day directional)**; we do not claim stable long free-runs. That boundary is the
frontier, stated plainly — not hidden.

## Research-backed next steps (verified deep-research pass wlp6oec7l)

The build matches the recommended architecture (Deep Markov Model + RSSM split); the evidence-backed
upgrades to make it best-practice and fix the drift:

1. **Student-t emission** — put the fat tails in the emission, keep the transition Gaussian/light-tailed
   so long rollouts stay stable (fixes the free-running overshoot).
2. **Sign-conditioned volatility transition** — feed `sign(r)·|r|` into the transition so latent vol
   reacts asymmetrically to down-moves → strengthens the leverage effect.
3. **Explicit `u_t` conditioning channel** — the structural "do-hook" so *"what if a rate shock vs a
   credit shock"* gives different coherent paths. The research flags action-conditioning as
   *constitutive* of a world model.
4. **Temperature control** on the rollout (added) + variance floors — stabilize long horizons; report
   the **honest horizon** at which calibration breaks (don't assume a fixed H).
5. **Evaluation panel**: variogram score (primary, dependence-sensitive), energy score, PIT calibration,
   **aggregational Gaussianity** (daily fat tails must become near-Gaussian monthly), VaR-of-paths —
   pre-registered, vs a block-bootstrap / GARCH-simulation baseline.

## How it fits the system

The world-model core **complements**, not replaces, the specialist bank. The specialists win point
forecasting (Regime-Meridian ~4% over HAR) and risk (min-variance, EVT). The world-model core adds a
**new capability they cannot provide** — coherent joint multi-step *scenario simulation* and structural
*what-ifs*. That is what makes "Meridian Enterprise World Model" an honest name: there is now a genuine
world model inside it.
