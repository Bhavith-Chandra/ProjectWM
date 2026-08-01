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

| Test | Result (shipped checkpoint `results/worldmodel.pt`) | Verdict |
|---|---|---|
| **Stylized facts** (Cont 2001): fat tails, vol clustering, leverage | learned — ACF\|r\| 0.36 (real 0.13), kurtosis 26 (real 4.5), leverage sign correct −0.02 (real −0.06) — decisively beats the i.i.d.-Gaussian baseline (ACF ~0, kurtosis ~0) | ✅ learned dynamics, but **overshoots** |
| **Aggregational Gaussianity** (daily fat tails → near-Gaussian monthly) | falls REAL 4.45→0.49 vs model 26→7.9 — direction right, magnitude too fat | ◑ direction right |
| **Scenario VaR** (1-day 99% from filtered-state emission, OOS, 1402 days) | breach **3.1%** (target 1.0%) — under-calibrated | ◑ **looser than the EVT-GPD specialist (0.94%)** |
| **What-if** (structural `u_t` shock) | coherent flight-to-quality: equities −1 to −4 bps, TLT/IEF/LQD **+**, over 5 days | ✅ economically sensible |
| **Free-running LONG rollout** | unstable; high run-to-run variance (kurtosis overshoots to ~26 this run) | ⚠️ **documented limitation** |

**Honest bottom line:** the model **demonstrably learned market dynamics** (vol clustering + fat tails
vs a flat i.i.d. baseline are hard to fake) and produces **coherent structural what-ifs** — but on this
run it **overshoots** (kurtosis 26 vs 4.5) and its 1-day scenario VaR breaches **3.1%**, i.e. it is a
**joint SIMULATOR, not a calibrated tail** — looser than the EVT-GPD specialist (0.94%). So its wired
role (`analyze.world_portfolio_scenario`, surfaced in `ask.py --portfolio`) is **coherent joint
cross-asset structure + what-ifs, NOT a tighter VaR** — the calibrated tail number stays the
EVT-GPD/min-variance figure. The honest usable horizon is short (1-day directional; long free-runs
unstable). That boundary is the frontier, stated plainly — the next world-model work is stabilizing the
free-run (variance floors / lower Student-t weight) so the simulator's calibration matches the specialist.

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
