# Meridian — the Deep-Learning / World-Model / Energy-Based core

Meridian's neural foundation is a genuine **energy-based JEPA world model**, not a bag of MLPs. This
document states what it *is*, what it is *validated* to deliver, and — honestly — what it does not.

## Architecture (`meridian/model.py`, `worldmodel.py`, `switching.py`, `continuous.py`)

```
 x_{t-L+1:t}  ──►  SSM belief core  ──►  belief state h_t ──┬─► JEPA predictor g(h_t) ─► ẑ_{t+1}
 (windowed        (DiagonalSSM,                             │            │  energy = ‖ẑ_{t+1} − sg(f̄(x_fut))‖²
  features)        S4/Mamba-lite;                           │            ▼         = SURPRISE
                   or Neural-ODE core)                      │      EMA target encoder f̄ (no-grad)
                                                            ├─► vol head  (log-variance, QLIKE-trained)
                                                            └─► regime head (K-state)
 anti-collapse: SIGReg (LeJEPA-style isotropic-Gaussian moment matching on h)
 generative:    MeridianWorldModel — latent SSM + Student-t emission + rollout + do-intervention
```

This is a **Joint-Embedding Predictive Architecture** (LeCun-style): predict *latent* representations of
the future, not pixels/returns; an **EMA target encoder** provides the prediction target (BYOL/JEPA);
the **latent prediction error is an energy** (energy-based model); **SIGReg** prevents representation
collapse; and a separate **generative world model** rolls the latent forward for coherent multi-asset
simulation and structural what-ifs. Backbones: diagonal state-space (S4/Mamba-lite) or Neural-ODE.

## What it is validated to deliver (`scripts/validate_jepa_ebm.py`, 40,509 OOS rows, 2012–2026)

| Claim | Result | Verdict |
|---|---|---|
| **Energy = surprise** (EBM) | top-decile-energy days carry **4.4× the realized vol** of the rest; Spearman(energy, realized log-RV)=+0.25, Spearman(energy, \|forecast error\|)=+0.24 | ✅ **genuine** stress/surprise detector |
| **Generative world model** | reproduces vol clustering + fat tails vs i.i.d.; coherent flight-to-quality what-ifs (`WORLDMODEL_CORE.md`) | ✅ real dynamics |
| **Representation as a linear vol code** | within-block linear probe OOS R² ≈ −0.09 | ❌ not a clean *linear* decoder |
| **Point forecasting vs HAR** | neural QLIKE ≫ HAR (the daily-frequency neural blow-up) | ❌ HAR wins |

**Honest bottom line.** The neural core is a real deep-learning **world model** whose validated value is
the **energy-based surprise signal** and **generative simulation** — *not* point volatility forecasting.
At daily frequency, classical HAR beats deep nets out-of-sample (our ablations and the literature agree),
so the production forecaster stays classical. The DL system earns its place on a *different* axis:
representation of market state, an energy that spikes on genuine surprise, and coherent counterfactual
rollouts the linear specialists cannot produce.

## Why neural-foundational *and* classical-production is the honest design

These are not in tension — they answer different questions:
- **"What will next-day variance be?"** → classical HAR + implied vol (the +10.3% specialist). Deep nets
  overfit the low signal-to-noise of daily vol; measured, they blow up QLIKE.
- **"Is the market being surprising / what regime are we in / simulate a coherent crisis"** → the
  energy-based JEPA world model. This is where learned representations and generative dynamics win.

Forcing the deep net to also win forecasting would be dishonest (it doesn't) and would regress production.

## EB-JEPA learned energy — tested and REJECTED (`ab_energy_fast.py`, `compare_energy.py`)

The first DL research advance was to replace the fixed L2 latent-prediction-error energy with a **learned**
energy head (EB-JEPA), trained by an in-batch contrastive objective. Apples-to-apples A/B (identical
config, only the energy differs):

| Energy | vol lift | ρ(E, realized vol) |
|---|---|---|
| **L2 (latent-MSE)** | **2.91×** | **+0.38** |
| EB-JEPA (learned, contrastive) | 0.13× | −0.68 |

**Verdict: rejected.** The learned contrastive energy is *anti*-correlated with volatility — decisively
worse than the simple L2 energy. **Why (the principle):** a contrastive energy is trained to *discriminate*
which future matches which prediction; that optimizes distinctiveness/ease-of-matching, not surprise. The
L2 latent **prediction error** *is* surprise by construction, which is exactly why it gives the 4.4× lift.
A discriminative learned energy is the wrong tool for this signal. (Caveat: this refutes the *contrastive*
learned energy; a non-contrastive regularized EB-JEPA energy is a separate, lower-priority test — but the
same principle predicts it is unlikely to beat direct prediction-error for a surprise score.) The L2
energy stays the production surprise signal.


## Energy-as-regime vs HMM — tested, does NOT beat the baseline (`energy_vs_hmm.py`)

The pre-registered claim: do JEPA-energy regimes beat a Gaussian HMM (persistence + economic value)?

| Regime detector | dwell (days) | econ QLIKE % |
|---|---|---|
| HMM on returns | 9.10 | −39.8 |
| JEPA energy | 7.86 (−13.6%, fails +10% bar) | −33.9 |
| belief state | 344 (degenerate) | −45.7 |

**Verdict: NOT met.** The energy is *less* persistent than the HMM — because surprise is spiky, not
sticky. This is not a failure of the signal; it is the signal's nature: the energy is a genuine
transient **surprise / early-warning spike** detector (4.4× vol lift, ~6-day lead), which is a different
and complementary tool from a persistent regime label. Use the HMM for sticky regimes, the energy for
spikes. The pre-registered regime-vs-HMM claim remains unmet by every learned detector tried.

## Research roadmap (from the JEPA-variants deep-research pass, ranked by plausible lift)

The neural core is the substrate for genuine world-model research. Status of the ranked items:
1. ~~**EB-JEPA learned energy**~~ — **TESTED, REJECTED** (see above): the learned contrastive energy is
   worse than L2 for surprise; discrimination ≠ surprise.
2. ~~**Multi-timescale / hierarchical JEPA (H-JEPA)**~~ — **TESTED, REJECTED** (`ab_hjepa.py`):
   multi-scale energy 2.25× vs single-scale 2.67× vol lift. Aggregating horizons raises broad
   correlation but blunts the sharp spike that makes a surprise signal useful. Single-scale wins.
3. ~~**Energy-as-regime trigger vs HMM**~~ — **TESTED, NOT MET** (see above): the energy is spiky, not
   sticky; it does not beat the HMM as a persistent regime label.
4. **Latent-space world-model calibration** — the emission-side tail experiment (`WORLDMODEL_CORE.md`) —
   untested; the architecture already delegates the calibrated tail to EVT-GPD, so low expected value.

**What the DL research thread established.** The neural core delivers exactly ONE validated, durable win:
the **L2 latent-prediction energy as a surprise / early-warning spike** (4.4× vol lift, ~6-day lead). Every
attempt to make it fancier — a learned energy, a persistent regime label, a neural point-forecaster — lost
to a simpler baseline (L2 energy, HMM, HAR respectively). For daily financial series the honest pattern is
consistent: the simplest formulation that directly measures the target wins. Each result above was built
and **validated honestly**, with dead ends logged — exactly as the rest of Meridian.
