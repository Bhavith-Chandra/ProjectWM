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

## Research roadmap (from the JEPA-variants deep-research pass, ranked by plausible lift)

The neural core is the substrate for genuine world-model research. Evidence-ranked next steps:
1. **EB-JEPA energy formulation** (Meta FAIR) — replace the plain latent-MSE energy with a learned
   energy function; ranked the top candidate to strengthen the surprise/regime signal.
2. **Multi-timescale / hierarchical JEPA (H-JEPA)** — daily/weekly/monthly latent predictors (HAR is
   itself multiscale); test whether hierarchy improves the representation and the energy's regime lead.
3. **Energy-as-regime trigger vs HMM** — formalize the +0.25 surprise correlation into a regime detector
   and test economic value vs a Gaussian HMM (the one pre-registered claim not yet met).
4. **Latent-space world-model calibration** — the emission-side tail experiment (`WORLDMODEL_CORE.md`).

Each will be built and **validated honestly** — reported by measured lift over the right baseline, with
dead ends logged, exactly as the rest of Meridian.
