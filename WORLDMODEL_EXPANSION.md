# Meridian World Model Expansion — Building Finance's Best

## Vision
Transform Meridian from a modular volatility/regime/tail forecaster into a comprehensive **generative world model** for financial markets that can:

1. **Imagine coherent futures** — generate 10k+ multi-asset scenarios with learned dependencies
2. **Reason about causation** — learn market DAG, trace shocks → effects (n-th order)
3. **Adapt continuously** — online learning from live ticks, news, economic data
4. **Compute risk intelligently** — VaR/CVaR/ES/tail-entropy from scenarios
5. **Optimize portfolios** — min-variance, risk-parity, max-Sharpe given model beliefs
6. **Explain decisions** — interpretable latent factors, causal attribution, what-if analysis

---

## Architecture (Modular Design)

```
┌─────────────────────────────────────────────────────────────────┐
│                   MERIDIAN WORLD MODEL CORE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  Representation  │    │   Causal Engine  │                  │
│  │  Learning        │    │   (DAG Discovery)│                  │
│  │  (JEPA/GLP)      │    │   (Intervention) │                  │
│  └────────┬─────────┘    └────────┬─────────┘                  │
│           │                       │                            │
│           ▼                       ▼                            │
│  ┌──────────────────────────────────────────┐                 │
│  │  Latent State z_t (factorized):         │                 │
│  │  z_trend, z_vol, z_sentiment,           │                 │
│  │  z_systemic, z_microstructure           │                 │
│  └────────┬─────────────────────┬──────────┘                  │
│           │                     │                            │
│  ┌────────▼─────────┐  ┌───────▼──────────┐                  │
│  │  Transition p()  │  │  Emission p()    │                  │
│  │  (with causal    │  │  (hybrid:        │                  │
│  │   structure)     │  │   parametric +   │                  │
│  │                  │  │   nonparametric) │                  │
│  └────────┬─────────┘  └───────┬──────────┘                  │
│           │                    │                            │
│           └────────┬───────────┘                            │
│                    ▼                                        │
│  ┌──────────────────────────────────────────┐               │
│  │  Scenario Generation (Monte Carlo)       │               │
│  │  → 10k paths with copula dependence      │               │
│  └───────┬──────────────────────────────────┘               │
│          ▼                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │  Risk Metrics Engine                     │               │
│  │  VaR, CVaR, ES, entropy, tail-index      │               │
│  └───────┬──────────────────────────────────┘               │
│          ▼                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │  Portfolio Optimizer                     │               │
│  │  min-var, risk-parity, max-Sharpe        │               │
│  └───────┬──────────────────────────────────┘               │
│          ▼                                                  │
│  ┌──────────────────────────────────────────┐               │
│  │  Interpretation & Visualization          │               │
│  │  Dashboards, causal graphs, what-ifs     │               │
│  └──────────────────────────────────────────┘               │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │        Continual Learning Harness                      │ │
│  │ (Replay buffer, priority weighting, regime detection) │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────────┘
         ▲                                          ▲
         │                                          │
    Live Market Data                        Action/Intervention
    (ticks, order flow,                    (portfolio actions,
     news, sentiment)                       what-if queries)
```

---

## Phase 1: Foundation (Enhanced Representation Learning)

### 1.1 JEPA vs GLP vs Current SSM — Decision Point

**Current**: CF-JEPA ensemble SSM core + QLIKE head + Student-t tail module

**JEPA advantages**:
- Joint-embedding → learns shared representation without explicit reconstruction loss
- Scales better to high-dim, sparse data (order books, news streams)
- Better suited to multi-scale temporal structure (1-day, 1-week, 1-month)

**GLP advantages**:
- Explicit latent factor model → interpretability (each factor = risk driver)
- Faster inference (smaller latent space)
- Natural connection to causal discovery

**Decision**: Build both as **alternatives** to existing SSM, measure on energy/variogram scores.

### 1.2 Latent Factorization

Structured latent code: `z_t = [z_trend, z_vol, z_sentiment, z_systemic, z_micro]`

- `z_trend` (4D): trend/momentum factors (decay ~5-20 days)
- `z_vol` (4D): realized volatility, vol clustering (decay ~30-60 days, leverage effect)
- `z_sentiment` (2D): market sentiment (crashes, rallies from news/order flow)
- `z_systemic` (2D): broad market risk (equity risk premium, risk-off regime)
- `z_micro` (1D): microstructure (bid-ask, volume regime)

**Why this structure**: Each latent factor has economic interpretation + separate half-life.

### 1.3 Loss Function

```
L_total = L_reconstruct + λ_kl * KL + λ_causal * L_causal + λ_calibration * L_calibration

L_reconstruct = E[variogram_score(x_pred, x_real)]  # Energy/variogram (proper scoring rules)
KL = KL_divergence(q(z|x) || p(z))
L_causal = MSE(intervene(z_i, shock) || observed_response)  # If we have A/B tests
L_calibration = calibration_loss(VaR_pred, actual_losses)  # Tail calibration
```

---

## Phase 2: Causal Discovery & Structure Learning

### 2.1 Learning the Market DAG

**Input**: Time series of `(returns, vol, sentiment, economic_data, news_sentiment)`

**Method**: 
- Start with PC algorithm (causal inference package) — works on observational data
- Refine with GES (Greedy Equivalence Search) if needed
- Validate with intervention experiments (where available)

**Output**: Directed acyclic graph showing:
```
Fed Decision → Bond Yields → Equity Duration Risk → Equity Vol
              ├→ Credit Spreads → High-Yield Returns
              └→ Inflation Expectations → TIPS Vol

Earnings Surprises → Sector Rotation → Beta-Crossover (SPY/QQQ)

Geopolitical News → Risk-Off → VIX Spike → Portfolio Rebalancing
```

### 2.2 N-th Order Causal Tracing

Given a shock (e.g., Fed rate hike), compute:
- **1st order**: Direct effect on yields
- **2nd order**: Yields → duration → equity betas
- **3rd order**: Equity betas → portfolio rebalancing → vol spike
- **4th order**: Vol spike → realized losses → margin calls
- **5th order**: Margin calls → fire sales → crowded-trade unwind

**Implementation**: 
- Traverse DAG forward from shock
- Use do-calculus to estimate path-specific effects
- Simulate 10k paths with shock at t=0, measure distributions at t+1, t+5, t+20

### 2.3 Structural What-Ifs

```python
# Example: "What if Fed raises 100bps?"
shock_scenario = {'fed_rate_delta': 0.01}  # 100bps
paths = model.intervene(shock_scenario, n_paths=10000, horizon=20)
# Returns: (returns_paths, vol_paths, sentiment_paths, ...)
# Can compute: median paths, confidence intervals, VaR, expected loss
```

---

## Phase 3: Scenario Generation & Risk Metrics

### 3.1 Scenario Generation Pipeline

```
Step 1: Condition model on recent history (past 120 days)
   → z_t = model.filter(returns[:120])

Step 2: Ancestral sample forward paths
   For n in 1..10000:
     for h in 0..H:
       z_{t+h} ~ p(z_{t+h} | z_{t+h-1})
       r_{t+h} ~ p(r | z_{t+h})  [hybrid parametric + FHS]
   → Returns matrix (10000 paths × H days)

Step 3: Preserve dependence structure
   Use copula transformation: rank correlation from real data
   applied to sample quantiles
```

### 3.2 Risk Metrics from Samples

```python
def risk_metrics(scenarios: ndarray):  # (n_paths, n_assets, horizon)
    """Compute risk metrics from generative samples."""
    # Portfolio value (equal-weighted, rebalanced daily)
    pnl = scenarios.cumsum(axis=0)  # (n_paths, horizon)
    
    metrics = {
        'var_95': np.percentile(pnl, 5, axis=0),
        'var_99': np.percentile(pnl, 1, axis=0),
        'cvar_95': pnl[pnl <= np.percentile(pnl, 5)].mean(axis=0),
        'tail_entropy': entropy(pnl <= np.percentile(pnl, 5)),
        'max_dd': (pnl - pnl.cummax()).min(axis=0),
        'jump_intensity': (pnl.diff() > 3*pnl.std()).sum(axis=0) / n_paths,
    }
    return metrics
```

### 3.3 Validation: Proper Scoring Rules

- **Energy Score**: `ES(x_sample, x_real) = E||X-x_real|| - 0.5*E||X-X'||`
- **Variogram Score**: measures calibration of multivariate tails
- **CRPS**: Continuous Ranked Probability Score for predictive distributions
- **Backtest**: VaR breach frequency, ES severity (regulatory)

---

## Phase 4: Portfolio Optimization

### 4.1 Mean-Variance with Learned Covariance

```python
def portfolio_optimize(model, returns_history, constraints=None):
    """Optimize portfolio using world-model estimated covariance."""
    # Step 1: Get model estimate of next-period covariance
    z_t = model.filter(returns_history)
    sigma_estimated = model.emit_cov(z_t)  # (n_assets, n_assets)
    
    # Step 2: Mean-variance optimization
    weights = cvxpy.Variable(n_assets)
    risk = cvxpy.quad_form(weights, sigma_estimated)
    
    # Expected return (use recent momentum or model-based forecast)
    mu = momentum(returns_history, window=20)  # or model.emit_mean()
    
    objective = cvxpy.Minimize(risk - λ * mu @ weights)
    problem = cvxpy.Problem(objective, [cvxpy.sum(weights) == 1, weights >= 0])
    problem.solve()
    
    return weights.value
```

### 4.2 Risk-Parity

```python
# Allocate so each asset contributes equally to portfolio volatility
weights_rp = 1 / sigma_estimated.diagonal()
weights_rp /= weights_rp.sum()
```

### 4.3 Max-Sharpe

```python
# Optimize: maximize (μ - rf) / σ
# subject to leverage, position limits, turnover constraints
```

---

## Phase 5: Continual Learning

### 5.1 Online Data Ingestion

```python
class LiveMarketHarness:
    def __init__(self, model, replay_buffer_size=10000):
        self.model = model
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        
    def ingest_tick(self, timestamp, returns, volume, order_book=None, news=None):
        """Ingest live market tick."""
        # Store in replay buffer with priority weight (recent + large moves)
        priority = 1.0 + abs(returns).mean()  # Prioritize large moves
        self.replay_buffer.append({
            'timestamp': timestamp,
            'returns': returns,
            'volume': volume,
            'order_book': order_book,
            'news': news,
            'priority': priority
        })
        
    def retrain_batch(self, batch_size=64, n_steps=100):
        """Perform online learning step."""
        # Sample from replay buffer with priority weighting
        batch = self._sample_buffer_prioritized(batch_size)
        
        # Frozen-target loss (prevent catastrophic forgetting)
        old_params = deepcopy(self.model.state_dict())
        
        for _ in range(n_steps):
            loss = self.model.elbo(batch)  # Or variogram loss
            loss.backward()
            # Constrain parameter updates to be small
            for (name, param), (_, old_param) in zip(
                self.model.named_parameters(), old_params.items()
            ):
                param.grad *= 0.01  # Small step size
            optimizer.step()
```

### 5.2 Regime Detection & Adaptation

```python
def detect_regime_shift(model, recent_returns, threshold=2.0):
    """Detect if model enters unfamiliar regime."""
    # Compute likelihood under model
    log_prob_old = model.log_prob(recent_returns)
    
    # If likelihood drops, model hasn't seen this regime
    if log_prob_old < -threshold:
        return True
    return False
```

### 5.3 Catastrophic Forgetting Prevention

**Strategy 1**: Elastic Weight Consolidation (EWC)
```
L = L_new + (λ/2) * Σ F_i * (θ_i - θ_old_i)²
```
where `F_i` is Fisher information (importance of parameter i).

**Strategy 2**: Replay buffer with importance weighting
- Recent data: high priority (adapt to new regime)
- Historical data: low priority (preserve learned structure)
- Balance: ~70% recent, ~30% historical

---

## Phase 6: Interpretation & Visualization

### 6.1 Latent Factor Dashboard

```
Panel 1: Latent Factors Over Time
├─ z_trend (4D): momentum, trend strength, mean-reversion
├─ z_vol (4D): realized vol, vol clustering, leverage
├─ z_sentiment (2D): risk-on/off, sentiment extremes
├─ z_systemic (2D): broad market risk, equity risk premium
└─ z_micro (1D): microstructure regime

Panel 2: Factor Attribution
├─ Correlation of each factor to realized returns
├─ Correlation to realized vol
├─ Forward predictability (lead/lag)
└─ Economic interpretation

Panel 3: Scenario Fan Charts
├─ 20-day forward paths (10th, 25th, 50th, 75th, 90th percentiles)
├─ Portfolio value distribution
├─ VaR/CVaR confidence intervals
└─ Risk attribution by factor
```

### 6.2 Causal Analysis Dashboard

```
Panel 1: Market DAG
└─ Interactive graph showing causal structure
   (click node to highlight incoming/outgoing edges)

Panel 2: Shock Simulation
├─ Select shock (Fed rate, geopolitical, earnings surprise)
├─ Specify magnitude
├─ Show N-th order propagation through DAG
└─ Display scenario paths and risk metrics

Panel 3: Historical Shocks
├─ Identify past major shocks from data
├─ Compare actual response vs model predictions
└─ Residual analysis (where did model miss?)
```

### 6.3 What-If Explorer

```
Interactive tool:
├─ "What if Fed raises 25bps?"
├─ "What if geopolitical crisis occurs?"
├─ "What if tech earnings collapse 20%?"
└─ → Returns portfolio paths, risk metrics, sector impacts
```

---

## Implementation Roadmap

| Phase | Duration | Key Milestones | Files |
|-------|----------|----------------|-------|
| **1: Representation** | 2-3 weeks | JEPA/GLP implementation, latent factorization, beat baseline | `meridian/jepa_encoder.py`, `meridian/glp.py` |
| **2: Causality** | 2-3 weeks | DAG learning (PC/GES), intervention validation, N-th order tracing | `meridian/causal_dag.py`, `meridian/causal_effects.py` |
| **3: Scenarios & Risk** | 1-2 weeks | 10k+ path generation, copula dependence, VaR/CVaR/ES backtesting | `meridian/scenario_gen.py`, `meridian/risk_metrics.py` |
| **4: Portfolio Opt** | 1 week | Min-var, risk-parity, max-Sharpe, live rebalancing | `meridian/portfolio_optimizer.py` |
| **5: Continual Learn** | 1 week | Online harness, replay buffer, regime detection, EWC | `meridian/continual_learning.py` |
| **6: Interpretation** | 1 week | Dashboards, causal viz, what-if tool | `demo/world_model_dashboard.py` |
| **Validation & Cleanup** | 1 week | Comprehensive backtests, documentation, edge cases | `scripts/comprehensive_benchmark.py` |

**Total: ~6-8 weeks for a world-class financial world model**

---

## Success Criteria

### Representation Learning
- [ ] Beat block-bootstrap baseline by 5-10% on energy/variogram scores
- [ ] JEPA/GLP edge over baseline SSM validated (Diebold-Mariano test)
- [ ] Latent factors are economically interpretable

### Causal Discovery
- [ ] DAG structure matches known market relationships
- [ ] N-th order shocks produce sensible scenario paths
- [ ] Historical shock responses align with model predictions (R² > 0.7)

### Risk Metrics
- [ ] VaR breaches at expected frequency (95% CI)
- [ ] ES severity passes regulatory test (Acerbi Z²)
- [ ] Scenario calibration: actual losses within confidence intervals

### Portfolio Optimization
- [ ] Backtested Sharpe ratio ≥ 0.8 on live data (2023-2024)
- [ ] Max drawdown ≤ -15% on equal-weighted portfolio
- [ ] Transaction cost < 2bps annually

### Interpretation
- [ ] Latent factors have >0.7 correlation with economic indicators
- [ ] Causal DAG is coherent and stable over time
- [ ] What-if scenarios match historical precedents (e.g., 2020 COVID, 2022 rate hikes)

---

## Data Requirements

- **Market data**: Daily OHLCV for 11 assets (existing)
- **Volatility data**: VIX, MOVE, sector-specific IV (existing)
- **Sentiment data**: News feeds (Reuters, Bloomberg, Twitter)
- **Economic calendar**: Fed decisions, CPI, employment (FRED API)
- **Order book**: Intraday bid-ask, depth (if available; otherwise fallback)

---

## Compute Resources

- **Training**: 2-4 GPU hours per JEPA/GLP variant (Apple MPS or cloud)
- **Online learning**: <1s per day for batch retraining
- **Scenario generation**: 10k paths in <500ms
- **Portfolio optimization**: <100ms

---

## References & Related Work

- **JEPA**: LeCun & Yildirim 2024 (vision); adapt to time series
- **World Models**: Ha & Schmidhuber 2018; Dreamer (Hafner et al. 2020)
- **Causal Inference**: Pearl 2009 (do-calculus); PC/GES algorithms
- **Factor Models**: Dynamic factor models (Stock & Watson 2005)
- **Risk**: Acerbi & Szekely 2014 (energy/variogram scores)
- **Financial**: Brini 2026 (foundation models beat baseline on benchmarks?)

