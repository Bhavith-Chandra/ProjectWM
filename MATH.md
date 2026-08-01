# Meridian — Complete Mathematical Reference

Every quantitative claim in the project, with its math, assumptions, and the committed script that
validates it. Notation is fixed once and reused. Equations use GitHub LaTeX.

**Symbols.** $r_t$ = daily log return $\ln(P_t/P_{t-1})$. $RV_t$ = realized variance (a variance
*proxy*). $\sigma_t^2$ = the (latent) conditional variance we forecast. $\hat\sigma^2$ = a forecast,
$\tilde\sigma^2$ = the proxy used to score it. $\mathbb{1}\{\cdot\}$ = indicator. $q$ = a quantile
level (0.99). $n$ = sample size. $w$ = portfolio weights. Bold = vectors/matrices.

---

## Part I — Volatility measurement and forecasting

### 1. Realized-variance estimators (`features.py`)

From daily OHLC we form three range-based variance proxies. With $o,h,l,c=\ln(O,H,L,C)$:

$$RV^{\text{cc}}_t = (c_t - c_{t-1})^2, \qquad RV^{\text{park}}_t = \frac{(h_t-l_t)^2}{4\ln 2},$$

$$RV^{\text{GK}}_t = \tfrac12 (h_t - l_t)^2 - (2\ln 2 - 1)\,(c_t - o_t)^2 \quad\text{(Garman–Klass)}.$$

Garman–Klass is up to ~7× more efficient than close-to-close because the intraday range $(h-l)$
carries more information about $\sigma$ than the single close-to-close increment. It is the primary
target basis $RV_t \equiv \max(RV^{\text{GK}}_t,\varepsilon)$, floored at $\varepsilon=10^{-12}$ so a
zero-range (halted-flat) day cannot produce $\ln 0=-\infty$ (see §22, robustness).

### 2. HAR-RV cascade (Corsi 2009) (`features.py`, `engine.py`)

The Heterogeneous AutoRegressive model captures long memory with three horizons — daily, weekly,
monthly — as a parsimonious linear cascade:

$$RV^{(w)}_t = \tfrac15\sum_{i=0}^{4} RV_{t-i},\qquad RV^{(m)}_t = \tfrac1{22}\sum_{i=0}^{21} RV_{t-i}.$$

We forecast in **logs** (variance is right-skewed and $\log RV$ is approximately Gaussian):

$$\ln RV_{t+1} = \beta_0 + \beta_d \ln RV_t + \beta_w \ln RV^{(w)}_t + \beta_m \ln RV^{(m)}_t
 + \beta_\ell \ln RS^-_t + \varepsilon_{t+1}.$$

**Realized semivariance** (Barndorff-Nielsen–Kinnebrock–Shephard 2010) splits variance by sign:

$$RS^-_t = \big(\min(r_t,0)\big)^2 \ \ (\text{bad vol}),\qquad RS^+_t = \big(\max(r_t,0)\big)^2\ \ (\text{good vol}).$$

The leverage term $\beta_\ell \ln RS^-$ encodes that **downside** moves forecast higher future vol than
upside moves of equal size. Validated: adding it improves OOS QLIKE +0.76% on average, concentrated in
equities (SPY +3.6%), ~0 in FX — exactly as the leverage effect predicts (`shar_validate.py`).

### 3. Jensen correction: log-forecast → level-forecast (`engine.py`, `benchmark_*.py`)

We fit $\hat y = \widehat{\mathbb{E}}[\ln RV_{t+1}]$ but need a forecast of the **level** $RV_{t+1}$.
Under conditional log-normality $\ln RV_{t+1}\sim\mathcal N(\hat y,\ s^2)$,

$$\widehat{RV}_{t+1} = \mathbb{E}[RV_{t+1}] = \exp\!\Big(\hat y + \tfrac12 s^2\Big),\qquad
 s^2 = \widehat{\operatorname{Var}}(\ln RV - \widehat{\ln RV}).$$

Omitting the $+\tfrac12 s^2$ (the "Jensen bump", `jb` in code) biases the level forecast **downward** by
$\exp(\tfrac12 s^2)$ — material since $s^2$ is not small. Every level forecast in the repo applies it.

---

## Part II — Loss functions and model comparison

### 4. QLIKE and proxy-robustness (`evalproto.py`)

The scoring loss is QLIKE, in its distance form (≥0, zero at equality):

$$L_{\text{QLIKE}}(\hat\sigma^2,\tilde\sigma^2) = \frac{\tilde\sigma^2}{\hat\sigma^2}
 - \ln\frac{\tilde\sigma^2}{\hat\sigma^2} - 1.$$

**Why QLIKE and not MSE-on-vol or MAE.** We never observe $\sigma_t^2$; we score against a *noisy proxy*
$\tilde\sigma^2$. Patton (2011) proves only a specific family of losses gives the **same forecast ranking
in expectation** whether scored on the true $\sigma^2$ or a conditionally-unbiased proxy
($\mathbb{E}[\tilde\sigma^2\mid\mathcal F]=\sigma^2$). QLIKE and MSE-on-variance are in that family; MAE and
MSE-on-*volatility* are **not** — they produce proxy-induced ranking reversals. QLIKE additionally
penalizes under-prediction of variance more heavily (crucial for risk), and is scale-free. Hence QLIKE is
the headline; MAE is banned from headline comparisons.

### 5. Diebold–Mariano test (`evalproto.py`)

For two forecasts with loss differential $d_t = L^{(1)}_t - L^{(2)}_t$,

$$\mathrm{DM} = \frac{\bar d}{\sqrt{\widehat{\operatorname{Var}}(\bar d)}}\ \xrightarrow{d}\ \mathcal N(0,1),
\qquad \widehat{\operatorname{Var}}(\bar d) = \frac1n\Big(\gamma_0 + 2\sum_{k=1}^{K}\big(1-\tfrac{k}{K+1}\big)\gamma_k\Big),$$

using a Newey–West (HAC) long-run variance because $d_t$ is autocorrelated. $|\mathrm{DM}|>1.96 \Rightarrow$
the loss difference is significant at 5%. All "+X% over HAR (p<0.001)" claims are DM p-values.

### 6. Model Confidence Set (Hansen–Lunde–Nason 2011)

The MCS returns the set $\widehat{\mathcal M}^*_{1-\alpha}$ of models statistically indistinguishable from
the best at confidence $1-\alpha$. It iteratively tests the equal-predictive-ability hypothesis
$H_0:\mathbb{E}[d_{ij}]=0\ \forall i,j\in\mathcal M$ and eliminates the worst until $H_0$ is not rejected.
"Only Meridian-lin survives the 90% MCS" means every other model was eliminated as significantly worse.

---

## Part III — Exogenous signal: implied volatility

### 7. Matched implied-vol family and the HARX model (`exog.py`)

Implied volatility is the market's own risk-neutral variance forecast. Each asset is paired with **its
own** vol index (SPY↔VIX, QQQ↔VXN, IWM↔RVX, USO↔OVX, GLD↔GVZ, …), not the generic VIX. Augmenting HAR:

$$\ln RV_{t+1} = \text{HAR}(\ldots) + \gamma_1 \ln IV_t + \gamma_2 \Delta\ln IV_t
 + \gamma_3\,\text{TS}_t + \gamma_4\,\text{VRP}_t + \varepsilon_{t+1}.$$

- **Term structure** $\text{TS}_t=\ln(\text{VIX9D}_t/\text{VIX3M}_t)$: slope/vol-of-vol a single level misses.
- **Variance risk premium** $\text{VRP}_t = \ln IV^2_t/252 - \ln RV_t$ (implied minus realized variance).

**Validated result** (`benchmark_exog.py`, 23,496 OOS forecasts, 7 matched-index ETFs):

| Model | QLIKE | vs HAR | DM vs HAR |
|---|---|---|---|
| HAR | 0.3488 | — | — |
| + matched IV | 0.3221 | **+7.64%** | <0.001 |
| + IV + TS + VRP | 0.3129 | **+10.30%** | <0.001 |

Deep-research (passes `w084stjkn`, `wlbzkb0a9`, 127 agents, all key claims verified against primary
sources) confirms IV is the single strongest free lever, that raw IV overstates next-day RV ~36% via the
VRP (hence gate on the *ratio*, not the level), and that the edge peaks ~1-week and decays by 1-month.

### 8. Market-RV commonality factor and the lag-invariance (leakage) theorem (`features.py`)

The common factor (Bollerslev "Risk Everywhere") is the cross-sectional mean of log-RV:

$$F_t = \frac1N\sum_{i=1}^N \ln RV_{i,t}.$$

**Claim (no lookahead).** $F_t$ is *contemporaneous* — every $RV_{i,t}$ is known at close $t$ — so using
$F_t$ to predict $RV_{i,t+1}$ is causal, not future-peeking. **Falsification test:** if the edge came from
contemporaneous cross-sectional leakage, replacing $F_t$ by the strictly-lagged $F_{t-1}$ would collapse
it. It does not (`leakage_mktrv_test.py`, 24 held-out assets, 117,928 rows):

$$\underbrace{+0.54\%}_{F_t\ \text{(contemp.)}} \approx \underbrace{+0.56\%}_{F_{t-1}\ \text{(lagged)}},
\quad\text{both DM-significant.}$$

Lag-invariance of the edge ⟹ the signal is the *persistent* commonality in volatility, not a
same-day artifact. Production uses `lag=1` as free hygiene (provably equivalent, removes all doubt).

---

## Part IV — Tail risk

### 9. Conditional EVT via Peaks-Over-Threshold (McNeil–Frey 2000) (`engine.py`)

Standardize losses $z_t = -r_t/\hat\sigma_t$. By the Pickands–Balkema–de Haan theorem, exceedances over a
high threshold $u$ converge to a **Generalized Pareto Distribution**:

$$G_{\xi,\beta}(y) = 1-\Big(1+\frac{\xi y}{\beta}\Big)^{-1/\xi},\qquad y = z-u > 0.$$

With $N_u$ exceedances out of $n$, the $q$-quantile (VaR) and Expected Shortfall are closed-form:

$$\mathrm{VaR}_q = u + \frac{\beta}{\xi}\Big[\Big(\frac{n}{N_u}(1-q)\Big)^{-\xi}-1\Big],\qquad
 \mathrm{ES}_q = \frac{\mathrm{VaR}_q}{1-\xi} + \frac{\beta-\xi u}{1-\xi}\quad(\xi<1).$$

Then $\mathrm{VaR}^{\text{1-day}}_q = \hat\sigma_t\cdot \mathrm{VaR}_q$. Tail shape $\xi$ is fit by MLE
(clamped $\xi\in[-0.4,0.9]$ for stability); ES exists only for $\xi<1$ (finite mean). $u$ = 90th
percentile of standardized losses. This models the tail *shape* separately from the conditional *scale*
$\hat\sigma_t$ — the conditional-EVT idea.

### 10. Why Cornish–Fisher must NOT replace EVT at 99% (`cf_vs_evt.py`)

Cornish–Fisher expands a tail quantile in the first four moments (skew $S$, excess kurtosis $K$):

$$z_{\text{CF}}(q) = z + \tfrac16(z^2-1)S + \tfrac1{24}(z^3-3z)K - \tfrac1{36}(2z^3-5z)S^2,\quad z=\Phi^{-1}(q).$$

**Domain-of-validity failure (Maillard 2018).** $z_{\text{CF}}(\cdot)$ must be *monotone increasing* to be
a valid quantile function; its derivative

$$\frac{dz_{\text{CF}}}{dz} = 1 + \tfrac13 z S + \tfrac1{24}(3z^2-3)K - \tfrac1{36}(6z^2-5)S^2$$

goes **negative** for large $(S,K)$ — precisely the fat-tailed regime where deep-tail risk lives. Measured:
CF was non-monotone on **9%** of windows, and its 99% VaR breach was **1.30%** vs EVT-GPD **1.18%**
(target 1.0%), across 10 assets. Verdict: EVT-GPD is best-calibrated; CF is at most a fast approximation,
never the deep-tail estimator. (And FHS runs in 3.5 ms, so there is no speed problem for CF to solve.)

### 11. VaR/ES backtests (`es_backtest.py`)

Let $x$ = number of breaches ($r_{t+1} < -\mathrm{VaR}_t$) in $n$ days, target rate $p=1-q$.

**Kupiec POF (unconditional coverage):**

$$LR_{\text{POF}} = -2\ln\frac{(1-p)^{n-x}p^{x}}{(1-\hat\pi)^{n-x}\hat\pi^{x}}\ \sim\ \chi^2_1,\quad \hat\pi=x/n.$$

**Christoffersen independence:** with $n_{ij}$ = counts of transitioning breach-state $i\to j$,

$$LR_{\text{ind}} = -2\ln\frac{(1-\hat\pi)^{n_{00}+n_{10}}\hat\pi^{n_{01}+n_{11}}}
{(1-\hat\pi_{01})^{n_{00}}\hat\pi_{01}^{n_{01}}(1-\hat\pi_{11})^{n_{10}}\hat\pi_{11}^{n_{11}}}\ \sim\ \chi^2_1,$$

testing whether a breach today is independent of a breach yesterday (no *clustering*).

**Acerbi–Székely Test 2 (ES):** with ES$_t>0$ the predicted shortfall magnitude and $\mathbb 1_t$ the breach,

$$Z_2 = \frac{1}{n\,p}\sum_t \frac{r_t\,\mathbb 1_t}{-\mathrm{ES}_t} + 1 \approx 0 \text{ if ES correct};\quad
 Z_2>0 \Rightarrow \text{ES too optimistic.}$$

*(Sign convention matters: $r_t<0$ on a breach and ES enters as a negative return, so the ratio is positive
and the sum $\approx -1$ when ES is exact. An earlier implementation double-negated the denominator,
yielding a spurious $Z_2\approx 2$; corrected, $Z_2\approx 0.02$ — the ES is well-sized.)*

**Results:** Kupiec **passes** (mean breach 0.94%), Acerbi $Z_2\approx0$ (**ES well-calibrated**); the only
gap is Christoffersen — breaches **cluster** — which a faster EWMA $\sigma$ partially fixes (independence
passes 1/8 → 3/8 assets; residual clustering is intrinsic to daily tail risk).

### 12. Filtered Historical Simulation (`analyze.py`)

Non-parametric horizon distribution: standardize historical returns $z_i = r_i/\hat\sigma$, resample with
replacement, rescale by the *current* vol forecast, and sum over the horizon $H$:

$$\tilde r^{(p)}_{t+1:t+H} = \sum_{h=1}^{H} \hat\sigma_t\, z_{I(p,h)},\quad I(p,h)\sim \text{Unif}\{1,\dots,n\},$$

giving 10,000 coherent paths → VaR/ES/drawdown quantiles. Keeps the empirical shock shape (fat tails,
asymmetry) while conditioning scale on today. Cost: **3.45 ms** for $10^4\times21$ (measured) — vectorized
`rng.choice` + one `percentile`, not "sorting thousands of records on the fly."

---

## Part V — Portfolio construction

### 13. Ledoit–Wolf shrinkage + minimum-variance (`analyze.py`)

The sample covariance $S$ is ill-conditioned when $N\!\sim\!T$. Shrink toward a structured target $F$:

$$\Sigma_{\text{LW}} = \delta^\star F + (1-\delta^\star)S,\qquad
 \delta^\star = \arg\min_\delta \mathbb{E}\lVert \Sigma_{\text{LW}} - \Sigma\rVert_F^2,$$

with $\delta^\star$ estimated in closed form. The **Global Minimum-Variance** portfolio is

$$w_{\text{GMV}} = \frac{\Sigma^{-1}\mathbf 1}{\mathbf 1^\top \Sigma^{-1}\mathbf 1},\qquad
 \sigma_p^2 = w^\top\Sigma w.$$

Measured −14% portfolio vol vs equal-weight on SPY/TLT/GLD/QQQ.

---

## Part VI — Shock propagation network

### 14. VAR, generalized IRF (Pesaran–Shin 1998), connectedness (`network.py`)

Fit a vector autoregression $\mathbf r_t = \sum_{k}A_k \mathbf r_{t-k} + \mathbf u_t$,
$\operatorname{Cov}(\mathbf u)=\Sigma$, with MA representation $\mathbf r_t=\sum_h \Theta_h \mathbf u_{t-h}$.
The **generalized** impulse response (order-invariant, unlike Cholesky) of variable $i$ to a shock in $j$:

$$\text{GIRF}_{i\leftarrow j}(h) = \frac{(\Theta_h \Sigma)_{ij}}{\sqrt{\Sigma_{jj}}}.$$

At impact ($h=0$, $\Theta_0=I$) this reduces to $\Sigma_{ij}/\sqrt{\Sigma_{jj}}$; scaling a unit shock in
$j$ to size $\delta$ gives the **co-move** $\Sigma_{ij}/\Sigma_{jj}\cdot\delta = \beta_{i|j}\,\delta$ — the
regression beta. This identity is the hinge of §15–16. Diebold–Yilmaz **connectedness** normalizes the
generalized FEVD to a "who-transmits-to-whom" table; net transmitter $=$ (to others) $-$ (from others).

### 15. Threshold-VAR and the leave-one-crisis-out methodology (`validate_network.py`) — *requested*

**The hypothesis.** In crises, correlations "gap toward 1", so a *stress-conditional* covariance
$\Sigma_{\text{stress}}$ should propagate shocks better than one linear $\Sigma$. A Threshold-VAR switches
covariance when a stress boundary is crossed.

**The methodological trap.** Build $\Sigma_{\text{stress}}$ from the 2008/2020/2022 windows and then score
on those same windows, and you are testing in-sample: the matrix has seen the very co-moves it is asked to
predict, so it "wins" by construction. This is data-snooping, not evidence.

**Leave-one-crisis-out (LOCO), chronological.** Order the crises $C_1$ (GFC), $C_2$ (COVID), $C_3$ (2022)
by time. To score crisis $C_k$:

$$\Sigma^{(k)}_{\text{stress}} = \operatorname{Cov}\Big(\!\!\bigcup_{j<k} C_j\Big),\qquad
 \Sigma^{(k)}_{\text{calm}} = \operatorname{Cov}\big(\{t : t<\min C_k,\ t\notin \textstyle\bigcup_{j<k}C_j\}\big).$$

That is: **the stress covariance is built only from crises strictly *before* the test crisis**, and *all*
data from $C_k$ onward is purged from estimation. Concretely — predict COVID using only the GFC footprint;
predict 2022 using GFC+COVID. For each day $t\in C_k$, given the realized market move $r_{m,t}$, predict
each asset $i$'s move two ways and score squared error vs realized $r_{i,t}$:

$$\hat r^{\text{lin}}_{i,t} = \beta^{\text{full}}_{i|m}\,r_{m,t},\qquad
 \hat r^{\text{thr}}_{i,t} = \beta^{\text{stress}}_{i|m}\,r_{m,t},\qquad
 \beta_{i|m}=\frac{\Sigma_{im}}{\Sigma_{mm}}.$$

A one-sided Diebold–Mariano (Newey–West, lag 5) tests whether the threshold RMSE is significantly lower.

**Result — the Threshold-VAR is defeated OOS.** RMSE $\Delta$: COVID **−0.7%**, 2022 **−1.1%** (negative =
threshold *worse*), DM $p=0.90,\,0.99$. Two independent designs agree (`reflexive_validate.py` rolling
stress betas: −2%/−5.8%, sign-test $p=1.00$). *Why:* the full-sample $\beta$ already contains the crises
(they are in the estimation window), so it already prices the average crash co-move; re-estimating on the
stress subset spends estimation variance (few crisis days) for no bias reduction. The stylized fact
(correlations rise) is real; it does **not** translate into better OOS *point* prediction of co-moves.

### 16. Why volatility, not correlation, carries portfolio tail risk (`validate_network.py`) — *requested*

Write the covariance as $\Sigma = D\,C\,D$ with $D=\operatorname{diag}(\sigma_1,\dots,\sigma_N)$ and $C$
the correlation matrix. Portfolio variance:

$$\sigma_p^2 = w^\top \Sigma w = \sum_{i,j} w_i w_j\, \sigma_i \sigma_j\, \rho_{ij}. \tag{16.1}$$

**Volatility scaling is multiplicative.** If every vol scales by a crisis factor $g$ ($\sigma_i\to g\sigma_i$)
with correlations unchanged, then from (16.1)

$$\sigma_p^2 \to g^2\,\sigma_p^2 \quad\Longrightarrow\quad \sigma_p \to g\,\sigma_p
 \quad\Longrightarrow\quad \mathrm{VaR}\to g\cdot\mathrm{VaR}. \tag{16.2}$$

VaR is **linear in $g$**: a 3× vol surge ⟹ 3× VaR (+200%).

**Correlation gap is bounded and sub-linear.** Take the homogeneous book (equal weights $w_i=1/N$, common
vol $\sigma$, common correlation $\rho$). Then (16.1) collapses to

$$\sigma_p^2 = \sigma^2\Big[\tfrac1N + \tfrac{N-1}{N}\rho\Big] \;\xrightarrow[N\to\infty]{}\; \sigma^2\rho,
\qquad \sigma_p \approx \sigma\sqrt{\rho}. \tag{16.3}$$

Under a correlation gap $\rho\to\rho'$ (vols fixed), the VaR ratio is

$$\frac{\mathrm{VaR}(\rho')}{\mathrm{VaR}(\rho)} = \sqrt{\frac{\rho'}{\rho}}. \tag{16.4}$$

**The key inequality.** Equities already co-move: $\rho\approx0.6$ in calm markets, and $\rho'\le 1$. Hence
the *entire* achievable correlation effect is capped:

$$\frac{\mathrm{VaR}(\rho')}{\mathrm{VaR}(\rho)} \le \sqrt{\frac{1}{\rho}} = \sqrt{\frac{1}{0.6}} \approx 1.29. \tag{16.5}$$

So correlation-going-to-1 can raise portfolio VaR by **at most ~29%**, while a typical crisis vol surge of
$g\approx3$ raises it **200%**. The two effects are not comparable: one is multiplicative and unbounded in
$g$, the other is $\sqrt{\cdot}$ and bounded by $1/\sqrt{\rho}$.

**Empirical confirmation** (COVID, equal-weight risk book, OOS):

| Scenario | VaR width | 99% breach |
|---|---|---|
| Full-sample cov (calm vol + calm corr) | 3.07% | 25.6% |
| Crisis **correlations only** (calm vol) | 3.16% | 23.1% |
| Crisis cov (crisis vol + crisis corr) | 8.43% | **7.7%** |

The correlation gap alone moves the breach 25.6%→23.1% (matching the ≤1.29× bound in (16.5)); crisis
**volatility** carries the entire coverage fix (→7.7%). $\blacksquare$

**Consequence.** The honest lever for crisis under-coverage is *dynamic volatility scaling* — which
Meridian already applies (HAR-lev+IV forecast + the IV early-warning gate) — not a crisis-correlation
overlay. This is why the ThresholdShockNetwork is retained as a diagnostic, not shipped as a default.

---

## Part VII — Leading indicators and actionable overlay

### 17. IV term-structure early-warning gate (`iv_earlywarning.py`)

Signal: term-structure inversion $\text{VIX9D}_t/\text{VIX3M}_t > 1$ (near-term fear exceeding
medium-term). Evaluated as a *leading* classifier of realized-vol stress onset (5-day RV crossing its
trailing-1y 85th percentile):

- **Sensitivity** (onsets preceded by inversion): **70%**.
- **Precision** (inversions followed by an onset ≤15d): **52%** vs a **13%** base rate ⟹ ~4× lift.
- **Lead time**: median **~6 trading days**, fired at-or-before onset in 100% of cases.

A backward HMM/percentile labeler lags by construction; the forward-looking IV gate leads — corroborated
by research (Albers 2025 DM-beats-HAR; Cavicchioli 2025 on HMM lag).

### 18. De-risking overlay and the Sortino ratio (`earlywarning_overlay.py`)

Position rule: hold exposure $1$, cut to $1-h$ for the next day when inverted (causal). Net of cost,
scored by downside-risk-adjusted return:

$$\text{Sortino} = \frac{\mathbb E[r]-r_f}{\sigma_{\text{down}}},\qquad
 \sigma_{\text{down}} = \sqrt{\mathbb E\big[\min(r,0)^2\big]}.$$

**Validated (SPY):** $h=0.5$ cuts max drawdown $-35.7\%\to-26.2\%$ (−27%), raises Sortino
$0.93\to0.98$, for ~2.5pp/yr of return. $h=1.0$ (full exit) **underperforms** (Sortino 0.78) — so a
*moderate* haircut only. Surfaced as a validated optional rule in the thesis, framed as systematic risk
management, not advice.

---

## Part VIII — The world model

### 19. Deep state-space generative model (`worldmodel.py`)

A latent-variable model of the joint multivariate market: latent state $z_t\in\mathbb R^{K}$, learned
stochastic transition (the "forward model"), and a heavy-tailed emission.

- **Inference** $q(z_t\mid r_{\le t})$: a GRU filter.
- **Transition** $p(z_t\mid z_{t-1}, c_{t-1})$: Gaussian, **sign-conditioned** on $c=\text{sign}(r)\cdot|r|$
 (the leverage channel) plus an exogenous do-hook $u_t$ (the structural intervention slot).
- **Emission** $p(r_t\mid z_t)$: multivariate **Student-$t$** with a **low-rank + diagonal** covariance
 (latent stochastic volatility with a factor structure):

$$r_t \sim t_\nu\big(0,\ \Sigma(z_t)\big),\qquad \Sigma(z_t)=D(z_t) + L(z_t)L(z_t)^\top,$$

with $D$ diagonal and $L\in\mathbb R^{N\times F}$ ($F\!\ll\!N$ factors). Student-$t$ emission gives fat
daily tails; a Gaussian *transition* keeps long rollouts from detonating (DreamerV3 stabilization).

### 20. Training objective (ELBO) and the Woodbury identity

Maximize the evidence lower bound:

$$\log p(R) \ \ge\ \underbrace{\mathbb E_{q}\big[\log p(R\mid z)\big]}_{\text{reconstruction}}
 - \underbrace{\mathrm{KL}\big(q(z\mid R)\,\|\,p(z)\big)}_{\text{regularizer}} \ =\ \text{ELBO},$$

with KL annealing $\beta:0\to1$ to avoid posterior collapse. The Student-$t$ NLL needs the quadratic form
$r^\top\Sigma^{-1}r$ and $\log\det\Sigma$. Computing these for $\Sigma=D+LL^\top$ naively is $O(N^3)$; the
**Woodbury identity** and **matrix determinant lemma** make it $O(NF^2)$:

$$(D+LL^\top)^{-1} = D^{-1} - D^{-1}L\,(I_F + L^\top D^{-1}L)^{-1}L^\top D^{-1},$$
$$\log\det(D+LL^\top) = \log\det(I_F + L^\top D^{-1}L) + \sum_i \log D_{ii}.$$

### 21. What the world model is validated to do (`train_worldmodel.py`, `world_stabilize.py`, `world_calib_validate.py`)

A world model earns the name by reproducing dynamics, not by point accuracy. **Stylized facts (Cont 2001)**
from the free-run simulation, measured *multi-path* (median over 8 paths, temperature 1.0):

| Statistic | Real (full history) | World model | i.i.d. Gaussian |
|---|---|---|---|
| Excess kurtosis (fat tails) | 13.3 | 7.9 | ~0 |
| ACF$|r|$ lag 1 (vol clustering) | 0.30 | 0.32 | ~0 |
| Leverage sign corr$(r,|r_{+1}|)$ | − | − (correct sign) | ~0 |

**Free-run stability.** A temperature sweep of the transition noise shows kurtosis falls $7.9\to5.3$ as
temp $1.0\to0.5$, but ACF$|r|$ falls **faster** ($0.32\to0.25$) — so temperature 1.0 is optimal and lowering
it degrades clustering. The earlier "kurtosis 26" was a **single-path artifact**; the median over paths is
7.9. The wired scenario averages 3,000 paths, resolving per-path variance.

**Joint 1-day VaR** (`world_calib_validate.py`): raw WM joint VaR is Kupiec-calibrated recently (1.6%
breach, $p=0.21$), looser over the full 2008–2020 stress test (~3.1%). A hybrid rescale to EWMA marginals
was tested and **rejected** (2.2%, worse — EWMA under-disperses in calm). **What-if:** a structural
$u_t$-shock yields coherent flight-to-quality (equities −, Treasuries/IG +). Honest scope: a coherent
short-horizon *joint simulator* + intervention engine; the single-book calibrated tail stays EVT-GPD.

---

## Part IX — Production robustness (`robustness_edge_tests.py`)

### 22. Numerical guards

- **Log-floor:** every $\ln(\cdot)$ of a variance/semivariance uses $\ln(x+\varepsilon)$, $\varepsilon=10^{-12}$,
 so a zero-print (halted-flat / one-sided limit) day yields a finite $\ln\varepsilon\approx-27.6$, never
 $-\infty$/NaN.
- **GPD guardrail:** the MLE is wrapped; on non-convergence or degenerate $(\xi,\beta)$ it falls back to the
 empirical quantile, and the standardized quantile is floored at $\Phi^{-1}(q)$ so a deep-quiet window
 cannot collapse VaR below Gaussian before a regime shift.
- **Freshness audit:** exogenous feeds carry $\text{age} = t_{\text{now}}-t_{\text{last print}}$; if
 $\text{age}>$ threshold the signal is marked stale and the early-warning is withheld (currently live: the
 VIX9D/3M feed is flagged stale).

---

## Appendix — Master claim ledger

| # | Claim | Math | Evidence (script) | Status |
|---|---|---|---|---|
| 1 | Garman–Klass > close-to-close efficiency | §1 | features.py | ✅ standard |
| 2 | HAR + downside semivariance beats HAR | §2 | shar_validate.py | ✅ +0.76% |
| 3 | Jensen correction removes level-forecast bias | §3 | engine.py | ✅ applied |
| 4 | QLIKE is proxy-robust; MAE is not | §4 | evalproto.py | ✅ Patton 2011 |
| 5 | Matched IV + TS + VRP beats HAR | §7 | benchmark_exog.py | ✅ +10.3%, p<0.001 |
| 6 | Market-RV edge is lag-invariant (no leakage) | §8 | leakage_mktrv_test.py | ✅ +0.54≈+0.56% |
| 7 | EVT-GPD calibrates the 99% tail | §9,11 | es_backtest.py | ✅ Kupiec pass, Acerbi≈0 |
| 8 | Cornish–Fisher fails in fat tails | §10 | cf_vs_evt.py | ✅ non-monotone 9% |
| 9 | FHS is not a bottleneck | §12 | analyze.py | ✅ 3.45 ms |
| 10 | Ledoit–Wolf GMV cuts portfolio vol | §13 | analyze.py | ✅ −14% |
| 11 | Threshold-VAR loses to linear GIRF OOS | §15 | validate_network.py | ✅ DM p=0.90/0.99 |
| 12 | Vol, not correlation, carries portfolio tail risk | §16 | validate_network.py | ✅ proof + data |
| 13 | IV term-structure leads stress onset | §17 | iv_earlywarning.py | ✅ 70%/52%/~6d |
| 14 | Moderate de-risk overlay improves Sortino | §18 | earlywarning_overlay.py | ✅ 0.93→0.98 |
| 15 | World model reproduces stylized facts | §21 | train/stabilize/calib | ✅ multi-path |
| 16 | World-model VaR calibrated in aggregate, not tighter than EVT | §21 | world_calib_validate.py | ◑ honest boundary |
| 17 | Numerical guards hold on edge cases | §22 | robustness_edge_tests.py | ✅ all pass |

Every row links a formula (this document) to a re-runnable script and a measured result. No claim here is
asserted without a committed test behind it.
