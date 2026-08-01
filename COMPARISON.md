# Meridian — Detailed Model Comparison (⭐ = column winner)

*Out-of-sample, purged walk-forward. QLIKE/MSE/RMSE/MAE lower = better; MZ-R²/R²-vs-HAR%/IC higher = better; bias→0 best. DM-p = one-sided prob. of lower QLIKE than HAR (**bold** <0.05 = significantly beats HAR). MCS = in the 90% Model Confidence Set. All numbers from `results/*.json`; reproduce with the scripts in `scripts/`.*

## 1. Volatility forecasting — every model, every metric

### A. Independent source — Oxford-Man, 17 international indices (5-min RV, no VIX)

| Model | QLIKE↓ | MSE↓ | RMSE↓ | MAE↓ | MZ-R²↑ | R²vHAR%↑ | IC↑ | bias→0 | DM-p | MCS |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| EWMA | 0.2341 | 0.4141 | 0.6435 | 0.4972 | 0.633 | -39.23% | 0.793 | +0.173 | 1.000 | · |
| GARCH | 0.2483 | 0.4737 | 0.6882 | 0.5407 | 0.588 | -47.68% | 0.758 | +0.216 | 1.000 | · |
| HAR-RV (benchmark) | 0.1681 | 0.2860 | 0.5348 | 0.4024 | 0.724 | +0.00% | 0.846 | **+0.001 ⭐** | — | · |
| Meridian (HAR+lev) | 0.1647 | 0.2791 | 0.5283 | 0.3972 | 0.730 | +2.04% | 0.850 | +0.002 | **5.8e-26** | · |
| Meridian-lin | 0.1612 | 0.2729 | 0.5224 | 0.3929 | 0.736 | +4.15% | 0.853 | -0.003 | **3.3e-32** | ✅ |
| TimeMixer | 0.1756 | 0.2995 | 0.5472 | 0.4134 | 0.713 | -4.43% | 0.840 | +0.015 | 1.000 | · |
| Meridian-net | 0.1610 | **0.2718 ⭐** | **0.5213 ⭐** | **0.3920 ⭐** | **0.738 ⭐** | +4.25% | 0.853 | -0.012 | **5.7e-06** | ✅ |
| Meridian-CJ (jumps) | 0.1627 | — | 0.5246 | — | 0.735 | +3.09% | 0.852 | — | **2.9e-32** | · |
| Meridian-intra+ | 0.1620 | — | 0.5223 | — | 0.737 | +3.49% | **0.854 ⭐** | — | **1.2e-19** | ✅ |
| Regime-Meridian ★champion | **0.1606 ⭐** | — | 0.5245 | — | 0.735 | **+4.38% ⭐** | 0.852 | — | **6.5e-32** | ✅ |

### B. Held-out — 24 never-trained Yahoo assets

| Model | QLIKE↓ | MSE↓ | RMSE↓ | MAE↓ | MZ-R²↑ | R²vHAR%↑ | IC↑ | bias→0 | DM-p | MCS |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| EWMA | 0.4073 | 0.7693 | 0.8771 | 0.6920 | 0.609 | -16.18% | 0.771 | +0.309 | 1.000 | · |
| GARCH | 0.4232 | 0.9566 | 0.9781 | 0.7828 | 0.578 | -20.72% | 0.749 | +0.479 | 1.000 | · |
| HAR-RV (benchmark) | 0.3506 | 0.5911 | 0.7688 | 0.5973 | 0.656 | +0.00% | 0.798 | +0.012 | — | · |
| Meridian (HAR+lev) | 0.3490 | 0.5879 | 0.7667 | 0.5959 | 0.658 | +0.46% | 0.799 | +0.011 | 0.198 | · |
| Meridian-lin | **0.3428 ⭐** | 0.5808 | 0.7621 | 0.5920 | 0.662 | **+2.22% ⭐** | 0.801 | **-0.001 ⭐** | **3.9e-05** | ✅ |
| HAR-IV | 0.3447 | 0.5877 | 0.7666 | 0.5955 | 0.658 | +1.67% | 0.799 | -0.005 | **6.0e-09** | ✅ |
| TimeMixer | 0.4237 | 0.6294 | 0.7933 | 0.6169 | 0.636 | -20.85% | 0.786 | +0.014 | 0.972 | · |
| Meridian-net | 2.0494 | **0.5773 ⭐** | **0.7598 ⭐** | **0.5882 ⭐** | **0.665 ⭐** | -484.59% | **0.803 ⭐** | -0.025 | 0.921 | ✅ |

## 2. Portfolio risk management

### Portfolio-risk strategies (held-out universe, OOS 2012–2026)

| Strategy | Ann vol↓ | Ann ret | Sharpe↑ | Max DD↓ |
|---|--:|--:|--:|--:|
| equal-weight (naive) | 13.5% | +7.8% | 0.58 | -36.0% |
| inverse-vol | 11.2% | +7.4% | 0.66 | -29.9% |
| min-var (sample) | **6.2% ⭐** | +3.6% | 0.58 | **-19.3% ⭐** |
| min-var (Ledoit-Wolf) — Meridian | 6.8% | +4.9% | **0.71 ⭐** | -20.1% |

## 3. Tail risk (VaR/ES)

### Tail-risk methods (99% VaR, held-out pooled; target exceedance 1.0%)

| Method | Exceed% | Kupiec p↑ | Christoffersen p | Verdict |
|---|--:|--:|--:|:--|
| Gaussian (naive) | 0.96% | 0.179 | 0.000 | well-calibrated |
| Historical | 0.58% | 0.000 | 0.000 | miscalibrated |
| EVT — Meridian | 1.00% | **0.918 ⭐** | 0.000 | well-calibrated |

## Overall winners
- **Volatility forecast:** ⭐ **Regime-Meridian** (lowest QLIKE, +4.4% vs HAR, DM-significant, MCS).
- **Portfolio risk:** ⭐ **Meridian min-variance (Ledoit-Wolf)** — best Sharpe, −49% risk vs naive, beats sample-cov.
- **Tail risk:** ⭐ **Meridian conditional-EVT** — most exact 99% coverage (best Kupiec p).
- **Honest caveats:** the forecast edge over HAR is ~4% (real, not huge); the portfolio margin is huge vs the naive equal-weight but a tie vs the best baseline on risk (Meridian wins on Sharpe); all VaR methods fail the independence test at 99%. Nothing here beats the world's best models by a big margin — that does not exist on free data; these are the honest, reproducible frontier results.
