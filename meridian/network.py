"""Network shock-propagation — Build #1 from the EWM research synthesis.

Upgrades the single-beta "what-if" to a MULTI-ENTITY network simulation using a VAR +
GENERALIZED impulse response (Pesaran-Shin 1998, order-invariant). Given a shock to one
entity, it propagates through the estimated dynamic network to every other entity —
answering "who gets hit and how hard", including indirect paths, not just one beta.

Two objects, each used for what it is FOR:
  - generalized FEVD  → the CONNECTEDNESS table (variance shares; net transmitter/receiver).
  - cumulative GIRF   → the SHOCK-RESPONSE (return magnitudes; how a δ move in j moves i).

Honest scope: a linear VAR first-order propagation. Real crises are nonlinear (betas and
correlations jump); magnitudes are validated empirically (scripts/network_scenario.py)
before being presented as more than directional.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.api import VAR

warnings.simplefilter("ignore", ValueWarning)      # tz-naive daily index has no freq; fine
EPS = 1e-12


def fit_var(returns: pd.DataFrame, lag: int = 2):
    return VAR(returns).fit(lag)


def girf_cumulative(res, H: int = 10) -> np.ndarray:
    """Cumulative generalized impulse response over horizon H.

    G[i, j] = cumulative response of variable i to a ONE-UNIT (size = 1.0 in the shocked
    variable's own units) generalized shock in variable j. At impact this equals
    Sigma_ij / Sigma_jj (the covariance-implied co-move); later horizons add the VAR's
    dynamic propagation through the whole network.
    """
    Sigma = res.sigma_u.to_numpy() if hasattr(res.sigma_u, "to_numpy") else np.asarray(res.sigma_u)
    N = Sigma.shape[0]
    Theta = res.ma_rep(maxn=H)                       # (H+1, N, N)
    G = np.zeros((N, N))
    for h in range(H + 1):
        # response of all i to unit generalized shock in each j: Theta_h @ Sigma / Sigma_jj
        G += (Theta[h] @ Sigma) / np.diag(Sigma)[None, :]
    return G                                          # column j = responses to a shock in j


def propagate(returns: pd.DataFrame, source: str, shock: float,
              lag: int = 2, H: int = 10) -> pd.Series:
    """Estimate each entity's cumulative response to a `shock` (e.g. -0.05) in `source`."""
    res = fit_var(returns, lag)
    names = list(returns.columns)
    j = names.index(source)
    G = girf_cumulative(res, H)
    # scale unit response to the requested shock size (shock is in source's return units)
    resp = G[:, j] * (shock / (G[j, j] if abs(G[j, j]) > EPS else 1.0))
    return pd.Series(resp, index=names)


def regime_sigmas(returns: pd.DataFrame, res, stress_q: float = 0.90):
    """Split VAR residuals into CALM vs STRESS days and return (Sigma_calm, Sigma_stress, frac_stress).

    Stress = days whose cross-sectional average |residual| is in the top `1-stress_q` (default top 10%),
    i.e. broad, large-magnitude co-shocks — the days where correlations empirically gap toward 1
    (Ang-Chen 2002; Longin-Solnik 2001). We estimate a SEPARATE residual covariance per regime but keep
    ONE shared VAR for the dynamics: covariance needs far fewer samples than a full second VAR, so this
    captures the reflexive correlation surge without the overfitting a full Threshold-VAR incurs on the
    handful of true crisis days in 2007-2026 (the research's explicit estimation caution)."""
    U = pd.DataFrame(res.resid, columns=returns.columns) if not isinstance(res.resid, pd.DataFrame) else res.resid
    U = U.dropna()
    sev = U.abs().mean(axis=1)                                  # daily broad-shock severity
    thr = sev.quantile(stress_q)
    stress = sev >= thr
    Uc, Us = U[~stress].to_numpy(), U[stress].to_numpy()
    # shrink toward the pooled covariance so the stress matrix (few obs) stays well-conditioned
    Sig_all = np.cov(U.to_numpy(), rowvar=False)
    def _cov(A, w=0.25):
        S = np.cov(A, rowvar=False) if len(A) > A.shape[1] + 2 else Sig_all
        return (1 - w) * S + w * Sig_all
    return _cov(Uc), _cov(Us), float(stress.mean())


def _girf_from_sigma(Sigma, Theta, H):
    N = Sigma.shape[0]; G = np.zeros((N, N))
    for h in range(H + 1):
        G += (Theta[h] @ Sigma) / np.diag(Sigma)[None, :]
    return G


def regime_propagate(returns: pd.DataFrame, source: str, shock: float, threshold: float = -0.03,
                     lag: int = 2, H: int = 10, stress_q: float = 0.90):
    """Reflexive shock propagation: use the STRESS covariance network when the shock crosses a
    downside `threshold` (default -3%), else the CALM network. Returns (response: Series, regime: str).

    This is the honest, estimable form of a Threshold-VAR for daily data: one VAR for dynamics, a
    regime-switched residual covariance for the correlation surge.

    VALIDATION OUTCOME (scripts/reflexive_validate.py, 19 assets, 147 real crisis days 2007-2026):
    regime/stress betas did NOT beat full-sample betas at predicting REALIZED crash-day co-moves
    out-of-sample (−2% broad-severity, −5.8% downside-conditioned; sign-test p=1.00). The
    correlation-surge stylized fact is real, but the full-sample beta already prices it (crashes are in
    the sample), and re-estimating on the stress subset costs more estimation variance than it recovers.
    The linear model does NOT systematically under-predict, so review-claim "linear severely
    underestimates contagion" was not supported. CONSEQUENCE: `propagate` (linear GIRF) stays the
    default; this function is retained as a DIAGNOSTIC to expose the calm-vs-stress correlation split,
    not as a validated point-forecast upgrade. Do not present its magnitudes as more accurate."""
    res = fit_var(returns, lag)
    names = list(returns.columns); j = names.index(source)
    Theta = res.ma_rep(maxn=H)
    Sig_calm, Sig_stress = regime_sigmas(returns, res, stress_q)[:2]
    stressed = shock <= threshold
    Sigma = Sig_stress if stressed else Sig_calm
    G = _girf_from_sigma(Sigma, Theta, H)
    resp = G[:, j] * (shock / (G[j, j] if abs(G[j, j]) > EPS else 1.0))
    return pd.Series(resp, index=names), ("stress" if stressed else "calm")


def crisis_calm_cov(returns: pd.DataFrame, crisis_windows: list[tuple[str, str]],
                    shrink: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """Pooled CRISIS covariance (union of the given date windows) and CALM covariance (everything else).
    Both Ledoit-style shrunk toward the pooled full-sample cov for conditioning. Grounds the crisis
    footprint in ACTUAL stress blocks (2008/2020/2022), per the review's 'historical stress grounding'."""
    idx = returns.index
    mask = pd.Series(False, index=idx)
    for a, b in crisis_windows:
        mask |= (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(b))
    Rc, Rk = returns[mask.values], returns[~mask.values]
    Sig_all = np.cov(returns.to_numpy(), rowvar=False)
    def _cov(R):
        if len(R) <= R.shape[1] + 2:
            return Sig_all
        S = np.cov(R.to_numpy(), rowvar=False)
        return (1 - shrink) * S + shrink * Sig_all
    return _cov(Rk), _cov(Rc)                                  # (calm_cov, crisis_cov)


class ThresholdShockNetwork:
    """Non-linear, regime-conditional shock propagation (external review #3, historical-grounding form).

    Keeps the linear generalized-IRF as the CALM baseline and switches to a CRISIS covariance when the
    impulse crosses a downside `stress_threshold`, so cross-asset betas gap up under stress. At impact
    the response of asset i to a shock in the source is beta_i = Cov_i,src / Cov_src,src, scaled by the
    shock — the Pesaran-Shin generalized impact, evaluated on the regime-appropriate covariance.

    VALIDATION OUTCOME (scripts/validate_network.py, leave-one-crisis-out, chronological):
      * Point co-move PROPAGATION: the crisis matrix did NOT beat linear GIRF out-of-sample on the
        held-out COVID-2020 / 2022 crises (−0.7% / −1.1% RMSE, one-sided DM p=0.90 / 0.99).
      * Portfolio stress-VaR: the full-sample covariance DOES badly under-cover a portfolio in a severe
        crisis (25.6% breach in COVID vs a 1% target), and the crisis covariance fixes it (7.7%) — BUT
        the decomposition shows this is ENTIRELY a crisis-VOLATILITY effect, not the correlation gap:
        crisis-correlations-with-calm-vols barely moves coverage (25.6% → 23.1%). The review's specific
        "correlations gap to 1.0" mechanism is real as a stylized fact but is NOT where risk is
        under-estimated; the volatility LEVEL is — and that is already handled by the dynamic vol
        forecast (HAR-lev+IV) and the IV term-structure early-warning gate (exog.term_structure_warning).
    CONSEQUENCE: this class is retained as a DIAGNOSTIC / scenario tool, not a validated default upgrade.
    `propagate` (linear GIRF) remains the propagation default. Do not present its magnitudes as more
    accurate — the honest lever for crisis risk is dynamic volatility, which the model already applies."""

    def __init__(self, asset_names: list, calm_cov: np.ndarray, crisis_cov: np.ndarray,
                 stress_threshold: float = -0.03):
        self.asset_names = list(asset_names)
        self.calm_cov = np.asarray(calm_cov)
        self.crisis_cov = np.asarray(crisis_cov)
        self.stress_threshold = stress_threshold

    def propagate_shock(self, target_asset: str, magnitude: float) -> dict:
        if target_asset not in self.asset_names:
            raise ValueError(f"Asset '{target_asset}' missing from network universe.")
        idx = self.asset_names.index(target_asset)
        cov = self.crisis_cov if magnitude <= self.stress_threshold else self.calm_cov
        sigma_ii = cov[idx, idx] + EPS
        impact = cov[:, idx] / sigma_ii * magnitude            # beta_i * shock (generalized impact)
        return {self.asset_names[i]: float(impact[i]) for i in range(len(self.asset_names))}

    def regime_of(self, magnitude: float) -> str:
        return "crisis" if magnitude <= self.stress_threshold else "calm"


def connectedness(returns: pd.DataFrame, lag: int = 2, H: int = 10):
    """Generalized-FEVD net transmitter/receiver table (%) — the connectedness object."""
    res = fit_var(returns, lag)
    Sigma = res.sigma_u.to_numpy() if hasattr(res.sigma_u, "to_numpy") else np.asarray(res.sigma_u)
    N = Sigma.shape[0]
    Theta = res.ma_rep(maxn=H); sig = np.sqrt(np.diag(Sigma))
    num = np.zeros((N, N)); den = np.zeros(N)
    for h in range(H):
        Th = Theta[h]; TS = Th @ Sigma
        for i in range(N):
            den[i] += Th[i] @ Sigma @ Th[i]
            for k in range(N):
                num[i, k] += (TS[i, k] ** 2) / (sig[k] ** 2)
    C = num / den[:, None]
    C = C / C.sum(1, keepdims=True) * 100
    frm = C.sum(1) - np.diag(C); to = C.sum(0) - np.diag(C)
    return pd.DataFrame({"to": to, "from": frm, "net": to - frm}, index=returns.columns)
