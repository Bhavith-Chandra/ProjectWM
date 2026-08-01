"""Unified analysis pipeline — one call runs the whole Meridian stack on any entity and returns
a structured, honest report: volatility forecast, regime, tail risk (VaR/ES), Monte-Carlo scenario
simulation, market beta, and a generated THESIS. Numbers come only from calibrated modules.

Monte Carlo = FILTERED HISTORICAL SIMULATION (the research-endorsed honest scenario engine): draw
standardized historical returns, scale by the current volatility forecast, roll forward → a full
distribution of future outcomes (VaR/ES, expected range, drawdown probability). For coherent
multi-asset joint scenarios use the world-model core (meridian/worldmodel.py); FHS here is the fast,
per-query, non-parametric baseline.

Portfolio path adds covariance + minimum-variance optimization (the −49%-risk module).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from meridian.engine import analyze as engine_analyze, resolve
from meridian.data import fetch_yahoo
from meridian.features import realized_variance
from meridian import connect

TRADING = 252


def monte_carlo(ret: np.ndarray, vol_now_ann: float, horizon: int = 21, n_paths: int = 10000, seed: int = 0):
    """Filtered historical simulation of horizon-day cumulative return. Returns a summary dict."""
    r = ret[np.isfinite(ret)]
    sd = np.std(r[-252:]) if len(r) >= 252 else np.std(r)
    z = r / (sd + 1e-9)                                     # standardized shocks (fat-tailed, real)
    daily = vol_now_ann / np.sqrt(TRADING)                 # scale to the current vol forecast
    rng = np.random.RandomState(seed)
    sims = rng.choice(z, size=(n_paths, horizon)) * daily  # FHS paths
    cum = sims.sum(1)                                       # horizon cumulative log-return
    q = np.percentile(cum, [1, 5, 25, 50, 75, 95, 99])
    return {"horizon_days": horizon, "median_pct": float(q[3] * 100),
            "p05_pct": float(q[1] * 100), "p95_pct": float(q[5] * 100),
            "var95_pct": float(q[1] * 100), "es95_pct": float(cum[cum <= q[1]].mean() * 100),
            "var99_pct": float(q[0] * 100), "prob_loss_gt_10pct": float((cum < -0.10).mean() * 100),
            "prob_up": float((cum > 0).mean() * 100)}


def full_analysis(entity: str, conn: connect.Connection | None = None,
                  horizon: int = 21, with_news: bool = False) -> dict:
    """Run the full stack on one entity. Returns a structured report (all numbers from modules)."""
    a = engine_analyze(entity)
    if isinstance(a, dict):                                 # need-data fallback
        return {"ok": False, "message": a.get("message", "no data")}
    rvf = realized_variance(fetch_yahoo(a.resolved["symbol"]))
    mc = monte_carlo(rvf["ret"].dropna().to_numpy(), a.vol_fc_1d_ann, horizon=horizon)
    out = {"ok": True, "entity": a.resolved, "as_of": a.last_date, "last_price": a.last_price,
           "volatility": {"now_ann_pct": round(a.vol_now_ann * 100, 1),
                          "forecast_1d_ann_pct": round(a.vol_fc_1d_ann * 100, 1),
                          "forecast_1w_ann_pct": round(a.vol_fc_5d_ann * 100, 1),
                          "percentile": round(a.vol_pct * 100)},
           "forecast_model": a.forecast_model, "iv_early_warning": a.iv_early_warning,
           "regime": a.regime, "market_beta": None if not np.isfinite(a.beta_mkt) else round(a.beta_mkt, 2),
           "tail_1d": {"var99_pct": round(a.var99 * 100, 1), "es99_pct": round(a.es99 * 100, 1)},
           "monte_carlo": {k: round(v, 1) for k, v in mc.items()},
           "trend": {"ret_1m_pct": None if not np.isfinite(a.ret_1m) else round(a.ret_1m * 100, 1),
                     "ret_12m_pct": None if not np.isfinite(a.ret_12m) else round(a.ret_12m * 100, 1),
                     "drawdown_pct": round(a.drawdown * 100, 1)}}
    if with_news and conn is not None:
        out["news"] = conn.news(a.resolved["name"] + " stock")[:6]
    out["thesis"] = _thesis(out)
    return out


def _thesis(o: dict) -> str:
    """Generate a readable thesis from the module outputs (every number traces to a module)."""
    v, mc, t = o["volatility"], o["monte_carlo"], o["trend"]
    e = o["entity"]
    L = [f"**{e['name']} ({e['symbol']}) — Meridian thesis** (as of {o['as_of']}, last {o['last_price']:,.2f})",
         f"- **Regime:** {o['regime']}; volatility {v['now_ann_pct']}% annualized "
         f"({v['percentile']}th percentile of its own history), forecast {v['forecast_1w_ann_pct']}% next week.",
         f"- **{mc['horizon_days']}-day outlook (Monte-Carlo, 10k paths):** median {mc['median_pct']:+.1f}%, "
         f"a 90% range of {mc['p05_pct']:+.1f}% to {mc['p95_pct']:+.1f}%; "
         f"{mc['prob_up']:.0f}% chance of a positive return.",
         f"- **Downside:** 1-day 99% VaR {o['tail_1d']['var99_pct']:.1f}% / ES {o['tail_1d']['es99_pct']:.1f}%; "
         f"over {mc['horizon_days']} days, {mc['prob_loss_gt_10pct']:.1f}% probability of a >10% drawdown.",
         f"- **Context:** {t['ret_1m_pct']:+.1f}% past month, {t['drawdown_pct']:+.1f}% from its high"
         + (f"; market beta {o['market_beta']}." if o['market_beta'] is not None else "."),
         *( [f"- **⚠ Early-warning:** VIX term structure is **inverted** "
             f"(VIX9D/VIX3M = {o['iv_early_warning'].get('ratio9d_3m')}) — the market is pricing "
             f"near-term stress. This signal leads realized-vol stress onset by a median ~6 trading "
             f"days (70% of onsets, 52% precision; scripts/iv_earlywarning.py).",
             "- **Optional de-risk rule (validated):** halving equity exposure while the structure is "
             "inverted historically cut max drawdown ~27% (−35.7%→−26.2%) and raised Sortino "
             "(0.93→0.98) for ~2.5pp/yr of return foregone — a *moderate* haircut only; a full exit "
             "underperformed (scripts/earlywarning_overlay.py). A systematic rule, not advice."]
            if o.get("iv_early_warning", {}).get("inverted") else
            [f"- **Note:** the VIX term-structure feed is stale "
             f"({o['iv_early_warning'].get('age_days')}d old) — early-warning signal withheld to avoid "
             f"acting on decayed data."] if o.get("iv_early_warning", {}).get("stale") else [] ),
         ("- **Forecaster:** implied-vol-augmented HAR (matched IV family + VIX term structure) — "
          "the +10.3%-over-HAR model validated OOS (DM p<0.001)."
          if o.get("forecast_model") == "HAR-lev+IV" else
          "- **Forecaster:** HAR-leverage (price-only) — no free matched implied-vol index for this "
          "asset, so the IV lever isn't available here."),
         "*Every figure is a calibrated-module output (no-lookahead). A measurement-and-forecast read, "
         "not investment advice — and not a claim to predict the market, only to quantify its risk.*"]
    return "\n".join(L)


CRISIS_WINDOWS = (("2008_GFC", "2008-09-01", "2008-11-30"),
                  ("2020_COVID", "2020-03-01", "2020-04-30"))


def crisis_stress_test(rets: dict, syms: list, weights: np.ndarray,
                       windows=CRISIS_WINDOWS, min_obs: int = 15) -> dict:
    """Historical VOLATILITY stress test: scale each asset's vol to its GFC-2008 / COVID-2020 level
    while PRESERVING the current correlation structure, then re-price the book's 99% VaR.

    Why vol-only (not crisis correlations): our own decomposition (scripts/validate_network.py) showed
    crisis correlations barely move portfolio coverage (25.6%→23.1%); crisis VOLATILITY carries it
    (→7.7%). So this scales the honest lever and does NOT fabricate a crisis correlation map.

    POST-GFC INCEPTION HANDLING (the real fix): an asset that IPO'd after a crisis window has no data
    there. Rather than emit NaN, we PROXY its crisis vol = current_vol × (median vol-surge of the assets
    that ARE present in that window) and flag it, so a 2021-listed name still gets a defensible stress
    vol instead of poisoning the portfolio variance."""
    Z99 = 2.326
    recent = pd.DataFrame({s: rets[s] for s in syms}).dropna().iloc[-252:]
    cur_vols = recent.std(ddof=1).to_numpy()
    cur_cov = recent.cov().to_numpy()
    dinv = np.diag(1.0 / (cur_vols + 1e-12)); corr = dinv @ cur_cov @ dinv
    cur_var99 = Z99 * float(np.sqrt(weights @ cur_cov @ weights))
    out = {"current_var99_pct": round(cur_var99 * 100, 2), "method": "Gaussian 99% on stressed vol; "
           "current correlations preserved", "scenarios": {}, "proxied": {}}
    for label, a, b in windows:
        cvol = np.full(len(syms), np.nan); present = np.zeros(len(syms), bool)
        for i, s in enumerate(syms):
            wv = rets[s].loc[a:b].dropna()
            if len(wv) >= min_obs:
                cvol[i] = wv.std(ddof=1); present[i] = True
        if present.sum() == 0:
            continue
        surge = float(np.median(cvol[present] / (cur_vols[present] + 1e-12)))   # market vol-surge
        proxied = [syms[i] for i in range(len(syms)) if not present[i]]
        for i in range(len(syms)):
            if not present[i]:
                cvol[i] = cur_vols[i] * surge                                   # honest proxy
        D = np.diag(cvol); stressed_cov = D @ corr @ D
        s_var99 = Z99 * float(np.sqrt(weights @ stressed_cov @ weights))
        out["scenarios"][label] = {"stressed_var99_pct": round(s_var99 * 100, 2),
                                   "multiplier": round(s_var99 / (cur_var99 + 1e-12), 2),
                                   "vol_surge": round(surge, 2)}
        if proxied:
            out["proxied"][label] = proxied
    return out


def world_portfolio_scenario(entities: list[str], weights=None, horizon: int = 1,
                             n_paths: int = 3000) -> dict:
    """JOINT multi-asset scenario from the WORLD-MODEL core — coherent cross-asset return paths sampled
    from the learned latent dynamics (the world model's unique capability), vs the per-asset FHS which
    ignores learned joint structure. Filters the current joint latent state from recent returns of the
    FULL trained universe, then emits/rolls the portfolio's assets jointly.

    Scope (honest): only for portfolios whose assets are ALL in the trained universe (major ETFs); else
    available=False and the caller uses FHS. Horizon: the world model is 1-day calibrated / multi-day
    directional (WORLDMODEL_CORE.md), so default horizon=1 for a calibrated joint VaR."""
    try:
        import torch
        from meridian.worldmodel import load_pretrained, WM_SCALE
    except Exception:
        return {"available": False, "reason": "torch unavailable"}
    wm, uni = load_pretrained()
    if wm is None:
        return {"available": False, "reason": "no trained world-model checkpoint (run scripts/train_worldmodel.py)"}
    syms = []
    for e in entities:
        r = resolve(e)
        if not r or not r.get("symbol"):
            return {"available": False, "reason": f"could not resolve {e}"}
        syms.append(r["symbol"])
    outside = [s for s in syms if s not in uni]
    if outside:
        return {"available": False, "reason": f"outside trained universe: {outside}"}
    try:
        cols = {a: np.log(fetch_yahoo(a)["adjclose"]).diff() for a in uni}
        R = pd.DataFrame(cols).dropna().iloc[-250:]
        Rt = torch.tensor(R.to_numpy() * WM_SCALE, dtype=torch.float32)
        z = wm.filter_state(Rt.unsqueeze(0))[0]
        if horizon <= 1:
            paths = wm.emit_sample(z, n_paths=n_paths).numpy() / WM_SCALE          # [P,N] 1-day
        else:
            paths = wm.rollout(z, steps=horizon, n_paths=n_paths).sum(1).numpy() / WM_SCALE  # cumulative
    except Exception as ex:
        return {"available": False, "reason": f"world-model sampling failed: {ex}"}
    idx = [uni.index(s) for s in syms]; n = len(syms)
    w = np.asarray(weights) if weights is not None else np.ones(n) / n
    port = paths[:, idx] @ w
    q = np.percentile(port, [1, 5, 50, 95])
    return {"available": True, "horizon_days": horizon, "n_paths": n_paths, "engine": "world-model (joint)",
            "joint_var99_pct": round(float(q[0]) * 100, 2), "joint_var95_pct": round(float(q[1]) * 100, 2),
            "joint_es99_pct": round(float(port[port <= q[0]].mean()) * 100, 2),
            "median_pct": round(float(q[2]) * 100, 2),
            # HONEST calibration: the world model is a joint SIMULATOR, not the calibrated tail. Its 1-day
            # scenario VaR breaches ~3% vs a 1% target (scripts/train_worldmodel.py test 2) — looser than
            # the EVT-GPD specialist (0.94%). Value = cross-asset COHERENCE + what-ifs, not a tighter VaR.
            "calibration": "directional joint scenario (~3% breach vs 1% target); for the calibrated tail "
                           "use the EVT-GPD/FHS number — the world model adds joint coherence, not tighter VaR"}


def portfolio_analysis(entities: list[str]) -> dict:
    """Covariance + minimum-variance optimization + crisis volatility stress-test for a basket."""
    from sklearn.covariance import LedoitWolf
    rets, syms = {}, []
    for q in entities:
        r = resolve(q)
        if not r or not r.get("symbol"):
            continue
        try:
            s = realized_variance(fetch_yahoo(r["symbol"]))["ret"].dropna()
        except Exception:
            continue
        if len(s) > 252:
            rets[r["symbol"]] = s; syms.append(r["symbol"])
    if len(rets) < 2:
        return {"ok": False, "message": "need ≥2 entities with data"}
    R = pd.DataFrame(rets).dropna().iloc[-756:]
    S = LedoitWolf().fit(R.to_numpy()).covariance_; n = len(syms)
    inv1 = np.linalg.inv(S + 1e-8 * np.eye(n)) @ np.ones(n); w = inv1 / inv1.sum()
    pv = lambda ww: float(np.sqrt(ww @ S @ ww * TRADING))
    ew = np.ones(n) / n
    return {"ok": True, "symbols": syms, "min_var_weights": {syms[i]: round(w[i] * 100) for i in range(n)},
            "min_var_vol_pct": round(pv(w) * 100, 1), "equal_weight_vol_pct": round(pv(ew) * 100, 1),
            "risk_reduction_pct": round((1 - pv(w) / pv(ew)) * 100),
            "crisis_stress": crisis_stress_test(rets, syms, w),
            "world_scenario": world_portfolio_scenario(syms, weights=w)}   # joint WM scenario if in-universe


if __name__ == "__main__":
    import json
    r = full_analysis("Apple")
    print(r["thesis"] if r["ok"] else r["message"])
    print("\n--- portfolio ---")
    p = portfolio_analysis(["SPY", "TLT", "GLD", "QQQ"])
    print(json.dumps(p, indent=1) if p["ok"] else p["message"])
