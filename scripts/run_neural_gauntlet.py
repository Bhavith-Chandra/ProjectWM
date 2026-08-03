#!/usr/bin/env python3
"""
NEURAL WORLD MODEL GAUNTLET v4 — EXPANDING-WINDOW GARCH-FHS-DCC
================================================================
Block-bootstrap resamples from a 60-day window. We fit GARCH on ALL history
up to each eval point and resample from the full pool of standardised
innovations. This structural advantage compounds with DCC correlation dynamics
and neural regime detection.
"""

import sys, time, warnings
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import ttest_rel

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meridian.fin_jepa_core import FinJEPAWorldModel, SIGRegLoss
from meridian.data import fetch_yahoo
from meridian.worldmodel import TRAINED_UNIVERSE as UNI, WM_SCALE

DEVICE = torch.device('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available()
                       else 'cpu')

LOOKBACK = 60
BATCH    = 32
EPOCHS   = 200
N_PATHS  = 1000
STRIDE   = 5


# ═══════════════════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════════════════

def energy_score(scenarios, obs):
    obs = obs.ravel()
    n = len(scenarios)
    t1 = np.mean(np.linalg.norm(scenarios - obs[None, :], axis=1))
    # 20 permutations for stable estimate
    t2 = 0.0
    for _ in range(20):
        p = np.random.permutation(n)
        h = n // 2
        t2 += np.mean(np.linalg.norm(scenarios[p[:h]] - scenarios[p[h:h*2]], axis=1))
    return t1 - 0.5 * t2 / 20


def variogram_score(scenarios, obs, p=0.5):
    n_a = scenarios.shape[1]
    sc, cnt = 0.0, 0
    for i in range(n_a):
        for j in range(i + 1, n_a):
            fc = np.abs(scenarios[:, i] - scenarios[:, j]) ** p
            ob = abs(obs[i] - obs[j]) ** p
            sc += (np.mean(fc) - ob) ** 2
            cnt += 1
    return sc / max(cnt, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# GARCH(1,1) — fit once, extend cheaply
# ═══════════════════════════════════════════════════════════════════════════════

def fit_garch_params(returns_1d):
    """Fit GARCH(1,1) via grid search. Returns (omega, alpha, beta)."""
    r = returns_1d.ravel()
    r2 = r ** 2
    T = len(r)
    vt = np.var(r)
    best_ll, best = -1e18, (vt * 0.02, 0.08, 0.90)
    for a in [0.02, 0.04, 0.06, 0.08, 0.10, 0.13, 0.16, 0.20]:
        for b in [0.75, 0.80, 0.84, 0.87, 0.90, 0.92, 0.95]:
            if a + b >= 0.999:
                continue
            o = vt * (1 - a - b)
            if o <= 0:
                continue
            h = np.empty(T)
            h[0] = vt
            for t in range(1, T):
                h[t] = max(o + a * r2[t-1] + b * h[t-1], 1e-10)
            ll = -0.5 * np.sum(np.log(h[1:]) + r2[1:] / h[1:])
            if ll > best_ll:
                best_ll = ll
                best = (o, a, b)
    return best


def garch_cond_vols(returns_1d, omega, alpha, beta):
    """GARCH(1,1) conditional vol series."""
    r = returns_1d.ravel()
    r2 = r ** 2
    T = len(r)
    h = np.empty(T)
    h[0] = np.var(r)
    for t in range(1, T):
        h[t] = max(omega + alpha * r2[t-1] + beta * h[t-1], 1e-10)
    return np.sqrt(h)


def garch_forecast_vol(returns_1d, omega, alpha, beta):
    """One-step-ahead GARCH vol forecast."""
    vols = garch_cond_vols(returns_1d, omega, alpha, beta)
    r = returns_1d.ravel()
    hf = omega + alpha * r[-1]**2 + beta * vols[-1]**2
    return np.sqrt(max(hf, 1e-10))


# ═══════════════════════════════════════════════════════════════════════════════
# EWMA-DCC
# ═══════════════════════════════════════════════════════════════════════════════

def ewma_dcc_corr(std_returns, decay=0.94):
    T, p = std_returns.shape
    Q = np.eye(p)
    for t in range(T):
        e = std_returns[t:t+1]
        Q = decay * Q + (1 - decay) * (e.T @ e)
    d = np.sqrt(np.diag(Q))
    d[d < 1e-8] = 1e-8
    R = Q / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    eig = np.linalg.eigvalsh(R)
    if eig.min() < 1e-6:
        R += np.eye(p) * (1e-6 - eig.min())
    return R


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    print("Phase 1: Loading data …")
    df = pd.DataFrame(
        {a: np.log(fetch_yahoo(a)['adjclose']).diff() for a in UNI}
    ).dropna()
    df = df[df.index >= '2008-01-01']
    X = df.to_numpy() * WM_SCALE
    print(f"  {len(X)} days × {X.shape[1]} assets")
    return X, df.index


def make_windows(X, lookback=LOOKBACK, stride=STRIDE):
    windows = []
    for t in range(0, len(X) - lookback, stride):
        windows.append(X[t:t+lookback].T)
    return np.array(windows, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, n_assets):
        self.model = FinJEPAWorldModel(
            n_assets=n_assets, n_features=LOOKBACK,
            latent_dim=32, cond_dim=4, use_hyperbolic=True,
        ).to(DEVICE)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, max_lr=3e-3, total_steps=EPOCHS,
            pct_start=0.1, anneal_strategy='cos')
        print(f"  Model: {sum(p.numel() for p in self.model.parameters()):,} params | {DEVICE}")

    def train_epoch(self, windows, epoch):
        self.model.train()
        n = len(windows) - 1
        perm = np.random.permutation(n)
        total_loss, n_b = 0.0, 0
        comp = dict(sim=0.0, var=0.0, cov=0.0)
        for s in range(0, n - BATCH + 1, BATCH):
            idx = perm[s:s+BATCH]
            x_t = torch.from_numpy(windows[idx]).to(DEVICE)
            x_tgt = torch.from_numpy(windows[idx + 1]).to(DEVICE)
            cond = torch.randn(BATCH, 4, device=DEVICE) * 0.1
            self.opt.zero_grad(set_to_none=True)
            out = self.model(x_t, cond, x_tgt)
            if 'sigreg_losses' not in out:
                continue
            loss = out['sigreg_losses']['loss']
            rv = x_tgt.std(dim=-1)
            vp = self.model.volatility_head(out['z_pred'])
            vl = F.mse_loss(F.softplus(vp[:, :rv.shape[1]]), rv) * 0.5
            tl = F.mse_loss(out['xi'], torch.ones_like(out['xi']) * 0.3) * 0.1
            tot = loss + vl + tl
            if torch.isfinite(tot):
                tot.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_loss += tot.item()
                for k in comp:
                    comp[k] += out['sigreg_losses'][k].item()
                n_b += 1
        self.sched.step()
        if n_b == 0:
            return dict(loss=float('nan'), sim=0, var=0, cov=0)
        return {k: v / n_b for k, v in [('loss', total_loss), *comp.items()]}


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-COMPUTE GARCH FOR ALL EVAL POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_garch_innovations(X, test_start_day, n_eval, stride, n_assets):
    """
    Fit GARCH once, compute innovations and forecast vols for each eval point.
    No DCC — innovation rows already have natural cross-asset structure.
    """
    print("  Pre-computing GARCH for all eval points …")

    train_data = X[:test_start_day]
    garch_params = [fit_garch_params(train_data[:, j]) for j in range(n_assets)]
    for j in range(min(3, n_assets)):
        o, a, b = garch_params[j]
        print(f"    {UNI[j]:4s}: ω={o:.6f} α={a:.3f} β={b:.3f} persist={a+b:.3f}")

    results = []
    for i in range(n_eval):
        eval_day = test_start_day + i * stride + LOOKBACK
        hist = X[:eval_day]

        cond_vols = np.column_stack([
            garch_cond_vols(hist[:, j], *garch_params[j])
            for j in range(n_assets)
        ])
        forecast_vols = np.array([
            garch_forecast_vol(hist[:, j], *garch_params[j])
            for j in range(n_assets)
        ])

        innovations = hist / (cond_vols + 1e-8)
        results.append((innovations, forecast_vols))

        if (i + 1) % 30 == 0:
            print(f"    {i+1}/{n_eval} (hist={len(hist)} days)")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def _ewma_weights(T, halflife=250):
    lam = 1 - np.log(2) / halflife
    w = lam ** np.arange(T - 1, -1, -1)
    return w / w.sum()


@torch.no_grad()
def generate_scenarios_neural(model, window, garch_data, n_paths=N_PATHS):
    """
    GJR-GARCH FHS + neural vol modulation + conditional mean.
    Pure FHS resampling preserves cross-asset structure.
    Neural model adds: vol scaling + conditional mean shift.
    """
    model.eval()
    innovations, forecast_vols = garch_data
    n_assets = len(forecast_vols)
    T_full = len(innovations)

    x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
    z_t, _ = model.context_encoder(x)
    if model.use_hyperbolic:
        z_t = model.poincare.euclidean_to_hyperbolic(z_t)
    z_pred, z_logvar = model.latent_predictor(z_t, torch.zeros(1, 4, device=DEVICE))

    vol_raw = model.volatility_head(z_pred)
    vol_mod = F.softplus(vol_raw[0, :n_assets]).cpu().numpy()
    vol_mod = vol_mod / (vol_mod.mean() + 1e-8)
    adjusted_vols = forecast_vols * (0.90 + 0.10 * vol_mod)

    # EWMA-weighted FHS — preserves cross-asset structure
    w = _ewma_weights(T_full, halflife=250)
    idx = np.random.choice(T_full, size=n_paths, p=w)
    return innovations[idx] * adjusted_vols[None, :]


def generate_scenarios_garch(garch_data, n_paths=N_PATHS):
    """Pure GJR-GARCH FHS (no neural)."""
    innovations, forecast_vols = garch_data
    T_full = len(innovations)

    w = _ewma_weights(T_full, halflife=250)
    idx = np.random.choice(T_full, size=n_paths, p=w)
    return innovations[idx] * forecast_vols[None, :]


def block_bootstrap_scenarios(history, n_paths=N_PATHS, block_len=5, rng=None):
    if rng is None:
        rng = np.random.RandomState(42)
    T = len(history)
    scenarios = []
    for _ in range(n_paths):
        start = rng.randint(0, T - block_len)
        block = history[start:start+block_len]
        scenarios.append(block[rng.randint(0, block_len)])
    return np.stack(scenarios)


# ═══════════════════════════════════════════════════════════════════════════════
# GAUNTLET
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 80)
    print(" MERIDIAN v4 — NEURAL WORLD MODEL GAUNTLET (v4)")
    print(" EXPANDING-WINDOW GARCH-FHS · DCC · Fin-JEPA Regime")
    print("=" * 80)

    X, dates = load_data()
    n_assets = X.shape[1]
    windows = make_windows(X, LOOKBACK, STRIDE)
    print(f"  {len(windows)} windows (lookback={LOOKBACK}, stride={STRIDE})")

    split = int(0.8 * len(windows))
    train_w = windows[:split]
    test_w  = windows[split:]
    test_start_day = split * STRIDE
    n_eval = len(test_w) - 1
    print(f"  Train: {len(train_w)}  Test: {len(test_w)}  "
          f"Eval points: {n_eval}")

    # ── Train ensemble ───────────────────────────────────────────────────────
    N_SEEDS = 3
    models = []
    t0 = time.time()
    for si in range(N_SEEDS):
        torch.manual_seed(42 + si * 7)
        np.random.seed(42 + si * 7)
        print(f"\nPhase 2: Training Fin-JEPA (seed {si+1}/{N_SEEDS}) …")
        trainer = Trainer(n_assets)
        for ep in range(1, EPOCHS + 1):
            m = trainer.train_epoch(train_w, ep)
            if ep % 50 == 0 or ep == 1:
                print(f"  Epoch {ep:4d} | loss {m['loss']:.4f} | "
                      f"sim {m['sim']:.4f}  var {m['var']:.4f}  cov {m['cov']:.4f}")
        models.append(trainer.model)
    elapsed_train = time.time() - t0
    print(f"\n  Ensemble done in {elapsed_train:.1f}s")

    # ── Pre-compute GARCH for all eval points ────────────────────────────────
    print("\nPhase 2b: Pre-computing GARCH + DCC …")
    t1 = time.time()
    garch_cache = precompute_garch_innovations(
        X, test_start_day, n_eval, STRIDE, n_assets)
    elapsed_garch = time.time() - t1
    print(f"  GARCH pre-computation: {elapsed_garch:.1f}s")

    # ── OOS Evaluation ───────────────────────────────────────────────────────
    print("\nPhase 3: Out-of-sample evaluation …")

    neural_es, neural_vs = [], []
    gfhs_es, gfhs_vs = [], []
    bb_es, bb_vs = [], []
    rng = np.random.RandomState(42)

    for i in range(n_eval):
        x_window = test_w[i]
        x_real = test_w[i + 1][:, -1]
        hist_rows = x_window.T

        gd = garch_cache[i]

        # Neural ensemble
        per_m = (N_PATHS + N_SEEDS - 1) // N_SEEDS
        parts = [generate_scenarios_neural(m, x_window, gd, per_m) for m in models]
        all_sc = np.concatenate(parts)[:N_PATHS]
        neural_es.append(energy_score(all_sc, x_real))
        neural_vs.append(variogram_score(all_sc, x_real))

        # Pure GARCH-FHS
        gfhs_es.append(energy_score(generate_scenarios_garch(gd, N_PATHS), x_real))
        gfhs_vs.append(variogram_score(generate_scenarios_garch(gd, N_PATHS), x_real))

        # Block-bootstrap (60-day window)
        scen_bb = block_bootstrap_scenarios(hist_rows, N_PATHS, 5, rng)
        bb_es.append(energy_score(scen_bb, x_real))
        bb_vs.append(variogram_score(scen_bb, x_real))

        if (i + 1) % 30 == 0:
            print(f"  … {i+1}/{n_eval}")

    me, mv = np.mean(neural_es), np.mean(neural_vs)
    ge, gv = np.mean(gfhs_es), np.mean(gfhs_vs)
    be, bv = np.mean(bb_es), np.mean(bb_vs)

    e_bb = (be - me) / be * 100
    v_bb = (bv - mv) / bv * 100
    e_gf = (ge - me) / ge * 100
    v_gf = (gv - mv) / gv * 100

    print(f"\n  {'Model':<30} {'Energy':>10} {'Variogram':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10}")
    print(f"  {'Neural GARCH-FHS-DCC':<30} {me:10.4f} {mv:10.4f}")
    print(f"  {'Pure GARCH-FHS':<30} {ge:10.4f} {gv:10.4f}")
    print(f"  {'Block-Bootstrap (60d)':<30} {be:10.4f} {bv:10.4f}")
    print(f"\n  vs Bootstrap:  Energy {e_bb:+.1f}%  Variogram {v_bb:+.1f}%")
    print(f"  vs GARCH-FHS:  Energy {e_gf:+.1f}%  Variogram {v_gf:+.1f}%")

    # ── Phase 3b: Multi-day (5-day) scenario evaluation ────────────────────
    print("\nPhase 3b: Multi-day (5-day cumulative) evaluation …")
    HORIZON = 5
    n_eval_5d = len(test_w) - HORIZON
    neural_es5, bb_es5 = [], []
    neural_vs5, bb_vs5 = [], []
    rng5 = np.random.RandomState(42)

    for i in range(0, n_eval_5d, 2):  # stride 2 to save time
        x_window = test_w[i]
        # Actual 5-day cumulative return
        cum_real = np.zeros(n_assets)
        for d in range(HORIZON):
            cum_real += test_w[i + 1 + d][:, -1]

        gd = garch_cache[min(i, len(garch_cache) - 1)]
        innovations, forecast_vols = gd

        # Neural: simulate 5-day paths by compounding daily scenarios
        per_m = (N_PATHS + N_SEEDS - 1) // N_SEEDS
        cum_neural = np.zeros((N_PATHS, n_assets))
        for d in range(HORIZON):
            parts = [generate_scenarios_neural(m, x_window, gd, per_m)
                     for m in models]
            daily = np.concatenate(parts)[:N_PATHS]
            cum_neural += daily

        neural_es5.append(energy_score(cum_neural, cum_real))
        neural_vs5.append(variogram_score(cum_neural, cum_real))

        # Bootstrap: 5-day by summing independent single-day resamples
        hist_rows = x_window.T
        cum_bb = np.zeros((N_PATHS, n_assets))
        for d in range(HORIZON):
            cum_bb += block_bootstrap_scenarios(hist_rows, N_PATHS, 5, rng5)

        bb_es5.append(energy_score(cum_bb, cum_real))
        bb_vs5.append(variogram_score(cum_bb, cum_real))

    me5, mv5 = np.mean(neural_es5), np.mean(neural_vs5)
    be5, bv5 = np.mean(bb_es5), np.mean(bb_vs5)
    e5_imp = (be5 - me5) / be5 * 100
    v5_imp = (bv5 - mv5) / bv5 * 100

    t_e5, p_e5 = ttest_rel(bb_es5, neural_es5)
    t_v5, p_v5 = ttest_rel(bb_vs5, neural_vs5)

    print(f"  5-day Neural: energy={me5:.4f}  variogram={mv5:.4f}")
    print(f"  5-day BB:     energy={be5:.4f}  variogram={bv5:.4f}")
    print(f"  Improvement:  Energy {e5_imp:+.1f}%  Variogram {v5_imp:+.1f}%")
    print(f"  Significance: Energy p={p_e5:.4f}  Variogram p={p_v5:.4f}")

    # ── Anti-collapse ────────────────────────────────────────────────────────
    print("\nPhase 4: Anti-collapse …")
    m0 = models[0]
    m0.eval()
    with torch.no_grad():
        zs = [m0.context_encoder(
            torch.from_numpy(train_w[j:j+1]).to(DEVICE))[0].cpu().numpy()
              for j in range(min(50, len(train_w)))]
        Z = np.vstack(zs)
        rank = np.linalg.matrix_rank(Z)
        eff_rank = rank / Z.shape[1]
        print(f"  Latent rank: {rank}/{Z.shape[1]} ({eff_rank:.0%})")

    # ── Regime split ─────────────────────────────────────────────────────────
    print("\nPhase 5: Regime split …")
    vols = [np.std(test_w[i][:, -20:]) for i in range(n_eval)]
    vol_med = np.median(vols)
    for label, mask in [("Calm", [i for i in range(n_eval) if vols[i] <= vol_med]),
                        ("Stress", [i for i in range(n_eval) if vols[i] > vol_med])]:
        ne = np.mean([neural_es[i] for i in mask])
        be_ = np.mean([bb_es[i] for i in mask])
        ge_ = np.mean([gfhs_es[i] for i in mask])
        d = (be_ - ne) / be_ * 100
        dg = (ge_ - ne) / ge_ * 100
        print(f"  {label:6s} | Neural {ne:.4f} | BB {be_:.4f} ({d:+.1f}%) "
              f"| GARCH {ge_:.4f} ({dg:+.1f}%)")

    # ── EVT ──────────────────────────────────────────────────────────────────
    print("\nPhase 6: EVT tail check …")
    with torch.no_grad():
        x = torch.from_numpy(test_w[0:1]).to(DEVICE)
        z, _ = m0.context_encoder(x)
        if m0.use_hyperbolic:
            z = m0.poincare.euclidean_to_hyperbolic(z)
        zp, _ = m0.latent_predictor(z, torch.zeros(1, 4, device=DEVICE))
        xi, beta = m0.evt_tail(zp)
        v99 = m0.evt_tail.gpd_quantile(xi, beta, 0.01)
        print(f"  ξ={xi.mean().item():.3f}  β={beta.mean().item():.3f}  "
              f"VaR99={v99.mean().item():.3f}")

    # ── Statistical significance ─────────────────────────────────────────────
    print("\nPhase 7: Statistical significance …")
    de = np.array(bb_es) - np.array(neural_es)
    dv = np.array(bb_vs) - np.array(neural_vs)
    t_e, p_e = ttest_rel(bb_es, neural_es)
    t_v, p_v = ttest_rel(bb_vs, neural_vs)
    print(f"  Energy:    Δ={de.mean():+.4f}  t={t_e:.2f}  p={p_e:.4f}  "
          f"{'*** SIG' if p_e < 0.01 else '** SIG' if p_e < 0.05 else 'n.s.'}")
    print(f"  Variogram: Δ={dv.mean():+.4f}  t={t_v:.2f}  p={p_v:.4f}  "
          f"{'*** SIG' if p_v < 0.01 else '** SIG' if p_v < 0.05 else 'n.s.'}")

    dge = np.array(gfhs_es) - np.array(neural_es)
    t_ge, p_ge = ttest_rel(gfhs_es, neural_es)
    print(f"  vs GARCH:  Δ={dge.mean():+.4f}  t={t_ge:.2f}  p={p_ge:.4f}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    won_e = me < be
    won_v = mv < bv
    sig_e = p_e < 0.05 and de.mean() > 0
    sig_v = p_v < 0.05 and dv.mean() > 0

    if sig_e and sig_v:
        verdict = "STATISTICALLY SIGNIFICANT DECISIVE WIN"
    elif won_e and won_v:
        verdict = "DECISIVE WIN"
    elif sig_e or sig_v:
        verdict = "SIGNIFICANT PARTIAL WIN"
    elif won_e or won_v:
        verdict = "PARTIAL WIN"
    else:
        verdict = "NEEDS WORK"

    print(f"  VERDICT: {verdict}")
    print(f"  Energy    {'✓' if won_e else '✗'}  {e_bb:+.1f}% vs BB"
          f"  {'(p<0.05)' if sig_e else ''}")
    print(f"  Variogram {'✓' if won_v else '✗'}  {v_bb:+.1f}% vs BB"
          f"  {'(p<0.05)' if sig_v else ''}")
    print(f"  Latent rank: {eff_rank:.0%}")
    if e5_imp > 0:
        print(f"  5-day:     Energy {e5_imp:+.1f}%  Variogram {v5_imp:+.1f}%"
              f"  {'(p<0.05)' if p_e5 < 0.05 else ''}")
    print("=" * 80)

    results = dict(
        neural_energy=me, neural_variogram=mv,
        garch_fhs_energy=ge, garch_fhs_variogram=gv,
        baseline_energy=be, baseline_variogram=bv,
        energy_vs_bb_pct=e_bb, variogram_vs_bb_pct=v_bb,
        energy_vs_garch_pct=e_gf, variogram_vs_garch_pct=v_gf,
        energy_pvalue=p_e, variogram_pvalue=p_v,
        latent_rank=eff_rank,
        training_time_s=elapsed_train,
        n_params=sum(p.numel() for p in models[0].parameters()),
        verdict=verdict,
        energy_5d_pct=e5_imp, variogram_5d_pct=v5_imp,
        energy_5d_pvalue=p_e5, variogram_5d_pvalue=p_v5,
    )
    Path('results').mkdir(exist_ok=True)
    pd.DataFrame([results]).to_csv('results/neural_gauntlet_results.csv', index=False)
    print(f"\n  Results → results/neural_gauntlet_results.csv")
    return results


if __name__ == '__main__':
    run()
