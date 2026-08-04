#!/usr/bin/env python3
"""
Meridian World Model V5 — Neural Regime + GARCH-FHS
================================================================

V3 LESSON: Neural correlation prediction HURTS (variogram -4.4%)
V4 LESSON: Neural vol multiplier HURTS (overfits, 21% scaling)

BOTH shared the same flaw: neural modifications to well-calibrated
GARCH scenarios introduce OOS noise that exceeds any signal.

V5 STRATEGY: Neural net does ONE thing — regime classification.

  MARGINALS: GJR-GARCH vol evolution (UNCHANGED, proven)
  DEPENDENCE: Natural block correlation (NO Iman-Conover)
  INNOVATION: Neural-regime-weighted block resampling (the ONLY neural part)
  VOL SCALING: None (trust GARCH)

Why regime classification IS the right neural target:
  - VIX percentiles are a 1D projection of regime space
  - True regimes depend on VIX + credit spreads + yield curve + momentum
  - A small neural net (~5K params) can learn this multi-factor regime
  - Small model → can't overfit, but CAN capture cross-feature interactions
  - Block selection is the ONLY step where better regime info helps

The model is honest: GARCH-FHS does the heavy lifting, neural regime
just picks which history to resample from.
"""

import sys, os, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging
from io import BytesIO, StringIO
from zipfile import ZipFile
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# UNIVERSE + CONSTANTS
# ══════════════════════════════════════════════════════════════════════

EQUITY_US = ['SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'IVV', 'RSP']
EQUITY_SECTOR = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE']
EQUITY_INTL = ['EFA', 'EEM', 'VWO', 'IEMG', 'VEA']
FIXED_INCOME = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'BND', 'AGG']
COMMODITIES = ['GLD', 'SLV', 'USO', 'DBC']
ALTERNATIVES = ['VNQ', 'VNQI']
UNIVERSE = EQUITY_US + EQUITY_SECTOR + EQUITY_INTL + FIXED_INCOME + COMMODITIES + ALTERNATIVES
WM_SCALE = 100.0

FRED_SERIES = {
    'VIXCLS': 'vix', 'DGS10': 'yield_10y', 'DGS2': 'yield_2y',
    'DGS30': 'yield_30y', 'DGS5': 'yield_5y',
    'T10Y2Y': 'spread_10y2y', 'T10Y3M': 'spread_10y3m',
    'BAMLH0A0HYM2': 'hy_spread', 'BAMLC0A4CBBB': 'bbb_spread',
    'DTWEXBGS': 'usd_index', 'DCOILWTICO': 'wti_oil',
    'DFEDTARU': 'fed_funds_upper', 'TEDRATE': 'ted_spread',
}

HORIZONS = [1, 5, 10, 20]
N_SC = 1000
N_EVAL = 160
STRIDE = 3


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def fetch_fama_french():
    url = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip'
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        zf = ZipFile(BytesIO(resp.read()))
        csv_name = [n for n in zf.namelist() if n.lower().endswith('.csv')][0]
        raw = zf.read(csv_name).decode('utf-8')
        lines = raw.strip().split('\n')
        header_idx = next((i for i, l in enumerate(lines) if 'Mkt-RF' in l), None)
        if header_idx is None:
            return None
        data_lines = []
        for line in lines[header_idx + 1:]:
            parts = line.strip().split(',')
            if len(parts) >= 5 and len(parts[0].strip()) == 8:
                data_lines.append(line)
            elif data_lines:
                break
        csv_str = lines[header_idx] + '\n' + '\n'.join(data_lines)
        df = pd.read_csv(StringIO(csv_str))
        df.columns = [c.strip() for c in df.columns]
        dc = df.columns[0]
        df[dc] = pd.to_datetime(df[dc].astype(str), format='%Y%m%d')
        df = df.set_index(dc).apply(pd.to_numeric, errors='coerce') / 100
        df.columns = ['mkt_rf', 'smb', 'hml', 'rf']
        log.info(f"  Fama-French: {len(df)} days")
        return df
    except Exception as e:
        log.warning(f"  Fama-French failed: {e}")
        return None


def load_data():
    import yfinance as yf
    log.info(f"Source 1: Yahoo ({len(UNIVERSE)} ETFs)...")
    pd_data = yf.download(UNIVERSE, period='max', auto_adjust=True, progress=False, threads=True)
    if isinstance(pd_data.columns, pd.MultiIndex):
        close = pd_data['Close']
    else:
        close = pd_data[['Close']].rename(columns={'Close': UNIVERSE[0]})

    valid = close.notna().sum(axis=1) >= int(len(UNIVERSE) * 0.7)
    close = close[valid].ffill().bfill()
    good = [a for a in UNIVERSE if close[a].notna().sum() > len(close) * 0.9]
    close = close[good].ffill().bfill()
    asset_names = list(close.columns)
    n_assets = len(asset_names)
    log.info(f"  {n_assets} assets, {len(close)} days")

    returns = np.log(close).diff().fillna(0).values
    T = len(returns)

    rv_22d = np.zeros(T)
    for i in range(22, T):
        rv_22d[i] = returns[i - 22:i].std(axis=0).mean()

    log.info("Source 2: FRED macro...")
    import pandas_datareader.data as web
    macro = {}
    for fid, name in FRED_SERIES.items():
        try:
            s = web.DataReader(fid, 'fred', start='2000-01-01').iloc[:, 0]
            macro[name] = s.reindex(close.index).ffill().bfill().values
            log.info(f"  {fid}: OK")
        except Exception:
            log.warning(f"  {fid}: failed")

    vix = np.nan_to_num(macro.get('vix', np.full(T, 20.0)), nan=20.0) / 100
    sp_10y2y = np.nan_to_num(macro.get('spread_10y2y', np.zeros(T)), nan=0.0) / 100
    hy_sp = np.nan_to_num(macro.get('hy_spread', np.full(T, 4.0)), nan=4.0) / 100
    bbb_sp = np.nan_to_num(macro.get('bbb_spread', np.full(T, 2.0)), nan=2.0) / 100
    usd = np.nan_to_num(macro.get('usd_index', np.full(T, 100.0)), nan=100.0)
    usd_chg = np.zeros(T)
    usd_chg[1:] = np.diff(usd) / np.maximum(np.abs(usd[:-1]), 1e-6)
    oil = np.nan_to_num(macro.get('wti_oil', np.full(T, 60.0)), nan=60.0)
    oil_chg = np.zeros(T)
    oil_chg[1:] = np.diff(oil) / np.maximum(np.abs(oil[:-1]), 1e-6)

    log.info("Source 3: Fama-French...")
    ff = fetch_fama_french()
    ff_feat = np.zeros((T, 4))
    if ff is not None:
        ff_feat = np.nan_to_num(ff.reindex(close.index).ffill().fillna(0)[['mkt_rf', 'smb', 'hml', 'rf']].values)

    # Macro feature matrix for regime classifier (per-day, not per-asset)
    macro_features = np.column_stack([
        vix,                    # VIX level
        np.gradient(vix),       # VIX momentum
        rv_22d,                 # Realized vol (22d)
        sp_10y2y,               # Yield curve slope
        hy_sp,                  # HY credit spread
        bbb_sp,                 # BBB credit spread
        usd_chg,                # USD momentum
        oil_chg,                # Oil momentum
        ff_feat[:, 0],          # Mkt-RF factor
        ff_feat[:, 1],          # SMB factor
        ff_feat[:, 2],          # HML factor
    ])
    macro_features = np.clip(np.nan_to_num(macro_features, nan=0.0), -10, 10)

    start = 60
    return returns[start:], asset_names, vix[start:], macro_features[start:]


# ══════════════════════════════════════════════════════════════════════
# GJR-GARCH
# ══════════════════════════════════════════════════════════════════════

def fit_gjr_garch(r):
    best_ll, best_p = -1e18, (0.04, 0.04, 0.90)
    v_unc = r.var()
    for a in [0.02, 0.04, 0.06, 0.08]:
        for g in [0.02, 0.04, 0.06, 0.08, 0.10]:
            for b in [0.82, 0.85, 0.88, 0.90, 0.92]:
                if a + g / 2 + b >= 0.999:
                    continue
                om = v_unc * (1 - a - g / 2 - b)
                if om < 1e-12:
                    continue
                s2 = np.full(len(r), v_unc)
                for t in range(1, len(r)):
                    r2 = r[t - 1] ** 2
                    s2[t] = max(om + a * r2 + g * r2 * (r[t - 1] < 0) + b * s2[t - 1], 1e-12)
                ll = -0.5 * np.sum(np.log(s2[100:]) + r[100:] ** 2 / s2[100:])
                if ll > best_ll:
                    best_ll, best_p = ll, (a, g, b)
    return best_p


_garch_cache = {}


def precompute_garch(returns, t_eval):
    if t_eval in _garch_cache:
        return _garch_cache[t_eval]
    hist = returns[:t_eval]
    T_h, N = hist.shape
    fvols = np.zeros(N)
    innov = np.zeros((T_h, N))
    gparams = np.zeros((N, 4))

    for j in range(N):
        r = hist[:, j]
        a, g, b = fit_gjr_garch(r)
        v_unc = r.var()
        om = v_unc * (1 - a - g / 2 - b)

        s2 = np.full(len(r), v_unc)
        for t in range(1, len(r)):
            r2 = r[t - 1] ** 2
            s2[t] = max(om + a * r2 + g * r2 * (r[t - 1] < 0) + b * s2[t - 1], 1e-12)
        cv = np.sqrt(s2)
        innov[:, j] = r / np.maximum(cv, 1e-8)

        rv_d = np.abs(r[-1])
        rv_w = r[-5:].std() if len(r) >= 5 else rv_d
        rv_m = r[-22:].std() if len(r) >= 22 else rv_w
        r2_last = r[-1] ** 2
        lev = r2_last * (r[-1] < 0)
        gfv = np.sqrt(om + a * r2_last + g * lev + b * cv[-1] ** 2)
        fvols[j] = max(0.3 * rv_d + 0.3 * rv_w + 0.2 * rv_m + 0.2 * gfv, gfv * 0.5)
        gparams[j] = [om, a, g, b]

    result = (innov, fvols, gparams)
    _garch_cache[t_eval] = result
    if len(_garch_cache) > 200:
        del _garch_cache[min(_garch_cache)]
    return result


# ══════════════════════════════════════════════════════════════════════
# NEURAL REGIME CLASSIFIER
# ══════════════════════════════════════════════════════════════════════

class NeuralRegimeClassifier(nn.Module):
    """
    Small neural net that maps macro features → regime embedding.
    Used to compute block selection weights for FHS resampling.

    ~5K params — deliberately tiny to prevent overfitting.
    """
    def __init__(self, n_macro_feat=11, n_regimes=6, hidden=32):
        super().__init__()
        self.n_regimes = n_regimes

        # GRU on recent macro history
        self.gru = nn.GRU(n_macro_feat, hidden, 1, batch_first=True)

        # Regime embedding head
        self.regime_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_regimes)
        )

        # Recency head: how much to bias toward recent history
        self.recency_head = nn.Sequential(
            nn.Linear(hidden, 1), nn.Sigmoid()
        )

    def forward(self, macro_seq):
        """
        macro_seq: (B, T, n_macro_feat)
        Returns:
            regime_embed: (B, n_regimes) — softmax probabilities
            recency: (B, 1) — how much to weight recent vs broad history
        """
        h, _ = self.gru(macro_seq)
        h_last = h[:, -1, :]  # (B, hidden)

        regime_logits = self.regime_head(h_last)
        regime_embed = torch.softmax(regime_logits, dim=-1)

        recency = self.recency_head(h_last)  # (B, 1) in [0, 1]

        return regime_embed, recency


def _compute_regime_labels(returns, macro_features, n_regimes=6, window=22):
    """Compute regime labels based on future realized vol + market direction."""
    T = len(returns)
    labels = np.zeros(T, dtype=int)
    fwd_vol = np.zeros(T)
    fwd_ret = np.zeros(T)

    for t in range(T - window):
        fwd_vol[t] = returns[t:t + window].std(axis=0).mean()
        fwd_ret[t] = returns[t:t + window, :7].mean()  # US equity avg

    # Cluster into regimes using vol and direction
    valid = fwd_vol > 0
    if valid.sum() < 100:
        return labels

    vol_pctiles = np.percentile(fwd_vol[valid], [25, 50, 75])
    for t in range(T - window):
        v = fwd_vol[t]
        r = fwd_ret[t]
        if v <= vol_pctiles[0]:
            labels[t] = 0 if r >= 0 else 1  # low vol, up/down
        elif v <= vol_pctiles[1]:
            labels[t] = 2 if r >= 0 else 3  # med vol, up/down
        else:
            labels[t] = 4 if r >= 0 else 5  # high vol, up/down

    return labels


def train_regime_model(model, macro_features, returns, device, n_epochs=80):
    model = model.to(device)
    T = len(macro_features)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Regime model params: {n_params:,}")

    labels = _compute_regime_labels(returns, macro_features)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    macro_t = torch.tensor(macro_features, dtype=torch.float32, device=device)

    SEQ = 22
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, n_epochs + 1):
        model.train()
        tloss, nb = 0.0, 0
        idx = np.random.permutation(T - SEQ - 25)[:512]

        for bs in range(0, len(idx), 32):
            bi = idx[bs:bs + 32]
            if len(bi) < 4:
                continue
            xb = torch.stack([macro_t[i:i + SEQ] for i in bi])
            yb = labels_t[bi + SEQ]

            regime_probs, recency = model(xb)
            loss = loss_fn(regime_probs, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            tloss += loss.item()
            nb += 1

        sched.step()
        if ep % 20 == 0:
            log.info(f"    Regime {ep}/{n_epochs}: loss={tloss / max(nb, 1):.4f}")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════
# SCENARIO GENERATION
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_neural_block_weights(model, macro_features, t_eval, device):
    """Compute block selection weights using neural regime classifier."""
    SEQ = min(22, t_eval)
    macro_ctx = torch.tensor(
        macro_features[t_eval - SEQ:t_eval],
        dtype=torch.float32, device=device
    ).unsqueeze(0)

    regime_probs, recency = model(macro_ctx)
    regime_probs = regime_probs[0].cpu().numpy()  # (n_regimes,)
    recency_val = recency[0, 0].item()  # scalar in [0, 1]

    return regime_probs, recency_val


_regime_embed_cache = None


@torch.no_grad()
def precompute_regime_embeddings(model, macro_features, device):
    """Precompute regime embeddings for all time periods (run once)."""
    global _regime_embed_cache
    T = len(macro_features)
    SEQ = 22
    n_regimes = model.n_regimes

    all_probs = np.zeros((T, n_regimes))
    all_recency = np.zeros(T)
    macro_t = torch.tensor(macro_features, dtype=torch.float32, device=device)

    batch_size = 512
    for start in range(SEQ, T, batch_size):
        end = min(start + batch_size, T)
        xb = torch.stack([macro_t[t - SEQ:t] for t in range(start, end)])
        probs, rec = model(xb)
        all_probs[start:end] = probs.cpu().numpy()
        all_recency[start:end] = rec[:, 0].cpu().numpy()

    _regime_embed_cache = (all_probs, all_recency)
    log.info(f"  Precomputed regime embeddings for {T} periods")
    return all_probs, all_recency


def generate_v5_scenarios(model, macro_features, innovations, fvols, gparams,
                          all_regime_probs, all_recency, t_eval, horizon,
                          n_sc, device, seed=42):
    """
    V5 scenario generation: GARCH-FHS with neural regime-weighted blocks.

    1. Neural regime → block selection weights (the ONLY neural part)
    2. GARCH vol evolution (pure GARCH, no neural modification)
    3. Block resampling with regime + EWMA weights
    4. Natural cross-asset correlation preserved (NO Iman-Conover)
    """
    N = innovations.shape[1]
    rng = np.random.default_rng(seed)

    # Current regime embedding (precomputed)
    regime_probs = all_regime_probs[t_eval - 1]
    recency_val = all_recency[t_eval - 1]

    # Regime similarity for all historical periods
    SEQ = 22
    n_inn = t_eval
    hist_regime_scores = np.dot(all_regime_probs[:n_inn], regime_probs)

    # Adaptive halflife based on neural recency
    base_hl = 250
    halflife = int(base_hl * (1.5 - recency_val))
    halflife = max(80, min(800, halflife))

    # Block selection weights: regime_similarity × EWMA decay
    bl = 1 if horizon <= 5 else (3 if horizon <= 10 else 5)
    decay = np.exp(-np.log(2) / halflife * np.arange(n_inn)[::-1])

    # Combine: decay × regime similarity (boosted)
    regime_boost = 0.5 + 1.5 * hist_regime_scores  # [0.5, 2.0]
    w = decay * regime_boost
    w[:SEQ] = 0  # Exclude early period (no valid regime scores)

    ms = max(n_inn - bl, 1)
    bw = np.array([w[i:i + bl].sum() for i in range(ms)])
    bw_sum = bw.sum()
    if bw_sum > 0:
        bw /= bw_sum
    else:
        bw = np.ones(ms) / ms

    # GARCH params
    omega, alpha_g, gamma, beta = gparams[:, 0], gparams[:, 1], gparams[:, 2], gparams[:, 3]

    # Generate scenarios: GARCH-FHS with vol evolution
    scenarios = np.zeros((n_sc, horizon, N))
    for s in range(n_sc):
        vol_t = fvols.copy()
        t = 0
        while t < horizon:
            st = rng.choice(ms, p=bw)
            chunk = min(bl, horizon - t)
            for d in range(chunk):
                z = innovations[min(st + d, n_inn - 1)]
                r = vol_t * z
                scenarios[s, t + d] = r
                r2 = r ** 2
                lev = r2 * (r < 0)
                vol_t = np.sqrt(np.maximum(
                    omega + alpha_g * r2 + gamma * lev + beta * vol_t ** 2, 1e-12))
            t += chunk

    return scenarios


def classify_regime_vix(vix_h):
    """Simple VIX-percentile regime for baselines."""
    if len(vix_h) < 10:
        return np.zeros(len(vix_h), dtype=int)
    good = vix_h[vix_h > 0]
    if len(good) < 10:
        return np.zeros(len(vix_h), dtype=int)
    thr = np.percentile(good, [33, 66, 90])
    return np.array([0 if v <= thr[0] else 1 if v <= thr[1] else 2 if v <= thr[2] else 3
                     for v in vix_h])


def resample_fhs_const_vol(innovations, fvols, regimes, cur_reg,
                           n_sc, horizon, bl=3, hl=250, seed=42):
    n_inn, N = innovations.shape
    rm = (regimes == cur_reg).astype(float)
    am = (np.abs(regimes - cur_reg) <= 1).astype(float)
    boost = 1.0 + 2.0 * rm + 0.5 * am
    decay = np.exp(-np.log(2) / hl * np.arange(n_inn)[::-1])
    w = decay * boost
    ms = n_inn - bl
    bw = np.array([w[i:i + bl].sum() for i in range(ms)])
    bw /= bw.sum()
    rng = np.random.default_rng(seed)
    sc = np.zeros((n_sc, horizon, N))
    for s in range(n_sc):
        t = 0
        while t < horizon:
            st = rng.choice(ms, p=bw)
            c = min(bl, horizon - t)
            sc[s, t:t + c] = innovations[st:st + c] * fvols[None, :]
            t += c
    return sc


def resample_fhs_vol_evolution(innovations, fvols, gparams, regimes, cur_reg,
                               n_sc, horizon, bl=3, hl=250, seed=42):
    n_inn, N = innovations.shape
    rm = (regimes == cur_reg).astype(float)
    am = (np.abs(regimes - cur_reg) <= 1).astype(float)
    boost = 1.0 + 2.0 * rm + 0.5 * am
    decay = np.exp(-np.log(2) / hl * np.arange(n_inn)[::-1])
    w = decay * boost
    ms = max(n_inn - bl, 1)
    bw = np.array([w[i:i + bl].sum() for i in range(ms)])
    bw /= bw.sum()

    omega, alpha, gamma, beta = gparams[:, 0], gparams[:, 1], gparams[:, 2], gparams[:, 3]
    rng = np.random.default_rng(seed)
    sc = np.zeros((n_sc, horizon, N))

    for s in range(n_sc):
        vol_t = fvols.copy()
        t = 0
        while t < horizon:
            st = rng.choice(ms, p=bw)
            chunk = min(bl, horizon - t)
            for d in range(chunk):
                z = innovations[min(st + d, n_inn - 1)]
                sc[s, t + d] = vol_t * z
                r = sc[s, t + d]
                r2 = r ** 2
                lev = r2 * (r < 0)
                vol_t = np.sqrt(np.maximum(
                    omega + alpha * r2 + gamma * lev + beta * vol_t ** 2, 1e-12))
            t += chunk

    return sc


# ══════════════════════════════════════════════════════════════════════
# SCORING
# ══════════════════════════════════════════════════════════════════════

def energy_score(scenarios, observed, n_pairs=500):
    n = len(scenarios)
    t1 = np.mean([np.linalg.norm(scenarios[i] - observed) for i in range(n)])
    rng = np.random.default_rng(42)
    pairs = rng.integers(0, n, size=(n_pairs, 2))
    t2 = np.mean([np.linalg.norm(scenarios[pairs[k, 0]] - scenarios[pairs[k, 1]])
                   for k in range(n_pairs)])
    return t1 - 0.5 * t2


def variogram_score(scenarios, observed, p=0.5):
    N = scenarios.shape[1]
    if N < 2:
        return 0.0
    tot, np_ = 0.0, 0
    for i in range(N):
        for j in range(i + 1, N):
            od = np.abs(observed[i] - observed[j]) ** p
            sd = (np.abs(scenarios[:, i] - scenarios[:, j]) ** p).mean()
            tot += (od - sd) ** 2
            np_ += 1
    return tot / max(np_, 1)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 95)
    log.info("WORLD MODEL V5 — NEURAL REGIME + GARCH-FHS")
    log.info("Key: Neural regime classifier for block selection, GARCH does the rest")
    log.info("Key: NO vol modification, NO correlation prediction, NO Iman-Conover")
    print("=" * 95)

    returns, asset_names, vix, macro_features = load_data()
    T, n_assets = returns.shape
    n_macro = macro_features.shape[1]
    dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    train_end = T - 504

    log.info(f"Data: {T} days | Train: {train_end} | Test: 504 | "
             f"Assets: {n_assets} | Macro feats: {n_macro} | Device: {dev}")

    # Train regime classifier
    log.info("Training Neural Regime Classifier...")
    model = NeuralRegimeClassifier(n_macro_feat=n_macro, n_regimes=6, hidden=32)
    model = train_regime_model(model, macro_features[:train_end], returns[:train_end],
                               dev, n_epochs=80)

    # Precompute all regime embeddings once
    log.info("Precomputing regime embeddings...")
    all_regime_probs, all_recency = precompute_regime_embeddings(
        model, macro_features, dev)

    # Eval
    results = {h: {k: [] for k in
                   ['wm_e', 'garch_e', 'garch_ve_e', 'bb_e',
                    'wm_v', 'garch_v', 'garch_ve_v', 'bb_v']}
               for h in HORIZONS}
    test_ret = returns[train_end:]

    log.info(f"Evaluating: {N_EVAL} windows × {N_SC} scenarios × {n_assets} assets")

    for idx in range(N_EVAL):
        t = idx * STRIDE
        if t + max(HORIZONS) > len(test_ret):
            break
        t_abs = train_end + t

        innov, fvols, gparams = precompute_garch(returns, t_abs)
        vh = vix[:t_abs]
        reg = classify_regime_vix(vh)
        cr = reg[-1]

        for h in HORIZONS:
            if t + h > len(test_ret):
                continue
            obs = test_ret[t:t + h].sum(0)

            bl = 1 if h <= 5 else (3 if h <= 10 else 5)
            hl = 200 if h <= 5 else 350

            # 1. World Model V5 (neural regime + GARCH-FHS vol evolution)
            wm_sc = generate_v5_scenarios(
                model, macro_features, innov, fvols, gparams,
                all_regime_probs, all_recency, t_abs, h, N_SC, dev, seed=42)

            # 2. GARCH-FHS (constant vol)
            g_sc = resample_fhs_const_vol(innov, fvols, reg, cr,
                                          N_SC, h, bl=bl, hl=hl, seed=42)

            # 3. GARCH-FHS with vol evolution
            gve_sc = resample_fhs_vol_evolution(innov, fvols, gparams, reg, cr,
                                                N_SC, h, bl=bl, hl=hl, seed=42)

            # 4. Block bootstrap
            ctx = returns[t_abs - 40:t_abs]
            rng_bb = np.random.default_rng(42 + idx * 100 + h)
            bb_sc = np.zeros((N_SC, h, n_assets))
            for s in range(N_SC):
                pos = 0
                while pos < h:
                    st = rng_bb.integers(0, max(1, len(ctx) - 10))
                    c = min(10, h - pos)
                    bb_sc[s, pos:pos + c] = ctx[st:st + c]
                    pos += c

            for pfx, sc in [('wm', wm_sc), ('garch', g_sc), ('garch_ve', gve_sc), ('bb', bb_sc)]:
                cum = sc[:, :h].sum(1)
                results[h][f'{pfx}_e'].append(energy_score(cum, obs))
                results[h][f'{pfx}_v'].append(variogram_score(cum, obs))

        if (idx + 1) % 10 == 0:
            log.info(f"  {idx + 1}/{N_EVAL} done ({time.time() - t0:.0f}s)")

    # ── Report ────────────────────────────────────────────────────
    from scipy.stats import ttest_rel

    np_total = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 95)
    print(f"  WORLD MODEL V5 CHAMPIONSHIP — {n_assets} assets × {n_macro} macro features")
    print(f"  Neural Regime Classification + GARCH-FHS Vol Evolution")
    print("=" * 95)
    print(f"  Regime model: {np_total:,} params | GRU + regime head")
    print(f"  Training: 80 regime classification epochs")
    print(f"  Eval: {N_EVAL} × {N_SC} scenarios | Stride: {STRIDE}")
    print("-" * 95)

    verdicts = []
    for h in HORIZONS:
        r = results[h]
        nw = len(r['garch_e'])
        if nw < 2:
            continue

        we = np.mean(r['wm_e'])
        ge = np.mean(r['garch_e'])
        gve = np.mean(r['garch_ve_e'])
        be = np.mean(r['bb_e'])
        wv = np.mean(r['wm_v'])
        gv = np.mean(r['garch_v'])
        gvev = np.mean(r['garch_ve_v'])
        bv = np.mean(r['bb_v'])

        we_pct = (be - we) / be * 100
        we_p = ttest_rel(r['bb_e'], r['wm_e']).pvalue

        ge_pct = (be - ge) / be * 100
        ge_p = ttest_rel(r['bb_e'], r['garch_e']).pvalue

        gve_pct = (be - gve) / be * 100
        gve_p = ttest_rel(r['bb_e'], r['garch_ve_e']).pvalue

        wg_pct = (ge - we) / ge * 100
        wg_p = ttest_rel(r['garch_e'], r['wm_e']).pvalue

        wgve_pct = (gve - we) / gve * 100
        wgve_p = ttest_rel(r['garch_ve_e'], r['wm_e']).pvalue

        wgv_pct = (gv - wv) / gv * 100
        wgv_p = ttest_rel(r['garch_v'], r['wm_v']).pvalue

        tag = "WIN" if we_pct > 0 and we_p < 0.05 else \
              "LOSE" if we_pct < 0 and we_p < 0.05 else "TIE"
        verdicts.append((h, tag, we_pct, we_p))

        print(f"\n  {h}-DAY HORIZON ({nw} windows) [{tag}]")
        print(f"  {'─' * 90}")
        print(f"    ENERGY SCORE (lower = better):")
        print(f"      World Model V5:         {we:.5f}  vs BB: {we_pct:+.1f}%  p={we_p:.6f}")
        print(f"      GARCH-FHS (const vol):  {ge:.5f}  vs BB: {ge_pct:+.1f}%  p={ge_p:.6f}")
        print(f"      GARCH-FHS (vol evol):   {gve:.5f}  vs BB: {gve_pct:+.1f}%  p={gve_p:.6f}")
        print(f"      Block Bootstrap:        {be:.5f}")
        print(f"      >>> V5 lift over GARCH (const):  {wg_pct:+.2f}%  p={wg_p:.4f}")
        print(f"      >>> V5 lift over GARCH (evol):   {wgve_pct:+.2f}%  p={wgve_p:.4f}")
        print(f"    VARIOGRAM SCORE (lower = better):")
        print(f"      World Model V5:         {wv:.6f}")
        print(f"      GARCH-FHS (const vol):  {gv:.6f}")
        print(f"      GARCH-FHS (vol evol):   {gvev:.6f}")
        print(f"      Block Bootstrap:        {bv:.6f}")
        print(f"      >>> V5 variogram lift over GARCH: {wgv_pct:+.2f}%  p={wgv_p:.4f}")

    total_time = time.time() - t0
    print(f"\n{'=' * 95}")
    v_str = ' | '.join(f'{h}d:{tag} ({pct:+.1f}%, p={pv:.4f})' for h, tag, pct, pv in verdicts)
    print(f"  VERDICT: {v_str}")
    print(f"  Time: {total_time:.0f}s")
    nw = sum(1 for v in verdicts if v[1] == 'WIN')
    print(f"  Score: {nw}/{len(verdicts)} horizons WON")
    print(f"{'=' * 95}")

    np.savez('results/wm_v5_benchmark.npz',
             **{f'{h}d_{k}': np.array(results[h][k]) for h in HORIZONS for k in results[h]})


if __name__ == '__main__':
    main()
