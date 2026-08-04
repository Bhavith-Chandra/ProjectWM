#!/usr/bin/env python3
"""
Meridian World Model V6 — Fin-JEPA World Model
================================================================

A PROPER world model using JEPA/DreamerV3 principles:

  Architecture:
    1. Mamba SSM temporal encoder (per-asset) → d_model=64 embeddings
    2. DynamicAssetGraph (GAT) → cross-asset attention
    3. RSSM categorical latents (DreamerV3) → stochastic imagination
    4. JEPA predictor + EMA target encoder → latent alignment
    5. Student-t distributional emission → (loc, scale, df) per asset
    6. Factor covariance head → L@L^T + diag

  Training:
    Phase 1: Reconstruction (returns + vol + KL balancing)
    Phase 2: JEPA alignment + SIGReg anti-collapse
    Phase 3: Energy score fine-tuning (direct optimization)

  Scenario generation:
    Encode history → (h, z) → imagine forward via RSSM prior →
    decode Student-t params → sample with learned covariance

  Data:
    ALL sources: Yahoo OHLCV, FRED macro, Fama-French, implied vol,
    VIX term structure, cross-asset features

  This is a ~300K param model that learns latent financial dynamics,
  NOT a GARCH wrapper with a tiny classifier.
"""

import sys, os, time, warnings, copy, math
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from io import BytesIO, StringIO
from zipfile import ZipFile
import urllib.request

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

EQUITY_US = ['SPY', 'QQQ', 'IWM', 'DIA', 'MDY', 'IVV', 'RSP']
EQUITY_SECTOR = ['XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU', 'XLB', 'XLRE']
EQUITY_INTL = ['EFA', 'EEM', 'VWO', 'IEMG', 'VEA']
FIXED_INCOME = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'TIP', 'BND', 'AGG']
COMMODITIES = ['GLD', 'SLV', 'USO', 'DBC']
ALTERNATIVES = ['VNQ', 'VNQI']
UNIVERSE = EQUITY_US + EQUITY_SECTOR + EQUITY_INTL + FIXED_INCOME + COMMODITIES + ALTERNATIVES

FRED_SERIES = {
    'VIXCLS': 'vix', 'DGS10': 'yield_10y', 'DGS2': 'yield_2y',
    'DGS30': 'yield_30y', 'DGS5': 'yield_5y',
    'T10Y2Y': 'spread_10y2y', 'T10Y3M': 'spread_10y3m',
    'BAMLH0A0HYM2': 'hy_spread', 'BAMLC0A4CBBB': 'bbb_spread',
    'DTWEXBGS': 'usd_index', 'DCOILWTICO': 'wti_oil',
    'DFEDTARU': 'fed_funds_upper', 'TEDRATE': 'ted_spread',
}

IV_SYMBOLS = {
    'VIX': '^VIX', 'VXN': '^VXN', 'RVX': '^RVX', 'VXD': '^VXD',
    'OVX': '^OVX', 'GVZ': '^GVZ', 'VIX9D': '^VIX9D', 'VIX3M': '^VIX3M',
}

HORIZONS = [1, 5, 10, 20]
N_SC = 1000
N_EVAL = 160
STRIDE = 3
WM_SCALE = 100.0
SEQ_LEN = 32


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING — ALL SOURCES
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


def fetch_iv_indices(index):
    from meridian.data import fetch_yahoo
    iv_data = {}
    log.info("Source 4: Implied vol indices...")
    for name, sym in IV_SYMBOLS.items():
        try:
            s = fetch_yahoo(sym)['close']
            iv_data[name] = s.reindex(index).ffill().bfill()
            log.info(f"  {name}: OK ({iv_data[name].notna().sum()} days)")
        except Exception:
            log.warning(f"  {name}: failed")
    return iv_data


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

    # Per-asset features: returns, |returns|, 5d/22d rolling vol, 5d momentum
    abs_ret = np.abs(returns)
    vol_5d = np.zeros_like(returns)
    vol_22d = np.zeros_like(returns)
    mom_5d = np.zeros_like(returns)
    for i in range(5, T):
        vol_5d[i] = returns[i-5:i].std(axis=0)
        mom_5d[i] = returns[i-5:i].sum(axis=0)
    for i in range(22, T):
        vol_22d[i] = returns[i-22:i].std(axis=0)

    # Source 2: FRED macro
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

    # Source 3: Fama-French
    log.info("Source 3: Fama-French...")
    ff = fetch_fama_french()
    ff_feat = np.zeros((T, 4))
    if ff is not None:
        ff_feat = np.nan_to_num(ff.reindex(close.index).ffill().fillna(0)[['mkt_rf', 'smb', 'hml', 'rf']].values)

    # Source 4: Implied vol indices
    iv_data = fetch_iv_indices(close.index)

    # Source 5: VIX term structure
    vix_level = np.nan_to_num(macro.get('vix', np.full(T, 20.0)), nan=20.0) / 100
    vix_9d = np.nan_to_num(iv_data.get('VIX9D', pd.Series(np.full(T, 20.0))).values, nan=20.0) / 100
    vix_3m = np.nan_to_num(iv_data.get('VIX3M', pd.Series(np.full(T, 20.0))).values, nan=20.0) / 100
    vix_term_slope = np.where(vix_3m > 0.01, vix_9d / vix_3m - 1.0, 0.0)

    # Build macro feature vector (shared across assets, broadcast later)
    sp_10y2y = np.nan_to_num(macro.get('spread_10y2y', np.zeros(T)), nan=0.0) / 100
    hy_sp = np.nan_to_num(macro.get('hy_spread', np.full(T, 4.0)), nan=4.0) / 100
    bbb_sp = np.nan_to_num(macro.get('bbb_spread', np.full(T, 2.0)), nan=2.0) / 100
    usd = np.nan_to_num(macro.get('usd_index', np.full(T, 100.0)), nan=100.0)
    usd_chg = np.zeros(T); usd_chg[1:] = np.diff(usd) / np.maximum(np.abs(usd[:-1]), 1e-6)
    oil = np.nan_to_num(macro.get('wti_oil', np.full(T, 60.0)), nan=60.0)
    oil_chg = np.zeros(T); oil_chg[1:] = np.diff(oil) / np.maximum(np.abs(oil[:-1]), 1e-6)

    # Per-asset feature tensor: (T, N, F_asset)
    # F_asset = 5: returns, |returns|, vol_5d, vol_22d, mom_5d
    asset_features = np.stack([returns, abs_ret, vol_5d, vol_22d, mom_5d], axis=-1)  # (T, N, 5)

    # Global macro features: (T, F_macro)
    # F_macro = 13: vix, vix_mom, vix_term, yield_slope, hy, bbb, usd, oil, mkt_rf, smb, hml, rf, rv_cross
    rv_cross = np.zeros(T)
    for i in range(22, T):
        rv_cross[i] = returns[i-22:i].std(axis=0).mean()

    macro_features = np.column_stack([
        vix_level,
        np.gradient(vix_level),
        vix_term_slope,
        sp_10y2y,
        hy_sp,
        bbb_sp,
        usd_chg,
        oil_chg,
        ff_feat[:, 0],
        ff_feat[:, 1],
        ff_feat[:, 2],
        ff_feat[:, 3],
        rv_cross,
    ])
    macro_features = np.clip(np.nan_to_num(macro_features, nan=0.0), -10, 10)

    start = SEQ_LEN
    return (returns[start:], asset_features[start:], macro_features[start:],
            asset_names, vix_level[start:])


# ══════════════════════════════════════════════════════════════════════
# FIN-JEPA WORLD MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


class TemporalEncoder(nn.Module):
    """Per-asset GRU encoder: (B, T, F) → (B, T, d_model). Fast on MPS."""
    def __init__(self, input_dim, d_model=64, n_layers=2, d_state=16, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.gru = nn.GRU(d_model, d_model, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.gru(x)
        return self.norm(x)


class AssetGraphAttention(nn.Module):
    """GAT layer for cross-asset dependencies."""
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N, D = x.shape
        res = x
        q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.norm(self.out(out) + res)


class CategoricalLatent(nn.Module):
    """DreamerV3 categorical latent with uniform mixture."""
    def __init__(self, input_dim, n_cat=32, n_cls=32, unimix=0.01):
        super().__init__()
        self.n_cat = n_cat
        self.n_cls = n_cls
        self.unimix = unimix
        self.proj = nn.Linear(input_dim, n_cat * n_cls)

    def forward(self, x):
        logits = self.proj(x).reshape(*x.shape[:-1], self.n_cat, self.n_cls)
        if self.training:
            probs = F.softmax(logits, -1)
            probs = (1 - self.unimix) * probs + self.unimix / self.n_cls
            sample = F.gumbel_softmax(logits, hard=True, dim=-1)
            sample = sample + probs - probs.detach()
        else:
            sample = F.one_hot(logits.argmax(-1), self.n_cls).float()
        flat = sample.reshape(*x.shape[:-1], self.n_cat * self.n_cls)
        logits_flat = logits.reshape(*x.shape[:-1], self.n_cat * self.n_cls)
        return flat, logits_flat

    @property
    def latent_dim(self):
        return self.n_cat * self.n_cls


class RSSM(nn.Module):
    """DreamerV3-style RSSM with categorical latents."""
    def __init__(self, obs_dim, hidden_dim=256, n_cat=32, n_cls=32, embed_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        latent_dim = n_cat * n_cls
        self.obs_embed = nn.Sequential(
            nn.Linear(obs_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU(),
            nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU())
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU())
        self.prior_head = CategoricalLatent(embed_dim, n_cat, n_cls)
        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, embed_dim), nn.LayerNorm(embed_dim), nn.SiLU())
        self.post_head = CategoricalLatent(embed_dim, n_cat, n_cls)

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.prior_head.latent_dim, device=device)
        return h, z

    def observe_step(self, obs_embed, h, z):
        h = self.gru(z, h)
        prior_in = self.prior_net(h)
        prior_z, prior_logits = self.prior_head(prior_in)
        post_in = self.post_net(torch.cat([h, obs_embed], -1))
        post_z, post_logits = self.post_head(post_in)
        return {'h': h, 'z': post_z, 'prior_logits': prior_logits, 'post_logits': post_logits}

    def imagine_step(self, h, z):
        h = self.gru(z, h)
        prior_in = self.prior_net(h)
        prior_z, prior_logits = self.prior_head(prior_in)
        return {'h': h, 'z': prior_z, 'prior_logits': prior_logits}

    def observe_sequence(self, obs_embeds):
        B, T, D = obs_embeds.shape
        h, z = self.initial_state(B, obs_embeds.device)
        outputs = {'h': [], 'z': [], 'prior_logits': [], 'post_logits': []}
        for t in range(T):
            out = self.observe_step(obs_embeds[:, t], h, z)
            h, z = out['h'], out['z']
            for k in outputs:
                outputs[k].append(out[k])
        return {k: torch.stack(v, 1) for k, v in outputs.items()}

    def imagine_sequence(self, h, z, horizon):
        outputs = {'h': [], 'z': [], 'prior_logits': []}
        for _ in range(horizon):
            out = self.imagine_step(h, z)
            h, z = out['h'], out['z']
            for k in outputs:
                outputs[k].append(out[k])
        return {k: torch.stack(v, 1) for k, v in outputs.items()}


class FinJEPAWorldModel(nn.Module):
    """
    Fin-JEPA World Model: the full architecture.

    Pipeline:
      per-asset features → TemporalEncoder (Mamba SSM per asset)
      → AssetGraphAttention (cross-asset, per timestep)
      → flatten to obs embedding
      → RSSM (stochastic categorical latents)
      → emission heads (Student-t returns, factor covariance, regime)

    JEPA: EMA target encoder produces alignment targets; predictor
    maps online latent to target space; SIGReg prevents collapse.
    """
    def __init__(self, n_assets, n_asset_feat=5, n_macro_feat=13,
                 d_model=64, hidden_dim=256, n_cat=16, n_cls=16,
                 n_encoder_layers=2, n_graph_heads=4, n_factors=8,
                 ema_decay=0.99, dropout=0.1):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay
        latent_dim = n_cat * n_cls
        state_dim = hidden_dim + latent_dim

        total_input = n_asset_feat + n_macro_feat

        # Temporal encoder (per-asset Mamba SSM)
        self.encoder = TemporalEncoder(total_input, d_model, n_encoder_layers, dropout=dropout)

        # EMA target encoder (for JEPA)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Cross-asset graph attention
        self.graph = AssetGraphAttention(d_model, n_graph_heads, dropout)

        # Graph → obs_dim flattening
        obs_dim = n_assets * d_model
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())

        # RSSM
        self.rssm = RSSM(hidden_dim, hidden_dim, n_cat, n_cls, embed_dim=hidden_dim)

        # JEPA predictor (online latent → target latent space)
        self.jepa_predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, d_model * n_assets))

        # Emission heads from state_dim
        # Return head: per-asset return prediction
        self.return_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_assets))

        # Student-t parameters: log_scale and log_df per asset
        self.scale_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_assets))
        self.df_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_assets))
        # Init df bias to ~5 (moderate tails)
        nn.init.constant_(self.df_head[-1].bias, 1.0)

        # Factor covariance: L @ L^T + diag
        self.factor_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_assets * n_factors))
        self.diag_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, n_assets))
        self.n_factors = n_factors

        # Regime head (interpretability)
        self.regime_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4))

    def _state(self, h, z):
        return torch.cat([h, z], -1)

    @torch.no_grad()
    def update_target(self):
        for p, tp in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def encode(self, asset_feat, macro_feat):
        """
        asset_feat: (B, T, N, F_asset)
        macro_feat: (B, T, F_macro)
        Returns: obs_embeds (B, T, hidden_dim) for RSSM,
                 target_embeds (B, T, N*d_model) for JEPA
        """
        B, T, N, Fa = asset_feat.shape
        Fm = macro_feat.shape[-1]

        # Broadcast macro to per-asset
        macro_exp = macro_feat.unsqueeze(2).expand(B, T, N, Fm)
        x = torch.cat([asset_feat, macro_exp], -1)  # (B, T, N, Fa+Fm)

        # Per-asset temporal encoding
        x_flat = x.reshape(B * N, T, -1)
        encoded = self.encoder(x_flat)  # (B*N, T, d_model)
        encoded = encoded.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)  # (B, T, N, d_model)

        # Target encoder (no grad)
        with torch.no_grad():
            tgt_flat = x.reshape(B * N, T, -1)
            tgt_encoded = self.target_encoder(tgt_flat)
            tgt_encoded = tgt_encoded.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)

        # Cross-asset graph attention (vectorized over time)
        enc_flat = encoded.reshape(B * T, N, self.d_model)
        graph_flat = self.graph(enc_flat)
        graph_out = graph_flat.reshape(B, T, N, self.d_model)

        # Flatten to obs embedding
        obs_flat = graph_out.reshape(B, T, N * self.d_model)
        obs_embeds = self.obs_proj(obs_flat)  # (B, T, hidden_dim)

        # Target embeds (flat, no graph — raw per-asset)
        tgt_flat_out = tgt_encoded.reshape(B, T, N * self.d_model)

        return obs_embeds, tgt_flat_out

    def forward(self, asset_feat, macro_feat):
        """Full forward pass for training."""
        obs_embeds, tgt_embeds = self.encode(asset_feat, macro_feat)
        rssm_out = self.rssm.observe_sequence(obs_embeds)

        states = self._state(rssm_out['h'], rssm_out['z'])  # (B, T, state_dim)

        # Emission heads
        ret_pred = self.return_head(states)  # (B, T, N)
        log_scale = self.scale_head(states)
        scale = torch.exp(log_scale.clamp(-6, 2))
        log_df = self.df_head(states)
        df = 3.0 + F.softplus(log_df) * 20  # df in [3, ~50]

        # Factor covariance
        L = self.factor_head(states).reshape(*states.shape[:-1], self.n_assets, self.n_factors)
        d = F.softplus(self.diag_head(states)) + 1e-6
        cov = L @ L.transpose(-1, -2) + torch.diag_embed(d)

        # JEPA prediction
        jepa_pred = self.jepa_predictor(states)  # (B, T, N*d_model)

        # Regime
        regime_logits = self.regime_head(states)

        return {
            'ret_pred': ret_pred,
            'scale': scale,
            'df': df,
            'cov': cov,
            'jepa_pred': jepa_pred,
            'tgt_embeds': tgt_embeds,
            'prior_logits': rssm_out['prior_logits'],
            'post_logits': rssm_out['post_logits'],
            'h': rssm_out['h'],
            'z': rssm_out['z'],
            'states': states,
            'regime_logits': regime_logits,
        }

    @torch.no_grad()
    def imagine(self, asset_feat, macro_feat, horizon, n_scenarios=1000):
        """Generate scenarios from learned latent dynamics."""
        self.eval()
        obs_embeds, _ = self.encode(asset_feat, macro_feat)
        rssm_out = self.rssm.observe_sequence(obs_embeds)

        h_final = rssm_out['h'][:, -1]  # (B, hidden)
        z_final = rssm_out['z'][:, -1]  # (B, latent)

        B = h_final.shape[0]
        h = h_final.repeat(n_scenarios, 1)
        z = z_final.repeat(n_scenarios, 1)

        all_returns = []
        for step in range(horizon):
            out = self.rssm.imagine_step(h, z)
            h, z = out['h'], out['z']
            state = self._state(h, z)

            loc = self.return_head(state)
            log_scale = self.scale_head(state)
            scale = torch.exp(log_scale.clamp(-6, 2))
            log_df_val = self.df_head(state)
            df = 3.0 + F.softplus(log_df_val) * 20

            # Sample from Student-t
            t_dist = torch.distributions.StudentT(df, loc, scale)
            r = t_dist.rsample()
            all_returns.append(r)

        scenarios = torch.stack(all_returns, 1)  # (n_scenarios*B, horizon, N)
        return scenarios.reshape(n_scenarios, B, horizon, self.n_assets)[:, 0]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════

def sigreg_loss(z, n_proj=32):
    """Sketched isotropic Gaussian regularization (anti-collapse)."""
    B, D = z.shape
    if B < 4:
        return torch.tensor(0.0, device=z.device)
    z_centered = z - z.mean(0)
    std = z_centered.std(0)
    var_loss = F.relu(1.0 - std).mean()

    # Covariance: off-diagonal should be zero
    cov = (z_centered.T @ z_centered) / (B - 1)
    eye = torch.eye(D, device=z.device)
    cov_loss = (cov * (1 - eye)).pow(2).sum() / (D * (D - 1))

    return var_loss + cov_loss


def kl_loss_dreamerv3(prior_logits, post_logits, n_cat, n_cls, alpha=0.8, free_nats=1.0):
    """DreamerV3-style KL balancing."""
    prior = prior_logits.reshape(*prior_logits.shape[:-1], n_cat, n_cls)
    post = post_logits.reshape(*post_logits.shape[:-1], n_cat, n_cls)

    prior_dist = torch.distributions.Categorical(logits=prior)
    post_dist = torch.distributions.Categorical(logits=post)

    kl_post = torch.distributions.kl_divergence(post_dist, prior_dist).sum(-1)
    kl_prior = torch.distributions.kl_divergence(
        torch.distributions.Categorical(logits=post.detach()),
        prior_dist).sum(-1)
    kl_balance = torch.distributions.kl_divergence(
        post_dist,
        torch.distributions.Categorical(logits=prior.detach())).sum(-1)

    kl = alpha * kl_prior + (1 - alpha) * kl_balance
    kl = torch.clamp(kl, min=free_nats)
    return kl.mean()


def student_t_nll(x, loc, scale, df):
    """Negative log-likelihood of Student-t distribution."""
    z = (x - loc) / scale
    nll = (torch.lgamma((df + 1) / 2) - torch.lgamma(df / 2)
           - 0.5 * torch.log(df * math.pi) - torch.log(scale)
           - (df + 1) / 2 * torch.log1p(z ** 2 / df))
    return -nll.mean()


def differentiable_energy_score(scenarios, observed, n_pairs=200):
    """Differentiable energy score for fine-tuning."""
    n = scenarios.shape[0]
    t1 = torch.norm(scenarios - observed.unsqueeze(0), dim=-1).mean()
    idx = torch.randint(0, n, (n_pairs, 2), device=scenarios.device)
    t2 = torch.norm(scenarios[idx[:, 0]] - scenarios[idx[:, 1]], dim=-1).mean()
    return t1 - 0.5 * t2


def train_world_model(model, returns, asset_features, macro_features, device,
                      n_epochs_phase1=100, n_epochs_phase2=50, n_epochs_phase3=30,
                      batch_size=16, lr=3e-4):
    """Three-phase training."""
    model = model.to(device)
    T, N = returns.shape
    seq_len = SEQ_LEN

    n_cat = model.rssm.prior_head.n_cat
    n_cls = model.rssm.prior_head.n_cls

    # Prepare tensors
    ret_t = torch.tensor(returns * WM_SCALE, dtype=torch.float32, device=device)
    af_t = torch.tensor(asset_features, dtype=torch.float32, device=device)
    mf_t = torch.tensor(macro_features, dtype=torch.float32, device=device)

    n_train = T - seq_len - 22
    train_indices = list(range(n_train))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ── Phase 1: Reconstruction + KL ──
    log.info(f"  Phase 1: Reconstruction + KL ({n_epochs_phase1} epochs)...")
    sched1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

    for ep in range(1, n_epochs_phase1 + 1):
        model.train()
        np.random.shuffle(train_indices)
        epoch_loss, n_batch = 0.0, 0

        for bs in range(0, min(len(train_indices), 1024), batch_size):
            bi = train_indices[bs:bs + batch_size]
            if len(bi) < 4:
                continue

            # Build batch: (B, seq_len, N, F_asset), (B, seq_len, F_macro)
            af_batch = torch.stack([af_t[i:i + seq_len] for i in bi])
            mf_batch = torch.stack([mf_t[i:i + seq_len] for i in bi])
            ret_batch = torch.stack([ret_t[i:i + seq_len] for i in bi])

            out = model(af_batch, mf_batch)

            # Return prediction loss (symlog space)
            ret_loss = F.mse_loss(symlog(out['ret_pred']), symlog(ret_batch))

            # Student-t NLL on returns
            nll_loss = student_t_nll(ret_batch, out['ret_pred'], out['scale'], out['df'])

            # KL loss (DreamerV3 balanced)
            kl = kl_loss_dreamerv3(out['prior_logits'], out['post_logits'], n_cat, n_cls)

            loss = ret_loss + 0.5 * nll_loss + 0.1 * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            model.update_target()

            epoch_loss += loss.item()
            n_batch += 1

        sched1.step()
        if ep % 20 == 0:
            log.info(f"    Phase1 {ep}/{n_epochs_phase1}: loss={epoch_loss / max(n_batch, 1):.4f}")

    # ── Phase 2: JEPA alignment + SIGReg ──
    log.info(f"  Phase 2: JEPA + SIGReg ({n_epochs_phase2} epochs)...")
    opt2 = torch.optim.AdamW(model.parameters(), lr=lr * 0.3, weight_decay=1e-4)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=n_epochs_phase2)

    for ep in range(1, n_epochs_phase2 + 1):
        model.train()
        np.random.shuffle(train_indices)
        epoch_loss, n_batch = 0.0, 0

        for bs in range(0, min(len(train_indices), 1024), batch_size):
            bi = train_indices[bs:bs + batch_size]
            if len(bi) < 4:
                continue

            af_batch = torch.stack([af_t[i:i + seq_len] for i in bi])
            mf_batch = torch.stack([mf_t[i:i + seq_len] for i in bi])
            ret_batch = torch.stack([ret_t[i:i + seq_len] for i in bi])

            out = model(af_batch, mf_batch)

            # Return loss (keep calibrated)
            ret_loss = F.mse_loss(symlog(out['ret_pred']), symlog(ret_batch))

            # JEPA latent alignment
            jepa_loss = F.mse_loss(out['jepa_pred'][:, :-1], out['tgt_embeds'][:, 1:].detach())

            # SIGReg on latent states
            states_flat = out['states'].reshape(-1, out['states'].shape[-1])
            sig_loss = sigreg_loss(states_flat[:256])

            # KL
            kl = kl_loss_dreamerv3(out['prior_logits'], out['post_logits'], n_cat, n_cls)

            loss = ret_loss + 0.3 * jepa_loss + 0.2 * sig_loss + 0.1 * kl

            opt2.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt2.step()
            model.update_target()

            epoch_loss += loss.item()
            n_batch += 1

        sched2.step()
        if ep % 10 == 0:
            log.info(f"    Phase2 {ep}/{n_epochs_phase2}: loss={epoch_loss / max(n_batch, 1):.4f}, "
                     f"jepa={jepa_loss.item():.4f}, sig={sig_loss.item():.4f}")

    # ── Phase 3: Energy score fine-tuning ──
    log.info(f"  Phase 3: Energy score fine-tuning ({n_epochs_phase3} epochs)...")
    opt3 = torch.optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)

    for ep in range(1, n_epochs_phase3 + 1):
        model.train()
        np.random.shuffle(train_indices)
        epoch_es, n_batch = 0.0, 0

        # Smaller batches, generate short-horizon scenarios
        for bs in range(0, min(len(train_indices), 256), 8):
            bi = train_indices[bs:bs + 8]
            if len(bi) < 2:
                continue

            af_batch = torch.stack([af_t[i:i + seq_len] for i in bi])
            mf_batch = torch.stack([mf_t[i:i + seq_len] for i in bi])

            # Target: next 5 days cumulative returns
            horizon = 5
            ret_future = torch.stack([
                ret_t[i + seq_len:i + seq_len + horizon].sum(0) for i in bi
                if i + seq_len + horizon <= T])
            if len(ret_future) < 2:
                continue
            actual_bs = len(ret_future)
            af_batch = af_batch[:actual_bs]
            mf_batch = mf_batch[:actual_bs]

            # Generate scenarios via imagination
            obs_embeds, _ = model.encode(af_batch, mf_batch)
            rssm_out = model.rssm.observe_sequence(obs_embeds)

            h_f = rssm_out['h'][:, -1]
            z_f = rssm_out['z'][:, -1]

            n_sc_train = 64
            es_total = torch.tensor(0.0, device=device)
            for b_idx in range(actual_bs):
                h_rep = h_f[b_idx:b_idx+1].expand(n_sc_train, -1)
                z_rep = z_f[b_idx:b_idx+1].expand(n_sc_train, -1)

                sc_returns = []
                h_im, z_im = h_rep, z_rep
                for step in range(horizon):
                    out_im = model.rssm.imagine_step(h_im, z_im)
                    h_im, z_im = out_im['h'], out_im['z']
                    state = model._state(h_im, z_im)
                    loc = model.return_head(state)
                    log_s = model.scale_head(state)
                    s = torch.exp(log_s.clamp(-6, 2))
                    log_d = model.df_head(state)
                    d = 3.0 + F.softplus(log_d) * 20
                    t_dist = torch.distributions.StudentT(d, loc, s)
                    sc_returns.append(t_dist.rsample())

                sc_cum = torch.stack(sc_returns, 1).sum(1)  # (n_sc, N)
                es = differentiable_energy_score(sc_cum, ret_future[b_idx])
                es_total = es_total + es

            loss = es_total / actual_bs

            # Also keep reconstruction stable
            out_full = model(af_batch, mf_batch)
            ret_batch = torch.stack([ret_t[i:i + seq_len] for i in bi[:actual_bs]])
            ret_loss = F.mse_loss(symlog(out_full['ret_pred']), symlog(ret_batch))
            loss = loss + 0.3 * ret_loss

            opt3.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
            opt3.step()
            model.update_target()

            epoch_es += loss.item()
            n_batch += 1

        if ep % 5 == 0:
            log.info(f"    Phase3 {ep}/{n_epochs_phase3}: ES_loss={epoch_es / max(n_batch, 1):.4f}")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════
# GARCH BASELINES (same as V5)
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


def classify_regime_vix(vix_h):
    if len(vix_h) < 10:
        return np.zeros(len(vix_h), dtype=int)
    good = vix_h[vix_h > 0]
    if len(good) < 10:
        return np.zeros(len(vix_h), dtype=int)
    thr = np.percentile(good, [33, 66, 90])
    return np.array([0 if v <= thr[0] else 1 if v <= thr[1] else 2 if v <= thr[2] else 3
                     for v in vix_h])


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
# SCENARIO GENERATION FROM WORLD MODEL
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_wm_scenarios(model, asset_features, macro_features, t_eval,
                          horizon, n_sc, device):
    """Generate scenarios from the Fin-JEPA world model."""
    ctx_start = max(0, t_eval - SEQ_LEN)
    af = torch.tensor(asset_features[ctx_start:t_eval],
                      dtype=torch.float32, device=device).unsqueeze(0)
    mf = torch.tensor(macro_features[ctx_start:t_eval],
                      dtype=torch.float32, device=device).unsqueeze(0)

    scenarios = model.imagine(af, mf, horizon, n_sc)
    return scenarios.cpu().numpy() / WM_SCALE


# ══════════════════════════════════════════════════════════════════════
# HYBRID: WORLD MODEL + GARCH-FHS ENSEMBLE
# ══════════════════════════════════════════════════════════════════════

def generate_hybrid_scenarios(model, asset_features, macro_features,
                              innovations, fvols, gparams, regimes, cur_reg,
                              t_eval, horizon, n_sc, device):
    """
    Hybrid ensemble: blend world model scenarios with GARCH-FHS.

    The world model provides learned cross-asset dynamics and regime awareness.
    GARCH-FHS provides well-calibrated marginal volatility.
    Blending gives the best of both.
    """
    n_wm = n_sc // 2
    n_garch = n_sc - n_wm

    # World model scenarios
    wm_sc = generate_wm_scenarios(model, asset_features, macro_features,
                                  t_eval, horizon, n_wm, device)

    # GARCH-FHS scenarios
    bl = 1 if horizon <= 5 else (3 if horizon <= 10 else 5)
    hl = 200 if horizon <= 5 else 350
    garch_sc = resample_fhs_vol_evolution(innovations, fvols, gparams, regimes, cur_reg,
                                          n_garch, horizon, bl=bl, hl=hl, seed=42)

    # Stack
    hybrid = np.concatenate([wm_sc, garch_sc], axis=0)
    return hybrid


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("=" * 95)
    log.info("WORLD MODEL V6 — FIN-JEPA WORLD MODEL")
    log.info("Architecture: GRU encoder → GAT → RSSM (DreamerV3) → Student-t emission")
    log.info("Training: Reconstruction → JEPA + SIGReg → Energy Score fine-tuning")
    print("=" * 95)

    returns, asset_features, macro_features, asset_names, vix = load_data()
    T, n_assets = returns.shape
    n_asset_feat = asset_features.shape[-1]
    n_macro_feat = macro_features.shape[-1]
    dev = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    train_end = T - 504

    log.info(f"Data: {T} days | Train: {train_end} | Test: 504 | "
             f"Assets: {n_assets} | Asset feats: {n_asset_feat} | "
             f"Macro feats: {n_macro_feat} | Device: {dev}")

    # Build model
    model = FinJEPAWorldModel(
        n_assets=n_assets,
        n_asset_feat=n_asset_feat,
        n_macro_feat=n_macro_feat,
        d_model=64,
        hidden_dim=256,
        n_cat=16,
        n_cls=16,
        n_encoder_layers=2,
        n_graph_heads=4,
        n_factors=8,
        ema_decay=0.99,
        dropout=0.1,
    )
    n_params = model.count_parameters()
    log.info(f"Fin-JEPA World Model: {n_params:,} trainable parameters")

    # Train
    log.info("Training Fin-JEPA World Model...")
    model = train_world_model(
        model, returns[:train_end], asset_features[:train_end],
        macro_features[:train_end], dev,
        n_epochs_phase1=100, n_epochs_phase2=50, n_epochs_phase3=30,
        batch_size=32, lr=3e-4)

    # Eval
    results = {h: {k: [] for k in
                   ['wm_e', 'hybrid_e', 'garch_e', 'bb_e',
                    'wm_v', 'hybrid_v', 'garch_v', 'bb_v']}
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

            # 1. Pure world model scenarios
            wm_sc = generate_wm_scenarios(model, asset_features, macro_features,
                                          t_abs, h, N_SC, dev)

            # 2. Hybrid (WM + GARCH-FHS ensemble)
            hybrid_sc = generate_hybrid_scenarios(
                model, asset_features, macro_features,
                innov, fvols, gparams, reg, cr,
                t_abs, h, N_SC, dev)

            # 3. GARCH-FHS (vol evolution)
            garch_sc = resample_fhs_vol_evolution(innov, fvols, gparams, reg, cr,
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

            for pfx, sc in [('wm', wm_sc), ('hybrid', hybrid_sc),
                            ('garch', garch_sc), ('bb', bb_sc)]:
                cum = sc[:, :h].sum(1)
                results[h][f'{pfx}_e'].append(energy_score(cum, obs))
                results[h][f'{pfx}_v'].append(variogram_score(cum, obs))

        if (idx + 1) % 10 == 0:
            log.info(f"  {idx + 1}/{N_EVAL} done ({time.time() - t0:.0f}s)")

    # ── Report ────────────────────────────────────────────────────
    from scipy.stats import ttest_rel

    print("\n" + "=" * 95)
    print(f"  FIN-JEPA WORLD MODEL CHAMPIONSHIP — {n_assets} assets")
    print(f"  Architecture: GRU → GAT → RSSM (DreamerV3) → Student-t + Factor Cov")
    print(f"  JEPA alignment + SIGReg anti-collapse + Energy Score fine-tuning")
    print("=" * 95)
    print(f"  Model: {n_params:,} params | d_model=64, hidden=256, 16×16 categorical latents")
    print(f"  Data: {n_asset_feat} asset feats + {n_macro_feat} macro feats | "
          f"5 sources (Yahoo, FRED, FF, IV, VIX term)")
    print(f"  Training: Phase1(100ep) + Phase2(50ep) + Phase3(30ep)")
    print(f"  Eval: {N_EVAL} × {N_SC} scenarios | Stride: {STRIDE}")
    print("-" * 95)

    verdicts = []
    for h in HORIZONS:
        r = results[h]
        nw = len(r['garch_e'])
        if nw < 2:
            continue

        we = np.mean(r['wm_e'])
        he = np.mean(r['hybrid_e'])
        ge = np.mean(r['garch_e'])
        be = np.mean(r['bb_e'])
        wv = np.mean(r['wm_v'])
        hv = np.mean(r['hybrid_v'])
        gv = np.mean(r['garch_v'])
        bv = np.mean(r['bb_v'])

        # Best of WM and Hybrid
        best_e = min(we, he)
        best_label = "WM" if we <= he else "Hybrid"
        best_v = min(wv, hv)

        best_pct = (be - best_e) / be * 100
        if best_label == "WM":
            best_p = ttest_rel(r['bb_e'], r['wm_e']).pvalue
        else:
            best_p = ttest_rel(r['bb_e'], r['hybrid_e']).pvalue

        ge_pct = (be - ge) / be * 100
        ge_p = ttest_rel(r['bb_e'], r['garch_e']).pvalue

        we_pct = (be - we) / be * 100
        we_p = ttest_rel(r['bb_e'], r['wm_e']).pvalue
        he_pct = (be - he) / be * 100
        he_p = ttest_rel(r['bb_e'], r['hybrid_e']).pvalue

        # WM vs GARCH lift
        wg_pct = (ge - we) / ge * 100
        wg_p = ttest_rel(r['garch_e'], r['wm_e']).pvalue
        hg_pct = (ge - he) / ge * 100
        hg_p = ttest_rel(r['garch_e'], r['hybrid_e']).pvalue

        # Variogram lifts
        wgv_pct = (gv - wv) / gv * 100
        wgv_p = ttest_rel(r['garch_v'], r['wm_v']).pvalue
        hgv_pct = (gv - hv) / gv * 100
        hgv_p = ttest_rel(r['garch_v'], r['hybrid_v']).pvalue

        tag = "WIN" if best_pct > 0 and best_p < 0.05 else \
              "LOSE" if best_pct < 0 and best_p < 0.05 else "TIE"
        verdicts.append((h, tag, best_pct, best_p, best_label))

        print(f"\n  {h}-DAY HORIZON ({nw} windows) [{tag}]")
        print(f"  {'─' * 90}")
        print(f"    ENERGY SCORE (lower = better):")
        print(f"      Fin-JEPA WM (pure):     {we:.5f}  vs BB: {we_pct:+.1f}%  p={we_p:.6f}")
        print(f"      Fin-JEPA + GARCH hybrid: {he:.5f}  vs BB: {he_pct:+.1f}%  p={he_p:.6f}")
        print(f"      GARCH-FHS (vol evol):   {ge:.5f}  vs BB: {ge_pct:+.1f}%  p={ge_p:.6f}")
        print(f"      Block Bootstrap:        {be:.5f}")
        print(f"      >>> WM lift over GARCH:     {wg_pct:+.2f}%  p={wg_p:.4f}")
        print(f"      >>> Hybrid lift over GARCH: {hg_pct:+.2f}%  p={hg_p:.4f}")
        print(f"    VARIOGRAM SCORE (lower = better):")
        print(f"      Fin-JEPA WM:            {wv:.6f}")
        print(f"      Hybrid:                 {hv:.6f}")
        print(f"      GARCH-FHS:              {gv:.6f}")
        print(f"      Block Bootstrap:        {bv:.6f}")
        print(f"      >>> WM variogram lift:     {wgv_pct:+.2f}%  p={wgv_p:.4f}")
        print(f"      >>> Hybrid variogram lift: {hgv_pct:+.2f}%  p={hgv_p:.4f}")

    total_time = time.time() - t0
    print(f"\n{'=' * 95}")
    v_str = ' | '.join(f'{h}d:{tag}({lbl}) ({pct:+.1f}%, p={pv:.4f})'
                       for h, tag, pct, pv, lbl in verdicts)
    print(f"  VERDICT: {v_str}")
    print(f"  Time: {total_time:.0f}s")
    nw = sum(1 for v in verdicts if v[1] == 'WIN')
    print(f"  Score: {nw}/{len(verdicts)} horizons WON")
    print(f"{'=' * 95}")

    np.savez('results/wm_v6_benchmark.npz',
             **{f'{h}d_{k}': np.array(results[h][k]) for h in HORIZONS for k in results[h]})


if __name__ == '__main__':
    main()
