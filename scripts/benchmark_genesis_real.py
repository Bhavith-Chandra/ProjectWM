#!/usr/bin/env python3
"""
Meridian Genesis — Real Data Benchmark
========================================
Trains Genesis on real market data (Yahoo Finance + FRED)
and evaluates against GARCH-FHS baseline.

Uses a smaller config to run on CPU in reasonable time.
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from meridian.data import fetch_yahoo, fetch_fred, DEFAULT_START
from meridian.features import build_asset_frame, realized_variance
from meridian.world_model.genesis import MeridianGenesis, GenesisConfig
from meridian.world_model.rssm import symlog, symexp
from meridian.world_model.conformal import AdaptiveConformalInference


# ─── Asset universe ─────────────────────────────────────────────────────

UNIVERSE = {
    'equity': ['SPY', 'QQQ', 'IWM', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP'],
    'fixed_income': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG'],
    'commodity': ['GLD', 'SLV', 'USO', 'DBC', 'DBA'],
    'fx': ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X'],
    'alternatives': ['VNQ', 'VNQI', 'IEMG', 'EFA', 'EEM'],
    'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD'],
}

ASSET_NAMES = []
ASSET_CLASS_MAP = {}
idx = 0
for cls, tickers in UNIVERSE.items():
    ASSET_CLASS_MAP[cls] = list(range(idx, idx + len(tickers)))
    ASSET_NAMES.extend(tickers)
    idx += len(tickers)

N_ASSETS = len(ASSET_NAMES)
FEATURE_COLS = ['ret', 'rv', 'log_rv', 'har_d', 'har_w', 'har_m',
                'ret_abs', 'ret_5', 'rv_cc']


# ─── Data loading ───────────────────────────────────────────────────────

def load_real_data(start='2015-01-01'):
    """Load real market data for all assets."""
    print(f'Fetching data for {N_ASSETS} assets from {start}...')

    macro_series = {}
    for series_id in ['VIXCLS', 'DGS10', 'DGS2']:
        try:
            macro_series[series_id] = fetch_fred(series_id, start=start)
        except Exception as e:
            print(f'  Warning: FRED {series_id} failed: {e}')

    if macro_series:
        macro = pd.DataFrame(macro_series).ffill().dropna()
    else:
        macro = None

    all_frames = {}
    failed = []
    for ticker in ASSET_NAMES:
        try:
            ohlc = fetch_yahoo(ticker, start=start)
            if len(ohlc) < 252:
                print(f'  {ticker}: only {len(ohlc)} rows, skipping')
                failed.append(ticker)
                continue
            frame = build_asset_frame(ohlc, macro)
            all_frames[ticker] = frame
            print(f'  {ticker}: {len(frame)} rows, {frame.columns.tolist()[:5]}...')
        except Exception as e:
            print(f'  {ticker}: FAILED ({e})')
            failed.append(ticker)

    if failed:
        print(f'\n  Failed tickers: {failed}')

    return all_frames, macro


class RealMarketDataset(Dataset):
    """
    Sliding window dataset from real market data.
    Groups assets by class for sheaf encoder input.
    """

    def __init__(self, frames: dict, window: int = 120,
                 stride: int = 5, feature_cols=None):
        self.window = window
        self.feature_cols = feature_cols or FEATURE_COLS

        common_dates = None
        for ticker, df in frames.items():
            dates = df.index
            if common_dates is None:
                common_dates = dates
            else:
                common_dates = common_dates.intersection(dates)

        common_dates = common_dates.sort_values()
        self.dates = common_dates
        print(f'  Common date range: {common_dates[0]} to {common_dates[-1]} ({len(common_dates)} days)')

        self.data = {}
        self.returns = {}
        for ticker, df in frames.items():
            df_aligned = df.loc[common_dates]
            feats = []
            for col in self.feature_cols:
                if col in df_aligned.columns:
                    feats.append(df_aligned[col].values)
                else:
                    feats.append(np.zeros(len(common_dates)))
            self.data[ticker] = np.stack(feats, axis=-1).astype(np.float32)
            self.returns[ticker] = df_aligned['ret'].values.astype(np.float32) if 'ret' in df_aligned.columns else np.zeros(len(common_dates), dtype=np.float32)

        self.targets = {}
        for ticker, df in frames.items():
            df_aligned = df.loc[common_dates]
            if 'y' in df_aligned.columns:
                self.targets[ticker] = df_aligned['y'].values.astype(np.float32)
            elif 'r_next' in df_aligned.columns:
                self.targets[ticker] = df_aligned['r_next'].values.astype(np.float32)
            else:
                self.targets[ticker] = np.zeros(len(common_dates), dtype=np.float32)

        T = len(common_dates)
        self.indices = list(range(window, T - 1, stride))
        print(f'  Samples: {len(self.indices)} (window={window}, stride={stride})')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        w = self.window

        features = {}
        for cls_name, asset_indices in ASSET_CLASS_MAP.items():
            tickers = [ASSET_NAMES[i] for i in asset_indices]
            available = [tk for tk in tickers if tk in self.data]
            if not available:
                n_feats = len(self.feature_cols)
                features[cls_name] = torch.zeros(w, len(tickers), n_feats)
                continue

            cls_data = []
            for tk in tickers:
                if tk in self.data:
                    cls_data.append(torch.from_numpy(self.data[tk][t-w:t]))
                else:
                    cls_data.append(torch.zeros(w, len(self.feature_cols)))
            features[cls_name] = torch.stack(cls_data, dim=1)

        ret_list = []
        for tk in ASSET_NAMES:
            if tk in self.returns:
                ret_list.append(torch.from_numpy(self.returns[tk][t-w:t]))
            else:
                ret_list.append(torch.zeros(w))
        returns_seq = torch.stack(ret_list, dim=-1)

        target_list = []
        for tk in ASSET_NAMES:
            if tk in self.targets:
                target_list.append(self.targets[tk][t])
            else:
                target_list.append(0.0)
        target = torch.tensor(target_list, dtype=torch.float32)

        return {
            'features': features,
            'returns_seq': returns_seq,
            'target': target,
            'date_idx': t,
        }


def collate_real(batch):
    features = {}
    for cls_name in batch[0]['features']:
        features[cls_name] = torch.stack([b['features'][cls_name] for b in batch])
    returns_seq = torch.stack([b['returns_seq'] for b in batch])
    target = torch.stack([b['target'] for b in batch])
    return {'features': features, 'returns_seq': returns_seq, 'target': target}


# ─── Small config for CPU training ──────────────────────────────────────

def make_small_config():
    """Smaller config that runs on CPU in minutes, not hours."""
    n_feats = len(FEATURE_COLS)
    config = GenesisConfig(
        n_assets=N_ASSETS,
        asset_class_map=ASSET_CLASS_MAP,
        input_dims={
            'equity': n_feats,
            'fixed_income': n_feats,
            'commodity': n_feats,
            'fx': n_feats,
            'alternatives': n_feats,
            'crypto': n_feats,
        },
        common_dim=64,
        n_diffusion_layers=2,
        h_dim=32,
        s_dim=16,
        e_dim=16,
        rg_n_heads=4,
        hidden_dim=256,
        topo_dim=32,
        belief_dim=256,
        n_scenarios=32,
        max_weight=0.15,
        n_factors=16,
        dropout=0.1,
        return_window=60,
    )
    return config


# ─── GARCH baseline ────────────────────────────────────────────────────

def garch_baseline(returns: np.ndarray) -> np.ndarray:
    """
    Simple GARCH(1,1) variance forecast as baseline.
    returns: (T, N) array of daily returns
    output: (T, N) array of one-step-ahead variance forecasts
    """
    T, N = returns.shape
    omega = 0.00001
    alpha = 0.05
    beta = 0.90
    var_forecast = np.zeros_like(returns)
    sigma2 = np.var(returns[:22], axis=0)

    for t in range(1, T):
        var_forecast[t] = omega + alpha * returns[t-1]**2 + beta * sigma2
        sigma2 = var_forecast[t]

    return var_forecast


# ─── Main ──────────────────────────────────────────────────────────────

def main():
    device = torch.device('cpu')
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print('=' * 60)
    print('MERIDIAN GENESIS — REAL DATA BENCHMARK')
    print('=' * 60)

    # Load data
    frames, macro = load_real_data(start='2015-01-01')
    available_assets = list(frames.keys())
    print(f'\nLoaded {len(available_assets)}/{N_ASSETS} assets')

    if len(available_assets) < 10:
        print('ERROR: Too few assets loaded. Check network connection.')
        return

    # Build dataset
    print('\nBuilding dataset...')
    window = 80
    dataset = RealMarketDataset(frames, window=window, stride=10)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = n_total - n_train

    train_ds = torch.utils.data.Subset(dataset, range(n_train))
    val_ds = torch.utils.data.Subset(dataset, range(n_train, n_total))

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True,
                          collate_fn=collate_real, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=4, collate_fn=collate_real,
                        drop_last=True)

    print(f'  Train: {n_train}, Val: {n_val}')

    # Build model
    config = make_small_config()
    model = MeridianGenesis(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {n_params:,} parameters (small config for CPU)')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,
                                   weight_decay=0.01)

    # Train
    n_epochs = 5
    print(f'\nTraining {n_epochs} epochs on REAL market data...')
    print('-' * 60)

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0
        n_batches = 0
        t0 = time.time()

        for batch in train_dl:
            features = {k: v.to(device) for k, v in batch['features'].items()}
            returns_seq = batch['returns_seq'].to(device)
            target = batch['target'].to(device)

            # Replace NaN with 0
            for k in features:
                features[k] = features[k].nan_to_num(0.0)
            returns_seq = returns_seq.nan_to_num(0.0)
            target = target.nan_to_num(0.0)

            try:
                output = model(features, returns_seq=returns_seq)
                losses = model.loss(output, {'returns': target})
                loss = losses['total']

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1
            except RuntimeError as e:
                print(f'  Batch error: {e}')
                continue

        dt = time.time() - t0
        avg_loss = total_loss / max(1, n_batches)
        print(f'  Epoch {epoch} | loss {avg_loss:.4f} | {n_batches} batches | {dt:.1f}s')

    # Evaluate
    print('\n' + '=' * 60)
    print('EVALUATION ON HELD-OUT DATA')
    print('=' * 60)

    model.eval()
    all_preds = []
    all_targets = []
    all_portfolio = []

    with torch.no_grad():
        for batch in val_dl:
            features = {k: v.to(device).nan_to_num(0.0) for k, v in batch['features'].items()}
            returns_seq = batch['returns_seq'].to(device).nan_to_num(0.0)
            target = batch['target'].to(device).nan_to_num(0.0)

            try:
                output = model(features, returns_seq=returns_seq)
                pred = symexp(output['returns'])
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                all_portfolio.append(output['portfolio']['weights'].cpu())
            except RuntimeError:
                continue

    if not all_preds:
        print('No valid predictions generated.')
        return

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    weights = torch.cat(all_portfolio)

    # Metrics
    mse = F.mse_loss(preds, targets).item()
    mae = (preds - targets).abs().mean().item()
    direction = ((preds.sign() == targets.sign()).float().mean()).item()

    # Per-asset correlation
    corrs = []
    for i in range(min(N_ASSETS, preds.shape[1])):
        p, t = preds[:, i], targets[:, i]
        if p.std() > 1e-8 and t.std() > 1e-8:
            c = torch.corrcoef(torch.stack([p, t]))[0, 1].item()
            if not np.isnan(c):
                corrs.append(c)

    avg_corr = np.mean(corrs) if corrs else 0.0

    # GARCH baseline
    returns_np = targets.numpy()
    garch_var = garch_baseline(returns_np)
    garch_mse = np.nanmean((garch_var - returns_np**2)**2)
    genesis_var_mse = F.mse_loss(preds**2, targets**2).item()

    # Portfolio metrics
    port_returns = (weights * targets).sum(dim=-1)
    sharpe = port_returns.mean() / port_returns.std().clamp(min=1e-6) * np.sqrt(252)
    max_dd = (port_returns.cumsum(0).cummax(0).values - port_returns.cumsum(0)).max()

    print(f'\n  Genesis MSE:        {mse:.6f}')
    print(f'  Genesis MAE:        {mae:.6f}')
    print(f'  Direction accuracy: {direction:.1%}')
    print(f'  Avg correlation:    {avg_corr:.4f}')
    print(f'  GARCH variance MSE: {garch_mse:.6f}')
    print(f'  Genesis var MSE:    {genesis_var_mse:.6f}')
    print(f'\n  Portfolio annualized Sharpe: {sharpe:.4f}')
    print(f'  Portfolio max drawdown:     {max_dd:.4f}')
    print(f'  Portfolio mean daily return: {port_returns.mean():.6f}')
    print(f'  Avg weight concentration:   {weights.max(dim=-1).values.mean():.4f}')

    # Module diagnostics from last batch
    print(f'\n=== MODULE DIAGNOSTICS (last batch) ===')
    with torch.no_grad():
        batch = next(iter(val_dl))
        features = {k: v.to(device).nan_to_num(0.0) for k, v in batch['features'].items()}
        returns_seq = batch['returns_seq'].to(device).nan_to_num(0.0)
        out = model(features, returns_seq=returns_seq)

        print(f'  Reflexivity rho:     [{out["rho"].min():.4f}, {out["rho"].max():.4f}]')
        print(f'  Hurst range:         [{out["hurst"].min():.3f}, {out["hurst"].max():.3f}]')
        print(f'  Systemic risk:       {out["systemic_risk"][:, -1].mean():.4f}')
        print(f'  KL loss:             {out["kl_loss"]:.4f}')
        print(f'  DAG acyclicity:      {out["acyclicity_loss"]:.2f}')
        regime = out['regime'].softmax(-1)
        print(f'  Regime probs:        {regime[0].tolist()}')
        print(f'  Portfolio VaR:       {out["portfolio"]["var"].mean():.4f}')
        print(f'  Portfolio CVaR:      {out["portfolio"]["cvar"].mean():.4f}')

    # Save
    results = {
        'model': 'MeridianGenesis',
        'params': n_params,
        'assets': available_assets,
        'n_assets': len(available_assets),
        'train_samples': n_train,
        'val_samples': n_val,
        'metrics': {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction,
            'avg_correlation': avg_corr,
            'garch_variance_mse': float(garch_mse),
            'genesis_variance_mse': genesis_var_mse,
            'portfolio_sharpe': sharpe.item(),
            'portfolio_max_drawdown': max_dd.item(),
        },
    }
    with open(results_dir / 'genesis_real_data_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f'\nResults saved to results/genesis_real_data_results.json')
    print('REAL DATA BENCHMARK COMPLETE')


if __name__ == '__main__':
    main()
