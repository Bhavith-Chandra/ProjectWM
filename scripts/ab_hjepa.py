"""H-JEPA A/B: does a HIERARCHICAL multi-timescale energy beat the single-scale surprise signal?

Hierarchical JEPA predicts the future latent at several horizons (h=1 daily, 5 weekly, 22 monthly) with
one predictor head per scale; the multi-scale energy is the mean prediction error across scales. We test
whether that multi-scale energy is a sharper surprise detector (vol lift) than the single-scale (h=1)
energy — same encoder, same data, same training, only the energy read-off differs.

Honest verdict either way. Self-contained: reuses the SSM Encoder + Predictor blocks from meridian.model.
"""
from __future__ import annotations

import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.model import Encoder, Predictor, sigreg_loss, MeridianConfig
from meridian.windows import FEATURES, asset_matrix, train_scaler

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L, SPLIT, EPOCHS, BATCH = 32, "2019-01-01", 6, 256
HORIZONS = [1, 5, 22]


class HJEPA(nn.Module):
    def __init__(self, cfg, horizons):
        super().__init__()
        torch.manual_seed(cfg.seed)
        self.encoder = Encoder(cfg)
        self.target = Encoder(cfg); self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.preds = nn.ModuleList([Predictor(cfg.d_model) for _ in horizons])  # one head per scale
        self.vol_head = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, 1))
        self.horizons = horizons; self.ema = cfg.ema

    @torch.no_grad()
    def update_target(self):
        for tp, p in zip(self.target.parameters(), self.encoder.parameters()):
            tp.mul_(self.ema).add_(p, alpha=1 - self.ema)

    def forward(self, ctx, futs):
        h = self.encoder(ctx)[:, -1]
        vol = self.vol_head(h).squeeze(-1)
        energies = []
        for pred, fut in zip(self.preds, futs):
            zp = pred(h)
            with torch.no_grad():
                zt = self.target(fut)[:, -1]
            energies.append((zp - zt).pow(2).mean(-1))          # per-scale surprise
        return h, vol, torch.stack(energies, -1)                 # energies: [B, n_scales]


def build():
    d = load_all(); macro = d["macro"]
    frames = {a: build_asset_frame(o, macro) for a, o in d["prices"].items()}
    ad = {a: asset_matrix(f) for a, f in frames.items()}
    mean, std = train_scaler(frames, ad, pd.Timestamp(SPLIT))
    hz = max(HORIZONS)
    cols = {"ctx": [], "y": [], "date": [], "asset": []}
    futs = {h: [] for h in HORIZONS}
    for a, (dates, X, y) in ad.items():
        Xs = (X - mean) / std
        T = len(Xs)
        for t in range(L - 1, T - hz - 1):
            if not np.isfinite(y[t]):
                continue
            cols["ctx"].append(Xs[t - L + 1:t + 1])
            for h in HORIZONS:
                futs[h].append(Xs[t - L + 1 + h:t + 1 + h])
            cols["y"].append(y[t]); cols["date"].append(dates[t]); cols["asset"].append(a)
    ctx = np.asarray(cols["ctx"], np.float32); y = np.asarray(cols["y"], np.float32)
    futs = {h: np.asarray(v, np.float32) for h, v in futs.items()}
    meta = pd.DataFrame({"date": pd.to_datetime(cols["date"]), "asset": cols["asset"], "y": y})
    return ctx, futs, meta


def main():
    t0 = time.time()
    ctx, futs, meta = build()
    tr = (meta["date"] < pd.Timestamp(SPLIT)).to_numpy(); te = ~tr
    cfg = MeridianConfig(n_features=len(FEATURES), window=L, seed=0)
    m = HJEPA(cfg, HORIZONS).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    C = torch.tensor(ctx[tr], device=DEV); Fs = [torch.tensor(futs[h][tr], device=DEV) for h in HORIZONS]
    Y = torch.tensor(meta["y"].to_numpy()[tr], device=DEV)
    n = len(Y); g = torch.Generator().manual_seed(0); m.train()
    print(f"H-JEPA A/B — horizons {HORIZONS}, train {tr.sum()} / test {te.sum()}, {EPOCHS} epochs, {DEV}\n")
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            h, vol, en = m(C[b], [f[b] for f in Fs])
            rv = torch.exp(Y[b]); ql = (rv * torch.exp(-vol) + vol - Y[b] - 1).mean()
            loss = ql + en.mean() + 0.5 * sigreg_loss(h)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); m.update_target()
    m.eval()
    with torch.no_grad():
        Cte = torch.tensor(ctx[te], device=DEV); Fte = [torch.tensor(futs[h][te], device=DEV) for h in HORIZONS]
        _, _, en = m(Cte, Fte)
    en = en.cpu().numpy()                                        # [N, n_scales]
    yv = meta["y"].to_numpy()[te]; rv = np.exp(yv)

    def lift(e):
        hi = e >= np.quantile(e, 0.9)
        return np.mean(rv[hi]) / np.mean(rv[~hi]), spearmanr(e, yv)[0]

    print(f"  {'energy':>22} {'vol lift':>9} {'ρ(E,vol)':>9}")
    l1, r1 = lift(en[:, 0])                                      # single-scale (h=1)
    print(f"  {'single-scale (h=1)':>22} {l1:>8.2f}x {r1:>+9.3f}")
    lm, rm = lift(en.mean(1))                                    # multi-scale mean
    print(f"  {'H-JEPA multi-scale':>22} {lm:>8.2f}x {rm:>+9.3f}")
    win = lm > l1 * 1.02
    print(f"\n  VERDICT [{time.time()-t0:.0f}s]: hierarchical multi-scale energy "
          f"{'BEATS' if win else 'does NOT beat'} single-scale ({lm:.2f}x vs {l1:.2f}x) → "
          f"{'adopt H-JEPA energy' if win else 'keep single-scale, report honestly'}.")


if __name__ == "__main__":
    main()
