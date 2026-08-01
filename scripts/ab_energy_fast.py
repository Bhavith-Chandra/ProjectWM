"""Fast apples-to-apples A/B: EB-JEPA LEARNED energy vs fixed latent-MSE energy, as a SURPRISE signal.

The full walk-forward is too slow for iteration (the contrastive + expanding window + ensemble). This
trains BOTH energy modes with an IDENTICAL light config on one train/test split (train < FIRST_TEST,
test after), pooled across the universe, and compares the test-set energy purely as a surprise detector:
vol lift (top-decile-energy realized vol ÷ rest), Spearman(energy, realized vol / |forecast error|).

Same seed, same epochs, same data — the only difference is the energy. Honest verdict either way.
"""
from __future__ import annotations

import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.model import Meridian, MeridianConfig
from meridian.windows import FEATURES, asset_matrix, build_windows, train_scaler

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
L, SPLIT, EPOCHS, BATCH = 32, "2019-01-01", 6, 256


def build_split():
    d = load_all(); macro = d["macro"]
    frames = {a: build_asset_frame(o, macro) for a, o in d["prices"].items()}
    ad = {a: asset_matrix(f) for a, f in frames.items()}
    mean, std = train_scaler(frames, ad, pd.Timestamp(SPLIT))
    tr, te = {k: [] for k in ("ctx", "fut", "y")}, {k: [] for k in ("ctx", "fut", "y")}
    te_meta = []
    for a, (dates, X, y) in ad.items():
        w = build_windows(dates, X, y, L, mean, std)
        if w is None:
            continue
        dt = pd.DatetimeIndex(w["dates"]); m = dt < pd.Timestamp(SPLIT)
        for k in ("ctx", "fut", "y"):
            tr[k].append(w[k][m]); te[k].append(w[k][~m])
        te_meta.append((a, dt[~m]))
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    te = {k: np.concatenate(v) for k, v in te.items()}
    meta = pd.concat([pd.DataFrame({"asset": a, "date": dts}) for a, dts in te_meta], ignore_index=True)
    return tr, te, meta


def train_eval(mode, tr, te):
    cfg = MeridianConfig(n_features=len(FEATURES), window=L, seed=0, loss_mode="qlike", energy_mode=mode)
    m = Meridian(cfg).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ctx, fut, y = (torch.tensor(tr[k], device=DEV) for k in ("ctx", "fut", "y"))
    n = len(y); g = torch.Generator().manual_seed(0); m.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); loss, _ = m.loss({"x_ctx": ctx[b], "x_fut": fut[b], "y": y[b]})
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); m.update_target()
    m.eval()
    with torch.no_grad():
        out = m.forward(torch.tensor(te["ctx"], device=DEV), torch.tensor(te["fut"], device=DEV))
    return out["energy"].cpu().numpy(), out["vol"].cpu().numpy()


def surprise_metrics(energy, y_true, y_pred):
    rv = np.exp(y_true); err = np.abs(y_true - y_pred)
    hi = energy >= np.quantile(energy, 0.9)
    return {"lift": np.mean(rv[hi]) / np.mean(rv[~hi]),
            "rho_vol": spearmanr(energy, y_true)[0], "rho_err": spearmanr(energy, err)[0]}


def main():
    t0 = time.time()
    tr, te, meta = build_split()
    print(f"A/B energy — train<{SPLIT} ({len(tr['y'])}), test≥{SPLIT} ({len(te['y'])}), {EPOCHS} epochs, dev={DEV}\n")
    y_true = te["y"]
    print(f"  {'energy':>18} {'vol lift':>9} {'ρ(E,vol)':>9} {'ρ(E,err)':>9}")
    res = {}
    for mode, name in [("l2", "L2 (latent-MSE)"), ("learned", "EB-JEPA (learned)")]:
        e, vp = train_eval(mode, tr, te)
        mts = surprise_metrics(e, y_true, vp); res[mode] = mts
        print(f"  {name:>18} {mts['lift']:>8.2f}x {mts['rho_vol']:>+9.3f} {mts['rho_err']:>+9.3f}")
    win = res["learned"]["lift"] > res["l2"]["lift"] * 1.02
    print(f"\n  VERDICT [{time.time()-t0:.0f}s]: EB-JEPA learned energy "
          f"{'BEATS' if win else 'does NOT beat'} latent-MSE as a surprise signal "
          f"({res['learned']['lift']:.2f}x vs {res['l2']['lift']:.2f}x) → "
          f"{'adopt learned energy' if win else 'keep the simpler L2 energy, report honestly'}.")


if __name__ == "__main__":
    main()
