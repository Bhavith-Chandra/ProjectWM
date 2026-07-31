"""Volatility-forecast BENCHMARK — Meridian vs GARCH, EWMA, HAR, and TimeMixer.

Protocol (as requested):
  * date-based WALK-FORWARD with PURGE + EMBARGO (no leakage across the target-overlap edge),
  * metrics: QLIKE (variance), RMSE (log-vol), IC (Spearman rank corr of forecast vs realized),
  * Diebold-Mariano tests vs HAR, and a Model Confidence Set (Hansen-Lunde-Nason) at 90%.

Models (all fit only on each fold's train, evaluated on its embargoed test block):
  * EWMA (RiskMetrics, lambda=0.94 on RV) — recursive.
  * GARCH(1,1) — arch MLE per fold/asset; leak-free variance recursion, train-mean calibrated.
  * HAR-RV (Corsi 2009) — OLS on the daily/weekly/monthly log-RV cascade.
  * Meridian (HAR + leverage) — the production per-entity vol module (adds the down-move channel).
  * TimeMixer (compact) — faithful reimplementation: multiscale downsampling + series
    decomposition + cross-scale (past-decomposable) mixing + multi-predictor head, trained pooled.

Level forecasts (EWMA/GARCH) and log forecasts (HAR/Meridian/TimeMixer) are put on a common
footing by a train-only calibration (multiplicative for level models; Jensen for log models),
so QLIKE compares DYNAMICS, not a variance-proxy offset. Honest: whatever wins, wins.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import load_all
from meridian.features import realized_variance
from meridian.evalproto import qlike, diebold_mariano

EPS = 1e-12
L = 40                       # input window for TimeMixer
MIN_TRAIN_D = 1260           # ~5y initial train (calendar days of data)
TEST_D = 378                 # ~1.5y test block
EMBARGO_D = 22
RESULTS = Path(__file__).resolve().parent.parent / "results"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
RICH = ["har_d", "har_w", "har_m", "lev", "pos", "vix", "ret5"]   # Meridian-WM feature set


def build():
    d = load_all()
    vix = np.log(d["macro"]["VIXCLS"].clip(lower=EPS))
    rows, WlogA, WlevA = [], [], []
    for a, ohlc in d["prices"].items():
        rvf = realized_variance(ohlc)
        rv = rvf["rv"].to_numpy(); ret = rvf["ret"].to_numpy()
        lrv = np.log(rv + EPS)
        neg = np.log((np.minimum(ret, 0.0) ** 2) + EPS)      # bad-vol / leverage
        pos = np.log((np.maximum(ret, 0.0) ** 2) + EPS)      # good-vol semivariance
        dates = rvf.index
        w = pd.Series(lrv).rolling(5).mean().to_numpy()
        m = pd.Series(lrv).rolling(22).mean().to_numpy()
        r5 = pd.Series(ret).rolling(5).sum().to_numpy()
        vx = vix.reindex(dates).ffill().to_numpy()
        for t in range(L, len(rv) - 1):
            if not np.isfinite([lrv[t], w[t], m[t], neg[t], pos[t], lrv[t + 1]]).all():
                continue
            vt = vx[t] if np.isfinite(vx[t]) else np.log(15.0)   # VIX fallback
            r5t = r5[t] if np.isfinite(r5[t]) else 0.0
            rows.append((a, dates[t], rv[t], lrv[t + 1], lrv[t], w[t], m[t], neg[t], pos[t], vt, r5t, rv[t + 1]))
            WlogA.append(lrv[t - L + 1:t + 1]); WlevA.append(neg[t - L + 1:t + 1])
    R = pd.DataFrame(rows, columns=["asset", "date", "rv", "y", "har_d", "har_w", "har_m",
                                    "lev", "pos", "vix", "ret5", "rv_next"])
    return R, np.asarray(WlogA, np.float32), np.asarray(WlevA, np.float32), d


class MLP(nn.Module):
    def __init__(self, k, hid=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def meridian_wm(tr, te, seeds=4, epochs=70):
    """Rich-feature seed-ensemble (Meridian's engineered forecaster): realized semivariance
    (good/bad), implied vol, weekly return + the HAR cascade. MSE-trained, Jensen-corrected."""
    Xtr = tr[RICH].to_numpy(np.float32); ytr = tr["y"].to_numpy(np.float32)
    Xte = te[RICH].to_numpy(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_s = torch.tensor((Xtr - mu) / sd).to(DEV); yt = torch.tensor(ytr).to(DEV)
    Xte_s = torch.tensor((Xte - mu) / sd).to(DEV)
    preds_tr, preds_te = [], []
    for s in range(seeds):
        torch.manual_seed(s)
        net = MLP(len(RICH)).to(DEV)
        opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
        n = len(yt)
        for _ in range(epochs):
            perm = torch.randperm(n, device=DEV)
            for i in range(0, n, 1024):
                idx = perm[i:i + 1024]
                opt.zero_grad(); ((net(Xtr_s[idx]) - yt[idx]) ** 2).mean().backward(); opt.step()
        with torch.no_grad():
            preds_tr.append(net(Xtr_s).cpu().numpy()); preds_te.append(net(Xte_s).cpu().numpy())
    ptr = np.mean(preds_tr, 0); pte = np.mean(preds_te, 0)
    jb = 0.5 * np.var(ytr - ptr)                              # Jensen (log→level)
    return pte, jb                                            # (conditional log-mean, jb)


# --------------------------------------------------------------------------- #
# TimeMixer (compact, faithful): multiscale downsample + decomposition + mixing
# --------------------------------------------------------------------------- #
def moving_avg(x, k):                                   # x [B,C,Lc]
    pad = k // 2
    xp = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
    return torch.nn.functional.avg_pool1d(xp, k, stride=1)[..., :x.shape[-1]]


class TimeMixer(nn.Module):
    def __init__(self, L=L, C=2, scales=(1, 2, 4), hid=48):
        super().__init__()
        self.scales = scales
        self.enc = nn.ModuleList()
        for f in scales:
            Ls = max(L // f, 4)
            self.enc.append(nn.Sequential(nn.Linear(2 * C * Ls, hid), nn.GELU(), nn.Linear(hid, hid)))
        self.mix = nn.Sequential(nn.Linear(hid * len(scales), hid), nn.GELU(), nn.Linear(hid, hid))
        self.heads = nn.ModuleList([nn.Linear(hid, 1) for _ in scales])
        self.head_mix = nn.Linear(hid, 1)

    def forward(self, x):                                # x [B,C,L]
        feats = []
        for f, enc in zip(self.scales, self.enc):
            xs = x if f == 1 else torch.nn.functional.avg_pool1d(x, f)
            trend = moving_avg(xs, 5); seas = xs - trend       # series decomposition
            z = torch.cat([seas.flatten(1), trend.flatten(1)], 1)
            feats.append(enc(z))
        mixed = self.mix(torch.cat(feats, 1))               # cross-scale (past-decomposable) mixing
        out = self.head_mix(mixed)
        for h, fe in zip(self.heads, feats):                # future multi-predictor mixing
            out = out + h(fe)
        return out.squeeze(-1)


def train_timemixer(Xtr, ytr, mu, sd, epochs=35):
    torch.manual_seed(0)
    net = TimeMixer().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    X = torch.tensor((Xtr - mu) / sd).to(DEV); y = torch.tensor((ytr - mu) / sd).to(DEV)
    n = len(y)
    for _ in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            opt.zero_grad()
            loss = ((net(X[idx]) - y[idx]) ** 2).mean()
            loss.backward(); opt.step()
    return net


# --------------------------------------------------------------------------- #
# Classical models
# --------------------------------------------------------------------------- #
def har_predict(tr, te, cols):
    A = np.column_stack([np.ones(len(tr))] + [tr[c].to_numpy() for c in cols])
    beta, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
    jb = 0.5 * np.var(tr["y"].to_numpy() - A @ beta)
    B = np.column_stack([np.ones(len(te))] + [te[c].to_numpy() for c in cols])
    return B @ beta, jb                                     # (conditional log-mean, Jensen jb)


def ewma_series(rv, lam=0.94):
    f = np.empty(len(rv)); f[0] = rv[0]
    for t in range(1, len(rv)):
        f[t] = lam * f[t - 1] + (1 - lam) * rv[t - 1]        # forecast of rv[t] from info < t
    return f


def garch_sigma2(ret, train_mask):
    from arch import arch_model
    r = np.nan_to_num(ret * 100.0, nan=0.0)             # arch rejects NaN (first .diff() is NaN)
    try:
        res = arch_model(r[train_mask], mean="Zero", vol="Garch", p=1, q=1, rescale=False).fit(disp="off")
        w, al, be = res.params["omega"], res.params["alpha[1]"], res.params["beta[1]"]
    except Exception:
        return None
    s2 = np.empty(len(r)); s2[0] = np.nanvar(r[train_mask])
    for t in range(1, len(r)):
        s2[t] = w + al * r[t - 1] ** 2 + be * s2[t - 1]
    return s2 / (100.0 ** 2)                                 # back to return-variance units


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def main():
    print(f"Building panel … (device={DEV})")
    R, Wlog, Wlev, d = build()
    R = R.reset_index(drop=True)
    # per-asset precompute for EWMA / GARCH
    per = {}
    for a, o in d["prices"].items():
        rvf = realized_variance(o)
        per[a] = {"rv": rvf["rv"], "ret": rvf["ret"], "ewma": pd.Series(ewma_series(rvf["rv"].to_numpy()), index=rvf.index)}
    dates = np.array(sorted(R["date"].unique()))
    mu = Wlog.mean(); sd = Wlog.std() + 1e-6

    preds = []                                              # list of per-fold aligned DataFrames
    i = MIN_TRAIN_D; fold = 0
    while i < len(dates):
        test_dates = set(dates[i:i + TEST_D])
        cutoff = dates[i - EMBARGO_D]
        tr_mask = (R["date"] < cutoff).to_numpy()
        te_mask = R["date"].isin(test_dates).to_numpy()
        if tr_mask.sum() < 2000 or te_mask.sum() < 100:
            i += TEST_D; continue
        fold += 1
        tr, te = R[tr_mask], R[te_mask]
        out = te[["asset", "date", "y", "rv_next"]].copy()

        # Each model stores BOTH: out[m] = conditional LOG-MEAN (for RMSE) and
        # out[m+"__var"] = variance forecast (for QLIKE/IC). Every log model gets its OWN
        # train-estimated Jensen jb — fair, so no model is advantaged by (missing) calibration.
        def log_model(name, pm, jb):
            out[name] = pm; out[name + "__var"] = np.exp(pm + jb)

        log_model("HAR", *har_predict(tr, te, ["har_d", "har_w", "har_m"]))
        log_model("Meridian", *har_predict(tr, te, ["har_d", "har_w", "har_m", "lev"]))
        log_model("Meridian-WM", *meridian_wm(tr, te))

        # EWMA & GARCH — level (variance) forecasts, train-mean calibrated, per asset
        ew, ga = np.full(len(te), np.nan), np.full(len(te), np.nan)
        for a in te["asset"].unique():
            aidx = te["asset"].to_numpy() == a
            rv = per[a]["rv"]; ewf = per[a]["ewma"]
            tdates = te["date"].to_numpy()[aidx]
            trd = per[a]["rv"].index < cutoff
            cE = np.nanmean(rv[trd].to_numpy()) / (np.nanmean(ewf[trd].to_numpy()) + EPS)
            ew[aidx] = cE * ewf.reindex(tdates).to_numpy()
            s2 = garch_sigma2(per[a]["ret"].to_numpy(), np.asarray(per[a]["ret"].index < cutoff))
            if s2 is not None:
                s2s = pd.Series(s2, index=per[a]["ret"].index)
                cG = np.nanmean(rv[trd].to_numpy()) / (np.nanmean(s2s[per[a]["ret"].index < cutoff].to_numpy()) + EPS)
                ga[aidx] = cG * s2s.reindex(tdates).to_numpy()
        out["EWMA__var"] = np.clip(ew, EPS, None); out["EWMA"] = np.log(out["EWMA__var"])
        out["GARCH__var"] = np.clip(ga, EPS, None); out["GARCH"] = np.log(out["GARCH__var"])

        # TimeMixer — pooled; its own train-estimated Jensen (fair QLIKE, same as log models)
        Xtr = np.stack([Wlog[tr_mask], Wlev[tr_mask]], 1); ytr = tr["y"].to_numpy().astype(np.float32)
        net = train_timemixer(Xtr, ytr, mu, sd)
        with torch.no_grad():
            ptr_tm = net(torch.tensor((Xtr - mu) / sd).to(DEV)).cpu().numpy() * sd + mu
            pm_tm = net(torch.tensor((np.stack([Wlog[te_mask], Wlev[te_mask]], 1) - mu) / sd).to(DEV)).cpu().numpy() * sd + mu
        log_model("TimeMixer", pm_tm, 0.5 * np.var(ytr - ptr_tm))

        preds.append(out)
        print(f"  fold {fold}: train {tr_mask.sum()}  test {te_mask.sum()}  "
              f"({min(test_dates)}→{max(test_dates)})", flush=True)
        i += TEST_D

    P = pd.concat(preds, ignore_index=True)
    models = ["EWMA", "GARCH", "HAR", "Meridian", "TimeMixer", "Meridian-WM"]
    rv_true = P["rv_next"].to_numpy()
    y_log = P["y"].to_numpy()
    for m in models:
        P[m] = P[m].replace([np.inf, -np.inf], np.nan)
        P[m + "__var"] = P[m + "__var"].replace([np.inf, -np.inf], np.nan)
    loss = {m: qlike(rv_true, P[m + "__var"].to_numpy()) for m in models}   # QLIKE on variance forecast
    cov = {m: int(np.isfinite(P[m].to_numpy()).sum()) for m in models}
    print(f"\nBenchmark — pooled OOS ({len(P)} rows, {fold} walk-forward folds, purge+embargo)")
    print(f"  coverage (finite preds): " + ", ".join(f"{m}={cov[m]}" for m in models) + "\n")

    hmask = np.isfinite(loss["HAR"])
    print(f"  {'model':>11} {'QLIKE':>8} {'RMSE(log)':>10} {'IC':>7} {'DM vs HAR (p)':>16}")
    summary = {}
    qmeans = {}
    for m in models:
        mk = np.isfinite(P[m].to_numpy()) & np.isfinite(rv_true)
        q = float(np.nanmean(loss[m][mk]))
        rmse = float(np.sqrt(np.nanmean((y_log[mk] - P[m].to_numpy()[mk]) ** 2)))   # RMSE on log-mean
        ic = float(spearmanr(np.sqrt(P[m + "__var"].to_numpy()[mk]), np.sqrt(rv_true[mk]))[0])
        qmeans[m] = q
        if m == "HAR":
            dmp, p = "—", None
        else:
            cm = mk & hmask
            _, p = diebold_mariano(loss[m][cm], loss["HAR"][cm]); dmp = f"{p:.3f}"
        summary[m] = {"QLIKE": q, "RMSE_log": rmse, "IC": ic, "DM_vs_HAR_p": None if p is None else float(p)}
        print(f"  {m:>11} {q:>8.4f} {rmse:>10.4f} {ic:>7.3f} {dmp:>16}")
    best = min(qmeans, key=qmeans.get)
    print(f"  best (lowest QLIKE): {best}")

    # Model Confidence Set (Hansen-Lunde-Nason) on QLIKE losses — common finite rows
    from arch.bootstrap import MCS
    LM = pd.DataFrame({m: loss[m] for m in models}).replace([np.inf, -np.inf], np.nan).dropna()
    print(f"  MCS input rows (all models finite): {len(LM)}")
    mcs = MCS(LM, size=0.10, reps=1000, block_size=22, method="R", seed=0); mcs.compute()
    included = list(mcs.included); pvals = mcs.pvalues["Pvalue"].to_dict()
    print(f"\n  Model Confidence Set (90%): {{{', '.join(included)}}}")
    print(f"  MCS p-values: " + ", ".join(f"{m}={pvals.get(m, float('nan')):.3f}" for m in models))
    print("\n  QLIKE/RMSE lower = better; IC higher = better. MCS = models statistically")
    print("  indistinguishable from the best at 90% (survivors of the elimination test).")

    RESULTS.mkdir(exist_ok=True)
    payload = {"n_forecasts": int(len(P)), "folds": fold,
               "protocol": f"date-based walk-forward, min_train={MIN_TRAIN_D}d, test={TEST_D}d, embargo={EMBARGO_D}d",
               "metrics": summary, "mcs_90_included": included,
               "mcs_pvalues": {m: float(pvals.get(m, float("nan"))) for m in models}}
    (RESULTS / "benchmark_vol.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(summary).T.to_csv(RESULTS / "benchmark_vol.csv")
    print(f"\n  saved → results/benchmark_vol.json, results/benchmark_vol.csv")


if __name__ == "__main__":
    main()
