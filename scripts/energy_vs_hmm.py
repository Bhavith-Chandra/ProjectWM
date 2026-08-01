"""Does the JEPA ENERGY carry a regime signal that beats a Gaussian HMM? (the pre-registered claim.)

Uses the validated surprise signal (the L2 latent-prediction energy; 4.4x vol lift) as a regime detector
and compares it head-to-head with a Gaussian HMM on returns, on the SAME fair metrics as compare_regimes:
  * persistence   : mean dwell time (days) — regimes should be sticky, not flicker
  * economic value: QLIKE improvement when the HAR forecast gets a per-regime bias (regime usefulness)

Three regime detectors, each decoded into K states, all scored identically:
  (a) HMM on returns          — the baseline
  (b) HMM on the JEPA energy  — energy-as-regime (the test)
  (c) HMM on the belief state — the learned representation (prior attempt)
Honest verdict: the energy wins only if it is MORE persistent and MORE economically useful than the HMM.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from meridian.data import load_all
from meridian.features import build_asset_frame
from meridian.regimes import fit_hmm_states, mean_dwell, regime_conditioned_qlike

RES = Path(__file__).resolve().parent.parent / "results"


def main(k=3):
    mer = pd.read_parquet(RES / "meridian_predictions.parquet"); mer["date"] = pd.to_datetime(mer["date"])
    belief = np.load(RES / "meridian_belief.npy")
    base = pd.read_parquet(RES / "baseline_predictions.parquet")
    har = base[base["model"] == "HAR-RV"].copy(); har["date"] = pd.to_datetime(har["date"])
    d = load_all(); frames = {a: build_asset_frame(o, d["macro"]) for a, o in d["prices"].items()}

    rows = []
    for a in mer["asset"].unique():
        m = mer[mer["asset"] == a].sort_values("date")
        h = har[har["asset"] == a].set_index("date")
        j = m.set_index("date").join(h[["y_pred_log", "logvar_bias"]], rsuffix="_har").dropna(subset=["y_pred_log_har"])
        if len(j) < 300:
            continue
        rets = frames[a]["ret"].reindex(j.index).ffill().to_numpy()
        energy = j["energy"].to_numpy()
        keep = m.set_index("date").loc[j.index, "_belief_row"].to_numpy().astype(int)
        bel = belief[keep]
        rv_true, hp, hb = j["y_true_log"].to_numpy(), j["y_pred_log_har"].to_numpy(), j["logvar_bias_har"].to_numpy()
        try:
            _, s_ret, _ = fit_hmm_states(rets, k=k)
            _, s_en, _ = fit_hmm_states(energy.reshape(-1, 1), k=k)
            _, s_bel, _ = fit_hmm_states(bel, k=k)
        except Exception:
            continue
        r = {"asset": a}
        for tag, s in [("ret", s_ret), ("energy", s_en), ("belief", s_bel)]:
            r[f"{tag}_dwell"] = mean_dwell(s)
            r[f"{tag}_econ"] = regime_conditioned_qlike(rv_true, hp, hb, s)[2]
        rows.append(r)

    df = pd.DataFrame(rows)
    ag = df.drop(columns="asset").mean()
    print(f"Energy-as-regime vs HMM (K={k}, {len(df)} assets, pooled means)\n")
    print(f"  {'detector':>18} {'dwell (days)':>13} {'econ QLIKE %':>13}")
    for tag, name in [("ret", "HMM on returns"), ("energy", "JEPA energy"), ("belief", "belief state")]:
        print(f"  {name:>18} {ag[f'{tag}_dwell']:>13.2f} {ag[f'{tag}_econ']:>13.2f}")
    dwell_gain = (ag["energy_dwell"] - ag["ret_dwell"]) / ag["ret_dwell"] * 100
    econ_ok = ag["energy_econ"] >= ag["ret_econ"]
    print(f"\n  persistence: energy {ag['energy_dwell']:.2f}d vs HMM {ag['ret_dwell']:.2f}d ({dwell_gain:+.1f}%; "
          f"pre-registered bar +10%) {'PASS' if dwell_gain >= 10 else 'FAIL'}")
    print(f"  economic:    energy {ag['energy_econ']:.2f}% vs HMM {ag['ret_econ']:.2f}% "
          f"{'PASS' if econ_ok else 'FAIL'}")
    print(f"\n  >>> JEPA-energy regimes {'BEAT' if (dwell_gain >= 10 and econ_ok) else 'DO NOT beat'} "
          f"the HMM by the pre-registered margin.")


if __name__ == "__main__":
    main()
