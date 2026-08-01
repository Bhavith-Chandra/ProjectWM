"""Meridian Enterprise World Model — interactive CLI.

    python3 scripts/ask.py "Apple"                         # quick read
    python3 scripts/ask.py --full "Apple"                  # full thesis + Monte-Carlo + news
    python3 scripts/ask.py "Apple vs Tesla"                # comparison
    python3 scripts/ask.py --scenario "Apple" -0.05        # what-if: market -5%
    python3 scripts/ask.py --world SPY -0.05 "Tesla"       # multi-entity shock propagation
    python3 scripts/ask.py --portfolio SPY TLT GLD NVDA    # min-variance basket

Resolves ANY entity, fetches its data live, runs the module bank, and explains.
With no argument, runs a short demo across several asset classes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from meridian.engine import ask, analyze, scenario, portfolio, world_scenario
from meridian.data import fetch_yahoo
from meridian.features import realized_variance


def market_series():
    """SPY daily returns as the market benchmark for connectedness."""
    try:
        return realized_variance(fetch_yahoo("SPY"))["ret"].dropna()
    except Exception:
        return None


def main():
    mkt = market_series()
    args = sys.argv[1:]
    if args and args[0] == "--full":
        from meridian.analyze import full_analysis
        r = full_analysis(" ".join(args[1:]), with_news=True)
        if not r["ok"]:
            print(r["message"]); return
        print(r["thesis"])
        if r.get("news"):
            print("\n**Recent news (context only — not a market call):**")
            for h in r["news"][:5]:
                print(f"  • {h['title'][:90]}")
        return
    if args and args[0] == "--scenario":
        entity = args[1]; shock = float(args[2])
        print(scenario(entity, shock, market=mkt))
        return
    if args and args[0] == "--portfolio":
        print(portfolio(args[1:], market=mkt))
        from meridian.analyze import portfolio_analysis
        pa = portfolio_analysis(args[1:])
        if pa.get("ok") and pa.get("crisis_stress", {}).get("scenarios"):
            cs = pa["crisis_stress"]
            print("\n" + "=" * 72)
            print("PORTFOLIO CRISIS VOLATILITY STRESS-TEST (Empirical Volatility Scaling)")
            print("=" * 72)
            print(f"Current Calibrated 99% Portfolio VaR:  {cs['current_var99_pct']:.2f}%")
            names = {"2008_GFC": "2008 GFC", "2020_COVID": "2020 COVID"}
            for k, sc in cs["scenarios"].items():
                print(f"  ↳ Stressed Scenario [{names.get(k, k):<10}]:   {sc['stressed_var99_pct']:.2f}%"
                      f"  (x{sc['multiplier']:.2f} VaR, vol surge x{sc['vol_surge']:.2f})")
                if cs.get("proxied", {}).get(k):
                    print(f"      · proxied (no crisis-window history): {', '.join(cs['proxied'][k])}")
            print("-" * 72)
            print("* Scales asset vols to historical crisis peaks; PRESERVES current correlations")
            print("  (validated: crisis vol drives portfolio tail risk, not correlation — see")
            print("  scripts/validate_network.py). Gaussian 99% on the stressed vol.")
        return
    if args and args[0] == "--world":
        source = args[1]; shock = float(args[2]); extra = args[3:]
        print(world_scenario(source, shock, extra))
        return
    if args:
        q = " ".join(args)
        print(ask(q, market=mkt))
        return
    # demo across asset classes
    print("Meridian Enterprise World Model — demo across asset classes\n")
    for q in ["Apple", "bitcoin", "gold", "euro", "Tesla"]:
        print("=" * 70)
        print(ask(q, market=mkt))
        print()


if __name__ == "__main__":
    main()
