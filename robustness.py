"""
Robustness check for the headline strategy (equal-weight base+orth long book).

A single backtest number can be a lucky parameter pick. This sweeps the two main
design knobs and reports the Sharpe for each, so you can see whether the edge is
stable or hinges on one setting:
  - TOP_FRAC : how many names the book holds (top 10% / 20% / 30% / 50%)
  - HORIZON  : forward/holding period (= rebalance spacing), 10 / 20 / 40 days

Only the fast equal-weight composite is swept (no ML refits), so this is cheap.
Run: python robustness.py   /   PANEL=factor_panel_large_ext.parquet python robustness.py
"""

import numpy as np
import pandas as pd

import backtest as bt


def sharpe_for(features, top_frac: float, horizon: int) -> float:
    # Override the design knobs the backtest functions read at runtime.
    bt.TOP_FRAC, bt.HORIZON, bt.REBAL = top_frac, horizon, horizon
    bt.PERIODS_PER_YEAR = 252 / horizon

    df = bt.load(bt.BASE + bt.ORTH)
    dates = np.sort(df["date"].unique())
    rebal = bt._rebalance_dates(dates)
    ext_z = [f + "_z" for f in features]
    n = df["code"].nunique()
    min_book = max(2, int(round(n * top_frac)) * 2)
    rebal = np.array([t for t in rebal
                      if df[df["date"] == t].dropna(subset=ext_z).shape[0] >= min_book])

    if len(rebal) < 4:          # too few rebalances to judge
        return np.nan
    scores = bt.score_equal_weight(df, rebal, features)
    long_r, _ = bt.portfolio_returns(scores)
    m = bt.metrics(long_r)
    return m.get("sharpe", np.nan) if m else np.nan


def main() -> None:
    print(f"Panel: {bt.PANEL}")
    print("Sharpe of the equal-weight base+orth long book across design knobs:\n")
    horizons = [10, 20, 40]
    fracs = [0.10, 0.20, 0.30, 0.50]
    header = "TOP_FRAC \\ HORIZON  " + "".join(f"{h:>8}d" for h in horizons)
    print(header)
    for fr in fracs:
        row = f"   top {int(fr*100):>2}%          "
        for h in horizons:
            row += f"{sharpe_for(bt.BASE + bt.ORTH, fr, h):>9.2f}"
        print(row)
    print("\nStable, positive Sharpe across the grid => the edge is not a single "
          "lucky parameter choice. The base case in README is HORIZON=20, top 20%.")


if __name__ == "__main__":
    main()
