"""
Transaction-cost / turnover sensitivity for the headline strategy.

A backtest Sharpe means little without knowing how much trading produces it. For
the equal-weight composite (the best model in backtest.py) this script measures:
  - annualised one-way turnover of the long book, and
  - gross vs net Sharpe at several per-side cost levels.

Net return per rebalance subtracts the cost of trading into the new book:
    cost = 2 * one_way_turnover * cost_per_side
(2x = sell the names leaving + buy the names entering). Run on either panel:
    python transaction_costs.py
    PANEL=factor_panel_large_ext.parquet python transaction_costs.py
"""

import numpy as np
import pandas as pd

import backtest as bt

COST_LEVELS_BPS = [0, 5, 10, 20]  # per side


def long_book(scores: pd.DataFrame):
    """Per-rebalance: top-quantile holding set and its gross return."""
    holdings, gross, dates = [], [], []
    for t, g in scores.groupby("date"):
        g = g.dropna(subset=["fwd_ret"]).sort_values("score", ascending=False)
        k = max(1, int(round(len(g) * bt.TOP_FRAC)))
        if len(g) < 2 * k:
            continue
        top = g.head(k)
        holdings.append(set(top["code"]))
        gross.append(top["fwd_ret"].mean())
        dates.append(t)
    return dates, holdings, pd.Series(gross, index=dates)


def one_way_turnover(holdings: list[set]) -> pd.Series:
    """Fraction of the book replaced each rebalance (1.0 to establish book)."""
    tno = [1.0]
    for prev, cur in zip(holdings[:-1], holdings[1:]):
        tno.append(len(cur - prev) / len(cur))
    return pd.Series(tno)


def net_sharpe(gross: pd.Series, tno: pd.Series, cost_per_side: float) -> tuple:
    net = gross.values - 2 * tno.values * cost_per_side
    net = pd.Series(net, index=gross.index)
    m = bt.metrics(net)
    return m["ann_return"], m["sharpe"]


def run(features, label: str):
    df = bt.load(bt.BASE + bt.ORTH)
    dates = np.sort(df["date"].unique())
    rebal = bt._rebalance_dates(dates)
    ext_z = [f + "_z" for f in bt.BASE + bt.ORTH]
    n = df["code"].nunique()
    min_book = max(2, int(round(n * bt.TOP_FRAC)) * 2)
    rebal = np.array([t for t in rebal
                      if df[df["date"] == t].dropna(subset=ext_z).shape[0] >= min_book])

    scores = bt.score_equal_weight(df, rebal, features)
    _, holdings, gross = long_book(scores)
    tno = one_way_turnover(holdings)
    ann_tno = tno.mean() * bt.PERIODS_PER_YEAR

    gm = bt.metrics(gross)
    print(f"\n{label}: EW composite long book ({len(features)} factors)")
    print(f"  annualised one-way turnover: {ann_tno:5.1f}x  "
          f"(avg {tno.mean()*100:4.1f}% of the book replaced each rebalance)")
    print(f"  {'cost/side':>10} {'ann_ret':>9} {'net Sharpe':>11}")
    for bps in COST_LEVELS_BPS:
        ann, sh = net_sharpe(gross, tno, bps / 1e4)
        tag = "  (gross)" if bps == 0 else ""
        print(f"  {bps:>7}bps {ann*100:>8.2f}% {sh:>11.2f}{tag}")


def main() -> None:
    print(f"Panel: {bt.PANEL}")
    run(bt.BASE, "base(3)")
    run(bt.BASE + bt.ORTH, "base+orth(9)")
    print("\nRead: if net Sharpe holds up at 10-20 bps/side, the edge survives "
          "realistic costs; if it collapses, the backtest was paying itself in "
          "turnover.")


if __name__ == "__main__":
    main()
