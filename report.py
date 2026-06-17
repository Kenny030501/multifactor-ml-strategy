"""
Reporting: equity curves + alpha/beta decomposition for the headline strategy.

Because the long book carries market beta, a high Sharpe is not automatically
"alpha". This script regresses each strategy's per-rebalance return on the
equal-weight benchmark (CAPM-style):
    r_strategy = alpha + beta * r_benchmark + e
and reports annualised alpha, beta, the alpha t-stat, and the information ratio
(annualised mean residual / annualised residual vol) -- the part of the edge
that is NOT just market exposure. It also saves a cumulative-return plot.

Run:
    python report.py
    PANEL=factor_panel_large_ext.parquet python report.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtest as bt


def rebal_dates(df):
    dates = np.sort(df["date"].unique())
    rebal = bt._rebalance_dates(dates)
    ext_z = [f + "_z" for f in bt.BASE + bt.ORTH]
    n = df["code"].nunique()
    min_book = max(2, int(round(n * bt.TOP_FRAC)) * 2)
    return np.array([t for t in rebal
                     if df[df["date"] == t].dropna(subset=ext_z).shape[0] >= min_book])


def alpha_beta(strat: pd.Series, bench: pd.Series) -> dict:
    """CAPM regression of strategy on benchmark, per-rebalance returns."""
    s, b = strat.align(bench, join="inner")
    X = np.column_stack([np.ones(len(b)), b.values])
    beta_hat, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    a, beta = beta_hat
    resid = s.values - X @ beta_hat
    n = len(s)
    se_a = np.sqrt((resid @ resid) / (n - 2) *
                   np.linalg.inv(X.T @ X)[0, 0])
    t_a = a / se_a if se_a > 0 else np.nan
    ppy = bt.PERIODS_PER_YEAR
    ann_alpha = a * ppy
    # Information ratio = annualised alpha / annualised tracking error.
    # (Residual mean is ~0 by OLS construction, so IR uses alpha, not resid mean.)
    tracking_error = resid.std(ddof=2) * np.sqrt(ppy)
    ir = ann_alpha / tracking_error if tracking_error > 0 else np.nan
    return {"ann_alpha": ann_alpha, "beta": beta, "alpha_t": t_a, "info_ratio": ir}


def main() -> None:
    df = bt.load(bt.BASE + bt.ORTH)
    rebal = rebal_dates(df)
    n = df["code"].nunique()
    tag = f"{n}stocks"

    bench = bt.benchmark_returns(df, rebal)
    series = {"Benchmark (EW all)": bench}
    for label, feats in [("EW base(3)", bt.BASE), ("EW base+orth(9)", bt.BASE + bt.ORTH)]:
        long_r, _ = bt.portfolio_returns(bt.score_equal_weight(df, rebal, feats))
        series[label] = long_r

    # --- alpha / beta vs benchmark ---
    print(f"Panel: {bt.PANEL} | {n} stocks | {len(rebal)} rebalances")
    print(f"\n{'strategy':<20} {'annAlpha':>9} {'beta':>6} {'alpha_t':>8} {'InfoRatio':>10}")
    for label in ("EW base(3)", "EW base+orth(9)"):
        ab = alpha_beta(series[label], bench)
        print(f"{label:<20} {ab['ann_alpha']*100:>8.2f}% {ab['beta']:>6.2f} "
              f"{ab['alpha_t']:>8.2f} {ab['info_ratio']:>10.2f}")

    # --- equity curves ---
    plt.figure(figsize=(10, 6))
    for label, r in series.items():
        eq = (1 + r.sort_index()).cumprod()
        style = "--" if label.startswith("Benchmark") else "-"
        plt.plot(eq.index, eq.values, style, label=f"{label}", linewidth=1.8)
    plt.title(f"Cumulative return — equal-weight long book ({n} stocks)")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = f"equity_curve_{tag}.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
