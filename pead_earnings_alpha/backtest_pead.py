"""
Walk-forward backtest of the PEAD signal — long-short and long-only.

Every REBAL trading days, rank stocks by the SUE factor, go long the top
quintile and short the bottom quintile (equal weight), hold to the next
rebalance. We report both the raw SUE factor and the version orthogonalised to
price momentum/volatility, and decompose returns into alpha/beta vs an
equal-weight benchmark. No look-ahead: the factor at date t uses only earnings
filed on or before t, and the label is the realised forward return.

Prices/labels come from pead_factor_panel.parquet (built by pead_factor.py).
"""

import numpy as np
import pandas as pd

HORIZON = 20
REBAL = 20
TOP_FRAC = 0.20
PERIODS_PER_YEAR = 252 / HORIZON


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 3:
        return {}
    eq = (1 + r).cumprod()
    ann = eq.iloc[-1] ** (PERIODS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
    dd = (eq / eq.cummax() - 1).min()
    return {"ann_return": ann, "sharpe": ann / vol if vol else np.nan,
            "max_drawdown": dd, "win_rate": (r > 0).mean(), "n": len(r)}


def alpha_beta(strat: pd.Series, bench: pd.Series) -> dict:
    s, b = strat.align(bench, join="inner")
    s, b = s.dropna(), b.reindex(s.dropna().index)
    X = np.column_stack([np.ones(len(s)), b.values])
    coef, *_ = np.linalg.lstsq(X, s.values, rcond=None)
    a, beta = coef
    resid = s.values - X @ coef
    te = resid.std(ddof=2) * np.sqrt(PERIODS_PER_YEAR)
    return {"ann_alpha": a * PERIODS_PER_YEAR, "beta": beta,
            "info_ratio": (a * PERIODS_PER_YEAR) / te if te else np.nan}


def backtest(df: pd.DataFrame, factor: str):
    dates = np.sort(df["date"].unique())
    rebal = dates[::REBAL]
    long_r, ls_r, bench_r, idx = [], [], [], []
    longs, shorts = [], []                       # holding sets, for turnover
    for t in rebal:
        g = df[(df["date"] == t)].dropna(subset=[factor, "fwd_ret"])
        k = max(1, int(round(len(g) * TOP_FRAC)))
        if len(g) < 2 * k:
            continue
        g = g.sort_values(factor, ascending=False)
        long_r.append(g.head(k)["fwd_ret"].mean())
        ls_r.append(g.head(k)["fwd_ret"].mean() - g.tail(k)["fwd_ret"].mean())
        bench_r.append(g["fwd_ret"].mean()); idx.append(t)
        longs.append(set(g.head(k)["code"])); shorts.append(set(g.tail(k)["code"]))
    return (pd.Series(long_r, idx), pd.Series(ls_r, idx), pd.Series(bench_r, idx),
            longs, shorts)


def turnover(holdings: list) -> pd.Series:
    """One-way fraction of the book replaced each rebalance (1.0 to establish)."""
    t = [1.0] + [len(c - p) / len(c) for p, c in zip(holdings[:-1], holdings[1:])]
    return pd.Series(t)


def net_long_short(ls: pd.Series, longs, shorts, bps: float) -> pd.Series:
    """Subtract trading cost from both legs each rebalance."""
    cost = (turnover(longs).values + turnover(shorts).values) * (bps / 1e4)
    return pd.Series(ls.values - cost, index=ls.index)


def main() -> None:
    df = pd.read_parquet("pead_factor_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    n = df["code"].nunique()
    print(f"Universe: {n} stocks | {df['date'].min().date()} -> {df['date'].max().date()} "
          f"| rebalance {REBAL}d, hold {HORIZON}d, top/bottom {int(TOP_FRAC*100)}%\n")

    rows = {}
    cost_rows = {}
    for factor in ["sue_z", "sue_orth"]:
        long_r, ls_r, bench, longs, shorts = backtest(df, factor)
        rows[f"{factor} | long"] = metrics(long_r)
        rows[f"{factor} | long-short"] = metrics(ls_r)
        if factor == "sue_z":
            rows["Benchmark (EW all)"] = metrics(bench)
        rows[f"{factor} | long"].update(alpha_beta(long_r, bench))
        # net-of-cost long-short Sharpe at several per-side cost levels
        ann_to = (turnover(longs).mean() + turnover(shorts).mean()) * PERIODS_PER_YEAR
        cost_rows[factor] = {"ann_turnover(x)": ann_to,
                             **{f"net Sharpe @{b}bps": metrics(
                                 net_long_short(ls_r, longs, shorts, b)).get("sharpe")
                                for b in [0, 10, 20]}}

    table = pd.DataFrame(rows).T
    for c in ["ann_return", "max_drawdown", "win_rate", "ann_alpha"]:
        if c in table:
            table[c] = (table[c] * 100).map(lambda v: f"{v:6.2f}%" if pd.notna(v) else "")
    for c in ["sharpe", "beta", "info_ratio"]:
        if c in table:
            table[c] = table[c].map(lambda v: f"{v:5.2f}" if pd.notna(v) else "")
    table["n"] = table["n"].astype("Int64")
    pd.set_option("display.width", 200)
    print(table.to_string())

    ct = pd.DataFrame(cost_rows).T
    for c in ct.columns:
        ct[c] = ct[c].map(lambda v: f"{v:5.2f}" if pd.notna(v) else "")
    print("\nLong-short turnover & net-of-cost Sharpe:")
    print(ct.to_string())
    print("\nlong-short isolates the earnings signal; the long book also carries beta.")


if __name__ == "__main__":
    main()
