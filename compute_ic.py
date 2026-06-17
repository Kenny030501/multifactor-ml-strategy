"""
Factor validation via the Information Coefficient (IC).

For a cross-sectional stock-selection strategy, the relevant question is NOT
"does the factor predict a single stock's future return over time" but
"on any given day, does ranking stocks by the factor rank them by their
future return". That is exactly what the rank IC measures: the cross-sectional
Spearman correlation between the factor value and the forward return, computed
date by date.

We report, per factor:
  - mean IC          : average daily rank IC (sign + magnitude of edge)
  - IC std           : stability of the edge
  - ICIR = mean/std  : risk-adjusted edge (information ratio of the IC series)
  - t-stat           : ICIR * sqrt(#days), rough significance
  - hit rate         : fraction of days with IC in the "expected" direction
"""

import numpy as np
import pandas as pd

HORIZON = 20  # predict the forward 20-trading-day return

FACTORS = ["mom_20", "rev_5", "vol_20"]
# Prior expectation on the sign of each factor's IC (for the hit-rate column).
EXPECTED_SIGN = {"mom_20": +1, "rev_5": -1, "vol_20": -1}


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet("factor_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # Forward return per stock (grouped so the window never crosses tickers).
    df["fwd_ret"] = (
        df.groupby("code")["close"].transform(lambda x: x.shift(-HORIZON) / x - 1)
    )
    return df


def daily_rank_ic(df: pd.DataFrame, factor: str) -> pd.Series:
    """Cross-sectional Spearman IC of `factor` vs forward return, per date."""
    sub = df.dropna(subset=[factor, "fwd_ret"])

    def _ic(g: pd.DataFrame) -> float:
        if len(g) < 5:  # need a meaningful cross-section
            return np.nan
        return g[factor].corr(g["fwd_ret"], method="spearman")

    return sub.groupby("date")[[factor, "fwd_ret"]].apply(_ic).dropna()


def summarize(ic: pd.Series, factor: str) -> dict:
    n = len(ic)
    mean = ic.mean()
    std = ic.std()
    icir = mean / std if std > 0 else np.nan
    tstat = icir * np.sqrt(n) if np.isfinite(icir) else np.nan
    exp = EXPECTED_SIGN[factor]
    hit = (np.sign(ic) == exp).mean()
    return {
        "factor": factor,
        "n_days": n,
        "mean_IC": mean,
        "IC_std": std,
        "ICIR": icir,
        "t_stat": tstat,
        f"hit_rate(sign={exp:+d})": hit,
    }


def main() -> None:
    df = load_panel()
    print(f"Universe: {df['code'].nunique()} stocks | "
          f"{df['date'].min().date()} -> {df['date'].max().date()} | "
          f"forward horizon = {HORIZON} trading days\n")

    rows = []
    for f in FACTORS:
        ic = daily_rank_ic(df, f)
        rows.append(summarize(ic, f))

    summary = pd.DataFrame(rows).set_index("factor")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("=== Rank-IC summary ===")
    print(summary)

    print("\nReading guide:")
    print("  |mean_IC| > ~0.02-0.03 is a usable cross-sectional signal.")
    print("  |ICIR|   > ~0.3-0.5 and |t_stat| > ~2 suggest the edge is stable.")
    print("  Sign of mean_IC tells you the direction to trade the factor.")


if __name__ == "__main__":
    main()
