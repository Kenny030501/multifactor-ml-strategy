"""
Compute SUE (Standardized Unexpected Earnings) per earnings event.

Definition (Bernard & Thomas 1989, seasonal-random-walk version):
    unexpected earnings = EPS_q - EPS_{q-4}      (vs the same quarter last year)
    SUE = unexpected earnings / std(those YoY differences over a trailing window)

This needs only reported EPS, so it works from EDGAR alone — no analyst data.
Input  : earnings_raw.parquet (code, period_end, filed, eps)
Output : sue_panel.parquet    (+ eps_yoy, sue)
"""

import pandas as pd

TRAIL = 8        # quarters of history to standardise over
MIN_TRAIL = 4    # require at least this many to emit a SUE


def main() -> None:
    df = pd.read_parquet("earnings_raw.parquet")
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filed"] = pd.to_datetime(df["filed"])
    df = df.sort_values(["code", "period_end"]).reset_index(drop=True)

    g = df.groupby("code")["eps"]
    # Year-on-year change (4 quarters back = same fiscal quarter last year).
    df["eps_yoy"] = g.transform(lambda x: x - x.shift(4))

    # Standardise each stock's surprise by its own trailing volatility of YoY
    # changes (shifted by 1 so the current surprise is excluded -> no look-ahead).
    def _std(x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(TRAIL, min_periods=MIN_TRAIL).std()
    df["yoy_std"] = df.groupby("code")["eps_yoy"].transform(_std)
    df["sue"] = df["eps_yoy"] / df["yoy_std"]

    out = df.dropna(subset=["sue"]).copy()
    out = out[["code", "period_end", "filed", "eps", "eps_yoy", "sue"]]
    out.to_parquet("sue_panel.parquet", index=False)

    print(f"sue_panel.parquet: {len(out)} events with a SUE "
          f"({out['code'].nunique()} stocks)")
    print(out["sue"].describe().round(3).to_string())
    print("\nSample:")
    print(out.head().to_string(index=False))


if __name__ == "__main__":
    main()
