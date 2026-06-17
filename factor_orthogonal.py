"""
Build candidate price/volume factors, measure their overlap with the existing
three factors, orthogonalise them, and test whether the orthogonal residuals
still carry an independent cross-sectional signal.

Why orthogonalise?
------------------
The base factors (mom_20, rev_5, vol_20) are nearly collinear technicals. A new
factor only *adds* value if the part of it that is NOT already explained by the
base factors predicts returns. So for each candidate we:
  1. z-score it cross-sectionally (per date),
  2. regress it on the base z-scores within each date and keep the RESIDUAL
     (this residual is, by construction, orthogonal to the base factors),
  3. measure the rank IC of that residual.
If the residual's IC survives, the factor brings genuinely new information.

Only OHLCV is used (no fundamentals), so every factor here is price/volume based.
All windows are computed per stock so they never cross tickers; all
orthogonalisation is cross-sectional (same-date only) -> no look-ahead.
"""

import numpy as np
import pandas as pd

HORIZON = 20
BASE = ["mom_20", "rev_5", "vol_20"]


def build_candidates(df: pd.DataFrame) -> list[str]:
    """Add candidate factor columns; return their names."""
    df.sort_values(["code", "date"], inplace=True)
    g = df.groupby("code")
    close = g["close"]
    ret = g["close"].transform(lambda x: x.pct_change())
    df["_ret"] = ret

    # 1. Long-horizon momentum (6 months) — trend at a different frequency.
    df["mom_120"] = close.transform(lambda x: x / x.shift(120) - 1)

    # 2. 12-1 momentum (t-252..t-21) — classic momentum that skips the last
    #    month to avoid short-term reversal contamination.
    df["mom_12_1"] = close.transform(lambda x: x.shift(21) / x.shift(252) - 1)

    # 3. MAX / lottery factor — largest single daily return over the past 20d
    #    (a well-known *negative* anomaly; lottery-like stocks underperform).
    df["max_20"] = g["close"].transform(
        lambda x: x.pct_change().rolling(20).max())

    # 4. Return skewness over 60d — distributional shape, distinct from vol.
    df["skew_60"] = df.groupby("code")["_ret"].transform(
        lambda x: x.rolling(60).skew())

    # 5. Amihud illiquidity — mean(|ret| / dollar-volume) over 20d.
    df["_dollar"] = df["close"] * df["volume"]
    df["illiq_20"] = df.groupby("code").apply(
        lambda s: (s["_ret"].abs() / s["_dollar"].replace(0, np.nan))
        .rolling(20).mean(), include_groups=False).reset_index(level=0, drop=True)

    # 6. 52-week-high proximity (George & Hwang) — close / 252d rolling max.
    df["hi_52w"] = close.transform(lambda x: x / x.rolling(252).max())

    # 7. Downside volatility — std of negative daily returns over 20d.
    #    min_periods so a window isn't all-NaN just because some days were up.
    df["dvol_20"] = df.groupby("code")["_ret"].transform(
        lambda x: x.where(x < 0).rolling(20, min_periods=5).std())

    df.drop(columns=["_ret", "_dollar"], inplace=True)
    return ["mom_120", "mom_12_1", "max_20", "skew_60", "illiq_20",
            "hi_52w", "dvol_20"]


def zscore_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        gg = df.groupby("date")[c]
        df[c + "_z"] = (df[c] - gg.transform("mean")) / gg.transform("std")
    return df


def orthogonalise(df: pd.DataFrame, factor_z: str, base_z: list[str]) -> pd.Series:
    """Per-date residual of factor_z after regressing on base_z (OLS)."""
    def _resid(g: pd.DataFrame) -> pd.Series:
        sub = g[[factor_z] + base_z].dropna()
        if len(sub) < len(base_z) + 2:
            return pd.Series(np.nan, index=g.index)
        X = np.column_stack([np.ones(len(sub))] + [sub[b].values for b in base_z])
        y = sub[factor_z].values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = pd.Series(np.nan, index=g.index)
        resid.loc[sub.index] = y - X @ beta
        return resid
    return df.groupby("date", group_keys=False).apply(_resid)


def rank_ic(df: pd.DataFrame, col: str) -> tuple[float, float]:
    sub = df[["date", col, "fwd_ret"]].dropna()
    ics = [g[col].corr(g["fwd_ret"], method="spearman")
           for _, g in sub.groupby("date") if len(g) >= 5]
    ic = pd.Series(ics, dtype="float64").dropna()
    if ic.empty:
        return np.nan, np.nan
    mean = ic.mean()
    t = (mean / ic.std()) * np.sqrt(len(ic)) if ic.std() > 0 else np.nan
    return mean, t


def main() -> None:
    df = pd.read_parquet("factor_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["code", "date"], inplace=True)
    df["fwd_ret"] = df.groupby("code")["close"].transform(
        lambda x: x.shift(-HORIZON) / x - 1)

    cands = build_candidates(df)
    df = zscore_by_date(df, BASE + cands)
    base_z = [b + "_z" for b in BASE]

    # ---- correlation of candidates with the base factors -------------------
    zcols = base_z + [c + "_z" for c in cands]
    corr = df[zcols].corr()
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== |correlation| of candidate factors with base factors ===")
    print(corr.loc[[c + "_z" for c in cands], base_z].abs().round(3).to_string())

    # ---- IC of raw vs orthogonalised residual ------------------------------
    print("\n=== Independent signal test (rank IC, t-stat) ===")
    print(f"{'factor':<10} {'raw IC':>9} {'raw t':>7}   "
          f"{'orth IC':>9} {'orth t':>7}   max|corr w/base|")
    rows = []
    for c in cands:
        raw_ic, raw_t = rank_ic(df, c + "_z")
        df[c + "_orth"] = orthogonalise(df, c + "_z", base_z)
        o_ic, o_t = rank_ic(df, c + "_orth")
        mx = corr.loc[c + "_z", base_z].abs().max()
        rows.append((c, raw_ic, raw_t, o_ic, o_t, mx))
        print(f"{c:<10} {raw_ic:>9.4f} {raw_t:>7.2f}   "
              f"{o_ic:>9.4f} {o_t:>7.2f}   {mx:>6.3f}")

    print("\nReading guide:")
    print("  Keep a factor if its ORTHOGONAL residual still has |t| > ~2.")
    print("  High max|corr w/base| means the raw factor mostly duplicates an")
    print("  existing one; the orth column shows what's left after removing it.")

    keep = [r[0] for r in rows if abs(r[4]) >= 2.0]
    print(f"\nFactors with independent signal (|orth t| >= 2): {keep}")

    out = df.drop(columns=[c for c in df.columns if c.endswith("_z")])
    out.to_parquet("factor_panel_ext.parquet", index=False)
    print("\nSaved extended panel -> factor_panel_ext.parquet "
          f"({out.shape[0]} rows, {out.shape[1]} cols)")


if __name__ == "__main__":
    main()
