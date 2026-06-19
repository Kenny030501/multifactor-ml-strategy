"""
Turn SUE into a daily cross-sectional factor and test that it is INDEPENDENT of
price/volume momentum and volatility.

A common objection to PEAD is "it's just price momentum near earnings". So we
z-score SUE cross-sectionally, then regress it on momentum and volatility within
each date and keep the residual; if the residual still has rank IC, the earnings
signal carries information the price factors do not.

On each trading day a stock carries its most recent SUE, provided that earnings
event is still "fresh" (filed within FRESH_DAYS). Output feeds backtest_pead.py.

Prices via PRICES env var (default synthetic).
"""

import os
import numpy as np
import pandas as pd

PRICES = os.environ.get("PRICES", "prices_synth.parquet")
FRESH_DAYS = 90          # a SUE is "active" for ~one quarter after filing
HORIZON = 20             # forward return for the IC check


def daily_panel() -> pd.DataFrame:
    px = pd.read_parquet(PRICES)
    px.columns = [c.lower() for c in px.columns]
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["code", "date"]).reset_index(drop=True)

    g = px.groupby("code")["close"]
    px["mom_20"] = g.transform(lambda x: x / x.shift(20) - 1)
    px["vol_20"] = g.transform(lambda x: x.pct_change().rolling(20).std())
    px["fwd_ret"] = g.transform(lambda x: x.shift(-HORIZON) / x - 1)
    return px


def attach_sue(px: pd.DataFrame) -> pd.DataFrame:
    sue = pd.read_parquet("sue_panel.parquet")[["code", "filed", "sue"]]
    sue["filed"] = pd.to_datetime(sue["filed"])
    sue = sue.sort_values("filed")
    px = px.sort_values("date")
    merged = pd.merge_asof(px, sue, by="code", left_on="date", right_on="filed",
                           direction="backward")
    # Drop stale signals (last earnings too far in the past).
    age = (merged["date"] - merged["filed"]).dt.days
    merged.loc[age > FRESH_DAYS, "sue"] = np.nan
    return merged


def zscore(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        g = df.groupby("date")[c]
        df[c + "_z"] = (df[c] - g.transform("mean")) / g.transform("std")
    return df


def orthogonalise(df, factor_z, base_z):
    def _resid(g):
        sub = g[[factor_z] + base_z].dropna()
        if len(sub) < len(base_z) + 2:
            return pd.Series(np.nan, index=g.index)
        X = np.column_stack([np.ones(len(sub))] + [sub[b].values for b in base_z])
        beta, *_ = np.linalg.lstsq(X, sub[factor_z].values, rcond=None)
        r = pd.Series(np.nan, index=g.index)
        r.loc[sub.index] = sub[factor_z].values - X @ beta
        return r
    return df.groupby("date", group_keys=False).apply(_resid)


def rank_ic(df, col):
    sub = df[["date", col, "fwd_ret"]].dropna()
    ics = [g[col].corr(g["fwd_ret"], method="spearman")
           for _, g in sub.groupby("date") if len(g) >= 5]
    ic = pd.Series(ics, dtype="float64").dropna()
    if ic.empty:
        return np.nan, np.nan
    return ic.mean(), ic.mean() / ic.std() * np.sqrt(len(ic))


def main() -> None:
    px = attach_sue(daily_panel())
    px = zscore(px, ["sue", "mom_20", "vol_20"])
    px["sue_orth"] = orthogonalise(px, "sue_z", ["mom_20_z", "vol_20_z"])

    print(f"Prices: {PRICES} | rows with a fresh SUE: {px['sue'].notna().sum()}")
    corr = px[["sue_z", "mom_20_z", "vol_20_z"]].corr().loc["sue_z"]
    print(f"\nSUE correlation with momentum {corr['mom_20_z']:+.3f}, "
          f"volatility {corr['vol_20_z']:+.3f}")

    raw_ic, raw_t = rank_ic(px, "sue_z")
    o_ic, o_t = rank_ic(px, "sue_orth")
    print("\nDaily cross-sectional rank IC vs forward 20d return:")
    print(f"  SUE (raw)               mean IC {raw_ic:+.4f}  t {raw_t:+.2f}")
    print(f"  SUE orthogonal to mom/vol  mean IC {o_ic:+.4f}  t {o_t:+.2f}")
    print("\nIf the orthogonal IC survives, PEAD is not just price momentum.")

    keep = ["date", "code", "close", "fwd_ret", "sue", "sue_z", "sue_orth"]
    px[keep].to_parquet("pead_factor_panel.parquet", index=False)
    print("\nSaved pead_factor_panel.parquet")


if __name__ == "__main__":
    main()
