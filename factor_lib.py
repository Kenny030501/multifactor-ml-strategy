"""
Shared, universe-agnostic factor library.

Centralises the factor definitions so the single-stock-pool flow and the
larger-universe flow compute identical factors. All windows are per stock
(never cross tickers); all cross-sectional operations (z-score,
orthogonalisation) use same-date information only, so nothing here looks ahead.
"""

import numpy as np
import pandas as pd

BASE = ["mom_20", "rev_5", "vol_20"]
# Orthogonal residuals kept after the independent-signal test in
# factor_orthogonal.py (skew_60 was dropped).
ORTH_SOURCES = ["mom_120", "mom_12_1", "max_20", "illiq_20", "hi_52w", "dvol_20"]
ORTH = [c + "_orth" for c in ORTH_SOURCES]


def normalise_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Lower-case OHLCV columns and ensure a datetime 'date'."""
    panel = panel.copy()
    panel.columns = [c.lower() for c in panel.columns]
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["code", "date"]).reset_index(drop=True)


def add_base_factors(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("code")["close"]
    df["mom_20"] = g.transform(lambda x: x / x.shift(20) - 1)
    df["rev_5"] = g.transform(lambda x: x / x.shift(5) - 1)
    df["vol_20"] = g.transform(lambda x: x.pct_change().rolling(20).std())
    return df


def add_candidate_factors(df: pd.DataFrame) -> pd.DataFrame:
    df.sort_values(["code", "date"], inplace=True)
    g = df.groupby("code")
    close = g["close"]
    df["_ret"] = close.transform(lambda x: x.pct_change())

    df["mom_120"] = close.transform(lambda x: x / x.shift(120) - 1)
    df["mom_12_1"] = close.transform(lambda x: x.shift(21) / x.shift(252) - 1)
    df["max_20"] = df.groupby("code")["_ret"].transform(
        lambda x: x.rolling(20).max())
    df["_dollar"] = df["close"] * df["volume"]
    df["illiq_20"] = df.groupby("code", group_keys=False).apply(
        lambda s: (s["_ret"].abs() / s["_dollar"].replace(0, np.nan))
        .rolling(20).mean(), include_groups=False)
    df["hi_52w"] = close.transform(lambda x: x / x.rolling(252).max())
    df["dvol_20"] = df.groupby("code")["_ret"].transform(
        lambda x: x.where(x < 0).rolling(20, min_periods=5).std())

    df.drop(columns=["_ret", "_dollar"], inplace=True)
    return df


def zscore_by_date(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        g = df.groupby("date")[c]
        df[c + "_z"] = (df[c] - g.transform("mean")) / g.transform("std")
    return df


def orthogonalise(df: pd.DataFrame, factor_z: str, base_z: list[str]) -> pd.Series:
    """Per-date OLS residual of factor_z after regressing on base_z."""
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


def build_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: OHLCV panel -> panel with base + orthogonal factors."""
    df = normalise_columns(panel)
    df = add_base_factors(df)
    df = add_candidate_factors(df)
    df = zscore_by_date(df, BASE + ORTH_SOURCES)
    base_z = [b + "_z" for b in BASE]
    for c in ORTH_SOURCES:
        df[c + "_orth"] = orthogonalise(df, c + "_z", base_z)
    return df.drop(columns=[c for c in df.columns if c.endswith("_z")])
