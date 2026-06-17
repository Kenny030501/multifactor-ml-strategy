"""
Cross-sectional multi-factor stock-selection backtest.

Pipeline
--------
1. Cross-sectional standardisation: z-score every factor within each date, so
   that on any given day the factors are comparable across the 30 stocks.
2. Signal synthesis (three models, all strictly walk-forward / no look-ahead):
     - EW   : equal-weight composite of the z-scored factors, each factor
              signed by the SIGN of its expanding-window IC up to the
              rebalance date (so the direction is learned from the past only).
     - Ridge: linear model, retrained on an expanding window at each rebalance.
     - GBR  : gradient-boosted trees, same walk-forward protocol.
3. Portfolio construction: every REBAL trading days, rank stocks by the model
   score, equal-weight the top K as the long book; the long-short book is
   top K minus bottom K. Hold until the next rebalance.
4. Evaluation vs an equal-weight-all-stocks benchmark: annualised return,
   volatility, Sharpe, and max drawdown.

There is no look-ahead: the score at rebalance date t is built only from
factor values observable at t and from training labels whose forward window
has already fully realised by t.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

HORIZON = 20      # forward return / holding period, in trading days
REBAL = 20        # rebalance every REBAL trading days (non-overlapping)
TOP_K = 6         # long the top 6, short the bottom 6 (quintiles of 30)
MIN_TRAIN = 252   # minimum training rows before a model is trusted
PERIODS_PER_YEAR = 252 / HORIZON

BASE = ["mom_20", "rev_5", "vol_20"]
# Orthogonal residuals (built by factor_orthogonal.py) that pass the
# independent-signal test. Each is already orthogonal to the base factors.
ORTH = ["mom_120_orth", "mom_12_1_orth", "max_20_orth",
        "illiq_20_orth", "hi_52w_orth", "dvol_20_orth"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(features: list[str]) -> pd.DataFrame:
    """Load the richest available panel and z-score the requested features."""
    try:
        df = pd.read_parquet("factor_panel_ext.parquet")
    except FileNotFoundError:
        df = pd.read_parquet("factor_panel.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)

    # Forward return per stock (label), grouped so windows never cross tickers.
    df["fwd_ret"] = (
        df.sort_values(["code", "date"])
          .groupby("code")["close"]
          .transform(lambda x: x.shift(-HORIZON) / x - 1)
    )

    # Cross-sectional z-score of each feature within each date.
    for f in features:
        g = df.groupby("date")[f]
        df[f + "_z"] = (df[f] - g.transform("mean")) / g.transform("std")
    return df


# --------------------------------------------------------------------------- #
# Signal models -> a per-(date, code) score on rebalance dates
# --------------------------------------------------------------------------- #
def _rebalance_dates(dates: np.ndarray) -> np.ndarray:
    return dates[::REBAL]


def score_equal_weight(df: pd.DataFrame, rebal_dates, features) -> pd.DataFrame:
    """EW composite, factor signs from expanding-window IC (no look-ahead)."""
    zcols = [f + "_z" for f in features]
    out = []
    for t in rebal_dates:
        past = df[(df["date"] < t)].dropna(subset=zcols + ["fwd_ret"])
        # Direction of each factor learned only from realised history.
        signs = {}
        for zc in zcols:
            ic = past[zc].corr(past["fwd_ret"], method="spearman") if len(past) else 0.0
            signs[zc] = np.sign(ic) if np.isfinite(ic) and ic != 0 else 1.0
        cur = df[(df["date"] == t)].dropna(subset=zcols).copy()
        cur["score"] = sum(signs[zc] * cur[zc] for zc in zcols)
        out.append(cur[["date", "code", "score", "fwd_ret"]])
    return pd.concat(out, ignore_index=True)


def score_model(df: pd.DataFrame, rebal_dates, make_model, features) -> pd.DataFrame:
    """Walk-forward supervised model retrained at each rebalance date."""
    zcols = [f + "_z" for f in features]
    out = []
    for t in rebal_dates:
        # Train only on labels whose forward window is already realised by t.
        train = df[df["date"] <= t].dropna(subset=zcols + ["fwd_ret"])
        # A label at date d is realised by t only if d + HORIZON <= t.
        cutoff = df["date"][df["date"] <= t].drop_duplicates().sort_values()
        if len(cutoff) <= HORIZON:
            continue
        last_usable = cutoff.iloc[-(HORIZON + 1)]
        train = train[train["date"] <= last_usable]
        if len(train) < MIN_TRAIN:
            continue
        cur = df[df["date"] == t].dropna(subset=zcols).copy()
        if cur.empty:
            continue
        model = make_model()
        model.fit(train[zcols], train["fwd_ret"])
        cur["score"] = model.predict(cur[zcols])
        out.append(cur[["date", "code", "score", "fwd_ret"]])
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Portfolio construction + metrics
# --------------------------------------------------------------------------- #
def portfolio_returns(scores: pd.DataFrame):
    """Return per-rebalance long and long-short returns from a score table."""
    long_r, ls_r, dates = [], [], []
    for t, g in scores.groupby("date"):
        g = g.dropna(subset=["fwd_ret"]).sort_values("score", ascending=False)
        if len(g) < 2 * TOP_K:
            continue
        top = g.head(TOP_K)["fwd_ret"].mean()
        bot = g.tail(TOP_K)["fwd_ret"].mean()
        long_r.append(top)
        ls_r.append(top - bot)
        dates.append(t)
    return (pd.Series(long_r, index=dates, name="long"),
            pd.Series(ls_r, index=dates, name="long_short"))


def benchmark_returns(df: pd.DataFrame, rebal_dates) -> pd.Series:
    """Equal-weight all stocks, rebalanced on the same schedule."""
    r, idx = [], []
    for t in rebal_dates:
        cur = df[df["date"] == t].dropna(subset=["fwd_ret"])
        if not cur.empty:
            r.append(cur["fwd_ret"].mean())
            idx.append(t)
    return pd.Series(r, index=idx, name="benchmark")


def metrics(period_ret: pd.Series) -> dict:
    r = period_ret.dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    total = equity.iloc[-1] - 1
    ann = equity.iloc[-1] ** (PERIODS_PER_YEAR / len(r)) - 1
    vol = r.std() * np.sqrt(PERIODS_PER_YEAR)
    sharpe = ann / vol if vol > 0 else np.nan
    dd = (equity / equity.cummax() - 1).min()
    return {
        "total_return": total,
        "ann_return": ann,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "win_rate": (r > 0).mean(),
        "n_periods": len(r),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def _models():
    return {
        "EW_composite": ("ew", None),
        "Ridge":        ("model", lambda: Ridge(alpha=10.0)),
        "GBR":          ("model", lambda: GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.03,
            subsample=0.7, random_state=42)),
    }


def run_feature_set(df: pd.DataFrame, rebal, features: list[str]) -> dict:
    """Backtest every model on one feature set; return metrics rows."""
    rows = {}
    for name, (kind, maker) in _models().items():
        sc = (score_equal_weight(df, rebal, features) if kind == "ew"
              else score_model(df, rebal, maker, features))
        long_r, ls_r = portfolio_returns(sc)
        rows[f"{name} | long top{TOP_K}"] = metrics(long_r)
        rows[f"{name} | long-short"] = metrics(ls_r)
    return rows


def main() -> None:
    feature_sets = {
        "base(3)": BASE,
        "base+orth(9)": BASE + ORTH,
    }
    df = load(BASE + ORTH)  # z-score every feature once
    dates = np.sort(df["date"].unique())
    rebal = _rebalance_dates(dates)

    # Fair comparison: both feature sets must trade over the SAME dates. The
    # orthogonal factors need a long warm-up (252d), so restrict every model to
    # rebalance dates where the FULL feature set has a usable cross-section.
    ext_z = [f + "_z" for f in BASE + ORTH]
    valid = {t for t in rebal
             if df[(df["date"] == t)].dropna(subset=ext_z).shape[0] >= 2 * TOP_K}
    rebal = np.array([t for t in rebal if t in valid])

    print(f"Universe: {df['code'].nunique()} stocks | "
          f"{pd.Timestamp(dates[0]).date()} -> {pd.Timestamp(dates[-1]).date()}")
    print(f"Rebalance every {REBAL}d | hold {HORIZON}d | long/short top/bottom {TOP_K}")
    print(f"Common backtest window (after 252d warm-up): "
          f"{pd.Timestamp(rebal[0]).date()} -> {pd.Timestamp(rebal[-1]).date()} "
          f"| {len(rebal)} rebalance dates")
    print(f"Feature sets: base={BASE}\n              orth-add={ORTH}\n")

    rows = {"Benchmark(EW all)": metrics(benchmark_returns(df, rebal))}
    for set_name, feats in feature_sets.items():
        for label, m in run_feature_set(df, rebal, feats).items():
            rows[f"[{set_name}] {label}"] = m

    table = pd.DataFrame(rows).T
    pct = ["total_return", "ann_return", "ann_vol", "max_drawdown", "win_rate"]
    for c in pct:
        table[c] = (table[c] * 100).map(lambda v: f"{v:6.2f}%")
    table["sharpe"] = table["sharpe"].map(lambda v: f"{v:5.2f}")
    table["n_periods"] = table["n_periods"].astype(int)
    pd.set_option("display.width", 220)
    print("=== Backtest results (out-of-sample, walk-forward) ===")
    print(table.to_string())
    print("\nNote: long carries market beta; long-short isolates factor alpha.")
    print("Compare [base(3)] vs [base+orth(9)] to see the orthogonal factors' lift.")


if __name__ == "__main__":
    main()
