"""
PEAD event study — the headline analysis.

For every earnings event we line up the stock's market-adjusted return in a
window around the filing day (day 0), then average the cumulative abnormal return
(CAR) within SUE quintiles. If the market under-reacts, high-SUE stocks should
keep drifting up after the announcement and low-SUE down — the classic PEAD fan.

Also reports IC decay: the rank correlation between SUE and the forward
market-adjusted return, at several horizons.

Prices come from PRICES (default the synthetic panel); set
    PRICES=../data_panel.parquet
to run on the real price panel once you have real earnings.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PRICES = os.environ.get("PRICES", "prices_synth.parquet")
PRE, POST = 5, 60        # event window: [-5, +60] trading days
N_Q = 5                  # SUE quintiles
IC_HORIZONS = [1, 5, 10, 20, 40, 60]


def load_abnormal_returns() -> pd.DataFrame:
    """date x code matrix of market-adjusted daily returns."""
    px = pd.read_parquet(PRICES)
    px.columns = [c.lower() for c in px.columns]
    px["date"] = pd.to_datetime(px["date"])
    wide = px.pivot_table(index="date", columns="code", values="close").sort_index()
    ret = wide.pct_change()
    abn = ret.sub(ret.mean(axis=1), axis=0)     # subtract equal-weight market
    return abn


def main() -> None:
    sue = pd.read_parquet("sue_panel.parquet")
    sue["filed"] = pd.to_datetime(sue["filed"])
    abn = load_abnormal_returns()
    dates = abn.index
    pos = {d: i for i, d in enumerate(dates)}

    # SUE quintile labels (1 = most negative surprise, 5 = most positive).
    sue = sue[sue["code"].isin(abn.columns)].copy()
    sue["q"] = pd.qcut(sue["sue"], N_Q, labels=False) + 1

    # Collect each event's abnormal-return path on the [-PRE, +POST] grid.
    rel = np.arange(-PRE, POST + 1)
    paths = {q: [] for q in range(1, N_Q + 1)}
    ic_rows = []          # (sue, {h: fwd abnormal return})
    for _, e in sue.iterrows():
        d = e["filed"]
        # map filing to the first trading day >= filed
        i = pos.get(d)
        if i is None:
            future = dates[dates >= d]
            if len(future) == 0:
                continue
            i = pos[future[0]]
        if i - PRE < 0 or i + POST >= len(dates):
            continue
        col = abn[e["code"]].values
        window = col[i - PRE: i + POST + 1]
        if np.isnan(window).any():
            continue
        car = np.nancumsum(window)
        car = car - car[PRE]                  # baseline CAR=0 at day 0 (filing)
        paths[e["q"]].append(car)
        fwd = {h: np.nansum(col[i + 1: i + 1 + h]) for h in IC_HORIZONS}
        ic_rows.append((e["sue"], fwd))

    # ---- IC decay ----------------------------------------------------------
    icdf = pd.DataFrame([{"sue": s, **f} for s, f in ic_rows])
    print(f"Events used: {len(icdf)} | prices: {PRICES}\n")
    print("IC decay — rank corr(SUE, forward abnormal return):")
    print(f"{'horizon(d)':>10} {'rank IC':>9}")
    for h in IC_HORIZONS:
        ic = icdf["sue"].corr(icdf[h], method="spearman")
        print(f"{h:>10} {ic:>9.3f}")

    # ---- CAR-by-quintile plot (the headline) -------------------------------
    plt.figure(figsize=(10, 6))
    for q in range(1, N_Q + 1):
        if paths[q]:
            mean_car = np.mean(paths[q], axis=0) * 100
            plt.plot(rel, mean_car, label=f"SUE Q{q} (n={len(paths[q])})",
                     linewidth=1.8)
    plt.axvline(0, color="k", ls=":", lw=1)
    plt.title("Post-earnings cumulative abnormal return by SUE quintile")
    plt.xlabel("Trading days relative to filing (day 0)")
    plt.ylabel("Cumulative abnormal return (%)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out = "pead_event_study.png"
    plt.savefig(out, dpi=120)
    print(f"\nSaved {out}")
    spread = (np.mean(paths[N_Q], axis=0)[-1] - np.mean(paths[1], axis=0)[-1]) * 100
    print(f"Q{N_Q}-minus-Q1 CAR at +{POST}d: {spread:.2f}%")


if __name__ == "__main__":
    main()
