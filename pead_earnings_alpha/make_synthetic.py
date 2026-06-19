"""
Synthetic data generator — FOR PIPELINE VERIFICATION ONLY.

This fabricates an earnings panel and a price panel in which a Post-Earnings-
Announcement Drift (PEAD) of KNOWN size is planted: after each earnings filing,
a stock drifts in the direction of its earnings surprise. We use it to prove the
pipeline (SUE -> event study -> factor -> backtest) correctly *recovers* a drift
that we put in. It says nothing about whether PEAD is real in markets — that
requires the real EDGAR data produced by `fetch_edgar.py`.

Outputs (the schema every other script expects):
  earnings_raw.parquet : one row per (code, fiscal quarter)
      code, period_end, filed (point-in-time announcement date), eps
  prices_synth.parquet : daily price panel
      date, code, close
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(7)
N_STOCKS = 40
START = "2015-01-01"
N_DAYS = 1600                  # ~6.3 years of trading days
QUARTER = 63                   # trading days per quarter
REPORT_LAG = 40               # trading days from quarter-end to filing
DRIFT_DAYS = 40               # how long the planted drift lasts
DRIFT_COEF = 0.04             # cumulative drift ~= 4% * standardized surprise
JUMP_COEF = 0.010             # announcement-day reaction ~= 1% * surprise


def main() -> None:
    cal = pd.bdate_range(START, periods=N_DAYS)          # trading calendar
    codes = [f"SYN{i:02d}" for i in range(N_STOCKS)]

    price_rows, earn_rows = [], []
    for code in codes:
        # --- latent EPS process: seasonal level + AR(1) surprises -----------
        base = RNG.uniform(0.5, 2.5)
        seasonal = RNG.normal(0, 0.15, 4)               # repeats every 4 quarters
        # event dates: quarter-ends on the calendar and their filing dates
        q_idx = list(range(QUARTER, N_DAYS - DRIFT_DAYS - 5, QUARTER))

        surprises = {}                                   # filed_day_index -> std surprise
        prev_eps = {}
        for k, qi in enumerate(q_idx):
            shock = RNG.normal(0, 0.20)                  # the genuine surprise
            eps = base + seasonal[k % 4] + shock
            filed_i = qi + REPORT_LAG
            if filed_i >= N_DAYS:
                continue
            earn_rows.append({
                "code": code,
                "period_end": cal[qi],
                "filed": cal[filed_i],
                "eps": round(eps, 4),
            })
            # standardised surprise vs same quarter last year (seasonal RW)
            yoy = eps - prev_eps.get(k % 4, eps)
            surprises[filed_i] = yoy
            prev_eps[k % 4] = eps

        # normalise this stock's surprises to ~N(0,1) for a clean planted effect
        if surprises:
            vals = np.array(list(surprises.values()))
            sd = vals.std() or 1.0
            surprises = {i: v / sd for i, v in surprises.items()}

        # --- daily returns: market + idiosyncratic + planted PEAD -----------
        mkt = RNG.normal(0.0003, 0.010, N_DAYS)          # shared-ish drift
        idio = RNG.normal(0.0, 0.015, N_DAYS)
        ret = mkt + idio
        for filed_i, s in surprises.items():
            ret[filed_i] += JUMP_COEF * s                # announcement reaction
            end = min(filed_i + 1 + DRIFT_DAYS, N_DAYS)  # drift starts NEXT day
            ret[filed_i + 1:end] += (DRIFT_COEF * s) / DRIFT_DAYS

        close = 100 * np.cumprod(1 + ret)
        price_rows.append(pd.DataFrame({"date": cal, "code": code, "close": close}))

    prices = pd.concat(price_rows, ignore_index=True)
    earnings = pd.DataFrame(earn_rows).sort_values(["code", "filed"]).reset_index(drop=True)

    prices.to_parquet("prices_synth.parquet", index=False)
    earnings.to_parquet("earnings_raw.parquet", index=False)
    print(f"prices_synth.parquet : {prices.shape[0]} rows, {prices['code'].nunique()} stocks")
    print(f"earnings_raw.parquet : {earnings.shape[0]} earnings events")
    print(f"Planted PEAD: ~{DRIFT_COEF*100:.0f}% cumulative drift per 1-sigma surprise, "
          f"over {DRIFT_DAYS} days. The pipeline should recover this.")


if __name__ == "__main__":
    main()
