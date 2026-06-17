# multifactor-ml-strategy

Multi-factor equity selection strategy with ML-based signal synthesis and
rigorous out-of-sample backtesting.

## Universe & data

- **Primary: 30 US large caps** (AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, …), daily
  OHLCV, **2020-01-02 → 2024-12-30** (`data_panel.parquet`).
- **Larger: 88 US names**, **2012–2017** (`data_panel_large.parquet`), built by
  `build_universe.py` from a public GitHub-hosted dataset — used to test the
  strategy at ~3× scale on an independent period.
- Stored as a long panel: one row per `(date, code)`.

> Data is fetched separately (e.g. via yfinance on Colab). The remote build
> environment cannot reach market-data APIs, so the panel is committed as a
> parquet file. `panel_data.py` is a **deprecated** synthetic scaffold — do not
> run it (it is guarded to avoid overwriting the real panel).

## Pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Build factor panel | `factor_panel.py` | `factor_panel.parquet` |
| 2. Validate factors (rank IC) | `compute_ic.py` | console report |
| 3. Build orthogonal factors | `factor_orthogonal.py` | `factor_panel_ext.parquet` |
| 4. Backtest the strategy | `backtest.py` | console report |
| (opt) Larger universe | `build_universe.py` | `factor_panel_large_ext.parquet` |

`factor_lib.py` holds the shared, universe-agnostic factor definitions used by
both the 30-stock flow and the larger-universe flow.

```bash
pip install pandas numpy scikit-learn pyarrow
python factor_panel.py       # base factors: mom_20, rev_5, vol_20
python compute_ic.py         # are the base factors predictive cross-sectionally?
python factor_orthogonal.py  # build + test orthogonal price/volume factors
python backtest.py           # walk-forward backtest, base vs base+orthogonal

# Larger universe (88 real US names from a GitHub-hosted dataset):
python build_universe.py
PANEL=factor_panel_large_ext.parquet python backtest.py
```

## Factors

**Base** factors (`factor_panel.py`), computed per stock so windows never cross
tickers:

- `mom_20` — 20-day price momentum.
- `rev_5`  — 5-day return (short-term reversal proxy).
- `vol_20` — 20-day realised volatility of daily returns.

**Orthogonal** candidates (`factor_orthogonal.py`) — all price/volume based
(no fundamentals are available). Each candidate is z-scored cross-sectionally,
then **regressed on the base factors within each date and replaced by its
residual**, so the part it shares with the base factors is removed and only its
independent information remains. A factor is kept if that residual still has a
significant rank IC:

| candidate | what it is | raw IC (t) | orthogonal IC (t) |
|-----------|------------|-----------:|------------------:|
| `illiq_20`  | Amihud illiquidity | −0.067 (−8.3) | **−0.050 (−7.4)** |
| `mom_120`   | 6-month momentum | +0.045 (+5.1) | **+0.025 (+3.6)** |
| `mom_12_1`  | 12-1 momentum (skip last month) | +0.038 (+4.2) | **+0.026 (+3.5)** |
| `hi_52w`    | 52-week-high proximity | +0.007 (+0.7) | **+0.032 (+4.5)** |
| `max_20`    | lottery / max daily return | +0.022 (+2.7) | **−0.030 (−5.8)** |
| `dvol_20`   | downside volatility | +0.027 (+3.5) | +0.019 (+3.3) |
| `skew_60`   | 60-day return skewness | +0.000 (0.1) | −0.007 (−1.1) — dropped |

Orthogonalisation is the point: `hi_52w` looks dead raw (t 0.7) but has a strong
**independent** signal once its overlap with momentum/vol is removed (t +4.5);
`max_20` even flips sign — the classic lottery effect only appears after
controlling for volatility (with which it correlates 0.81).

## Methodology (why it is honest out-of-sample)

- **Cross-sectional** problem: on each date, factors are z-scored across the 30
  stocks, then used to rank stocks against each other.
- **Rank IC** (`compute_ic.py`): daily Spearman correlation between each factor
  and the forward 20-day return — measures real cross-sectional edge.
- **Walk-forward, no look-ahead** (`backtest.py`): at each rebalance the model
  sees only factor values observable that day and training labels whose forward
  window has already fully realised. Rebalance every 20 trading days; hold 20.
- **Portfolio**: equal-weight the top quintile (long book); long-short = top
  quintile − bottom quintile. Book size scales with the universe, so the same
  code runs on 30 or hundreds of names. Benchmark = equal-weight all stocks.
- Four signal models: equal-weight composite (factor signs from expanding-window
  IC), **IC-weighted composite** (weight each factor by its expanding-window mean
  IC — sign *and* magnitude), Ridge, and gradient-boosted trees.

## Results (2020–2024, out-of-sample)

Rank IC — only volatility carries a stable cross-sectional signal in this
universe/period (high-vol names led the post-2020 tech bull):

| factor | mean IC | t-stat |
|--------|--------:|-------:|
| vol_20 | **+0.040** | **+4.59** |
| mom_20 | −0.010 | −1.26 |
| rev_5  | +0.003 | +0.42 |

Backtest — **base (3 factors) vs base+orthogonal (9 factors)**, long book, over
an identical window (both restricted to the same rebalance dates after the
252-day warm-up, so the comparison is not confounded by the period):

| strategy (same 49-period window) | ann. return | Sharpe | ann. vol | max DD |
|----------------------------------|------------:|-------:|---------:|-------:|
| Benchmark (EW all 30) | 14.6% | 0.82 | 17.8% | −27% |
| base — EW composite | 26.9% | 0.90 | 29.9% | −41% |
| **base+orth — EW composite** | 21.4% | **1.00** | **21.4%** | **−36%** |
| base — IC-weighted | 3.7% | 0.14 | 27.5% | −49% |
| base+orth — IC-weighted | 10.8% | 0.42 | 26.0% | −49% |
| base — GBR | 24.0% | 0.86 | 28.0% | −34% |
| base+orth — GBR | 13.1% | 0.58 | 22.6% | −38% |

## Larger universe (88 stocks, 2013–2017)

`build_universe.py` pulls a real 88-stock US panel (StockNet dataset, committed
on GitHub and reachable from the sandbox) — ~3× the names, an independent period.
Long book, same identical-window protocol:

| strategy (88 stocks, 49-period window) | ann. return | Sharpe | ann. vol | max DD |
|----------------------------------------|------------:|-------:|---------:|-------:|
| Benchmark (EW all 88) | 12.8% | 1.52 | 8.4% | −10% |
| base — EW composite | 12.0% | 1.01 | 11.9% | −15% |
| **base+orth — EW composite** | 13.8% | **1.70** | **8.1%** | **−8%** |
| base — IC-weighted | 11.2% | 0.92 | 12.2% | −20% |
| base+orth — IC-weighted | 10.2% | 1.01 | 10.1% | −10% |
| base — GBR | 15.8% | 1.31 | 12.0% | −14% |
| base+orth — GBR | 8.2% | 0.82 | 9.9% | −15% |

**Takeaways (honest, across both universes):**

- **Orthogonal factors help the equal-weight long composite in both universes**
  (Sharpe 0.90 → 1.00 on 30 names; **1.01 → 1.70** on 88 names). The lift is
  larger on the bigger universe — diversification needs breadth — and there the
  base+orth composite (Sharpe 1.70) *beats the strong benchmark* (1.52).
- **IC-weighting did not beat plain equal-weighting in either universe.**
  Weighting by IC *magnitude* amplifies the noise in the IC estimates; with few,
  noisy factors the sign-only equal weight is more robust. An honest negative
  result for the IC-weighting idea as implemented.
- **Long-short alpha is weak-to-negative** everywhere: these large-cap universes
  have low cross-sectional dispersion, so most of the edge (and a lot of market
  beta) lives in the long book, not in shorting the bottom.
- A bigger universe sharply raises the benchmark's Sharpe (less idiosyncratic
  noise) and makes the orthogonal-factor edge clearer — breadth matters more than
  model complexity.

## Possible next steps

- Even bigger universe (hundreds of names) and a higher-dispersion universe
  (small/mid caps) so the long-short book has something to short.
- Try **ICIR-weighting** (mean IC / IC std) or shrinkage instead of raw
  IC-magnitude weighting, which was too noisy here.
- Fundamental factors (value, quality, earnings) for IC that is truly
  independent of the price/volume block — needs a fundamentals data source.
- Transaction costs / turnover control, and risk-adjusted (vol-target) sizing.
