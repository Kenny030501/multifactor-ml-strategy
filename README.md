# multifactor-ml-strategy

Multi-factor equity selection strategy with ML-based signal synthesis and
rigorous out-of-sample backtesting.

## Universe & data

- **30 US large caps** (AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, …), daily OHLCV.
- **2020-01-02 → 2024-12-30**, 1257 trading days per name (`data_panel.parquet`).
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
| 3. Backtest the strategy | `backtest.py` | console report |

```bash
pip install pandas numpy scikit-learn pyarrow
python factor_panel.py   # mom_20, rev_5, vol_20 per stock
python compute_ic.py     # are the factors predictive cross-sectionally?
python backtest.py       # walk-forward portfolio backtest
```

## Factors

Computed per stock (grouped so windows never cross tickers):

- `mom_20` — 20-day price momentum.
- `rev_5`  — 5-day return (short-term reversal proxy).
- `vol_20` — 20-day realised volatility of daily returns.

## Methodology (why it is honest out-of-sample)

- **Cross-sectional** problem: on each date, factors are z-scored across the 30
  stocks, then used to rank stocks against each other.
- **Rank IC** (`compute_ic.py`): daily Spearman correlation between each factor
  and the forward 20-day return — measures real cross-sectional edge.
- **Walk-forward, no look-ahead** (`backtest.py`): at each rebalance the model
  sees only factor values observable that day and training labels whose forward
  window has already fully realised. Rebalance every 20 trading days; hold 20.
- **Portfolio**: equal-weight the top 6 (long book); long-short = top 6 − bottom
  6. Benchmark = equal-weight all 30.
- Three signal models: equal-weight composite (factor signs from expanding-window
  IC), Ridge, and gradient-boosted trees.

## Results (2020–2024, out-of-sample)

Rank IC — only volatility carries a stable cross-sectional signal in this
universe/period (high-vol names led the post-2020 tech bull):

| factor | mean IC | t-stat |
|--------|--------:|-------:|
| vol_20 | **+0.040** | **+4.59** |
| mom_20 | −0.010 | −1.26 |
| rev_5  | +0.003 | +0.42 |

Backtest:

| strategy | ann. return | Sharpe | max DD |
|----------|------------:|-------:|-------:|
| Benchmark (EW all 30) | 18.5% | 0.99 | −27% |
| EW composite — long top 6 | **35.1%** | **1.23** | −41% |
| EW composite — long-short | 21.0% | 0.74 | −36% |
| Ridge — long top 6 | 29.1% | 0.97 | −50% |
| GBR — long top 6 | 31.8% | 1.14 | −39% |

**Takeaway:** the simple equal-weight composite beats both the benchmark and the
ML models here. The ML long-short alpha is weak — consistent with the IC result
that essentially one factor (volatility) drives the edge, so a flexible model on
30 names mostly fits noise. Complexity is not free.

## Possible next steps

- Add factors with independent IC (value, quality, earnings-based) — the current
  three are nearly collinear technicals.
- Larger universe (hundreds of names) so cross-sectional ranking and ML have room
  to work.
- Transaction costs / turnover control, and risk-adjusted (vol-target) sizing.
