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
| 3. Build orthogonal factors | `factor_orthogonal.py` | `factor_panel_ext.parquet` |
| 4. Backtest the strategy | `backtest.py` | console report |

```bash
pip install pandas numpy scikit-learn pyarrow
python factor_panel.py       # base factors: mom_20, rev_5, vol_20
python compute_ic.py         # are the base factors predictive cross-sectionally?
python factor_orthogonal.py  # build + test orthogonal price/volume factors
python backtest.py           # walk-forward backtest, base vs base+orthogonal
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

Backtest — **base (3 factors) vs base+orthogonal (9 factors)** over an identical
window (both restricted to the same rebalance dates after the 252-day warm-up,
so the comparison is not confounded by the period):

| strategy (same 49-period window) | ann. return | Sharpe | ann. vol | max DD |
|----------------------------------|------------:|-------:|---------:|-------:|
| Benchmark (EW all 30) | 14.6% | 0.82 | 17.8% | −27% |
| base — EW composite, long top 6 | 26.9% | 0.90 | 29.9% | −41% |
| **base+orth — EW composite, long top 6** | 21.4% | **1.00** | **21.4%** | **−36%** |
| base — GBR, long top 6 | 20.4% | 0.71 | 28.6% | −39% |
| base+orth — GBR, long top 6 | 16.8% | 0.64 | 26.2% | −40% |
| base — EW composite, long-short | 17.4% | 0.61 | 28.5% | −36% |
| base+orth — EW composite, long-short | 2.4% | 0.12 | 20.8% | −44% |

**Takeaways (honest):**

- The orthogonal factors improve the **long-only, risk-adjusted** result via
  diversification: the equal-weight composite's Sharpe rises 0.90 → **1.00** and
  volatility falls 29.9% → 21.4% — better risk, *not* higher raw return.
- They **hurt the ML and long-short** versions: 9 features on only 30 stocks
  overfit, and the long-short alpha collapses. More factors need a bigger
  universe.
- Directional return is still dominated by one or two strong factors
  (volatility / illiquidity); the simple composite remains hard to beat.
- General lesson: orthogonalisation buys *diversification*, not free return, and
  model complexity is not free on a small cross-section.

## Possible next steps

- **Bigger universe** (hundreds of names) — the single biggest lever; it gives
  both cross-sectional ranking and ML room to work, and lets the orthogonal
  factors pay off in the long-short book.
- Fundamental factors (value, quality, earnings) for IC that is truly
  independent of the price/volume block — needs a fundamentals data source.
- Transaction costs / turnover control, and risk-adjusted (vol-target) sizing.
- IC-weighted (not equal-weight) factor blending so strong factors aren't diluted.
