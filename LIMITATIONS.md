# Limitations & Future Improvements

This document states, as honestly as possible, what this project does **not**
establish and why. The backtest produces encouraging numbers (e.g. a gross
Sharpe of 1.70 and ~4.4% annual alpha on the 88-stock universe), but several
statistical, data, and methodological caveats mean these should be read as a
**rigorous demonstration of process**, not as evidence of a deployable edge.

---

## 1. Statistical limitations

**The alpha is not statistically significant.**
The CAPM-style regression of the equal-weight base+orth long book on the
benchmark gives an alpha *t*-stat of **1.07 (30-stock) and 1.45 (88-stock)** —
both below the conventional |t| > 2 threshold. With ~50 non-overlapping monthly
rebalances per universe, we cannot reject "the alpha is zero" at the 95% level.
The information ratios (0.56 and 0.80) are economically interesting but the
sample is too short to call the alpha real.

**IC t-stats are overstated.**
The rank-IC t-stats in `compute_ic.py` (e.g. vol_20 at +4.59) treat each daily
cross-sectional IC as independent. They are not: the forward return is a rolling
20-day window, so consecutive daily ICs are heavily autocorrelated. A
Newey-West / HAC correction would shrink these t-stats materially. The IC signs
and rough magnitudes are informative; the significance is inflated.

**Multiple-testing / selection bias.**
Several factors, four-to-five synthesis models, and two feature sets were
evaluated, and the orthogonal factor set was chosen using full-sample IC. The
"best" configuration was then highlighted. This is a textbook setting for
data-snooping: some of the apparent performance is selection. None of the
headline numbers are adjusted for the number of trials (e.g. via a deflated
Sharpe ratio), so they are upward-biased.

**Small number of effective observations.**
Sharpe, drawdown, and turnover are estimated from ~50 periods. Standard errors on
a Sharpe of 1.70 from 50 monthly observations are wide (roughly ±0.4), so the
gap between, say, 1.70 and 1.45 is well within noise.

---

## 2. Data limitations

**OHLCV only — no fundamentals.**
Every factor is price/volume derived. Even after orthogonalisation the factors
come from the same data-generating block, so the "independent" breadth is
limited; there is no value, quality, profitability, or earnings information.
This caps how much genuinely orthogonal signal can exist.

**Small, low-dispersion universes.**
30 and 88 large-cap names give a thin cross-section. Large caps move together
(pairwise return correlation ~0.4), so cross-sectional dispersion is low and the
**long-short book has little to short** — which is exactly what we observe
(long-short alpha is weak-to-negative everywhere). Cross-sectional ranks
estimated on 30–88 names are also noisy.

**Two disjoint periods, both bull markets.**
Universe 1 is 2020–2024 (benchmark Sharpe 0.82); universe 2 is 2012–2017
(benchmark Sharpe 1.52). They do not overlap, so they cannot be pooled, and
neither contains a full bear/crisis regime (2008, 2000, or a sustained
2022-style drawdown). Performance in a regime unlike the training periods is
untested.

**Survivorship bias.**
Both universes are fixed lists of names that existed and were liquid enough to be
in the source datasets for the whole window. Delisted, merged, or failed names
are absent, which flatters returns — especially for any momentum-type factor.

**No point-in-time integrity for membership.**
Universe membership is static and chosen with hindsight (these *are* large caps
today / in the dataset). A real backtest needs point-in-time index constituents.

---

## 3. Methodological limitations

**Factor-selection look-ahead.**
The *scoring* is strictly walk-forward (no look-ahead in signals or labels), but
the **decision of which orthogonal factors to keep** used full-sample IC
t-stats. That choice leaks information from the whole sample into the factor set.
A purged/combinatorial walk-forward that re-selects factors out-of-sample would
be stricter.

**Fixed, conventional hyperparameters.**
Factor windows (20/120/252 days), the 20-day horizon, quintile book size, and
model settings are chosen by convention, not validated out-of-sample. The
robustness sweep (`robustness.py`) shows the result is stable across these knobs,
which is reassuring but not the same as proper nested cross-validation.

**Backtest realism gaps.**
- Transaction costs are a flat per-side bps with no market impact, slippage, or
  size dependence.
- Shorting assumes free, available borrow — unrealistic, and the long-short
  results should be read accordingly.
- No position limits, capacity, financing, dividends-timing, or tax modelling.
- The benchmark is an equal-weight basket of the same universe, not an investable
  index; the long book also carries market beta (~0.7–1.3), so part of the raw
  return is simply beta.
- Rebalance period equals the forward horizon (clean, non-overlapping) but
  discards data; overlapping windows would need HAC standard errors.

---

## 4. What would meaningfully improve this — with better data

| Limitation | Fix when better data is available |
|------------|-----------------------------------|
| Tiny, biased universe | Point-in-time, survivorship-bias-free constituents of Russell 1000/3000 — hundreds of names, real dispersion, a working long-short book. |
| OHLCV only | Point-in-time fundamentals (Compustat/SF1) for value/quality/earnings factors with genuinely independent IC. |
| Short, single-regime sample | 20+ years spanning 2000, 2008, 2020 to test regime robustness and give enough periods for significance. |
| Inflated significance | Newey-West/HAC t-stats for overlapping returns; deflated Sharpe ratio (Bailey & López de Prado) to penalise the number of trials; block-bootstrap confidence intervals. |
| Factor-selection look-ahead | Combinatorial purged cross-validation (López de Prado) so factor selection itself is out-of-sample. |
| Cost realism | Borrow-fee, slippage, and square-root market-impact cost model calibrated to ADV. |
| Long-only beta | Beta-hedged or dollar-neutral construction so reported alpha is exposure-adjusted by design. |

---

## One-line summary

The value of this project is the **process** — IC validation, orthogonalisation,
strictly walk-forward backtesting, cost and alpha/beta decomposition, and a
robustness sweep — together with the discipline to report honest negative
results (simple equal-weight beats complex ML; IC/ICIR-weighting does not beat
equal-weight; the alpha is not yet statistically significant). The numbers are a
demonstration, not a deployable signal.
