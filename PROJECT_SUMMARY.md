# Project Summary — Multi-Factor Equity Selection (one page)

**What it is.** An end-to-end cross-sectional multi-factor stock-selection study
in Python (pandas / scikit-learn), built to demonstrate a *rigorous, honest*
quant research process rather than to maximise a backtest number.

**Data.** Daily OHLCV for two independent US universes — 30 large caps
(2020–2024) and 88 names (2012–2017) — as long panels. OHLCV only; no
fundamentals.

**Pipeline (each a script, shared logic in `factor_lib.py`):**
1. **Factors** — base price/volume factors (momentum, reversal, volatility).
2. **IC validation** — daily cross-sectional rank IC vs forward 20-day return.
3. **Orthogonalisation** — regress each candidate factor on the base set
   *within each date* and keep the residual, so only independent signal remains;
   retain factors whose residual IC is still significant.
4. **Walk-forward backtest** — no look-ahead; rebalance/hold 20 days; equal-weight
   top-quintile long book and long-short; benchmark = equal-weight universe;
   five synthesis models (equal-weight, IC-, ICIR-weighted, Ridge, GBR).
5. **Cost & risk** — turnover, net-of-cost Sharpe, CAPM alpha/beta decomposition.
6. **Robustness** — sweep book size × horizon.

**Headline result.** The equal-weight base+orthogonal long book reaches a gross
Sharpe of **1.70** (88-stock universe) vs a 1.52 benchmark, **1.45 net of 10 bps**
costs, with **+4.4%/yr alpha at beta 0.71** (info ratio 0.80), stable across the
parameter grid (Sharpe 1.26–1.85).

**What I learned to say honestly:**
- **Simple > complex** — the sign-only equal-weight composite beats IC-weighting,
  ICIR-weighting, Ridge, and gradient-boosted trees on a small cross-section.
- **Alpha, not beta** — base factors alone are negative-alpha leveraged beta; the
  orthogonal factors are what create genuine, lower-beta alpha.
- **Not yet significant** — the alpha *t*-stat is 1.1–1.5 (< 2) over ~50 periods;
  IC *t*-stats are inflated by overlapping returns; factor selection used
  full-sample IC; universes are small, large-cap, survivorship-biased, single-
  regime. Full caveats and fixes in `LIMITATIONS.md`.

**Skills shown.** Cross-sectional factor research, leakage-free walk-forward
design, factor orthogonalisation, IC/ICIR signal blending, ML for return
prediction, transaction-cost and alpha/beta attribution, robustness testing, and
— the part interviewers probe — knowing precisely where the results are weak.
