# PEAD project — Limitations & honest caveats

As with project 1, this records what the project does **not** establish. The
pipeline is verified to be correct; the economic result depends on real EDGAR
data and carries the caveats below.

## Verification vs. evidence
- The headline numbers in this repo come from **synthetic data with a planted
  drift** and only prove the code recovers a known effect. They are **not**
  evidence that PEAD pays — that requires running `fetch_edgar.py` on real
  filings and re-running the pipeline.

## Signal / accounting caveats
- **Announcement-date proxy.** The event date is the 10-Q/10-K `filed` date. The
  actual earnings press release (8-K) usually precedes the formal filing by a few
  days, so using `filed` is slightly *late* — conservative for capturing drift,
  but not the true announcement instant. A stricter version would parse 8-K
  earnings items.
- **Q4 is derived** as annual EPS − (Q1+Q2+Q3). This is standard but introduces
  small errors when fiscal calendars shift or restatements occur.
- **Seasonal-random-walk SUE**, not analyst-based. Analyst-expectation surprises
  (and revisions) are a stronger, more timely signal but need paid estimate data;
  the EPS-only SUE is cruder.
- **EPS only.** No revenue surprise, guidance, or earnings-call text — all known
  to add to PEAD.
- **As-filed values.** `companyfacts` gives values as reported; later amendments
  exist but are not separately handled.

## Data / universe caveats
- **Small universe** (the 30-stock project-1 set). Plenty of *events* for an
  event study, but a thin daily cross-section for a quintile long-short.
- **Survivorship bias.** The ticker list is current large caps; delisted names
  are absent. Static membership, not point-in-time index constituents.
- **Single regime / short sample** inherited from the price panels.

## Methodology caveats
- **No transaction costs yet** in `backtest_pead.py` (project 1's
  `transaction_costs.py` pattern should be applied; PEAD turnover is high because
  the signal refreshes every quarter).
- **Significance not yet adjusted** for overlapping returns or multiple testing
  (same issues documented in project 1's `LIMITATIONS.md`).
- **Long book carries beta**; the long-short row is the cleaner read of the
  earnings signal.

## What would strengthen it
- 8-K-based announcement timestamps; analyst-expectation SUE and revisions.
- A larger, point-in-time, survivorship-bias-free universe (Russell 1000).
- Transaction-cost and capacity analysis; combine with project-1 factors and test
  whether the blended, orthogonal book improves the information ratio.
