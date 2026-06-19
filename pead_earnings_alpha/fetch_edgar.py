"""
Fetch REAL point-in-time quarterly EPS from SEC EDGAR.

Run this where the network can reach SEC (e.g. Google Colab / your laptop) — the
build sandbox blocks data.sec.gov. EDGAR is free and needs no API key, only a
descriptive User-Agent with a contact email (SEC policy; rate limit 10 req/s).

Why EDGAR = point-in-time: each XBRL fact carries the `filed` date — the day the
number became public. Using `filed` as the event date means the backtest never
sees a number before the market did. This is the project's core integrity claim.

Output: earnings_raw.parquet  (code, period_end, filed, eps)
        — the exact schema the rest of the pipeline expects, so it drops in for
        prices_synth/earnings_raw with no other change.

Usage:
    python fetch_edgar.py            # default = the 30-stock project-1 universe
Edit TICKERS / set CONTACT to your email before running.
"""

import io
import json
import time
import urllib.request

import pandas as pd

CONTACT = "your_name your_email@example.com"   # <-- put your real email (SEC requires it)
HEADERS = {"User-Agent": CONTACT}

# Default universe = the US domestic filers from project 1 (all file 10-Q/10-K).
TICKERS = [
    "AAPL", "ABT", "ADBE", "AMZN", "BAC", "CSCO", "CVX", "DIS", "GOOGL", "GS",
    "HD", "INTC", "JPM", "KO", "MA", "MCD", "META", "MRK", "MSFT", "NFLX",
    "NKE", "NVDA", "PEP", "PFE", "PYPL", "SBUX", "TSLA", "V", "WMT", "XOM",
]
EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]
QTR_MIN, QTR_MAX = 80, 100      # a quarterly EPS fact spans ~90 days
YR_MIN, YR_MAX = 350, 380       # an annual EPS fact spans ~365 days


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(url, headers=HEADERS), timeout=30).read()
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def ticker_to_cik() -> dict:
    data = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    return {row["ticker"]: f"{int(row['cik_str']):010d}" for row in data.values()}


def eps_facts(cik: str) -> pd.DataFrame:
    """All EPS duration facts for one company, with filing dates."""
    raw = json.loads(_get(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"))
    facts = raw.get("facts", {}).get("us-gaap", {})
    rows = []
    for tag in EPS_TAGS:
        if tag not in facts:
            continue
        for unit_vals in facts[tag]["units"].values():       # usually 'USD/shares'
            for f in unit_vals:
                if "start" not in f or "end" not in f:
                    continue
                span = (pd.Timestamp(f["end"]) - pd.Timestamp(f["start"])).days
                rows.append({"start": f["start"], "end": f["end"], "val": f["val"],
                             "filed": f["filed"], "form": f.get("form"),
                             "fy": f.get("fy"), "fp": f.get("fp"), "span": span,
                             "tag": tag})
        if rows:
            break          # prefer diluted; fall back to basic only if diluted missing
    return pd.DataFrame(rows)


def quarterly_eps(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one quarterly EPS per period; derive Q4 = FY - (Q1+Q2+Q3)."""
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    df["filed"] = pd.to_datetime(df["filed"])

    q = df[(df["span"] >= QTR_MIN) & (df["span"] <= QTR_MAX)].copy()
    # earliest filing for each fiscal-quarter end (the announcement)
    q = q.sort_values("filed").drop_duplicates(subset=["end"], keep="first")

    # derive Q4 from annual minus the three reported quarters of that FY
    yr = df[(df["span"] >= YR_MIN) & (df["span"] <= YR_MAX)].copy()
    yr = yr.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    derived = []
    for _, a in yr.iterrows():
        fy_start = a["end"] - pd.Timedelta(days=365)
        prior = q[(q["end"] > fy_start) & (q["end"] < a["end"])]
        if len(prior) == 3:
            derived.append({"end": a["end"], "val": a["val"] - prior["val"].sum(),
                            "filed": a["filed"]})
    out = pd.concat([q[["end", "val", "filed"]],
                     pd.DataFrame(derived)], ignore_index=True)
    return out.sort_values("end").drop_duplicates(subset=["end"], keep="first")


def main() -> None:
    cik_map = ticker_to_cik()
    frames = []
    for i, tk in enumerate(TICKERS, 1):
        cik = cik_map.get(tk)
        if cik is None:
            print(f"  ! {tk}: no CIK"); continue
        try:
            qe = quarterly_eps(eps_facts(cik))
        except Exception as e:
            print(f"  ! {tk}: {e}"); continue
        if qe.empty:
            print(f"  ! {tk}: no quarterly EPS"); continue
        qe["code"] = tk
        frames.append(qe.rename(columns={"end": "period_end", "val": "eps"}))
        print(f"  {tk}: {len(qe)} quarters")
        time.sleep(0.15)        # be polite to SEC (<10 req/s)

    out = pd.concat(frames, ignore_index=True)[["code", "period_end", "filed", "eps"]]
    out = out.sort_values(["code", "period_end"]).reset_index(drop=True)
    out.to_parquet("earnings_raw.parquet", index=False)
    print(f"\nSaved earnings_raw.parquet: {len(out)} events, "
          f"{out['code'].nunique()} stocks, "
          f"{out['period_end'].min().date()} -> {out['period_end'].max().date()}")


if __name__ == "__main__":
    main()
