"""
Build a larger real stock universe and its factor panel.

The execution sandbox cannot reach any live market-data API (Yahoo, Stooq, etc.
are network-blocked), so a bigger universe is sourced from REAL daily OHLCV CSVs
that are committed to a public GitHub repo and therefore reachable via
raw.githubusercontent.com. We use the StockNet dataset (Xu & Cohen, ACL 2018):
88 large US names, ~5 years of daily data each.

Output:
  data_panel_large.parquet         (date, code, OHLCV)
  factor_panel_large_ext.parquet   (+ base and orthogonal factors)

Note: this universe spans a different period than the 30-stock panel
(roughly 2012-2017 vs 2020-2024). It is used to test the strategy at ~3x scale
on independent data, not to extend the original sample.
"""

import io
import sys
import urllib.request

import pandas as pd

import factor_lib

RAW_BASE = ("https://raw.githubusercontent.com/yumoxu/stocknet-dataset/"
            "master/price/raw/")
TICKERS = [
    "AAPL", "ABB", "ABBV", "AEP", "AGFS", "AMGN", "AMZN", "BA", "BABA", "BAC",
    "BBL", "BCH", "BHP", "BP", "BRK-A", "BSAC", "BUD", "C", "CAT", "CELG",
    "CHL", "CHTR", "CMCSA", "CODI", "CSCO", "CVX", "D", "DHR", "DIS", "DUK",
    "EXC", "FB", "GD", "GE", "GMRE", "GOOG", "HD", "HON", "HRG", "HSBC",
    "IEP", "INTC", "JNJ", "JPM", "KO", "LMT", "MA", "MCD", "MDT", "MMM",
    "MO", "MRK", "MSFT", "NEE", "NGG", "NVS", "ORCL", "PCG", "PCLN", "PEP",
    "PFE", "PG", "PICO", "PM", "PPL", "PTR", "RDS-B", "REX", "SLB", "SNP",
    "SNY", "SO", "SPLP", "SRE", "T", "TM", "TOT", "TSM", "UL", "UN",
    "UNH", "UPS", "UTX", "V", "VZ", "WFC", "WMT", "XOM",
]


def fetch_ticker(ticker: str) -> pd.DataFrame | None:
    url = RAW_BASE + ticker + ".csv"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        raw = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    except Exception as e:  # skip a ticker rather than abort the whole build
        print(f"  ! {ticker}: {e}")
        return None
    df = pd.read_csv(io.StringIO(raw))
    df["code"] = ticker
    return df


def main() -> None:
    print(f"Downloading {len(TICKERS)} tickers from StockNet (GitHub raw)...")
    frames = []
    for i, t in enumerate(TICKERS, 1):
        df = fetch_ticker(t)
        if df is not None and len(df) > 260:  # need >1y of history
            frames.append(df)
        if i % 20 == 0:
            print(f"  ...{i}/{len(TICKERS)}")
    if not frames:
        sys.exit("No data downloaded (network blocked?).")

    panel = pd.concat(frames, ignore_index=True)
    # Use split/dividend-adjusted close for return-based factors.
    if "Adj Close" in panel.columns:
        panel["Close"] = panel["Adj Close"]
    panel = panel[["Date", "code", "Open", "High", "Low", "Close", "Volume"]]
    panel.to_parquet("data_panel_large.parquet", index=False)
    print(f"\ndata_panel_large.parquet: {panel.shape[0]} rows, "
          f"{panel['code'].nunique()} stocks, "
          f"{pd.to_datetime(panel['Date']).min().date()} -> "
          f"{pd.to_datetime(panel['Date']).max().date()}")

    factors = factor_lib.build_factor_panel(panel)
    factors.to_parquet("factor_panel_large_ext.parquet", index=False)
    print(f"factor_panel_large_ext.parquet: {factors.shape[0]} rows, "
          f"{factors.shape[1]} cols "
          f"(base {factor_lib.BASE} + orth {factor_lib.ORTH})")


if __name__ == "__main__":
    main()
