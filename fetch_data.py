import akshare as ak

# Daily bars for Kweichow Moutai (600519), forward-adjusted.
df = ak.stock_zh_a_hist(
    symbol="600519",
    period="daily",
    start_date="20200101",
    end_date="20241231",
    adjust="qfq"
)

# akshare returns Chinese column names; rename to English for downstream use.
df = df.rename(columns={
    "日期": "date", "股票代码": "code", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    "振幅": "amplitude", "涨跌幅": "pct_chg", "涨跌额": "change", "换手率": "turnover",
})

print(df.head())
print(f"\n{len(df)} rows")
print(f"\nColumns: {list(df.columns)}")

df.to_csv("data_600519.csv", index=False)
print("\nSaved to data_600519.csv")
