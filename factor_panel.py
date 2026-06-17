import pandas as pd

panel = pd.read_parquet("data_panel.parquet")
panel.columns = [c.lower() for c in panel.columns]  
panel = panel.sort_values(["code", "date"]).reset_index(drop=True)

g = panel.groupby("code")["close"]

# momentum: 20-day return (groupby so windows don't cross stocks)
panel["mom_20"] = g.transform(lambda x: x / x.shift(20) - 1)

# reversal: 5-day return (expect a NEGATIVE relation to future return)
panel["rev_5"] = g.transform(lambda x: x / x.shift(5) - 1)

# volatility: 20-day std of daily returns
panel["vol_20"] = g.transform(lambda x: x.pct_change().rolling(20).std())

panel.to_parquet("factor_panel.parquet", index=False)
print(panel[["date", "code", "mom_20", "rev_5", "vol_20"]].dropna().head())
print("shape:", panel.shape, "| NaN in mom_20:", panel["mom_20"].isna().sum())