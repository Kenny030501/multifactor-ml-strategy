import pandas as pd

df = pd.read_csv("factor_600519.csv")
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

# Factor 2: short-term reversal
df["reversal_5d"] = -(df["收盘"] / df["收盘"].shift(5) - 1)

# Factor 3: 20-day volatility
df["daily_return"] = df["收盘"] / df["收盘"].shift(1) - 1
df["volatility_20d"] = df["daily_return"].rolling(20).std()

print("Current factor columns:")
print(df[["日期", "收盘", "momentum_20d", "reversal_5d", "volatility_20d"]].tail(10))

print("\nDescriptive stats of the three factors:")
print(df[["momentum_20d", "reversal_5d", "volatility_20d"]].describe())

df.to_csv("factor_600519.csv", index=False)
print("\nFactor file updated.")