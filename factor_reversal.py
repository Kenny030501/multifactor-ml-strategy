import pandas as pd

df = pd.read_csv("factor_600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# Factor 2: short-term reversal
df["reversal_5d"] = -(df["close"] / df["close"].shift(5) - 1)

# Factor 3: 20-day volatility
df["daily_return"] = df["close"] / df["close"].shift(1) - 1
df["volatility_20d"] = df["daily_return"].rolling(20).std()

print("Current factor columns:")
print(df[["date", "close", "momentum_20d", "reversal_5d", "volatility_20d"]].tail(10))

print("\nDescriptive stats of the three factors:")
print(df[["momentum_20d", "reversal_5d", "volatility_20d"]].describe())

df.to_csv("factor_600519.csv", index=False)
print("\nFactor file updated.")