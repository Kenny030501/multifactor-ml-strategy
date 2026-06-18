import pandas as pd

# Load the data saved earlier.
df = pd.read_csv("data_600519.csv")

# Keep only the two columns we need (date and close) and sort by date.
df = df[["date", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# === Momentum factor ===
# Momentum = cumulative return over the past 20 trading days,
# i.e. today's close / close 20 days ago - 1.
df["momentum_20d"] = df["close"] / df["close"].shift(20) - 1

# Inspect the result.
print(df.head(25))
print("\nMomentum factor summary:")
print(df["momentum_20d"].describe())

df.to_csv("factor_600519.csv", index=False)
print("\nSaved data with factor")
