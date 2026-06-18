import pandas as pd

df = pd.read_csv("factor_600519.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

df["future_return_20d"] = df["close"].shift(-20) / df["close"] - 1

df_clean = df.dropna(subset=["momentum_20d", "future_return_20d"])

correlation = df_clean["momentum_20d"].corr(df_clean["future_return_20d"])

print(f"Valid samples: {len(df_clean)}")
print(f"\nCorrelation between momentum and future 20-day return: {correlation:.4f}")

df_clean = df_clean.copy()
df_clean["momentum_bucket"] = pd.qcut(
    df_clean["momentum_20d"], 5,
    labels=["lowest", "low", "medium", "high", "highest"])
print("\nMomentum bucket -> mean future 20-day return:")
print(df_clean.groupby("momentum_bucket", observed=True)["future_return_20d"].mean())
