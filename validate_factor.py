import pandas as pd

df = pd.read_csv("factor_600519.csv")
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

df["future_return_20d"] = df["收盘"].shift(-20) / df["收盘"] - 1

df_clean = df.dropna(subset=["momentum_20d", "future_return_20d"])

correlation = df_clean["momentum_20d"].corr(df_clean["future_return_20d"])

print(f"有效样本数:{len(df_clean)}")
print(f"\n动量因子 与 未来20天收益 的相关系数:{correlation:.4f}")

df_clean = df_clean.copy()
df_clean["动量分档"] = pd.qcut(df_clean["momentum_20d"], 5,
                            labels=["最低", "较低", "中等", "较高", "最高"])
print("\n各动量档位 → 平均未来20天收益:")
print(df_clean.groupby("动量分档", observed=True)["future_return_20d"].mean())