import pandas as pd

# 读入刚才存的数据
df = pd.read_csv("data_600519.csv")

# 只保留我们需要的两列:日期 和 收盘价,并按日期排序
df = df[["日期", "收盘"]].copy()
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

# === 计算动量因子 ===
# 动量 = 过去20个交易日的累计收益率
# 思路:今天的收盘价 / 20天前的收盘价 - 1
df["momentum_20d"] = df["收盘"] / df["收盘"].shift(20) - 1

# 看看结果
print(df.head(25))
print(f"\n动量因子的统计描述:")
print(df["momentum_20d"].describe())

df.to_csv("factor_600519.csv", index=False)
print("\n已保存带因子的数据 ✅")