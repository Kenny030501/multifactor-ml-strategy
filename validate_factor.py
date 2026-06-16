import pandas as pd

# 读入带因子的数据
df = pd.read_csv("factor_600519.csv")
df["日期"] = pd.to_datetime(df["日期"])
df = df.sort_values("日期").reset_index(drop=True)

# === 算"未来20天收益率"——注意这里用 shift(-20)！===
# shift(-20) = 往上挪,把"20天后的价格"拉到当前行
# 这是故意用未来数据,但只用于"验证因子",绝不能混进因子本身
df["future_return_20d"] = df["收盘"].shift(-20) / df["收盘"] - 1

# 现在每一行有两个关键数字:
#   momentum_20d      = 过去20天涨了多少(因子,用过去算)
#   future_return_20d = 未来20天会涨多少(我们想预测的目标,用未来算)

# 去掉有空值的行(开头20行没动量,结尾20行没未来收益)
df_clean = df.dropna(subset=["momentum_20d", "future_return_20d"])

# === 核心验证:动量因子和未来收益,有没有关系？===
correlation = df_clean["momentum_20d"].corr(df_clean["future_return_20d"])

print(f"有效样本数:{len(df_clean)}")
print(f"\n动量因子 与 未来20天收益 的相关系数:{correlation:.4f}")
print("\n解读:")
print("  接近 +1 → 过去涨得多,未来也涨得多(动量有效)")
print("  接近  0 → 没关系(因子无效)")
print("  接近 -1 → 过去涨得多,未来反而跌(反转效应)")

# 再看得直观一点:把动量分成5档,看每档对应的平均未来收益
df_clean = df_clean.copy()
df_clean["动量分档"] = pd.qcut(df_clean["momentum_20d"], 5,
                            labels=["最低", "较低", "中等", "较高", "最高"])
print("\n各动量档位 → 平均未来20天收益:")
print(df_clean.groupby("动量分档", observed=True)["future_return_20d"].mean())