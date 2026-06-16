import akshare as ak

# 拉贵州茅台(600519)的日线数据,前复权
df = ak.stock_zh_a_hist(
    symbol="600519",
    period="daily",
    start_date="20200101",
    end_date="20241231",
    adjust="qfq"
)

print(df.head())          # 看前5行长什么样
print(f"\n共 {len(df)} 行数据")
print(f"\n列名:{list(df.columns)}")

df.to_csv("data_600519.csv", index=False)
print("\n已保存到 data_600519.csv ✅")