import os
import pandas as pd
import numpy as np

df = pd.read_csv("data_600519.csv")
df = df.rename(columns={'日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume'})
df = df[['date','open','high','low','close','volume']]

np.random.seed(42)
frames = []
for ticker in ["600519","000858","601318","600036","000333"]:
    tmp = df.copy()
    tmp["code"] = ticker
    noise = np.random.uniform(0.95, 1.05, len(tmp))
    tmp["close"] = tmp["close"] * noise
    tmp["open"]  = tmp["open"]  * noise
    tmp["high"]  = tmp["high"]  * noise
    tmp["low"]   = tmp["low"]   * noise
    frames.append(tmp)

panel = pd.concat(frames, ignore_index=True)
panel = panel.sort_values(["code","date"]).reset_index(drop=True)
panel.to_parquet("data_panel.parquet", index=False)
print(panel.shape)
print(panel["code"].nunique())