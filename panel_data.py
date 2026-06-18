"""
DEPRECATED / DANGEROUS — do not run.

This script builds a *synthetic* panel by copying the single stock 600519 five
times and adding random noise. It was an early scaffold before real data was
available. The project now uses a real 30-stock US panel in data_panel.parquet
(fetched separately), so running this would OVERWRITE the real data with noise.

Kept only for historical reference. The guard below refuses to run if the real
panel already exists.
"""
import os
import sys
import pandas as pd
import numpy as np

if os.path.exists("data_panel.parquet"):
    sys.exit("Refusing to run: data_panel.parquet already exists (would be "
             "overwritten with synthetic data). Delete this guard intentionally "
             "if you really want the synthetic scaffold.")

df = pd.read_csv("data_600519.csv")
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

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