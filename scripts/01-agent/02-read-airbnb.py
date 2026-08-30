"""01-agent / 02-read-airbnb：读取并初步查看课程主数据集。

运行：
    python scripts/01-agent/02-read-airbnb.py
"""
import pandas as pd

df = pd.read_csv("data/01-agent/nyc_airbnb.csv")
print("shape:", df.shape)
print()
print(df.dtypes)
print()
print("缺失值:")
print(df.isna().sum())
print()
print(df.head())
