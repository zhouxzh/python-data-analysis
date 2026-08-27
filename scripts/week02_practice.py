"""Week 2 实践：pandas 数据结构与读取。

运行：
    python scripts/week02_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 读取 CSV 并检查数据")
print("=" * 60)

air = pd.read_csv("data/air_quality_simple.csv")
print("shape:", air.shape)
print()
print(air.head())
print()
print(air.dtypes)
print()
print("缺失值:")
print(air.isna().sum())

print()
print("=" * 60)
print("2. 选列、过滤和新增列")
print("=" * 60)

print("PM25 前 5 行:")
print(air["PM25"].head())

print()
print("PM25 > 50 的城市:")
high = air[air["PM25"] > 50]
print(high[["city", "PM25"]])

air2 = air.copy()
air2["PM25"] = air2["PM25"].fillna(air2["PM25"].median())
air2["pollution_index"] = air2[["PM25", "PM10", "NO2", "SO2"]].sum(axis=1)
air2 = air2.sort_values("pollution_index", ascending=False)

print()
print("污染指数最高的城市:")
print(air2[["city", "pollution_index"]])

print()
print("=" * 60)
print("3. Mini case：PM2.5 最高的 3 个城市")
print("=" * 60)

summary = (
    air.groupby("city")["PM25"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
    .round(1)
)
print(summary)
print()
print("前 3 名（丢弃缺失后）:")
print(summary.dropna().head(3))
