"""Week 6 实践：合并、分组与迷你项目。

运行：
    python scripts/week06_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 合并 Airbnb 数据与行政区信息")
print("=" * 60)

df = pd.read_csv("data/nyc_airbnb.csv")
info = pd.read_csv("data/nyc_boroughs.csv")

grouped = (
    df.groupby("neighbourhood_group")["price"]
    .agg(listings="size", mean_price="mean", median_price="median")
    .round(2)
    .reset_index()
)

merged = grouped.merge(
    info,
    left_on="neighbourhood_group",
    right_on="borough",
    how="left"
)
print("merged shape:", merged.shape)
print("合并后缺失值:", int(merged.isna().sum().sum()))
print(merged.to_string(index=False))

print()
print("=" * 60)
print("2. 价格与收入、面积的关系")
print("=" * 60)

print(merged[["mean_price", "population_2020",
              "land_area_sqmi", "median_household_income_usd"]]
      .corr()["mean_price"]
      .round(3))
print()
print("解读：只有 5 个行政区，样本太少；相关 0.802 只能作为线索。")
