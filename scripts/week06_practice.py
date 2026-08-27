"""Week 6 实践：合并、分组与迷你项目。

运行：
    python scripts/week06_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 合并空气数据与城市信息")
print("=" * 60)

air = pd.read_csv("data/air_quality_simple.csv")
info = pd.read_csv("data/city_info.csv")

merged = air.merge(info, on="city", how="left")
print("merged shape:", merged.shape)
print()
print("缺失值:")
print(merged.isna().sum())

print()
print("=" * 60)
print("2. 按 region 汇总")
print("=" * 60)

merged["weighted_pm25"] = merged["PM25"] * merged["population_million"]

region = merged.groupby("region").apply(
    lambda g: pd.Series({
        "cities": g["city"].nunique(),
        "mean_pm25": g["PM25"].mean(),
        "population_weighted_pm25": g["weighted_pm25"].sum() / g["population_million"].sum(),
        "total_population_million": g["population_million"].sum()
    }),
    include_groups=False
).round(2)

print(region)
print()
print("解读：简单平均和人口加权平均不同，报告必须写清指标定义。")
