"""Week 1 实践：从认识数据到第一个数据问题。

运行：
    python scripts/week01_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. Python 基础：列表与循环")
print("=" * 60)

room_types = ["Entire home/apt", "Private room", "Shared room"]
prices = [211.79, 89.78, 70.13]

print("房型列表:", room_types)
print("第一个房型:", room_types[0])
print("最高平均价格:", max(prices))
print("最低平均价格:", min(prices))

total = 0
for p in prices:
    total = total + p
print("平均价格之和:", round(total, 2))

print()
print("=" * 60)
print("2. 读取课程主数据集 nyc_airbnb.csv")
print("=" * 60)

df = pd.read_csv("data/nyc_airbnb.csv")
print("shape:", df.shape)
print()
print(df.dtypes)
print()
print("缺失值:")
print(df.isna().sum())
print()
print(df.head())

print()
print("=" * 60)
print("3. 第一次数据问题：哪种房型平均价格最高")
print("=" * 60)

summary = (
    df.groupby("room_type")["price"]
    .agg(["mean", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
)
print(summary)
print()
print("结论：Entire home/apt 平均价格最高；每个分组都要报告样本量 count。")
