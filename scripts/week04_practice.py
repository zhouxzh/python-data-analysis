"""Week 4 实践：EDA 与提出假设。

运行：
    python scripts/week04_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 读取并检查数据")
print("=" * 60)

df = pd.read_csv("data/nyc_airbnb.csv")
print("shape:", df.shape)
print()
print("缺失值:")
print(df[["neighbourhood_group", "room_type", "price", "reviews_per_month"]]
      .isna().sum())

print()
print("=" * 60)
print("2. 按行政区和房型做 EDA")
print("=" * 60)

print("按行政区:")
print(df.groupby("neighbourhood_group")["price"]
      .agg(["mean", "median", "count"])
      .round(2)
      .sort_values("mean", ascending=False))
print()
print("按房型:")
print(df.groupby("room_type")["price"]
      .agg(["mean", "median", "count"])
      .round(2)
      .sort_values("mean", ascending=False))

print()
print("=" * 60)
print("3. 高价比例与缺失")
print("=" * 60)

df["high_price"] = df["price"] > 200
print("高价房源比例:")
print(df.groupby("room_type")["high_price"]
      .agg(["mean", "count"])
      .round(4))
print()
print("reviews_per_month 缺失数:")
print(df.groupby("room_type")["reviews_per_month"]
      .apply(lambda s: s.isna().sum()))

print()
print("=" * 60)
print("4. 相关性（相关不等于因果）")
print("=" * 60)

print(df[["price", "minimum_nights", "number_of_reviews", "availability_365"]]
      .corr()["price"]
      .round(3))
print()
print("发现：行政区与房型价差明显；缺失和无评论状态必须分开报告。")
