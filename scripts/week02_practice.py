"""Week 2 实践：pandas 数据结构与读取。

运行：
    python scripts/week02_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 读取 CSV 并检查数据")
print("=" * 60)

df = pd.read_csv("data/nyc_airbnb.csv")
print("shape:", df.shape)
print()
print(df.head())
print()
print(df.dtypes)
print()
print("缺失值:")
print(df.isna().sum())

print()
print("=" * 60)
print("2. 选列、过滤和新增列")
print("=" * 60)

print("price 前 5 行:")
print(df["price"].head())

print()
print("Manhattan 房源数:")
manhattan = df[df["neighbourhood_group"] == "Manhattan"]
print(len(manhattan))

df2 = df.copy()
df2["high_price"] = df2["price"] > 200
df2 = df2.sort_values("price", ascending=False)

print()
print("价格最高的 5 个房源:")
print(df2[["name", "room_type", "neighbourhood_group", "price"]].head(5))

print()
print("=" * 60)
print("3. Mini case：哪种房型平均价格最高")
print("=" * 60)

summary = (
    df.groupby("room_type")["price"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
    .round(2)
)
print(summary)
print()
print("提醒：mean 是平均值，count 是样本量；只看平均会忽略小样本和异常价格。")
