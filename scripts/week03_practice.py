"""Week 3 实践：pandas 基础。

运行：
    python scripts/week03_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 电商订单：olist_orders_45d.csv")
print("=" * 60)

orders = pd.read_csv(
    "data/03-pandas/olist_orders_45d.csv",
    parse_dates=["purchase_date"],
)
print("shape:", orders.shape)
print(orders.dtypes)
print()
daily = (
    orders.groupby("purchase_date")["quantity"]
    .sum()
    .sort_values(ascending=False)
)
print("订单数量最高的日期:")
print(daily.head())

print()
print("=" * 60)
print("2. 教育数据：College.csv")
print("=" * 60)

college = pd.read_csv("data/03-pandas/College.csv")
private = college[college["Private"] == "Yes"]
public = college[college["Private"] == "No"]
print("公立学校数:", len(public), "私立学校数:", len(private))
print()
print("私立学校平均 Outstate:", round(private["Outstate"].mean(), 2))
print("公立学校平均 Outstate:", round(public["Outstate"].mean(), 2))
print("毕业率最高的 5 所学校:")
print(college.sort_values("Grad.Rate", ascending=False).head())

print()
print("=" * 60)
print("3. 汽车数据：Cars93.csv")
print("=" * 60)

cars = pd.read_csv("data/03-pandas/Cars93.csv")
by_type = (
    cars.groupby("Type")["Price"]
    .agg(["mean", "count"])
    .round(2)
    .sort_values("mean", ascending=False)
)
print(by_type)
