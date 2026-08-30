"""05-cleaning-merge / 05-norway-merge-demo：演示不 strip 和 strip 后合并差异。"""
import pandas as pd

make = pd.read_csv("data/06-merge/norway_new_car_sales_by_make.csv")
model = pd.read_csv("data/06-merge/norway_new_car_sales_by_model.csv")

print("不 strip 直接合并行数:",
      len(pd.merge(model, make, on=["Year", "Month", "Make"], how="inner")))

model["Make"] = model["Make"].str.strip()
print("strip 后合并行数:",
      len(pd.merge(model, make, on=["Year", "Month", "Make"], how="inner")))

market = pd.read_csv("data/06-merge/MarketArrivals.csv")
print(market.groupby("state")["quantity"].sum().sort_values(ascending=False).head(3))
print(market.pivot_table(index="month", columns="year", values="quantity", aggfunc="sum").shape)
