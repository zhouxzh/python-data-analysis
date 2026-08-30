"""05-cleaning-merge / 03-norway-merge：清洗品牌名并合并两张汽车销售表。

运行：
    python scripts/05-cleaning-merge/03-norway-merge.py
"""
import pandas as pd

make = pd.read_csv("data/06-merge/norway_new_car_sales_by_make.csv")
model = pd.read_csv("data/06-merge/norway_new_car_sales_by_model.csv")

for frame in (make, model):
    frame["Make"] = frame["Make"].astype(str).str.strip()

merged = model.merge(
    make,
    on=["Year", "Month", "Make"],
    suffixes=("_model", "_make"),
    how="left",
)
print("model shape:", model.shape, "make shape:", make.shape)
print("merged shape:", merged.shape)

make_total = make.groupby("Make")["Quantity"].sum().sort_values(ascending=False)
model_total = model.groupby("Make")["Quantity"].sum().sort_values(ascending=False)
compare = pd.DataFrame({"by_make": make_total, "by_model": model_total}).round(2)
print("按品牌汇总对比前 5:")
print(compare.head())
