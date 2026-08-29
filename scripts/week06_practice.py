"""Week 6 实践：合并、分组与迷你项目。

运行：
    python scripts/week06_practice.py
"""
import pandas as pd

print("=" * 60)
print("1. 合并两张挪威汽车销售表")
print("=" * 60)

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
compare = pd.DataFrame(
    {"by_make": make_total, "by_model": model_total}
).round(2)
print("按品牌汇总对比前 5:")
print(compare.head())

print()
print("=" * 60)
print("2. 市场到货量分组与透视：MarketArrivals.csv")
print("=" * 60)

market = pd.read_csv("data/06-merge/MarketArrivals.csv")
by_state = (
    market.groupby("state")["quantity"]
    .agg(["sum", "mean", "count"])
    .round(2)
    .sort_values("sum", ascending=False)
)
print("各州到货量汇总:")
print(by_state.head())
pivot = market.pivot_table(
    index="month",
    columns="year",
    values="quantity",
    aggfunc="sum",
)
print("月份 x 年份到货量透视表:")
print(pivot)

print()
print("=" * 60)
print("3. 营销漏斗：email_campaign_funnel.csv")
print("=" * 60)

funnel = pd.read_csv("data/06-merge/email_campaign_funnel.csv")
stage_rows = funnel.groupby("Stage").size()
stage_users = funnel.groupby("Stage")["Users"].sum()
negative_stages = (stage_users < 0).sum()
duplicate_stage18 = int((funnel["Stage"] == "Stage 18: 5th Purchase").sum())
print("各阶段记录数:")
print(stage_rows)
print("各阶段 Users 汇总:")
print(stage_users)
print("负值阶段数:", negative_stages)
print("Stage 18 记录数:", duplicate_stage18)
print("提示：这不是可直接计算转化率的标准漏斗，需要先审计负值和重复阶段。")
