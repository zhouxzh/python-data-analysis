"""05-cleaning-merge / 04-market-arrivals：分组汇总和 pivot_table。

运行：
    python scripts/05-cleaning-merge/04-market-arrivals.py
"""
import pandas as pd

market = pd.read_csv("data/06-merge/MarketArrivals.csv")
by_state = (
    market.groupby("state")["quantity"]
    .agg(["sum", "mean", "count"])
    .round(2)
    .sort_values("sum", ascending=False)
)
print("各州到货量汇总前 5:")
print(by_state.head())

pivot = market.pivot_table(
    index="month",
    columns="year",
    values="quantity",
    aggfunc="sum",
)
print("月份 x 年份到货量透视表:")
print(pivot)
