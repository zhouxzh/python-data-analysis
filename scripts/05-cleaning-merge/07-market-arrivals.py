"""05-cleaning-merge / 07-market-arrivals：groupby 和 pivot_table 汇总到货数据。"""
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
