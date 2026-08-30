"""03-pandas / 01-olist-orders：读取电商订单，找订单数量最高的日期。

运行：
    python scripts/03-pandas/01-olist-orders.py
"""
import pandas as pd

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
