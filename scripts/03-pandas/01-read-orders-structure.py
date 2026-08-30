"""03-pandas / 01-read-orders-structure：读取订单表并查看结构。"""
import pandas as pd

df = pd.read_csv('data/03-pandas/olist_orders_45d.csv')

print(df.shape)
print(df.columns.tolist())
print(df.dtypes)
